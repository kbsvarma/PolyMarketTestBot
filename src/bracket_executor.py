"""
Bracket Executor — live order placement for the 15-minute bracket strategy.

Converts BracketSignalEvents into actual Polymarket orders and manages the
full position lifecycle:

  Phase 1  ─ Buy momentum_side (YES or NO) when signal fires.
  Phase 2  ─ Buy opposite_side when it bottoms, confirms reversal, and the
              bracket equation locks in guaranteed profit:
                  x_cost*(1+fee_x) + y_cost*(1+fee_y) < 1.00
  Settlement – Both legs auto-settle at window close (binary markets pay $1/$0).

SAFETY
------
  - `execute_enabled` in config MUST be set to true explicitly.
  - `LIVE_TRADING_ENABLED=true` MUST be in the .env file.
  - Max one bracket position per asset per window.
  - `max_concurrent_brackets` cap across all assets.
  - All actions are written to `bracket_audit_log_path` (JSONL).

Integration
-----------
Pass an instance of BracketExecutor to run_bracket_signal_observer():

    executor = BracketExecutor(config.crypto_direction, client)
    await run_bracket_signal_observer(config, client, executor=executor)

The observer calls:
    await executor.on_signal(event)           # Phase 1 on signal fire
    await executor.tick(asset, yes_ask, no_ask)  # every poll cycle
    executor.on_window_close(window_ts, asset, yes_won)  # on rollover
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

try:
    from loguru import logger  # type: ignore[import]
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from src.fees import taker_fee
from src.models import BracketSignalEvent


# ---------------------------------------------------------------------------
# Position lifecycle
# ---------------------------------------------------------------------------

class BracketPhase(str, Enum):
    PHASE1_PENDING    = "PHASE1_PENDING"      # order submitted, awaiting fill
    PHASE1_FILLED     = "PHASE1_FILLED"       # P1 filled; monitoring for P2
    PHASE2_PENDING    = "PHASE2_PENDING"      # P2 order submitted
    BRACKET_COMPLETE  = "BRACKET_COMPLETE"    # both legs filled — profit locked
    PHASE1_ONLY_CLOSED = "PHASE1_ONLY_CLOSED" # window closed with only P1 filled
    CANCELLED         = "CANCELLED"           # order failed / timed out


@dataclass
class BracketPosition:
    """Full state for one bracket trade."""
    position_id:    str
    event_id:       str          # links to BracketSignalEvent
    asset:          str          # "BTC" or "ETH"
    window_ts:      int
    window_close_ts: int
    momentum_side:  str          # "YES" or "NO"

    # Phase 1
    p1_token_id:     str
    p1_price:        float       # limit price placed
    p1_shares:       float
    p1_notional_usd: float
    p1_order_id:     str = ""
    p1_fill_price:   float = 0.0

    # Phase 2 — populated when reversal confirmed
    p2_token_id:     str = ""
    p2_price:        float = 0.0
    p2_shares:       float = 0.0
    p2_notional_usd: float = 0.0
    p2_order_id:     str = ""
    p2_fill_price:   float = 0.0

    # Monitoring
    phase:               BracketPhase = BracketPhase.PHASE1_PENDING
    min_opposite_price:  float = 999.0   # floor of opposite side after P1 placed
    opened_at:           float = field(default_factory=time.time)
    closed_at:           float = 0.0
    fill_check_attempts: int = 0

    # Outcome
    outcome:        str   = ""    # "YES_WINS" | "NO_WINS" | "UNKNOWN"
    actual_pnl_usd: float = 0.0


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class BracketExecutor:
    """
    Manages live order placement and position tracking for the bracket strategy.

    Designed to be called from run_bracket_signal_observer() on three hooks:
      - on_signal(event)                    → Phase 1 order
      - tick(asset, yes_ask, no_ask)        → Phase 2 monitoring / fill checks
      - on_window_close(ts, asset, yes_won) → settlement + P&L accounting
    """

    def __init__(self, cfg, client) -> None:
        """
        cfg    : CryptoDirectionConfig (from AppConfig.crypto_direction)
        client : PolymarketClient instance (must have place_buy_order,
                 get_order_status)
        """
        self._cfg = cfg
        self._client = client
        self._positions: dict[str, BracketPosition] = {}    # position_id → pos
        self._window_pos:  dict[str, str] = {}              # "BTC_173..." → position_id

        audit_path = Path(cfg.bracket_audit_log_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_path = audit_path

        logger.info(
            "BracketExecutor initialised execute_enabled={} max_concurrent={}",
            cfg.execute_enabled, cfg.max_concurrent_brackets,
        )

    # ------------------------------------------------------------------ #
    # Phase 1 — entry on signal fire
    # ------------------------------------------------------------------ #

    async def on_signal(self, event: BracketSignalEvent) -> bool:
        """
        Place Phase 1 (momentum_side) buy order when a signal fires.

        Returns True if an order was submitted, False if skipped.
        Token IDs (yes_token_id, no_token_id) come from the signal event itself.
        """
        if not self._cfg.execute_enabled:
            logger.debug("BracketExecutor: execute_enabled=false — signal ignored")
            return False

        # One bracket per asset per window
        window_key = f"{event.asset}_{event.window_open_ts}"
        if window_key in self._window_pos:
            logger.warning(
                "BracketExecutor: duplicate signal asset={} window_ts={} — skipping",
                event.asset, event.window_open_ts,
            )
            return False

        # Max concurrent positions
        active_count = sum(
            1 for p in self._positions.values()
            if p.phase in (
                BracketPhase.PHASE1_PENDING,
                BracketPhase.PHASE1_FILLED,
                BracketPhase.PHASE2_PENDING,
                BracketPhase.BRACKET_COMPLETE,
            )
        )
        if active_count >= self._cfg.max_concurrent_brackets:
            logger.warning(
                "BracketExecutor: max_concurrent_brackets={} reached ({} active) — skipping",
                self._cfg.max_concurrent_brackets, active_count,
            )
            return False

        # Resolve token IDs from the signal event
        momentum_token_id = (
            event.yes_token_id if event.momentum_side == "YES" else event.no_token_id
        )
        opposite_token_id = (
            event.no_token_id if event.momentum_side == "YES" else event.yes_token_id
        )

        if not momentum_token_id or momentum_token_id == "MISSING":
            logger.error(
                "BracketExecutor: momentum token_id is missing in signal event_id={}",
                event.event_id,
            )
            return False

        # Size
        entry_price = round(event.momentum_price, 4)
        shares = max(
            round(self._cfg.phase1_bet_size_usd / entry_price, 1),
            self._cfg.min_bracket_shares,
        )

        import uuid
        position_id = str(uuid.uuid4())[:12]

        logger.info(
            "BracketExecutor: Phase 1 — asset={} side={} price={} shares={} "
            "notional=${:.2f} token={}",
            event.asset, event.momentum_side, entry_price, shares,
            entry_price * shares, momentum_token_id,
        )

        # Place order
        order_id = ""
        try:
            result = await self._client.place_buy_order(
                token_id=momentum_token_id,
                price=entry_price,
                size=shares,
                entry_style=self._cfg.bracket_entry_style,
                client_order_id=position_id,
            )
            order_id = result.get("exchange_order_id", "")
            logger.info(
                "BracketExecutor: Phase 1 order placed order_id={} status={}",
                order_id, result.get("status"),
            )
        except Exception as exc:
            logger.error("BracketExecutor: Phase 1 order FAILED — {}", exc)
            self._audit({
                "type": "PHASE1_ORDER_FAILED",
                "position_id": position_id,
                "event_id": event.event_id,
                "asset": event.asset,
                "error": str(exc),
            })
            return False

        # Record position
        pos = BracketPosition(
            position_id=position_id,
            event_id=event.event_id,
            asset=event.asset,
            window_ts=event.window_open_ts,
            window_close_ts=event.window_close_ts,
            momentum_side=event.momentum_side,
            p1_token_id=momentum_token_id,
            p1_price=entry_price,
            p1_shares=shares,
            p1_notional_usd=round(entry_price * shares, 4),
            p1_order_id=order_id,
            p2_token_id=opposite_token_id,
        )
        self._positions[position_id] = pos
        self._window_pos[window_key] = position_id

        self._audit({
            "type": "PHASE1_ORDER_PLACED",
            "position_id": position_id,
            "event_id": event.event_id,
            "asset": event.asset,
            "momentum_side": event.momentum_side,
            "p1_token_id": momentum_token_id,
            "p1_price": entry_price,
            "p1_shares": shares,
            "p1_order_id": order_id,
            "window_ts": event.window_open_ts,
        })
        return True

    # ------------------------------------------------------------------ #
    # Poll-cycle hook — fill checks + Phase 2 trigger
    # ------------------------------------------------------------------ #

    async def tick(self, asset: str, yes_ask: float, no_ask: float) -> None:
        """
        Called every poll cycle per asset.

        - PHASE1_PENDING : checks if the order has been filled
        - PHASE1_FILLED  : tracks opposite price and triggers Phase 2 if ready
        """
        for pos in list(self._positions.values()):
            if pos.asset != asset:
                continue
            if pos.phase == BracketPhase.PHASE1_PENDING:
                await self._check_phase1_fill(pos)
            elif pos.phase == BracketPhase.PHASE1_FILLED:
                await self._check_phase2_trigger(pos, yes_ask, no_ask)

    async def _check_phase1_fill(self, pos: BracketPosition) -> None:
        """Poll the exchange for Phase 1 fill status."""
        if not pos.p1_order_id:
            return

        # Don't hammer the API — check at most every 5 ticks (5 seconds at 1s poll)
        pos.fill_check_attempts += 1
        if pos.fill_check_attempts % 5 != 1:
            return

        try:
            status = await self._client.get_order_status(pos.p1_order_id)
            state = str(status.get("status") or status.get("state") or "").upper()
            fill = float(
                status.get("fill_price")
                or status.get("avg_price")
                or status.get("avgPrice")
                or 0.0
            )

            if state in ("FILLED", "MATCHED"):
                pos.phase = BracketPhase.PHASE1_FILLED
                pos.p1_fill_price = fill or pos.p1_price
                logger.info(
                    "BracketExecutor: Phase 1 FILLED asset={} fill_price={:.4f} position_id={}",
                    pos.asset, pos.p1_fill_price, pos.position_id,
                )
                self._audit({
                    "type": "PHASE1_FILLED",
                    "position_id": pos.position_id,
                    "asset": pos.asset,
                    "fill_price": pos.p1_fill_price,
                })
            elif state in ("CANCELLED", "EXPIRED", "FAILED"):
                pos.phase = BracketPhase.CANCELLED
                logger.warning(
                    "BracketExecutor: Phase 1 order {} asset={} order_id={}",
                    state, pos.asset, pos.p1_order_id,
                )
                self._audit({
                    "type": f"PHASE1_{state}",
                    "position_id": pos.position_id,
                    "asset": pos.asset,
                })
        except Exception as exc:
            logger.debug("BracketExecutor: fill check failed (will retry): {}", exc)

    async def _check_phase2_trigger(
        self, pos: BracketPosition, yes_ask: float, no_ask: float
    ) -> None:
        """
        Evaluate Phase 2 entry conditions and place order if ready.

        Conditions (all must hold):
          1. opposite side has reached (or passed) target_y_price
          2. it has bounced back >= phase2_reversal_threshold from its floor
          3. the full bracket equation is profitable after fees
        """
        momentum_side = pos.momentum_side
        y_price = no_ask if momentum_side == "YES" else yes_ask

        if y_price <= 0:
            return

        # Update the floor
        if y_price < pos.min_opposite_price:
            pos.min_opposite_price = y_price

        floor = pos.min_opposite_price
        target = self._cfg.target_y_price
        reversal = self._cfg.phase2_reversal_threshold

        # Has y reached (within 5% of) target and bounced?
        near_target = floor <= target * 1.05
        bounced = y_price >= floor + reversal
        if not (near_target and bounced):
            return

        # Verify the bracket equation is profitable
        x_cost = pos.p1_fill_price or pos.p1_price
        fee_x = taker_fee(x_cost, category="crypto price")
        fee_y = taker_fee(y_price, category="crypto price")
        net_margin = 1.0 - x_cost * (1.0 + fee_x) - y_price * (1.0 + fee_y)

        if net_margin <= 0:
            logger.debug(
                "BracketExecutor: Phase 2 conditions met but margin negative "
                "margin={:.4f} asset={} — waiting for lower y",
                net_margin, pos.asset,
            )
            return

        logger.info(
            "BracketExecutor: Phase 2 trigger asset={} y_floor={:.4f} y_now={:.4f} "
            "net_margin={:.4f} position_id={}",
            pos.asset, floor, y_price, net_margin, pos.position_id,
        )
        await self._place_phase2(pos, y_price)

    async def _place_phase2(self, pos: BracketPosition, y_price: float) -> None:
        """Place Phase 2 (opposite_side) buy order."""
        # Mark pending immediately to prevent re-entry next tick
        pos.phase = BracketPhase.PHASE2_PENDING

        # Add a small buffer above the reversal price for fill probability
        entry_price = round(min(y_price + 0.005, 0.95), 4)
        shares = pos.p1_shares  # symmetric sizing

        logger.info(
            "BracketExecutor: placing Phase 2 order asset={} token={} "
            "price={:.4f} shares={:.1f}",
            pos.asset, pos.p2_token_id, entry_price, shares,
        )

        try:
            result = await self._client.place_buy_order(
                token_id=pos.p2_token_id,
                price=entry_price,
                size=shares,
                entry_style=self._cfg.bracket_entry_style,
                client_order_id=f"{pos.position_id}-p2",
            )
            pos.p2_order_id = result.get("exchange_order_id", "")
            pos.p2_price = entry_price
            pos.p2_shares = shares
            pos.p2_notional_usd = round(entry_price * shares, 4)
            pos.phase = BracketPhase.BRACKET_COMPLETE

            logger.info(
                "BracketExecutor: Phase 2 PLACED order_id={} — bracket LOCKED! "
                "asset={} total_cost={:.4f} guaranteed_margin={:.4f}",
                pos.p2_order_id, pos.asset,
                (pos.p1_fill_price or pos.p1_price) + entry_price,
                1.0 - (pos.p1_fill_price or pos.p1_price)
                    * (1 + taker_fee(pos.p1_fill_price or pos.p1_price, "crypto price"))
                    - entry_price
                    * (1 + taker_fee(entry_price, "crypto price")),
            )
            self._audit({
                "type": "PHASE2_ORDER_PLACED",
                "position_id": pos.position_id,
                "asset": pos.asset,
                "p2_token_id": pos.p2_token_id,
                "p2_price": entry_price,
                "p2_shares": shares,
                "p2_order_id": pos.p2_order_id,
            })

        except Exception as exc:
            # Revert so we retry next tick
            pos.phase = BracketPhase.PHASE1_FILLED
            logger.error("BracketExecutor: Phase 2 order FAILED — {}", exc)
            self._audit({
                "type": "PHASE2_ORDER_FAILED",
                "position_id": pos.position_id,
                "asset": pos.asset,
                "error": str(exc),
            })

    # ------------------------------------------------------------------ #
    # Window close — settlement accounting
    # ------------------------------------------------------------------ #

    def on_window_close(
        self, window_ts: int, asset: str, yes_won: bool | None
    ) -> None:
        """
        Called when a 15-minute window closes.

        Sets the outcome on all open positions for this asset+window and
        calculates actual P&L.  Binary markets auto-settle so no sell orders
        are needed — one leg pays $1.00 per share, the other pays $0.
        """
        for pos in list(self._positions.values()):
            if pos.window_ts != window_ts or pos.asset != asset:
                continue
            if pos.phase in (BracketPhase.PHASE1_ONLY_CLOSED, BracketPhase.CANCELLED):
                continue

            pos.closed_at = time.time()
            if yes_won is True:
                pos.outcome = "YES_WINS"
            elif yes_won is False:
                pos.outcome = "NO_WINS"
            else:
                pos.outcome = "UNKNOWN"

            if pos.phase == BracketPhase.BRACKET_COMPLETE:
                # Both legs filled — guaranteed profit regardless of which side wins.
                # One token pays $1/share, the other pays $0/share → net receipt = $1/share.
                # Cost = p1_cost*(1+fee_p1) + p2_cost*(1+fee_p2)
                x = pos.p1_fill_price or pos.p1_price
                y = pos.p2_fill_price or pos.p2_price
                cost = x * (1.0 + taker_fee(x, "crypto price")) + y * (
                    1.0 + taker_fee(y, "crypto price")
                )
                pos.actual_pnl_usd = round((1.0 - cost) * pos.p1_shares, 4)
                logger.info(
                    "BracketExecutor: BRACKET SETTLED asset={} outcome={} "
                    "pnl=${:.4f} position_id={}",
                    asset, pos.outcome, pos.actual_pnl_usd, pos.position_id,
                )

            else:
                # Phase 1 only — win or loss depends on which side won
                pos.phase = BracketPhase.PHASE1_ONLY_CLOSED
                x = pos.p1_fill_price or pos.p1_price
                cost = x * (1.0 + taker_fee(x, "crypto price"))

                if pos.outcome == "UNKNOWN":
                    pnl_per_share = 0.0
                elif (
                    (pos.momentum_side == "YES" and pos.outcome == "YES_WINS")
                    or (pos.momentum_side == "NO" and pos.outcome == "NO_WINS")
                ):
                    pnl_per_share = 1.0 - cost   # momentum leg paid $1
                else:
                    pnl_per_share = -cost         # momentum leg paid $0

                pos.actual_pnl_usd = round(pnl_per_share * pos.p1_shares, 4)
                logger.info(
                    "BracketExecutor: PHASE1-ONLY SETTLED asset={} outcome={} "
                    "momentum_side={} pnl=${:.4f} position_id={}",
                    asset, pos.outcome, pos.momentum_side,
                    pos.actual_pnl_usd, pos.position_id,
                )

            self._audit({
                "type": "POSITION_SETTLED",
                "position_id": pos.position_id,
                "asset": pos.asset,
                "window_ts": window_ts,
                "phase": pos.phase.value,
                "outcome": pos.outcome,
                "actual_pnl_usd": pos.actual_pnl_usd,
                "p1_price": pos.p1_price,
                "p1_fill_price": pos.p1_fill_price,
                "p2_price": pos.p2_price,
                "p2_fill_price": pos.p2_fill_price,
            })

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #

    def active_positions(self) -> list[BracketPosition]:
        """Positions that are still open (not yet settled)."""
        return [
            p for p in self._positions.values()
            if p.phase in (
                BracketPhase.PHASE1_PENDING,
                BracketPhase.PHASE1_FILLED,
                BracketPhase.PHASE2_PENDING,
                BracketPhase.BRACKET_COMPLETE,
            )
        ]

    def session_pnl_usd(self) -> float:
        """Total realised P&L across all settled positions this session."""
        return sum(p.actual_pnl_usd for p in self._positions.values())

    def session_summary(self) -> dict:
        all_pos = list(self._positions.values())
        settled = [
            p for p in all_pos
            if p.phase in (
                BracketPhase.BRACKET_COMPLETE,
                BracketPhase.PHASE1_ONLY_CLOSED,
                BracketPhase.CANCELLED,
            )
        ]
        brackets = [p for p in settled if p.phase == BracketPhase.BRACKET_COMPLETE]
        phase1_only = [p for p in settled if p.phase == BracketPhase.PHASE1_ONLY_CLOSED]
        wins = [p for p in phase1_only if p.actual_pnl_usd > 0]
        return {
            "total_positions": len(all_pos),
            "active": len(self.active_positions()),
            "settled": len(settled),
            "brackets_complete": len(brackets),
            "phase1_only": len(phase1_only),
            "phase1_wins": len(wins),
            "phase1_losses": len(phase1_only) - len(wins),
            "session_pnl_usd": round(self.session_pnl_usd(), 4),
        }

    # ------------------------------------------------------------------ #
    # Audit log
    # ------------------------------------------------------------------ #

    def _audit(self, record: dict) -> None:
        record.setdefault("ts", time.time())
        record.setdefault("ts_iso", datetime.now(timezone.utc).isoformat())
        try:
            with self._audit_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("BracketExecutor: audit write failed: {}", exc)
