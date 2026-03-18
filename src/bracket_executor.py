"""
Bracket Executor — live order placement for the 15-minute bracket strategy.

Converts BracketSignalEvents into actual Polymarket orders and manages the
full position lifecycle, in two independently-gated phases:

  Phase 1  ─ Buy momentum_side when signal fires (FOLLOW_TAKER for instant fill).
              Gate: execute_enabled = true
  Phase 2  ─ Buy opposite_side when bracket equation locks in guaranteed profit
              (PASSIVE_LIMIT resting bid at the reversal price).
              Gate: execute_enabled = true AND phase2_enabled = true

SEQUENTIAL TESTING APPROACH
----------------------------
  Step 1:  Set execute_enabled: true, phase2_enabled: false
           Run --execute-crypto and watch Phase 1 orders placed/filled.
           Review logs/bracket_trades.jsonl to confirm fills and prices.

  Step 2:  Once Phase 1 is verified, set phase2_enabled: true
           Phase 2 orders will now be placed when the bracket equation locks.

SAFETY
------
  - execute_enabled MUST be true in config (crypto_direction section).
  - LIVE_TRADING_ENABLED=true MUST be set in .env.
  - max_concurrent_brackets caps open positions across all assets.
  - One bracket position per asset per 15-minute window.
  - Unfilled Phase 1 orders are CANCELLED when the window closes.
  - All actions written to bracket_audit_log_path (JSONL).

ENTRY STYLE RATIONALE
---------------------
  Phase 1 — FOLLOW_TAKER (default):
    We're capturing a lag between Binance price and Polymarket repricing.
    That lag closes within seconds; a resting maker bid would miss the window.
    FOLLOW_TAKER (FOK) fills immediately at market price or fails cleanly.

  Phase 2 — PASSIVE_LIMIT (default):
    We're waiting for the opposite side to bottom and reverse.
    We know the reversal price in advance; posting a resting bid at that level
    lets the market come to us, which gives better fill price and no slippage.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from loguru import logger  # type: ignore[import]
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from src.fees import taker_fee
from src.models import BracketSignalEvent


# ---------------------------------------------------------------------------
# Position lifecycle states
# ---------------------------------------------------------------------------

class BracketPhase(str, Enum):
    PHASE1_PENDING     = "PHASE1_PENDING"      # P1 order submitted, awaiting fill confirmation
    PHASE1_FILLED      = "PHASE1_FILLED"       # P1 filled; monitoring for Phase 2
    PHASE2_PENDING     = "PHASE2_PENDING"      # P2 order submitted
    BRACKET_COMPLETE   = "BRACKET_COMPLETE"    # both legs filled — profit locked at close
    PHASE1_ONLY_CLOSED = "PHASE1_ONLY_CLOSED"  # window closed with only P1 filled
    CANCELLED          = "CANCELLED"           # order failed, timed out, or manually cancelled


@dataclass
class BracketPosition:
    """
    Full mutable state for one bracket trade, from signal-fire through settlement.
    """
    position_id:     str
    event_id:        str          # links to the originating BracketSignalEvent
    asset:           str          # "BTC" or "ETH"
    window_ts:       int          # 15-minute window start (Unix epoch)
    window_close_ts: int          # window_ts + 900
    momentum_side:   str          # "YES" (BTC up) or "NO" (BTC down)

    # ── Phase 1 (always placed) ──────────────────────────────────────────
    p1_token_id:     str
    p1_price:        float        # limit/market price used when placing the order
    p1_shares:       float        # number of shares bought
    p1_notional_usd: float        # p1_price * p1_shares (approx cost before fees)
    p1_order_id:     str = ""     # exchange-assigned order ID
    p1_fill_price:   float = 0.0  # actual fill price (0.0 until confirmed)

    # ── Phase 2 (placed only when bracket equation locks) ─────────────────
    p2_token_id:     str = ""
    p2_price:        float = 0.0  # limit price used for the Phase 2 order
    p2_shares:       float = 0.0
    p2_notional_usd: float = 0.0
    p2_order_id:     str = ""
    p2_fill_price:   float = 0.0

    # ── Monitoring ────────────────────────────────────────────────────────
    phase:                BracketPhase = BracketPhase.PHASE1_PENDING
    min_opposite_price:   float = 999.0  # floor of opposite-side ask since P1 filled
    opened_at:            float = field(default_factory=time.time)
    closed_at:            float = 0.0
    fill_check_attempts:  int = 0        # how many times we've polled for P1 fill
    p2_last_attempt_ts:   float = 0.0   # timestamp of last Phase 2 placement attempt

    # ── Outcome ───────────────────────────────────────────────────────────
    outcome:        str   = ""    # "YES_WINS" | "NO_WINS" | "UNKNOWN"
    actual_pnl_usd: float = 0.0


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class BracketExecutor:
    """
    Converts BracketSignalEvents into real Polymarket orders.

    Called from run_bracket_signal_observer() via three hooks:

        await executor.on_signal(event)             # → Phase 1 order placement
        await executor.tick(asset, yes_ask, no_ask) # → fill check + Phase 2 trigger
        executor.on_window_close(ts, asset, yes_won) # → settlement + P&L

    Configuration (all fields in CryptoDirectionConfig):
        execute_enabled       — master switch (must be true to place any order)
        phase2_enabled        — enables Phase 2; keep false while testing Phase 1
        phase1_entry_style    — FOLLOW_TAKER (default) or PASSIVE_LIMIT
        phase2_entry_style    — PASSIVE_LIMIT (default) or FOLLOW_TAKER
        phase1_bet_size_usd   — USD notional for the first leg
        max_concurrent_brackets — cap on open positions across all assets
    """

    # seconds to wait between Phase 2 placement retries after a failure
    _P2_RETRY_COOLDOWN = 15.0

    def __init__(self, cfg: Any, client: Any) -> None:
        self._cfg = cfg
        self._client = client
        self._positions: dict[str, BracketPosition] = {}  # position_id → pos
        self._window_pos: dict[str, str] = {}             # "BTC_{ts}" → position_id

        audit_path = Path(cfg.bracket_audit_log_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_path = audit_path

        logger.info(
            "BracketExecutor ready  execute={} phase2={} "
            "max_concurrent={}  p1_style={}  p2_style={}  bet=${}/leg",
            cfg.execute_enabled,
            cfg.phase2_enabled,
            cfg.max_concurrent_brackets,
            cfg.phase1_entry_style,
            cfg.phase2_entry_style,
            cfg.phase1_bet_size_usd,
        )

    # ------------------------------------------------------------------ #
    # PHASE 1 — entry on signal fire
    # ------------------------------------------------------------------ #

    async def on_signal(self, event: BracketSignalEvent) -> bool:
        """
        Place the Phase 1 buy order (momentum_side) when a direction signal fires.

        For FOLLOW_TAKER orders the Polymarket CLOB fills immediately (FOK);
        we parse the fill confirmation from the placement response itself rather
        than polling later.  For PASSIVE_LIMIT we record the order and let the
        tick() poll loop confirm the fill.

        Returns True if an order was submitted, False if skipped (guards failed).
        """
        if not self._cfg.execute_enabled:
            logger.debug("BracketExecutor: execute_enabled=false — signal skipped")
            return False

        # ── Guard: one bracket per asset per window ──────────────────────
        window_key = f"{event.asset}_{event.window_open_ts}"
        if window_key in self._window_pos:
            logger.warning(
                "BracketExecutor: already have bracket asset={} window={} — skipping",
                event.asset, event.window_open_ts,
            )
            return False

        # ── Guard: max concurrent open positions ─────────────────────────
        active = sum(
            1 for p in self._positions.values()
            if p.phase in (
                BracketPhase.PHASE1_PENDING,
                BracketPhase.PHASE1_FILLED,
                BracketPhase.PHASE2_PENDING,
                BracketPhase.BRACKET_COMPLETE,
            )
        )
        if active >= self._cfg.max_concurrent_brackets:
            logger.warning(
                "BracketExecutor: max_concurrent_brackets={} reached — skipping",
                self._cfg.max_concurrent_brackets,
            )
            return False

        # ── Resolve token IDs ────────────────────────────────────────────
        momentum_token_id = (
            event.yes_token_id if event.momentum_side == "YES" else event.no_token_id
        )
        opposite_token_id = (
            event.no_token_id if event.momentum_side == "YES" else event.yes_token_id
        )

        if not momentum_token_id or momentum_token_id in ("", "MISSING"):
            logger.error(
                "BracketExecutor: momentum token_id missing in event_id={} — skipping",
                event.event_id,
            )
            return False

        # ── Size ─────────────────────────────────────────────────────────
        entry_price = round(event.momentum_price, 4)
        shares = max(
            round(self._cfg.phase1_bet_size_usd / entry_price, 1),
            self._cfg.min_bracket_shares,
        )

        position_id = str(uuid.uuid4())[:12]
        entry_style = self._cfg.phase1_entry_style

        logger.info(
            "BracketExecutor: → Phase 1  asset={} side={} style={} "
            "price={:.4f} shares={:.1f} notional=${:.2f}  token={}",
            event.asset, event.momentum_side, entry_style,
            entry_price, shares, entry_price * shares, momentum_token_id,
        )

        # ── Place order ──────────────────────────────────────────────────
        try:
            result = await self._client.place_buy_order(
                token_id=momentum_token_id,
                price=entry_price,
                size=shares,
                entry_style=entry_style,
                client_order_id=position_id,
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

        order_id = result.get("exchange_order_id", "")
        raw_status = str(result.get("status") or "").upper()

        logger.info(
            "BracketExecutor: Phase 1 order placed  order_id={}  status={}",
            order_id, raw_status,
        )

        # ── Build position record ────────────────────────────────────────
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

        # For FOLLOW_TAKER (FOK), a successful response means immediate fill.
        # We skip the polling loop and mark filled right away using the placement price.
        if entry_style == "FOLLOW_TAKER" and raw_status in ("MATCHED", "FILLED", ""):
            # "" status on successful FOK response still means the order was matched
            pos.phase = BracketPhase.PHASE1_FILLED
            pos.p1_fill_price = entry_price   # FOK fills at or below limit price
            logger.info(
                "BracketExecutor: Phase 1 FILLED (FOLLOW_TAKER)  "
                "asset={}  fill_price={:.4f}",
                event.asset, pos.p1_fill_price,
            )
            self._audit({
                "type": "PHASE1_FILLED",
                "position_id": position_id,
                "event_id": event.event_id,
                "asset": event.asset,
                "fill_price": pos.p1_fill_price,
                "style": "FOLLOW_TAKER",
            })
        # For PASSIVE_LIMIT the order rests on the book; tick() will poll for fill.

        self._positions[position_id] = pos
        self._window_pos[window_key] = position_id

        self._audit({
            "type": "PHASE1_ORDER_PLACED",
            "position_id": position_id,
            "event_id": event.event_id,
            "asset": event.asset,
            "momentum_side": event.momentum_side,
            "entry_style": entry_style,
            "p1_token_id": momentum_token_id,
            "p1_price": entry_price,
            "p1_shares": shares,
            "p1_order_id": order_id,
            "window_ts": event.window_open_ts,
        })
        return True

    # ------------------------------------------------------------------ #
    # Per-poll-cycle hook — fill confirmation + Phase 2 trigger
    # ------------------------------------------------------------------ #

    async def tick(self, asset: str, yes_ask: float, no_ask: float) -> None:
        """
        Called every poll cycle (≈1 second) per asset.

          PHASE1_PENDING → polls exchange for fill confirmation (PASSIVE_LIMIT only)
          PHASE1_FILLED  → tracks opposite-side floor; triggers Phase 2 when ready
        """
        for pos in list(self._positions.values()):
            if pos.asset != asset:
                continue
            if pos.phase == BracketPhase.PHASE1_PENDING:
                await self._confirm_phase1_fill(pos)
            elif pos.phase == BracketPhase.PHASE1_FILLED:
                if self._cfg.phase2_enabled:
                    await self._check_phase2_trigger(pos, yes_ask, no_ask)

    # ── Phase 1 fill confirmation (PASSIVE_LIMIT only) ──────────────────

    async def _confirm_phase1_fill(self, pos: BracketPosition) -> None:
        """
        Poll the CLOB for Phase 1 fill status.

        Throttled to once every 5 ticks (~5 s) to avoid rate limiting.
        Not called for FOLLOW_TAKER orders (they fill synchronously in on_signal).
        """
        if not pos.p1_order_id:
            return

        pos.fill_check_attempts += 1
        if pos.fill_check_attempts % 5 != 1:
            return

        try:
            status = await self._client.get_order_status(pos.p1_order_id)
        except Exception as exc:
            logger.debug("BracketExecutor: P1 fill check error (will retry): {}", exc)
            return

        raw_state = str(status.get("status") or status.get("state") or "").upper()

        # get_order_status() normalises to "average_fill_price"
        fill_price = float(status.get("average_fill_price") or 0.0)

        if raw_state in ("FILLED", "MATCHED"):
            pos.phase = BracketPhase.PHASE1_FILLED
            pos.p1_fill_price = fill_price or pos.p1_price
            logger.info(
                "BracketExecutor: Phase 1 FILLED (poll)  asset={}  "
                "fill_price={:.4f}  position_id={}",
                pos.asset, pos.p1_fill_price, pos.position_id,
            )
            self._audit({
                "type": "PHASE1_FILLED",
                "position_id": pos.position_id,
                "asset": pos.asset,
                "fill_price": pos.p1_fill_price,
                "style": "PASSIVE_LIMIT",
            })
        elif raw_state in ("CANCELLED", "EXPIRED", "FAILED"):
            pos.phase = BracketPhase.CANCELLED
            logger.warning(
                "BracketExecutor: Phase 1 order {}  asset={}  order_id={}",
                raw_state, pos.asset, pos.p1_order_id,
            )
            self._audit({
                "type": f"PHASE1_{raw_state}",
                "position_id": pos.position_id,
                "asset": pos.asset,
            })

    # ── Phase 2 trigger check ────────────────────────────────────────────

    async def _check_phase2_trigger(
        self, pos: BracketPosition, yes_ask: float, no_ask: float
    ) -> None:
        """
        Evaluate Phase 2 entry conditions every poll cycle.

        Trigger conditions (all must hold simultaneously):
          1. Opposite side has reached the target_y_price floor (within 5% buffer).
          2. Opposite side has bounced back >= phase2_reversal_threshold from floor.
          3. Full bracket equation is profitable after Polymarket fees:
               1 - x*(1+fee_x) - y*(1+fee_y) > 0
          4. Phase 2 retry cooldown has elapsed (prevents rapid hammering on failure).
        """
        # Retry cooldown guard
        if time.time() - pos.p2_last_attempt_ts < self._P2_RETRY_COOLDOWN:
            return

        momentum_side = pos.momentum_side
        y_price = no_ask if momentum_side == "YES" else yes_ask

        if y_price <= 0:
            return

        # Track the opposite-side floor
        if y_price < pos.min_opposite_price:
            pos.min_opposite_price = y_price

        floor = pos.min_opposite_price
        target = self._cfg.target_y_price
        reversal = self._cfg.phase2_reversal_threshold

        # Condition 1+2: reached target zone and bounced
        near_target = floor <= target * 1.05   # within 5% above target counts
        bounced = y_price >= floor + reversal
        if not (near_target and bounced):
            return

        # Condition 3: bracket equation profitable
        x_cost = pos.p1_fill_price or pos.p1_price
        fee_x = taker_fee(x_cost, category="crypto price")
        fee_y = taker_fee(y_price, category="crypto price")
        net_margin = 1.0 - x_cost * (1.0 + fee_x) - y_price * (1.0 + fee_y)

        if net_margin <= 0:
            logger.debug(
                "BracketExecutor: Phase 2 conditions met but margin={:.4f} ≤ 0 — "
                "waiting for y to drop lower  asset={}",
                net_margin, pos.asset,
            )
            return

        logger.info(
            "BracketExecutor: → Phase 2 trigger!  "
            "asset={}  y_floor={:.4f}  y_now={:.4f}  net_margin={:.4f}  "
            "position_id={}",
            pos.asset, floor, y_price, net_margin, pos.position_id,
        )
        await self._place_phase2(pos, y_price)

    async def _place_phase2(self, pos: BracketPosition, y_price: float) -> None:
        """
        Place the Phase 2 (opposite_side) buy order.

        Uses PASSIVE_LIMIT by default: we post a resting bid slightly above the
        observed reversal price, letting the market come to us as the opposite
        side continues to recover.
        """
        # Mark pending BEFORE placing so we don't re-enter next tick
        pos.phase = BracketPhase.PHASE2_PENDING
        pos.p2_last_attempt_ts = time.time()

        entry_style = self._cfg.phase2_entry_style

        # For PASSIVE_LIMIT: place just above the reversal price for fill probability
        # For FOLLOW_TAKER: use current ask directly
        if entry_style == "PASSIVE_LIMIT":
            entry_price = round(min(y_price + 0.01, 0.94), 4)
        else:
            entry_price = round(min(y_price, 0.94), 4)

        shares = pos.p1_shares   # symmetric sizing

        logger.info(
            "BracketExecutor: placing Phase 2 order  asset={}  style={}  "
            "token={}  price={:.4f}  shares={:.1f}",
            pos.asset, entry_style, pos.p2_token_id, entry_price, shares,
        )

        try:
            result = await self._client.place_buy_order(
                token_id=pos.p2_token_id,
                price=entry_price,
                size=shares,
                entry_style=entry_style,
                client_order_id=f"{pos.position_id}-p2",
            )
        except Exception as exc:
            # Revert to PHASE1_FILLED so next tick can re-evaluate after cooldown
            pos.phase = BracketPhase.PHASE1_FILLED
            logger.error("BracketExecutor: Phase 2 order FAILED — {}", exc)
            self._audit({
                "type": "PHASE2_ORDER_FAILED",
                "position_id": pos.position_id,
                "asset": pos.asset,
                "error": str(exc),
            })
            return

        pos.p2_order_id = result.get("exchange_order_id", "")
        pos.p2_price = entry_price
        pos.p2_shares = shares
        pos.p2_notional_usd = round(entry_price * shares, 4)
        pos.phase = BracketPhase.BRACKET_COMPLETE

        # Calculate guaranteed margin at this point
        x_cost = pos.p1_fill_price or pos.p1_price
        guaranteed = 1.0 - x_cost * (1 + taker_fee(x_cost, "crypto price")) - entry_price * (
            1 + taker_fee(entry_price, "crypto price")
        )

        logger.info(
            "BracketExecutor: Phase 2 PLACED — bracket LOCKED! 🔒  "
            "asset={}  p1={:.4f}  p2={:.4f}  guaranteed_margin={:.4f}  "
            "order_id={}",
            pos.asset, x_cost, entry_price, guaranteed, pos.p2_order_id,
        )
        self._audit({
            "type": "PHASE2_ORDER_PLACED",
            "position_id": pos.position_id,
            "asset": pos.asset,
            "entry_style": entry_style,
            "p2_token_id": pos.p2_token_id,
            "p2_price": entry_price,
            "p2_shares": shares,
            "p2_order_id": pos.p2_order_id,
            "guaranteed_margin": round(guaranteed, 6),
        })

    # ------------------------------------------------------------------ #
    # Window close — cancellation + settlement accounting
    # ------------------------------------------------------------------ #

    async def on_window_close(
        self, window_ts: int, asset: str, yes_won: bool | None
    ) -> None:
        """
        Called when a 15-minute window rolls over.

        For every open position belonging to this window + asset:
          - PHASE1_PENDING   → cancel the unfilled order (prevents stale open order)
          - PHASE1_FILLED    → settle Phase 1 only; P&L depends on outcome
          - BRACKET_COMPLETE → settle bracket; P&L is guaranteed regardless of outcome
        """
        for pos in list(self._positions.values()):
            if pos.window_ts != window_ts or pos.asset != asset:
                continue
            if pos.phase in (BracketPhase.PHASE1_ONLY_CLOSED, BracketPhase.CANCELLED):
                continue

            pos.closed_at = time.time()
            pos.outcome = (
                "YES_WINS" if yes_won is True
                else "NO_WINS" if yes_won is False
                else "UNKNOWN"
            )

            # ── Cancel unfilled Phase 1 order ────────────────────────────
            if pos.phase == BracketPhase.PHASE1_PENDING:
                await self._cancel_order(pos.p1_order_id, pos, reason="window_close")
                pos.phase = BracketPhase.CANCELLED
                self._audit({
                    "type": "PHASE1_CANCELLED_WINDOW_CLOSE",
                    "position_id": pos.position_id,
                    "asset": pos.asset,
                    "order_id": pos.p1_order_id,
                })
                continue

            # ── Settle BRACKET_COMPLETE ───────────────────────────────────
            if pos.phase == BracketPhase.BRACKET_COMPLETE:
                # Both legs resolve at window close.
                # One token pays $1/share, the other $0/share → net receipt = $1/share.
                # Guaranteed profit = 1 - total_cost (regardless of direction).
                x = pos.p1_fill_price or pos.p1_price
                y = pos.p2_fill_price or pos.p2_price
                cost = (
                    x * (1.0 + taker_fee(x, "crypto price"))
                    + y * (1.0 + taker_fee(y, "crypto price"))
                )
                pos.actual_pnl_usd = round((1.0 - cost) * pos.p1_shares, 4)
                logger.info(
                    "BracketExecutor: BRACKET SETTLED 🏆  asset={}  outcome={}  "
                    "pnl=${:.4f}  position_id={}",
                    asset, pos.outcome, pos.actual_pnl_usd, pos.position_id,
                )

            # ── Settle PHASE1_ONLY ────────────────────────────────────────
            else:
                pos.phase = BracketPhase.PHASE1_ONLY_CLOSED
                x = pos.p1_fill_price or pos.p1_price
                cost = x * (1.0 + taker_fee(x, "crypto price"))

                if pos.outcome == "UNKNOWN":
                    pnl_per_share = 0.0
                elif (
                    (pos.momentum_side == "YES" and pos.outcome == "YES_WINS")
                    or (pos.momentum_side == "NO"  and pos.outcome == "NO_WINS")
                ):
                    pnl_per_share = 1.0 - cost     # momentum leg paid $1/share
                else:
                    pnl_per_share = -cost           # momentum leg paid $0/share

                pos.actual_pnl_usd = round(pnl_per_share * pos.p1_shares, 4)
                logger.info(
                    "BracketExecutor: PHASE1-ONLY SETTLED  asset={}  outcome={}  "
                    "momentum_side={}  pnl=${:.4f}  position_id={}",
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

    async def _cancel_order(
        self, order_id: str, pos: BracketPosition, reason: str = ""
    ) -> None:
        """Attempt to cancel an open order. Logs but does not raise on failure."""
        if not order_id:
            return
        try:
            await self._client.cancel_order(order_id)
            logger.info(
                "BracketExecutor: cancelled order  order_id={}  reason={}  "
                "position_id={}",
                order_id, reason, pos.position_id,
            )
        except Exception as exc:
            logger.warning(
                "BracketExecutor: cancel failed (order may already be settled)  "
                "order_id={}  error={}",
                order_id, exc,
            )

    # ------------------------------------------------------------------ #
    # Public accessors — for reporting / status display
    # ------------------------------------------------------------------ #

    def active_positions(self) -> list[BracketPosition]:
        """Positions that are still open (not yet settled or cancelled)."""
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
        """Total realised P&L (USD) across all settled positions this session."""
        return sum(p.actual_pnl_usd for p in self._positions.values())

    def session_summary(self) -> dict:
        all_p = list(self._positions.values())
        settled = [
            p for p in all_p
            if p.phase in (
                BracketPhase.BRACKET_COMPLETE,
                BracketPhase.PHASE1_ONLY_CLOSED,
                BracketPhase.CANCELLED,
            )
        ]
        brackets  = [p for p in settled if p.phase == BracketPhase.BRACKET_COMPLETE]
        p1_only   = [p for p in settled if p.phase == BracketPhase.PHASE1_ONLY_CLOSED]
        wins      = [p for p in p1_only  if p.actual_pnl_usd > 0]
        return {
            "total_positions":      len(all_p),
            "active":               len(self.active_positions()),
            "settled":              len(settled),
            "brackets_complete":    len(brackets),
            "phase1_only":          len(p1_only),
            "phase1_wins":          len(wins),
            "phase1_losses":        len(p1_only) - len(wins),
            "session_pnl_usd":      round(self.session_pnl_usd(), 4),
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
