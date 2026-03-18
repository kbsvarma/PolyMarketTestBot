"""
Bracket Executor — live order placement for the 15-minute bracket strategy.

Full lifecycle, both phases, end-to-end:

  Phase 1  ─ Buy momentum_side (YES if BTC up, NO if BTC down) when the
              direction signal fires.  Entry style: FOLLOW_TAKER (FOK).
              A successful placement response means the order filled immediately.
              On failure the position is not created; the window continues.

  Phase 2  ─ Buy opposite_side once the bracket equation locks profit:
                  x_cost*(1+fee_x) + y_cost*(1+fee_y) < 1.00
              Watches every poll cycle for the opposite side to bottom and
              bounce by phase2_reversal_threshold.  Entry style: FOLLOW_TAKER.
              On placement success, BRACKET_COMPLETE is set immediately.

  Close    ─ Binary markets auto-resolve at window close.  Both legs pay
              $1 or $0 per share; net receipt = $1 regardless of direction.
              Profit = 1 - x*(1+fee_x) - y*(1+fee_y) per share.

SAFETY GATES (in order)
-----------------------
  1. execute_enabled must be true in config.yaml (crypto_direction section)
  2. LIVE_TRADING_ENABLED=true must be in .env
  3. max_concurrent_brackets caps total open positions
  4. One bracket per asset per 15-minute window
  5. Unfilled Phase 1 orders are CANCELLED at window close
  6. Pending Phase 2 orders are CANCELLED at window close
  7. 15-second retry cooldown on Phase 2 failures

ENTRY STYLE
-----------
Both phases use FOLLOW_TAKER (Fill-Or-Kill):
  - Phase 1: Speed matters — we're capturing a Binance-to-Polymarket lag that
    closes in seconds.  FOLLOW_TAKER fills instantly or fails cleanly.
  - Phase 2: We want immediate lock-in the moment the bracket equation turns
    positive.  A resting PASSIVE_LIMIT bid at the reversal price could miss the
    fill window if the opposite side quickly bounces back above our bid.

All actions written to logs/bracket_trades.jsonl.
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
    PHASE1_PENDING     = "PHASE1_PENDING"      # Phase 1 order submitted (PASSIVE_LIMIT waiting)
    PHASE1_FILLED      = "PHASE1_FILLED"       # Phase 1 confirmed filled; monitoring for Phase 2
    PHASE2_PENDING     = "PHASE2_PENDING"      # Phase 2 order submitted but not yet confirmed
    BRACKET_COMPLETE   = "BRACKET_COMPLETE"    # Both legs filled — profit locked at window close
    PHASE1_ONLY_CLOSED = "PHASE1_ONLY_CLOSED"  # Window closed with Phase 1 filled, no Phase 2
    CANCELLED          = "CANCELLED"           # Order failed, timed out, or window-close cancelled


@dataclass
class BracketPosition:
    """
    Complete mutable state for one bracket position, from signal-fire
    through settlement.  One instance per 15-minute window per asset.
    """
    position_id:     str
    event_id:        str          # links to the originating BracketSignalEvent
    asset:           str          # "BTC" or "ETH"
    window_ts:       int          # 15-minute window start (Unix epoch seconds)
    window_close_ts: int          # window_ts + 900
    momentum_side:   str          # "YES" (BTC/ETH up) or "NO" (BTC/ETH down)

    # ── Phase 1 ──────────────────────────────────────────────────────────
    p1_token_id:     str
    p1_price:        float        # price at which the order was placed
    p1_shares:       float        # number of shares
    p1_notional_usd: float        # p1_price * p1_shares
    p1_order_id:     str = ""
    p1_fill_price:   float = 0.0  # actual fill price (= p1_price for FOLLOW_TAKER)

    # ── Phase 2 ──────────────────────────────────────────────────────────
    p2_token_id:     str = ""
    p2_price:        float = 0.0
    p2_shares:       float = 0.0
    p2_notional_usd: float = 0.0
    p2_order_id:     str = ""
    p2_fill_price:   float = 0.0  # = p2_price for FOLLOW_TAKER

    # ── Runtime monitoring ───────────────────────────────────────────────
    phase:                BracketPhase = BracketPhase.PHASE1_PENDING
    min_opposite_price:   float = 999.0  # running floor of opposite-side ask
    opened_at:            float = field(default_factory=time.time)
    closed_at:            float = 0.0
    fill_check_attempts:  int = 0        # poll counter for PASSIVE_LIMIT P1
    p2_last_attempt_ts:   float = 0.0   # last Phase 2 placement attempt time

    # ── Outcome (filled at window close) ─────────────────────────────────
    outcome:        str   = ""    # "YES_WINS" | "NO_WINS" | "UNKNOWN"
    actual_pnl_usd: float = 0.0


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class BracketExecutor:
    """
    Converts BracketSignalEvents into real Polymarket orders and manages
    the complete bracket position lifecycle.

    Called from run_bracket_signal_observer() via three hooks:

        await executor.on_signal(event)              # signal fires → Phase 1
        await executor.tick(asset, yes_ask, no_ask)  # every 1s → P1 fill + P2 trigger
        await executor.on_window_close(ts, asset, yes_won)  # window rolls → settle
    """

    _P2_RETRY_COOLDOWN = 15.0   # seconds between Phase 2 retry attempts per position

    def __init__(self, cfg: Any, client: Any) -> None:
        self._cfg = cfg
        self._client = client
        self._positions: dict[str, BracketPosition] = {}  # position_id → pos
        self._window_pos: dict[str, str] = {}             # "BTC_{ts}" → position_id

        audit_path = Path(cfg.bracket_audit_log_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_path = audit_path

        logger.info(
            "BracketExecutor ready  "
            "execute={} phase2={} max_concurrent={}  "
            "p1_style={} p2_style={}  bet=${}/leg",
            cfg.execute_enabled,
            cfg.phase2_enabled,
            cfg.max_concurrent_brackets,
            cfg.phase1_entry_style,
            cfg.phase2_entry_style,
            cfg.phase1_bet_size_usd,
        )

    # ================================================================== #
    # PHASE 1 — entry on signal fire
    # ================================================================== #

    async def on_signal(self, event: BracketSignalEvent) -> bool:
        """
        Place the Phase 1 buy order (momentum_side) when a signal fires.

        FOLLOW_TAKER (default): FOK order fills immediately on a successful
        response from the CLOB.  No exception = filled.  We mark PHASE1_FILLED
        right away without polling.

        PASSIVE_LIMIT (fallback): order rests on the book.  tick() polls for
        fill confirmation via get_order_status().

        Returns True if an order was submitted, False if skipped.
        """
        if not self._cfg.execute_enabled:
            logger.debug("BracketExecutor: execute_enabled=false — signal skipped")
            return False

        # ── Guard: one bracket per asset per window ──────────────────────
        window_key = f"{event.asset}_{event.window_open_ts}"
        if window_key in self._window_pos:
            logger.warning(
                "BracketExecutor: bracket already open  asset={} window={} — skip",
                event.asset, event.window_open_ts,
            )
            return False

        # ── Guard: max concurrent positions ──────────────────────────────
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
                "BracketExecutor: max_concurrent_brackets={} reached — skip",
                self._cfg.max_concurrent_brackets,
            )
            return False

        # ── Token IDs from signal event ───────────────────────────────────
        momentum_token_id = (
            event.yes_token_id if event.momentum_side == "YES" else event.no_token_id
        )
        opposite_token_id = (
            event.no_token_id if event.momentum_side == "YES" else event.yes_token_id
        )

        if not momentum_token_id or momentum_token_id in ("", "MISSING"):
            logger.error(
                "BracketExecutor: momentum token missing  event_id={} — skip",
                event.event_id,
            )
            return False

        # ── Size: USD notional ÷ unit price ──────────────────────────────
        entry_price = round(event.momentum_price, 4)
        shares = max(
            round(self._cfg.phase1_bet_size_usd / entry_price, 1),
            self._cfg.min_bracket_shares,
        )
        position_id = str(uuid.uuid4())[:12]
        entry_style = self._cfg.phase1_entry_style

        logger.info(
            "BracketExecutor: ▶ Phase 1  asset={} side={} style={} "
            "price={:.4f} shares={:.1f} notional=${:.2f}",
            event.asset, event.momentum_side, entry_style,
            entry_price, shares, entry_price * shares,
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
            logger.error("BracketExecutor: Phase 1 FAILED — {}", exc)
            self._audit({
                "type": "PHASE1_ORDER_FAILED",
                "position_id": position_id,
                "event_id": event.event_id,
                "asset": event.asset,
                "entry_price": entry_price,
                "shares": shares,
                "error": str(exc),
            })
            return False

        order_id = result.get("exchange_order_id", "")
        raw_status = str(result.get("status") or "").upper()
        logger.info(
            "BracketExecutor: Phase 1 placed  order_id={}  raw_status={}",
            order_id, raw_status,
        )

        # ── Record position ───────────────────────────────────────────────
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

        # FIX #3: FOLLOW_TAKER = FOK.  No exception → order matched and filled.
        # Do not gate on raw_status — Polymarket may return "LIVE", "MATCHED",
        # or any other string.  The absence of an exception is the fill signal.
        if entry_style == "FOLLOW_TAKER":
            pos.phase = BracketPhase.PHASE1_FILLED
            pos.p1_fill_price = entry_price   # FOK fills at ≤ limit price
            logger.info(
                "BracketExecutor: Phase 1 FILLED (FOLLOW_TAKER)  "
                "asset={}  fill_price={:.4f}  position_id={}",
                event.asset, pos.p1_fill_price, position_id,
            )
            self._audit({
                "type": "PHASE1_FILLED",
                "position_id": position_id,
                "event_id": event.event_id,
                "asset": event.asset,
                "fill_price": pos.p1_fill_price,
                "style": "FOLLOW_TAKER",
            })
        # PASSIVE_LIMIT: stays PHASE1_PENDING; tick() polls for fill.

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
            "window_close_ts": event.window_close_ts,
        })
        return True

    # ================================================================== #
    # Per-poll hook — Phase 1 fill polling + Phase 2 trigger
    # ================================================================== #

    async def tick(self, asset: str, yes_ask: float, no_ask: float) -> None:
        """
        Called every poll cycle (~1 s) per asset.

          PHASE1_PENDING → poll CLOB for fill (PASSIVE_LIMIT only)
          PHASE1_FILLED  → update opposite-side floor; trigger Phase 2 if ready
        """
        for pos in list(self._positions.values()):
            if pos.asset != asset:
                continue
            if pos.phase == BracketPhase.PHASE1_PENDING:
                await self._poll_phase1_fill(pos)
            elif pos.phase == BracketPhase.PHASE1_FILLED:
                if self._cfg.phase2_enabled:
                    await self._check_phase2(pos, yes_ask, no_ask)

    # ── Phase 1 fill polling (PASSIVE_LIMIT only) ────────────────────────

    async def _poll_phase1_fill(self, pos: BracketPosition) -> None:
        """
        Poll the CLOB for Phase 1 fill status.  Throttled to every 5 ticks
        (~5 s) to avoid rate-limiting.  Not used for FOLLOW_TAKER positions
        (they are already PHASE1_FILLED by the time they leave on_signal).
        """
        if not pos.p1_order_id:
            return

        pos.fill_check_attempts += 1
        if pos.fill_check_attempts % 5 != 1:
            return   # skip this tick

        try:
            status = await self._client.get_order_status(pos.p1_order_id)
        except Exception as exc:
            logger.debug("BracketExecutor: P1 fill poll error (retry later): {}", exc)
            return

        raw_state = str(status.get("status") or status.get("state") or "").upper()
        # FIX #4: get_order_status() normalises to "average_fill_price"
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
                "BracketExecutor: Phase 1 order {} asset={} order_id={}",
                raw_state, pos.asset, pos.p1_order_id,
            )
            self._audit({
                "type": f"PHASE1_{raw_state}",
                "position_id": pos.position_id,
                "asset": pos.asset,
            })

    # ── Phase 2 — trigger evaluation ─────────────────────────────────────

    async def _check_phase2(
        self, pos: BracketPosition, yes_ask: float, no_ask: float
    ) -> None:
        """
        Evaluate Phase 2 entry conditions every poll cycle.

        Trigger logic:
          1. Track running minimum of opposite-side ask (the floor).
             FIX #1: always update floor BEFORE the cooldown gate so we
             never miss the true bottom even while retrying.
          2. The opposite side must have bounced >= phase2_reversal_threshold
             from its floor (confirms the reversal, not just noise).
          3. FIX #2: Remove the `near_target` price gate entirely.
             The bracket equation (condition 4) is the real gate.
          4. Full bracket equation must be profitable after fees:
                 1 - x*(1+fee_x) - y*(1+fee_y) > 0
          5. Phase 2 retry cooldown must have elapsed.
        """
        y_price = no_ask if pos.momentum_side == "YES" else yes_ask
        if y_price <= 0:
            return

        # FIX #1: always track the floor regardless of cooldown
        if y_price < pos.min_opposite_price:
            pos.min_opposite_price = y_price

        # Cooldown gate (AFTER floor update)
        if time.time() - pos.p2_last_attempt_ts < self._P2_RETRY_COOLDOWN:
            return

        floor = pos.min_opposite_price
        reversal = self._cfg.phase2_reversal_threshold

        # Condition 2: confirmed reversal from floor
        if y_price < floor + reversal:
            return

        # Condition 4 (FIX #2: sole price gate — bracket equation must be profitable)
        x_cost = pos.p1_fill_price or pos.p1_price
        fee_x = taker_fee(x_cost, category="crypto price")
        fee_y = taker_fee(y_price, category="crypto price")
        net_margin = 1.0 - x_cost * (1.0 + fee_x) - y_price * (1.0 + fee_y)

        if net_margin <= 0:
            # Profitable bracket not possible yet at this y_price; keep waiting
            return

        logger.info(
            "BracketExecutor: ▶ Phase 2 trigger!  "
            "asset={}  y_floor={:.4f}  y_now={:.4f}  "
            "x_cost={:.4f}  net_margin={:.4f}  position_id={}",
            pos.asset, floor, y_price, x_cost, net_margin, pos.position_id,
        )
        await self._place_phase2(pos, y_price)

    async def _place_phase2(self, pos: BracketPosition, y_price: float) -> None:
        """
        Place the Phase 2 (opposite_side) buy order.

        FIX #4 + #5: use FOLLOW_TAKER for Phase 2.
          - When the trigger fires we need to lock the bracket NOW before the
            opposite side bounces back.  A PASSIVE_LIMIT resting bid at the
            reversal price risks missing the fill window entirely.
          - FOLLOW_TAKER fills at current market price or fails immediately.
          - On success we mark BRACKET_COMPLETE immediately (no polling needed).
        """
        pos.phase = BracketPhase.PHASE2_PENDING   # prevent re-entry next tick
        pos.p2_last_attempt_ts = time.time()

        entry_style = self._cfg.phase2_entry_style
        # Clamp to valid Polymarket price range (0.01–0.99)
        entry_price = round(min(max(y_price, 0.01), 0.99), 4)
        shares = pos.p1_shares   # symmetric sizing matches both legs

        logger.info(
            "BracketExecutor: placing Phase 2  asset={}  style={}  "
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
            # Revert to PHASE1_FILLED so next tick re-evaluates after cooldown
            pos.phase = BracketPhase.PHASE1_FILLED
            logger.error("BracketExecutor: Phase 2 FAILED — {} (retry in {}s)",
                         exc, self._P2_RETRY_COOLDOWN)
            self._audit({
                "type": "PHASE2_ORDER_FAILED",
                "position_id": pos.position_id,
                "asset": pos.asset,
                "attempted_price": entry_price,
                "error": str(exc),
            })
            return

        pos.p2_order_id = result.get("exchange_order_id", "")
        pos.p2_price = entry_price
        pos.p2_shares = shares
        pos.p2_notional_usd = round(entry_price * shares, 4)

        # FIX #5: FOLLOW_TAKER → no exception = immediate fill
        if entry_style == "FOLLOW_TAKER":
            pos.p2_fill_price = entry_price
            pos.phase = BracketPhase.BRACKET_COMPLETE
        else:
            # PASSIVE_LIMIT: order placed, will be confirmed later via poll
            # Phase stays PHASE2_PENDING until we verify
            # (Phase 2 PASSIVE_LIMIT poll is handled in _poll_phase2_fill)
            pos.p2_fill_price = 0.0

        x_cost = pos.p1_fill_price or pos.p1_price
        guaranteed = round(
            1.0
            - x_cost * (1.0 + taker_fee(x_cost, category="crypto price"))
            - entry_price * (1.0 + taker_fee(entry_price, category="crypto price")),
            6,
        )

        logger.info(
            "BracketExecutor: Phase 2 PLACED — bracket LOCKED 🔒  "
            "asset={}  p1={:.4f}  p2={:.4f}  "
            "guaranteed_margin={:.4f}  order_id={}",
            pos.asset, x_cost, entry_price, guaranteed, pos.p2_order_id,
        )
        self._audit({
            "type": "PHASE2_ORDER_PLACED",
            "position_id": pos.position_id,
            "asset": pos.asset,
            "entry_style": entry_style,
            "p2_token_id": pos.p2_token_id,
            "p2_price": entry_price,
            "p2_fill_price": pos.p2_fill_price,
            "p2_shares": shares,
            "p2_order_id": pos.p2_order_id,
            "p1_fill_price": pos.p1_fill_price or pos.p1_price,
            "guaranteed_margin": guaranteed,
        })

    # ================================================================== #
    # Window close — cancellation + settlement
    # ================================================================== #

    async def on_window_close(
        self, window_ts: int, asset: str, yes_won: bool | None
    ) -> None:
        """
        Called when a 15-minute window rolls over to the next.

        Actions per position state:
          PHASE1_PENDING   → CANCEL the open order (prevents stale CLOB orders)
          PHASE2_PENDING   → CANCEL the open Phase 2 order; settle as Phase1-only
          PHASE1_FILLED    → settle as Phase 1-only; P&L depends on direction
          BRACKET_COMPLETE → settle bracket; P&L guaranteed regardless of direction

        Binary markets resolve automatically — we don't need to place sell orders.
        Tokens that resolve $1 are claimed by the CLOB automatically.
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

            # ── Cancel unfilled Phase 1 ───────────────────────────────────
            if pos.phase == BracketPhase.PHASE1_PENDING:
                await self._cancel_order(pos.p1_order_id, pos, "window_close_p1")
                pos.phase = BracketPhase.CANCELLED
                self._audit({
                    "type": "PHASE1_CANCELLED_WINDOW_CLOSE",
                    "position_id": pos.position_id,
                    "asset": pos.asset,
                    "window_ts": window_ts,
                })
                continue

            # FIX #6: Cancel pending Phase 2 order before settling
            if pos.phase == BracketPhase.PHASE2_PENDING:
                await self._cancel_order(pos.p2_order_id, pos, "window_close_p2")
                pos.phase = BracketPhase.PHASE1_FILLED   # fall through to Phase1-only settle

            # ── Settle BRACKET_COMPLETE ───────────────────────────────────
            if pos.phase == BracketPhase.BRACKET_COMPLETE:
                x = pos.p1_fill_price or pos.p1_price
                y = pos.p2_fill_price or pos.p2_price
                cost = (
                    x * (1.0 + taker_fee(x, category="crypto price"))
                    + y * (1.0 + taker_fee(y, category="crypto price"))
                )
                # One token pays $1/share, other pays $0/share → net = $1/share
                pos.actual_pnl_usd = round((1.0 - cost) * pos.p1_shares, 4)
                logger.info(
                    "BracketExecutor: BRACKET SETTLED 🏆  asset={}  outcome={}  "
                    "pnl=${:.4f}  position_id={}",
                    asset, pos.outcome, pos.actual_pnl_usd, pos.position_id,
                )

            # ── Settle Phase 1 only ───────────────────────────────────────
            else:
                pos.phase = BracketPhase.PHASE1_ONLY_CLOSED
                x = pos.p1_fill_price or pos.p1_price
                cost = x * (1.0 + taker_fee(x, category="crypto price"))

                if pos.outcome == "UNKNOWN":
                    pnl_per_share = 0.0
                elif (
                    (pos.momentum_side == "YES" and pos.outcome == "YES_WINS")
                    or (pos.momentum_side == "NO"  and pos.outcome == "NO_WINS")
                ):
                    pnl_per_share = 1.0 - cost   # won: received $1, paid cost
                else:
                    pnl_per_share = -cost          # lost: received $0, paid cost

                pos.actual_pnl_usd = round(pnl_per_share * pos.p1_shares, 4)
                logger.info(
                    "BracketExecutor: P1-ONLY SETTLED  asset={}  outcome={}  "
                    "momentum={}  pnl=${:.4f}  position_id={}",
                    asset, pos.outcome, pos.momentum_side,
                    pos.actual_pnl_usd, pos.position_id,
                )

            self._audit({
                "type": "POSITION_SETTLED",
                "position_id": pos.position_id,
                "asset": pos.asset,
                "window_ts": window_ts,
                "phase_at_close": pos.phase.value,
                "outcome": pos.outcome,
                "actual_pnl_usd": pos.actual_pnl_usd,
                "p1_price": pos.p1_price,
                "p1_fill_price": pos.p1_fill_price,
                "p1_shares": pos.p1_shares,
                "p2_price": pos.p2_price,
                "p2_fill_price": pos.p2_fill_price,
            })

    async def _cancel_order(
        self, order_id: str, pos: BracketPosition, reason: str = ""
    ) -> None:
        """Attempt to cancel an open order.  Logs but does not raise on failure."""
        if not order_id:
            return
        try:
            await self._client.cancel_order(order_id)
            logger.info(
                "BracketExecutor: cancelled order  order_id={}  reason={}",
                order_id, reason,
            )
        except Exception as exc:
            # Common: order already filled or expired — not an error
            logger.debug(
                "BracketExecutor: cancel order_id={} failed (likely already settled): {}",
                order_id, exc,
            )

    # ================================================================== #
    # Public accessors — for reporting and external monitoring
    # ================================================================== #

    def active_positions(self) -> list[BracketPosition]:
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
        return sum(p.actual_pnl_usd for p in self._positions.values())

    def session_summary(self) -> dict:
        all_p     = list(self._positions.values())
        settled   = [p for p in all_p if p.phase in (
            BracketPhase.BRACKET_COMPLETE,
            BracketPhase.PHASE1_ONLY_CLOSED,
            BracketPhase.CANCELLED,
        )]
        brackets  = [p for p in settled if p.phase == BracketPhase.BRACKET_COMPLETE]
        p1_only   = [p for p in settled if p.phase == BracketPhase.PHASE1_ONLY_CLOSED]
        wins      = [p for p in p1_only  if p.actual_pnl_usd > 0]
        return {
            "total_positions":   len(all_p),
            "active":            len(self.active_positions()),
            "settled":           len(settled),
            "brackets_complete": len(brackets),
            "phase1_only":       len(p1_only),
            "phase1_wins":       len(wins),
            "phase1_losses":     len(p1_only) - len(wins),
            "session_pnl_usd":   round(self.session_pnl_usd(), 4),
        }

    # ================================================================== #
    # Audit log
    # ================================================================== #

    def _audit(self, record: dict) -> None:
        record.setdefault("ts", time.time())
        record.setdefault("ts_iso", datetime.now(timezone.utc).isoformat())
        try:
            with self._audit_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("BracketExecutor: audit write failed: {}", exc)
