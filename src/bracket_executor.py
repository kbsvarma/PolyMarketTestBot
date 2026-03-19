"""
Bracket Executor — live order placement for the 15-minute bracket strategy.

Full lifecycle, both phases, end-to-end:

  Phase 1  ─ Buy momentum_side (YES if BTC up, NO if BTC down) when the
              direction signal fires.  Entry style: FOLLOW_TAKER (FOK).
              A successful placement response is not enough — we explicitly
              confirm the fill before the position is opened. On failure the
              position is not created; the window continues.

  Phase 2  ─ Arm a remembered safe opposite-side level (y_safe) once the
              opposite ask first reaches the configured profitable zone.
              Then wait for the move to continue below y_safe and only buy
              when price later reclaims y_safe on reversal. Entry style:
              FOLLOW_TAKER. We only mark BRACKET_COMPLETE after the second leg
              is explicitly confirmed filled.

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
  - Phase 2: We still use immediate execution, but only AFTER the opposite side
    has first reached a remembered safe level, extended lower, and then
    reclaimed that safe level on reversal.

All actions written to logs/bracket_trades.jsonl.
"""
from __future__ import annotations

import asyncio
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
    BRACKET_COMPLETE   = "BRACKET_COMPLETE"    # Both legs filled — profit locked, window still open
    BRACKET_SETTLED    = "BRACKET_SETTLED"     # Bracket settled at window close — terminal state
    PHASE1_ONLY_CLOSED = "PHASE1_ONLY_CLOSED"  # Window closed with Phase 1 filled, no Phase 2
    HARD_EXITED        = "HARD_EXITED"         # Sold Phase-1 mid-window: 50c stop or final-30s cut
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
    entry_model:     str          # "lag" or "continuation"
    signal_price:    float        # original signal price before execution chase

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
    safe_opposite_price:  float = 0.0    # first observed safe y price
    dipped_below_safe_price: bool = False  # y later extended below safe_opposite_price
    opened_at:            float = field(default_factory=time.time)
    p1_filled_at:         float = 0.0
    closed_at:            float = 0.0
    fill_check_attempts:  int = 0        # poll counter for PASSIVE_LIMIT P1
    p2_last_attempt_ts:   float = 0.0   # last Phase 2 placement attempt time
    hard_exit_attempted:  bool = False   # prevents repeated hard-exit attempts
    hard_exit_last_attempt_ts: float = 0.0
    hard_exit_reason:     str = ""
    hard_exit_fill_price: float = 0.0

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
        self.execution_mode = str(getattr(client, "execution_mode", "live")).lower()
        self._positions: dict[str, BracketPosition] = {}  # position_id → pos
        self._window_pos: dict[str, str] = {}             # "BTC_{ts}" → position_id
        self._closed_window_summaries: dict[tuple[int, str], dict[str, Any]] = {}

        audit_path = Path(cfg.bracket_audit_log_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_path = audit_path

        logger.info(
            "BracketExecutor ready  mode={}  "
            "execute={} phase2={} max_concurrent={}  "
            "p1_style={} p2_style={}  shares={}/leg",
            self.execution_mode,
            cfg.execute_enabled,
            cfg.phase2_enabled,
            cfg.max_concurrent_brackets,
            cfg.phase1_entry_style,
            cfg.phase2_entry_style,
            cfg.phase1_shares,
        )

    @staticmethod
    def _filled_enough(filled_size: float, expected_size: float) -> bool:
        return filled_size >= max(expected_size - 1e-6, expected_size * 0.999)

    @staticmethod
    def _normalized_status(payload: dict[str, Any]) -> str:
        return str(payload.get("status") or payload.get("state") or "").upper()

    async def _confirm_follow_taker_fill(
        self,
        *,
        order_result: dict[str, Any],
        order_id: str,
        expected_size: float,
        fallback_price: float,
    ) -> tuple[bool, float, dict[str, Any]]:
        """
        Confirm that a FOLLOW_TAKER / FOK submission actually filled.

        Submission success alone is not enough. We first inspect the immediate
        placement payload, then poll order status briefly if needed.
        """
        last_payload = dict(order_result)
        attempts = max(int(self._cfg.fill_confirmation_attempts), 1)
        delay_seconds = max(float(self._cfg.fill_confirmation_delay_seconds), 0.0)

        for attempt in range(attempts):
            status = self._normalized_status(last_payload)
            filled_size = float(last_payload.get("filled_size") or 0.0)
            avg_fill = float(last_payload.get("average_fill_price") or 0.0) or fallback_price
            remaining = float(last_payload.get("remaining_size") or 0.0)

            if self._filled_enough(filled_size, expected_size):
                return True, avg_fill, last_payload
            if status in {"FILLED", "MATCHED"} and remaining <= 1e-6:
                return True, avg_fill, last_payload
            if status in {"CANCELLED", "EXPIRED", "FAILED", "REJECTED"}:
                return False, 0.0, last_payload

            if attempt + 1 >= attempts or not order_id or not hasattr(self._client, "get_order_status"):
                break
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            try:
                last_payload = await self._client.get_order_status(order_id)
            except Exception as exc:
                last_payload = {
                    "status": "UNKNOWN",
                    "error": str(exc),
                    "exchange_order_id": order_id,
                }
                break

        return False, 0.0, last_payload

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

        # ── Size: fixed share count for parity with the observer report ──
        signal_price = round(event.momentum_price, 4)
        shares = max(
            round(self._cfg.phase1_shares, 1),
            self._cfg.min_bracket_shares,
        )
        position_id = str(uuid.uuid4())[:12]
        entry_style = self._cfg.phase1_entry_style
        chase_cents = max(float(getattr(self._cfg, "phase1_max_chase_cents", 0.0) or 0.0), 0.0)
        entry_price = signal_price
        if entry_style == "FOLLOW_TAKER" and chase_cents > 0:
            # Small controlled chase so we can still lock x when the book lifts
            # by a tick between signal snapshot and order submission, but never
            # above the strategy's own entry band ceiling.
            entry_price = round(
                min(
                    0.99,
                    float(getattr(self._cfg, "entry_range_high", 0.99) or 0.99),
                    signal_price + chase_cents,
                ),
                4,
            )

        logger.info(
            "BracketExecutor: ▶ Phase 1  asset={} side={} style={} "
            "signal_price={:.4f} submit_limit={:.4f} chase={:.4f} "
            "shares={:.1f} notional=${:.2f}",
            event.asset, event.momentum_side, entry_style,
            signal_price, entry_price, entry_price - signal_price,
            shares, entry_price * shares,
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
            self._cache_signal_attempt_summary(
                event=event,
                position_id=position_id,
                phase="PHASE1_ORDER_FAILED",
                shares=shares,
                requested_price=entry_price,
            )
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
            entry_model=event.entry_model,
            signal_price=signal_price,
            p1_token_id=momentum_token_id,
            p1_price=entry_price,
            p1_shares=shares,
            p1_notional_usd=round(entry_price * shares, 4),
            p1_order_id=order_id,
            p2_token_id=opposite_token_id,
        )

        # FOLLOW_TAKER = FOK. Submission success is not enough; explicitly
        # confirm that the order filled before we credit the position.
        if entry_style == "FOLLOW_TAKER":
            filled, fill_price, fill_payload = await self._confirm_follow_taker_fill(
                order_result=result,
                order_id=order_id,
                expected_size=shares,
                fallback_price=entry_price,
            )
            if not filled:
                logger.warning(
                    "BracketExecutor: Phase 1 not filled after submission  "
                    "asset={} status={} order_id={} position_id={}",
                    event.asset,
                    self._normalized_status(fill_payload),
                    order_id,
                    position_id,
                )
                self._audit({
                    "type": "PHASE1_NOT_FILLED",
                    "position_id": position_id,
                    "event_id": event.event_id,
                    "asset": event.asset,
                    "signal_price": signal_price,
                    "submit_limit_price": entry_price,
                    "order_id": order_id,
                    "status": self._normalized_status(fill_payload),
                    "fill_payload": fill_payload,
                })
                self._cache_signal_attempt_summary(
                    event=event,
                    position_id=position_id,
                    phase="PHASE1_NOT_FILLED",
                    shares=shares,
                    requested_price=entry_price,
                )
                return False

            pos.phase = BracketPhase.PHASE1_FILLED
            pos.p1_fill_price = fill_price
            pos.p1_filled_at = time.time()
            pos.p1_notional_usd = round(pos.p1_fill_price * shares, 4)
            logger.info(
                "BracketExecutor: Phase 1 FILLED (FOLLOW_TAKER)  "
                "asset={}  fill_price={:.4f}  signal_price={:.4f} "
                "submit_limit={:.4f}  position_id={}",
                event.asset, pos.p1_fill_price, signal_price, entry_price, position_id,
            )
            self._audit({
                "type": "PHASE1_FILLED",
                "position_id": position_id,
                "event_id": event.event_id,
                "asset": event.asset,
                "fill_price": pos.p1_fill_price,
                "signal_price": signal_price,
                "submit_limit_price": entry_price,
                "style": "FOLLOW_TAKER",
                "execution_mode": self.execution_mode,
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
            "signal_price": signal_price,
            "p1_price": entry_price,
            "p1_shares": shares,
            "p1_order_id": order_id,
            "window_ts": event.window_open_ts,
            "window_close_ts": event.window_close_ts,
            "execution_mode": self.execution_mode,
        })
        return True

    def _cache_signal_attempt_summary(
        self,
        *,
        event: BracketSignalEvent,
        position_id: str,
        phase: str,
        shares: float,
        requested_price: float,
    ) -> None:
        """
        Cache a compact per-window summary even when Phase 1 never fills.

        Without this, the report falls back to the optimistic signal snapshot
        and can over-credit shadow/live windows where the executor correctly
        rejected the order as unfilled.
        """
        self._closed_window_summaries[(event.window_open_ts, event.asset)] = {
            "execution_mode": self.execution_mode,
            "position_id": position_id,
            "event_id": event.event_id,
            "asset": event.asset,
            "window_ts": event.window_open_ts,
            "phase": phase,
            "signal_side": event.momentum_side,
            "signal_price": event.momentum_price,
            "signal_entry_model": event.entry_model,
            "phase1_filled": False,
            "phase2_filled": False,
            "hard_exited": False,
            "hard_exit_reason": "",
            "hard_exit_fill_price": 0.0,
            "cancelled": phase in {"PHASE1_NOT_FILLED", "CANCELLED"},
            "actual_pnl_usd": 0.0,
            "p1_shares": shares,
            "p1_fill_price": 0.0,
            "p1_requested_price": requested_price,
            "p2_fill_price": 0.0,
            "safe_opposite_price": 0.0,
            "min_opposite_price": 999.0,
        }

    # ================================================================== #
    # Per-poll hook — Phase 1 fill polling + Phase 2 trigger
    # ================================================================== #

    async def tick(
        self,
        asset: str,
        yes_ask: float,
        no_ask: float,
        *,
        yes_bid: float = 0.0,
        no_bid: float = 0.0,
    ) -> None:
        """
        Called every poll cycle (~1 s) per asset.

          PHASE1_PENDING → poll CLOB for fill (PASSIVE_LIMIT only)
          PHASE1_FILLED  → check hard exit first, then Phase 2 if still open
        """
        for pos in list(self._positions.values()):
            if pos.asset != asset:
                continue
            if pos.phase == BracketPhase.PHASE1_PENDING:
                await self._poll_phase1_fill(pos)
            elif pos.phase == BracketPhase.PHASE1_FILLED:
                # Hard exit check takes priority over Phase 2.
                # If it fires, phase becomes HARD_EXITED and Phase 2 is skipped.
                await self._check_hard_exit(pos, yes_ask, no_ask, yes_bid=yes_bid, no_bid=no_bid)
                if pos.phase == BracketPhase.PHASE1_FILLED and self._cfg.phase2_enabled:
                    await self._check_phase2(pos, yes_ask, no_ask)

    # ── Hard exit — cut losses mid-window ────────────────────────────────

    async def _check_hard_exit(
        self,
        pos: BracketPosition,
        yes_ask: float,
        no_ask: float,
        *,
        yes_bid: float = 0.0,
        no_bid: float = 0.0,
    ) -> None:
        """
        Sell the Phase-1 leg mid-window to cap losses.

        Triggers when either:
          (a) The momentum-side mark drops to hard_exit_stop_price (default 0.50)
              — we paid above 0.50, so the market now thinks we're losing.
          (b) We're in the final hard_exit_final_seconds and still below entry
              — don't hold to binary resolution without a bracket.

        Uses the momentum-side best bid when available, because that's the
        executable sell-side price. Falls back to ask only if bid data is
        unavailable.
        """
        if pos.phase == BracketPhase.HARD_EXITED:
            return

        now_ts = time.time()
        retry_cooldown = float(
            getattr(self._cfg, "hard_exit_retry_cooldown_seconds", 0.0) or 0.0
        )
        if (
            pos.hard_exit_last_attempt_ts > 0
            and retry_cooldown > 0
            and (now_ts - pos.hard_exit_last_attempt_ts) < retry_cooldown
        ):
            return

        sellable_bid = yes_bid if pos.momentum_side == "YES" else no_bid
        ask_fallback = yes_ask if pos.momentum_side == "YES" else no_ask
        mark = sellable_bid if sellable_bid > 0 else ask_fallback
        if mark <= 0:
            return

        entry = pos.p1_fill_price or pos.p1_price
        seconds_to_close = pos.window_close_ts - now_ts
        # Once the window has ended, let settlement own the position. The
        # orderbook can collapse to closeout prices around rollover and should
        # not trigger fresh hard-exit logic.
        if seconds_to_close <= 0:
            return
        seconds_since_fill = max(0.0, now_ts - (pos.p1_filled_at or pos.opened_at))

        stop_trigger_price = float(self._cfg.hard_exit_stop_price) + float(
            getattr(self._cfg, "hard_exit_trigger_buffer_cents", 0.0) or 0.0
        )
        stop_hit = mark <= stop_trigger_price
        final_window_loss = (
            seconds_to_close <= self._cfg.hard_exit_final_seconds
            and mark < entry
        )

        continuation_grace_active = (
            pos.entry_model == "continuation"
            and seconds_since_fill < float(
                getattr(self._cfg, "continuation_hard_exit_grace_seconds", 0.0) or 0.0
            )
        )
        safe_arm_stop_protection = (
            bool(getattr(self._cfg, "safe_arm_suspend_stop", True))
            and pos.safe_opposite_price > 0
        )
        catastrophic_floor = 0.0
        if continuation_grace_active:
            catastrophic_floor = max(
                catastrophic_floor,
                float(getattr(self._cfg, "continuation_catastrophic_stop_price", 0.0) or 0.0),
            )
        if safe_arm_stop_protection:
            catastrophic_floor = max(
                catastrophic_floor,
                float(getattr(self._cfg, "safe_arm_catastrophic_stop_price", 0.0) or 0.0),
            )
        catastrophic_stop = catastrophic_floor > 0 and mark <= catastrophic_floor

        if (continuation_grace_active or safe_arm_stop_protection) and stop_hit and not catastrophic_stop and not final_window_loss:
            logger.info(
                "BracketExecutor: protected from ordinary hard stop  asset={} mark={:.4f} "
                "entry={:.4f} elapsed={:.1f}s continuation_grace={} safe_arm={} "
                "safe_price={:.4f} catastrophic_floor={:.4f} position_id={}",
                pos.asset,
                mark,
                entry,
                seconds_since_fill,
                continuation_grace_active,
                safe_arm_stop_protection,
                pos.safe_opposite_price,
                catastrophic_floor,
                pos.position_id,
            )
            return

        if not (stop_hit or final_window_loss):
            return

        if final_window_loss and not stop_hit:
            reason = "FINAL_30S_LOSS"
        elif catastrophic_stop and (continuation_grace_active or safe_arm_stop_protection):
            reason = "STOP_CATASTROPHIC"
        else:
            reason = "STOP_50C"
        pos.hard_exit_last_attempt_ts = now_ts
        pos.hard_exit_reason = reason

        primary_sell_price = round(max(0.01, float(self._cfg.hard_exit_stop_price)), 4)
        market_through_cents = max(
            float(getattr(self._cfg, "hard_exit_market_through_cents", 0.02) or 0.0),
            0.0,
        )
        market_through_price = round(
            max(
                0.01,
                float(self._cfg.hard_exit_stop_price) - market_through_cents,
            ),
            4,
        )
        if reason == "STOP_50C":
            attempt_prices = [primary_sell_price]
            # Optional fallback cross. When disabled (0.0c), keep retrying the
            # intended 50c stop instead of chasing into a much worse live loss.
            if market_through_cents > 0 and market_through_price < primary_sell_price:
                attempt_prices.append(market_through_price)
        else:
            attempt_prices = [market_through_price]
        sell_price = attempt_prices[-1]
        logger.info(
            "BracketExecutor: ⚠ HARD EXIT triggered  "
            "asset={}  reason={}  mark={:.4f}  entry={:.4f}  "
            "trigger_price={:.4f}  attempt_prices={}  seconds_to_close={:.1f}  position_id={}",
            pos.asset, reason, mark, entry, stop_trigger_price, attempt_prices, seconds_to_close, pos.position_id,
        )

        fill_payload: dict[str, object] = {}
        try:
            filled = False
            fill_price = 0.0
            last_error: Exception | None = None
            for idx, attempt_price in enumerate(attempt_prices):
                result = await self._client.place_sell_order(
                    token_id=pos.p1_token_id,
                    price=attempt_price,
                    size=pos.p1_shares,
                    entry_style="FOLLOW_TAKER",
                    client_order_id=f"{pos.position_id}-hard-exit-{idx + 1}",
                )
                order_id = str(result.get("exchange_order_id") or "")
                filled, fill_price, fill_payload = await self._confirm_follow_taker_fill(
                    order_result=result,
                    order_id=order_id,
                    expected_size=pos.p1_shares,
                    fallback_price=attempt_price,
                )
                sell_price = attempt_price
                if filled:
                    break
                last_error = RuntimeError(
                    f"hard exit not filled status={self._normalized_status(fill_payload)}"
                )
                if idx + 1 < len(attempt_prices):
                    logger.warning(
                        "BracketExecutor: hard exit retrying lower  asset={} reason={} "
                        "attempt_price={:.4f} next_price={:.4f} position_id={}",
                        pos.asset,
                        reason,
                        attempt_price,
                        attempt_prices[idx + 1],
                        pos.position_id,
                    )
            if not filled:
                raise last_error or RuntimeError("hard exit not filled")
            entry_cost = entry * (1.0 + taker_fee(entry, category="crypto price"))
            exit_proceeds = fill_price * (1.0 - taker_fee(fill_price, category="crypto price"))
            pnl = round((exit_proceeds - entry_cost) * pos.p1_shares, 4)
            pos.phase = BracketPhase.HARD_EXITED
            pos.closed_at = time.time()
            pos.hard_exit_attempted = True
            pos.hard_exit_fill_price = fill_price
            pos.actual_pnl_usd = pnl
            logger.info(
                "BracketExecutor: HARD EXIT filled  "
                "asset={}  sell_price={:.4f}  reason={}  pnl=${:.4f}  position_id={}",
                pos.asset, fill_price, reason, pnl, pos.position_id,
            )
        except Exception as exc:
            logger.error(
                "BracketExecutor: HARD EXIT sell FAILED (will resolve at window close)  "
                "asset={}  error={}  position_id={}",
                pos.asset, exc, pos.position_id,
            )

        self._audit({
            "type": "HARD_EXIT",
            "position_id": pos.position_id,
            "asset": pos.asset,
            "reason": reason,
            "mark": mark,
            "trigger_price": stop_trigger_price,
            "attempted_sell_prices": attempt_prices,
            "sell_price": sell_price,
            "fill_price": pos.hard_exit_fill_price,
            "entry_price": entry,
            "actual_pnl_usd": pos.actual_pnl_usd,
            "phase_after": pos.phase.value,
            "execution_mode": self.execution_mode,
        })

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
            pos.p1_filled_at = time.time()
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
          1. Track the running minimum of the opposite-side ask.
          2. Save y_safe the FIRST time the opposite side reaches the configured
             safe zone and the bracket is profitable after fees.
          3. Require the move to continue below y_safe by at least
             phase2_reversal_threshold.
          4. Only then buy when the opposite side reclaims y_safe on reversal.
          5. The bracket must still be profitable at the observed reclaim price.
          6. Phase 2 retry cooldown must have elapsed.
        """
        y_price = no_ask if pos.momentum_side == "YES" else yes_ask
        if y_price <= 0:
            return

        # Always track the true floor, even while waiting or cooling down.
        if y_price < pos.min_opposite_price:
            pos.min_opposite_price = y_price

        x_cost = pos.p1_fill_price or pos.p1_price
        fee_x = taker_fee(x_cost, category="crypto price")
        fee_y = taker_fee(y_price, category="crypto price")
        net_margin = 1.0 - x_cost * (1.0 + fee_x) - y_price * (1.0 + fee_y)

        # Arm the remembered safe price exactly once.
        if pos.safe_opposite_price <= 0:
            if y_price <= self._cfg.target_y_price and net_margin > 0:
                pos.safe_opposite_price = round(y_price, 6)
                logger.info(
                    "BracketExecutor: Phase 2 armed  asset={}  safe_y={:.4f}  "
                    "x_cost={:.4f}  net_margin={:.4f}  position_id={}",
                    pos.asset, pos.safe_opposite_price, x_cost, net_margin, pos.position_id,
                )
                self._audit({
                    "type": "PHASE2_SAFE_ARMED",
                    "position_id": pos.position_id,
                    "asset": pos.asset,
                    "safe_opposite_price": pos.safe_opposite_price,
                    "net_margin": round(net_margin, 6),
                })
            return

        reversal = self._cfg.phase2_reversal_threshold
        safe_price = pos.safe_opposite_price

        # Require continuation below the remembered safe level before we count
        # a reclaim as meaningful.
        if y_price <= safe_price - reversal:
            if not pos.dipped_below_safe_price:
                logger.info(
                    "BracketExecutor: Phase 2 continuation confirmed  asset={}  "
                    "safe_y={:.4f}  floor_now={:.4f}  position_id={}",
                    pos.asset, safe_price, y_price, pos.position_id,
                )
                self._audit({
                    "type": "PHASE2_SAFE_BREACHED",
                    "position_id": pos.position_id,
                    "asset": pos.asset,
                    "safe_opposite_price": safe_price,
                    "y_price": round(y_price, 6),
                })
            pos.dipped_below_safe_price = True
            return

        if not pos.dipped_below_safe_price:
            return

        # Cooldown gate (after tracking safe-level state)
        if time.time() - pos.p2_last_attempt_ts < self._P2_RETRY_COOLDOWN:
            return

        # Reclaim must come back to the remembered safe level without chasing
        # materially above it.
        if y_price < safe_price or y_price > safe_price + reversal:
            return
        if net_margin <= 0:
            return

        logger.info(
            "BracketExecutor: ▶ Phase 2 reclaim trigger!  "
            "asset={}  safe_y={:.4f}  y_floor={:.4f}  y_now={:.4f}  "
            "x_cost={:.4f}  net_margin={:.4f}  position_id={}",
            pos.asset, safe_price, pos.min_opposite_price, y_price,
            x_cost, net_margin, pos.position_id,
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

        # FOLLOW_TAKER must be explicitly confirmed as filled.
        if entry_style == "FOLLOW_TAKER":
            filled, fill_price, fill_payload = await self._confirm_follow_taker_fill(
                order_result=result,
                order_id=pos.p2_order_id,
                expected_size=shares,
                fallback_price=entry_price,
            )
            if not filled:
                pos.phase = BracketPhase.PHASE1_FILLED
                logger.warning(
                    "BracketExecutor: Phase 2 not filled after submission  "
                    "asset={} status={} position_id={}",
                    pos.asset,
                    self._normalized_status(fill_payload),
                    pos.position_id,
                )
                self._audit({
                    "type": "PHASE2_NOT_FILLED",
                    "position_id": pos.position_id,
                    "asset": pos.asset,
                    "order_id": pos.p2_order_id,
                    "status": self._normalized_status(fill_payload),
                    "fill_payload": fill_payload,
                })
                return
            pos.p2_fill_price = fill_price
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
            "execution_mode": self.execution_mode,
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
            if pos.phase in (
                BracketPhase.PHASE1_ONLY_CLOSED,
                BracketPhase.HARD_EXITED,
                BracketPhase.CANCELLED,
            ):
                self._store_window_summary(pos)
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
                self._store_window_summary(pos)
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
                # Terminal state — frees concurrency slot for next window
                pos.phase = BracketPhase.BRACKET_SETTLED
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
                "phase_at_close": pos.phase.value,   # BRACKET_SETTLED or PHASE1_ONLY_CLOSED
                "outcome": pos.outcome,
                "actual_pnl_usd": pos.actual_pnl_usd,
                "p1_price": pos.p1_price,
                "p1_fill_price": pos.p1_fill_price,
                "p1_shares": pos.p1_shares,
                "p2_price": pos.p2_price,
                "p2_fill_price": pos.p2_fill_price,
            })
            self._store_window_summary(pos)

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
        """Positions still open (orders live or awaiting window close)."""
        return [
            p for p in self._positions.values()
            if p.phase in (
                BracketPhase.PHASE1_PENDING,
                BracketPhase.PHASE1_FILLED,
                BracketPhase.PHASE2_PENDING,
                BracketPhase.BRACKET_COMPLETE,   # both legs filled, window not yet closed
            )
        ]

    def session_pnl_usd(self) -> float:
        return sum(p.actual_pnl_usd for p in self._positions.values())

    def session_summary(self) -> dict:
        all_p     = list(self._positions.values())
        settled   = [p for p in all_p if p.phase in (
            BracketPhase.BRACKET_SETTLED,
            BracketPhase.PHASE1_ONLY_CLOSED,
            BracketPhase.CANCELLED,
        )]
        brackets  = [p for p in settled if p.phase == BracketPhase.BRACKET_SETTLED]
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

    def take_window_summary(self, window_ts: int, asset: str) -> dict[str, Any] | None:
        """Return and clear the cached execution summary for one closed window."""
        return self._closed_window_summaries.pop((window_ts, asset), None)

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

    def _store_window_summary(self, pos: BracketPosition) -> None:
        """Cache a compact execution summary for report rendering."""
        self._closed_window_summaries[(pos.window_ts, pos.asset)] = {
            "execution_mode": self.execution_mode,
            "position_id": pos.position_id,
            "event_id": pos.event_id,
            "asset": pos.asset,
            "window_ts": pos.window_ts,
            "phase": pos.phase.value,
            "signal_side": pos.momentum_side,
            "signal_price": pos.signal_price,
            "signal_entry_model": pos.entry_model,
            "phase1_filled": bool(
                pos.p1_fill_price
                or pos.phase not in {BracketPhase.PHASE1_PENDING, BracketPhase.CANCELLED}
            ),
            "phase2_filled": bool(
                pos.p2_fill_price
                or pos.phase in {BracketPhase.BRACKET_COMPLETE, BracketPhase.BRACKET_SETTLED}
            ),
            "hard_exited": pos.phase == BracketPhase.HARD_EXITED,
            "hard_exit_reason": pos.hard_exit_reason,
            "hard_exit_fill_price": pos.hard_exit_fill_price,
            "cancelled": pos.phase == BracketPhase.CANCELLED,
            "actual_pnl_usd": pos.actual_pnl_usd,
            "p1_shares": pos.p1_shares,
            "p1_fill_price": pos.p1_fill_price or pos.p1_price,
            "p2_fill_price": pos.p2_fill_price or pos.p2_price,
            "safe_opposite_price": pos.safe_opposite_price,
            "min_opposite_price": pos.min_opposite_price,
            "dipped_below_safe_price": pos.dipped_below_safe_price,
            "phase2_reclaim_seen": bool(pos.p2_price > 0 or pos.p2_fill_price > 0),
        }
