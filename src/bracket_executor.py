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

from src.fees import max_profitable_opposite_price, taker_fee
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
    p1_initial_shares: float      # original number of shares requested
    p1_price:        float        # price at which the order was placed
    p1_shares:       float        # remaining open shares
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
    hard_exit_filled_shares: float = 0.0
    hard_exit_order_ids:  list[str] = field(default_factory=list)

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
    _SELL_POSITION_VISIBILITY_WAIT_SECONDS = 0.75
    _SELL_POSITION_VISIBILITY_POLL_SECONDS = 0.10

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

    @staticmethod
    def _reported_filled_size(payload: dict[str, Any], fallback_size: float = 0.0) -> float:
        try:
            filled_size = float(payload.get("filled_size") or 0.0)
        except (TypeError, ValueError):
            filled_size = 0.0
        if filled_size > 1e-6:
            return filled_size
        return max(float(fallback_size or 0.0), 0.0)

    @staticmethod
    def _phase1_stepdown_size(current_shares: float, min_shares: float) -> float:
        current = max(float(current_shares or 0.0), 0.0)
        minimum = max(float(min_shares or 0.0), 0.0)
        # The live crypto lanes intentionally use a 5–10 share window. Only
        # step down inside that regime; falling from 5 to 1 shares in tests or
        # other tiny-share configs would change behavior more than intended.
        if minimum < 5.0 - 1e-9:
            return 0.0
        if current <= minimum + 1e-9:
            return 0.0
        return round(minimum, 6)

    async def _confirm_follow_taker_fill(
        self,
        *,
        order_result: dict[str, Any],
        order_id: str,
        expected_size: float,
        fallback_price: float,
        allow_partial: bool = False,
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
            if allow_partial and filled_size > 1e-6:
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

    async def _get_orderbook_for_retry(self, token_id: str):
        getter = getattr(self._client, "get_orderbook", None)
        if callable(getter):
            return await getter(token_id)
        inner_client = getattr(self._client, "_client", None)
        inner_getter = getattr(inner_client, "get_orderbook", None)
        if callable(inner_getter):
            return await inner_getter(token_id)
        return None

    async def _marketable_book_context(
        self,
        *,
        token_id: str,
        limit_price: float,
        shares: float,
        side: str,
    ) -> dict[str, Any]:
        try:
            orderbook = await self._get_orderbook_for_retry(token_id)
        except Exception as exc:
            return {"orderbook_error": str(exc)}
        if orderbook is None:
            return {}

        if side == "SELL":
            levels = list(getattr(orderbook, "bids", []) or [])
        else:
            levels = list(getattr(orderbook, "asks", []) or [])
        best_price = float(levels[0].price) if levels else 0.0
        marketable_depth = 0.0
        for level in levels:
            level_price = float(level.price)
            marketable = level_price >= limit_price if side == "SELL" else level_price <= limit_price
            if not marketable:
                break
            marketable_depth += float(level.size)
        executable_shares = min(float(shares), marketable_depth)

        return {
            "best_price": round(best_price, 6),
            "marketable_depth": round(marketable_depth, 6),
            "executable_shares": round(executable_shares, 6),
            "marketable_now": bool(best_price > 0 and executable_shares + 1e-9 >= shares),
        }

    async def _phase1_retry_context(
        self,
        *,
        token_id: str,
        limit_price: float,
        shares: float,
    ) -> dict[str, Any]:
        context = await self._marketable_book_context(
            token_id=token_id,
            limit_price=limit_price,
            shares=shares,
            side="BUY",
        )
        if not context:
            return context
        return {
            "best_ask": context.get("best_price", 0.0),
            "marketable_depth": context.get("marketable_depth", 0.0),
            "marketable_now": bool(context.get("marketable_now", False)),
            "executable_shares": context.get("executable_shares", 0.0),
        }

    def _phase1_retry_ceiling(self, event: BracketSignalEvent, initial_limit: float) -> float:
        retry_to_cap = bool(getattr(self._cfg, "phase1_follow_taker_retry_to_strategy_cap", False))

        # Resolve the strategy-level cap once; both retry_to_cap and the
        # lag-conditioned path use it, and both must respect the same
        # continuation / band-touch sub-caps.
        strategy_cap = float(getattr(self._cfg, "entry_range_high", initial_limit) or initial_limit)
        entry_model = str(event.entry_model or "").lower()
        if entry_model == "continuation":
            continuation_cap = float(
                getattr(self._cfg, "continuation_max_momentum_price", strategy_cap) or strategy_cap
            )
            strategy_cap = min(strategy_cap, continuation_cap)
        elif entry_model == "band_touch":
            band_cap = float(
                getattr(self._cfg, "immediate_band_entry_high", strategy_cap) or strategy_cap
            )
            strategy_cap = min(strategy_cap, band_cap)

        if retry_to_cap:
            return round(max(initial_limit, min(0.99, strategy_cap)), 4)

        # Lag-conditioned retry ceiling: if the signal had a strong GBM lag gap,
        # the retry is allowed to chase up to signal_price + lag_gap * multiplier.
        # Budget reasoning: if lag_gap = 0.04 and multiplier = 0.5, we pay at most
        # 2c above signal price — still 2c below GBM fair value, edge is positive.
        # Hard cap at entry_range_high so the strategy ceiling is never crossed.
        # Disabled (multiplier = 0.0) by default; opt-in via phase1_lag_retry_multiplier.
        lag_retry_multiplier = float(
            getattr(self._cfg, "phase1_lag_retry_multiplier", 0.0) or 0.0
        )
        if lag_retry_multiplier > 0:
            lag_gap = float(getattr(event, "lag_gap", 0.0) or 0.0)
            if lag_gap > 0:
                signal_price = float(
                    getattr(event, "momentum_price", initial_limit) or initial_limit
                )
                lag_ceiling = round(signal_price + lag_gap * lag_retry_multiplier, 4)
                lag_ceiling = min(lag_ceiling, strategy_cap, 0.99)
                lag_ceiling = max(lag_ceiling, initial_limit)
                return round(lag_ceiling, 4)

        return round(initial_limit, 4)

    @staticmethod
    def _exception_diagnostics(exc: BaseException) -> dict[str, str]:
        cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
        details = {
            "exception_type": exc.__class__.__name__,
            "exception_source": f"{exc.__class__.__module__}.{exc.__class__.__name__}",
            "exception_message": str(exc),
            "cause_type": "",
            "cause_source": "",
            "cause_message": "",
        }
        if cause is not None:
            details["cause_type"] = cause.__class__.__name__
            details["cause_source"] = f"{cause.__class__.__module__}.{cause.__class__.__name__}"
            details["cause_message"] = str(cause)
        return details

    @staticmethod
    def _is_request_exception_failure(*, error: str = "", details: dict[str, str] | None = None) -> bool:
        text = (error or "").lower()
        if "request exception" in text or "status_code=none" in text:
            return True
        if not details:
            return False
        cause_message = str(details.get("cause_message") or "").lower()
        return "request exception" in cause_message or "status_code=none" in cause_message

    async def _refresh_client_for_retry(self) -> dict[str, Any]:
        refresher = getattr(self._client, "refresh_live_order_session", None)
        if not callable(refresher):
            return {"attempted": False, "ok": False, "reason": "refresh_unavailable"}
        try:
            payload = await refresher()
            if isinstance(payload, dict):
                return {
                    "attempted": True,
                    "ok": bool(payload.get("refreshed", True)),
                    **payload,
                }
            return {"attempted": True, "ok": True}
        except Exception as exc:
            details = self._exception_diagnostics(exc)
            return {
                "attempted": True,
                "ok": False,
                **details,
            }

    @staticmethod
    def _should_retry_follow_taker_failure(*, status: str = "", error: str = "") -> bool:
        text = (error or "").lower()
        normalized = (status or "").upper()
        if "fully filled or killed" in text or "couldn't be fully filled" in text:
            return True
        if normalized in {"CANCELLED", "EXPIRED"}:
            return True
        return False

    @staticmethod
    def _is_allowance_failure(error: str) -> bool:
        text = (error or "").lower()
        return "not enough balance / allowance" in text or "not enough balance" in text

    @staticmethod
    def _is_fak_no_match_failure(error: str) -> bool:
        text = (error or "").lower()
        return (
            "no orders found to match with fak order" in text
            or "fak orders are partially filled or killed if no match is found" in text
        )

    @staticmethod
    def _round_shares(size: float) -> float:
        return round(max(size, 0.0), 3)

    @staticmethod
    def _extract_position_quantity(payload: dict[str, Any]) -> float:
        candidates = [
            payload.get("quantity"),
            payload.get("size"),
            payload.get("shares"),
            payload.get("balance"),
            payload.get("remaining_size"),
            payload.get("filled_size"),
        ]
        raw = payload.get("raw")
        if isinstance(raw, dict):
            candidates.extend(
                [
                    raw.get("quantity"),
                    raw.get("size"),
                    raw.get("shares"),
                    raw.get("balance"),
                ]
            )
        for candidate in candidates:
            try:
                value = float(candidate or 0.0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return round(value, 6)
        return 0.0

    @staticmethod
    def _extract_position_average_price(payload: dict[str, Any], fallback_price: float) -> float:
        candidates = [
            payload.get("avg_price"),
            payload.get("avgPrice"),
            payload.get("average_price"),
            payload.get("averagePrice"),
            payload.get("average_fill_price"),
            payload.get("entry_price"),
            payload.get("entryPrice"),
            payload.get("price"),
        ]
        raw = payload.get("raw")
        if isinstance(raw, dict):
            candidates.extend(
                [
                    raw.get("avg_price"),
                    raw.get("avgPrice"),
                    raw.get("average_price"),
                    raw.get("averagePrice"),
                    raw.get("entry_price"),
                    raw.get("entryPrice"),
                    raw.get("price"),
                ]
            )
        for candidate in candidates:
            try:
                value = float(candidate or 0.0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return round(value, 6)
        return round(max(fallback_price, 0.0), 6)

    async def _available_token_position_shares(self, token_id: str) -> float | None:
        """
        Return the currently visible wallet balance for a conditional token.

        This is used by the live hard-exit path so we do not keep trying to
        sell more shares than Polymarket currently recognizes as spendable.
        Returns:
          - float >= 0 when position visibility is available
          - None when position visibility could not be queried at all
        """
        get_positions = getattr(self._client, "get_positions", None)
        if not callable(get_positions) or not token_id:
            return None

        try:
            positions = await get_positions()
        except Exception:
            return None

        best_qty = 0.0
        saw_match = False
        for item in positions or []:
            item_token_id = str(
                item.get("token_id")
                or item.get("asset")
                or item.get("asset_id")
                or item.get("tokenId")
                or ""
            )
            if item_token_id != token_id:
                continue
            saw_match = True
            qty = self._extract_position_quantity(item)
            if qty > best_qty:
                best_qty = qty
        if not saw_match:
            return None
        return round(best_qty, 6)

    async def _wait_for_visible_token_position(
        self,
        token_id: str,
        *,
        min_shares: float = 0.001,
        timeout_seconds: float | None = None,
    ) -> float | None:
        """
        Wait briefly for a freshly bought conditional token to become spendable.

        Phase 1 can fill before the token balance is immediately visible to
        Polymarket's sell path. Hard exits should therefore poll for spendable
        token visibility after an allowance refresh instead of immediately
        assuming the position cannot be sold.
        """
        if not token_id:
            return None

        timeout = (
            self._SELL_POSITION_VISIBILITY_WAIT_SECONDS
            if timeout_seconds is None
            else max(float(timeout_seconds), 0.0)
        )
        poll_seconds = max(float(self._SELL_POSITION_VISIBILITY_POLL_SECONDS), 0.0)
        deadline = time.monotonic() + timeout
        last_visible: float | None = None

        while True:
            visible = await self._available_token_position_shares(token_id)
            if visible is not None:
                last_visible = visible
                if visible + 1e-6 >= max(min_shares, 0.0):
                    return visible
            if time.monotonic() >= deadline or poll_seconds <= 0:
                break
            await asyncio.sleep(poll_seconds)

        return last_visible

    async def _prime_token_sell_readiness(self, token_id: str) -> None:
        """
        Best-effort sell-side allowance refresh right after Phase 1 fills.

        This keeps the signal path intact while improving the odds that a fast
        5-minute hard exit can sell the newly bought token immediately.

        Calls ensure_allowance first (approves the USDC spend), then waits for
        the token balance to propagate into the portfolio API.  Without the
        wait, the stop could fire in the next tick and see visible_position=0
        (token still settling), causing every sell attempt to break immediately
        before placing any order.
        """
        ensure_allowance = getattr(self._client, "ensure_token_sell_allowance", None)
        if not token_id:
            return
        if callable(ensure_allowance):
            try:
                await ensure_allowance(token_id)
            except Exception as exc:
                logger.warning(
                    "BracketExecutor: post-fill sell readiness refresh failed  token={} error={}",
                    token_id,
                    exc,
                )
        # Wait for the token to appear as spendable in the portfolio.  The CLOB
        # fill completes before the balance is visible; without this wait a hard
        # exit firing seconds later will see 0 shares and never place any order.
        visible = await self._wait_for_visible_token_position(token_id)
        if visible is None or visible <= 1e-6:
            logger.warning(
                "BracketExecutor: token not yet visible after fill readiness wait  token={}",
                token_id,
            )

    async def _reconcile_ambiguous_phase1_submission(
        self,
        *,
        token_id: str,
        market_id: str,
        client_order_id: str,
        min_shares: float,
        fallback_price: float,
    ) -> dict[str, Any] | None:
        get_open_orders = getattr(self._client, "get_open_orders", None)
        if callable(get_open_orders):
            try:
                open_orders = await get_open_orders()
            except Exception:
                open_orders = []
            for item in open_orders or []:
                if str(item.get("client_order_id") or "") != client_order_id:
                    continue
                exchange_order_id = str(item.get("exchange_order_id") or "")
                if exchange_order_id and hasattr(self._client, "get_order_status"):
                    try:
                        status = await self._client.get_order_status(exchange_order_id)
                    except Exception:
                        status = {}
                    filled_size = float(status.get("filled_size") or 0.0)
                    if filled_size + 1e-9 >= min_shares:
                        return {
                            "exchange_order_id": exchange_order_id,
                            "status": str(status.get("status") or "MATCHED"),
                            "filled_size": filled_size,
                            "average_fill_price": float(
                                status.get("average_fill_price") or fallback_price
                            ),
                            "remaining_size": float(status.get("remaining_size") or 0.0),
                            "raw": status.get("raw") or status,
                            "reconciled_source": "open_order_status",
                        }

                filled_size = float(item.get("filled_size") or 0.0)
                if filled_size + 1e-9 >= min_shares:
                    return {
                        "exchange_order_id": exchange_order_id,
                        "status": str(item.get("status") or "MATCHED"),
                        "filled_size": filled_size,
                        "average_fill_price": float(item.get("price") or fallback_price),
                        "remaining_size": float(item.get("remaining_size") or 0.0),
                        "raw": item,
                        "reconciled_source": "open_order",
                    }

        get_positions = getattr(self._client, "get_positions", None)
        if callable(get_positions):
            try:
                positions = await get_positions()
            except Exception:
                positions = []
            best_match: dict[str, Any] | None = None
            best_qty = 0.0
            for item in positions or []:
                item_market_id = str(
                    item.get("market_id")
                    or item.get("conditionId")
                    or item.get("market")
                    or ""
                )
                item_token_id = str(
                    item.get("token_id")
                    or item.get("asset")
                    or item.get("asset_id")
                    or item.get("tokenId")
                    or ""
                )
                if market_id and item_market_id and item_market_id != market_id:
                    continue
                if item_token_id != token_id:
                    continue
                qty = self._extract_position_quantity(item)
                if qty > best_qty:
                    best_match = item
                    best_qty = qty
            if best_match is not None and best_qty + 1e-9 >= min_shares:
                return {
                    "exchange_order_id": "",
                    "status": "MATCHED",
                    "filled_size": best_qty,
                    "average_fill_price": self._extract_position_average_price(
                        best_match,
                        fallback_price,
                    ),
                    "remaining_size": 0.0,
                    "raw": best_match,
                    "reconciled_source": "positions",
                }

        return None

    def _build_hard_exit_attempt_prices(
        self, primary_sell_price: float, *, include_emergency: bool = False
    ) -> list[float]:
        primary_sell_price = round(max(0.01, primary_sell_price), 4)
        market_through_cents = max(
            float(getattr(self._cfg, "hard_exit_market_through_cents", 0.02) or 0.0),
            0.0,
        )
        fallback_step_cents = max(
            float(getattr(self._cfg, "hard_exit_fallback_step_cents", 0.0) or 0.0),
            0.0,
        )
        min_sell_price = round(
            max(
                0.01,
                float(getattr(self._cfg, "hard_exit_min_sell_price", 0.01) or 0.01),
            ),
            4,
        )
        attempt_prices = [primary_sell_price]
        fallback_prices: list[float] = []
        if market_through_cents > 0:
            fallback_prices.append(
                round(max(min_sell_price, primary_sell_price - market_through_cents), 4)
            )
        has_fallback_ladder = market_through_cents > 0 or fallback_step_cents > 0
        ladder_floor = min_sell_price
        if fallback_step_cents > 0 and ladder_floor < primary_sell_price:
            next_price = primary_sell_price - fallback_step_cents
            while next_price >= ladder_floor - 1e-9:
                fallback_prices.append(round(max(ladder_floor, next_price), 4))
                next_price -= fallback_step_cents
        for candidate in fallback_prices:
            if candidate < primary_sell_price and candidate not in attempt_prices:
                attempt_prices.append(candidate)
        return attempt_prices

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
        target_shares = max(
            round(self._cfg.phase1_shares, 3),
            self._cfg.min_bracket_shares,
        )
        min_shares = max(round(self._cfg.min_bracket_shares, 3), 0.001)
        shares = target_shares
        position_id = str(uuid.uuid4())[:12]
        entry_style = self._cfg.phase1_entry_style
        chase_cents = max(float(getattr(self._cfg, "phase1_max_chase_cents", 0.0) or 0.0), 0.0)
        entry_price = signal_price
        strategy_entry_cap = float(getattr(self._cfg, "entry_range_high", 0.99) or 0.99)
        entry_model = str(event.entry_model or "").lower()
        if entry_model == "continuation":
            strategy_entry_cap = min(
                strategy_entry_cap,
                float(getattr(self._cfg, "continuation_max_momentum_price", strategy_entry_cap) or strategy_entry_cap),
            )
        elif entry_model == "band_touch":
            strategy_entry_cap = min(
                strategy_entry_cap,
                float(getattr(self._cfg, "immediate_band_entry_high", strategy_entry_cap) or strategy_entry_cap),
            )
        if entry_style == "FOLLOW_TAKER" and chase_cents > 0:
            # Small controlled chase so we can still lock x when the book lifts
            # by a tick between signal snapshot and order submission, but never
            # above the strategy's own entry band ceiling.
            entry_price = round(
                min(
                    0.99,
                    strategy_entry_cap,
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

        retry_budget = max(int(getattr(self._cfg, "phase1_follow_taker_retry_attempts", 0) or 0), 0)
        retry_delay = max(float(getattr(self._cfg, "phase1_follow_taker_retry_delay_seconds", 0.0) or 0.0), 0.0)
        retry_ceiling_price = self._phase1_retry_ceiling(event, entry_price)
        attempt_log: list[dict[str, Any]] = []
        result: dict[str, Any] | None = None
        order_id = ""
        fill_payload: dict[str, Any] = {}
        fill_price = 0.0
        phase1_filled = entry_style != "FOLLOW_TAKER"

        for attempt_index in range(retry_budget + 1):
            attempt_price = entry_price if attempt_index == 0 else retry_ceiling_price

            # On the first FOLLOW_TAKER attempt, optionally skip the pre-submission
            # orderbook fetch (phase1_skip_depth_precheck).  Submitting the FOK
            # immediately saves ~100–300 ms on the hot path; the exchange 400
            # rejection is the safety net if depth isn't there.  On retries we
            # always fetch so the lag-conditioned ceiling gate works correctly.
            skip_depth_precheck = (
                attempt_index == 0
                and entry_style == "FOLLOW_TAKER"
                and bool(getattr(self._cfg, "phase1_skip_depth_precheck", False))
            )
            if skip_depth_precheck:
                retry_context: dict[str, Any] = {}
                shares = target_shares
            else:
                retry_context = await self._phase1_retry_context(
                    token_id=momentum_token_id,
                    limit_price=attempt_price,
                    shares=target_shares,
                )
                executable_shares = self._round_shares(float(retry_context.get("executable_shares") or 0.0))
                if executable_shares + 1e-9 < min_shares:
                    best_ask = float(retry_context.get("best_ask") or 0.0)
                    can_retry_for_lift = (
                        attempt_index < retry_budget
                        and retry_ceiling_price > attempt_price + 1e-9
                        and best_ask > 0.0
                        and best_ask <= retry_ceiling_price + 1e-9
                    )
                    logger.info(
                        "BracketExecutor: Phase 1 skipped for thin depth  asset={} position_id={} "
                        "attempt={} best_ask={} marketable_depth={:.2f} executable={:.3f} "
                        "min_required={:.3f} limit={:.4f}",
                        event.asset,
                        position_id,
                        attempt_index + 1,
                        retry_context.get("best_ask", 0.0),
                        float(retry_context.get("marketable_depth") or 0.0),
                        executable_shares,
                        min_shares,
                        attempt_price,
                    )
                    attempt_log.append(
                        {
                            "attempt": attempt_index + 1,
                            "submit_limit_price": attempt_price,
                            "status": "RETRY_SKIPPED" if attempt_index > 0 else "DEPTH_TOO_THIN",
                            "min_required_shares": min_shares,
                            **retry_context,
                        }
                    )
                    if can_retry_for_lift:
                        logger.info(
                            "BracketExecutor: Phase 1 retrying after thin depth  asset={} "
                            "position_id={} attempt={} best_ask={:.4f} retry_limit={:.4f}",
                            event.asset,
                            position_id,
                            attempt_index + 1,
                            best_ask,
                            retry_ceiling_price,
                        )
                        if retry_delay > 0:
                            await asyncio.sleep(retry_delay)
                        continue
                    break
                shares = executable_shares

            try:
                result = await self._client.place_buy_order(
                    token_id=momentum_token_id,
                    price=attempt_price,
                    size=shares,
                    entry_style=entry_style,
                    client_order_id=position_id,
                )
            except Exception as exc:
                error_text = str(exc)
                error_details = self._exception_diagnostics(exc)
                request_exception_failure = self._is_request_exception_failure(
                    error=error_text,
                    details=error_details,
                )
                reconciled_result: dict[str, Any] | None = None
                retryable = (
                    entry_style == "FOLLOW_TAKER"
                    and attempt_index < retry_budget
                    and not request_exception_failure
                    and self._should_retry_follow_taker_failure(error=error_text)
                )
                refresh_payload: dict[str, Any] = {}
                if request_exception_failure:
                    refresh_payload = await self._refresh_client_for_retry()
                    reconciled_result = await self._reconcile_ambiguous_phase1_submission(
                        token_id=momentum_token_id,
                        market_id=event.market_id,
                        client_order_id=position_id,
                        min_shares=min_shares,
                        fallback_price=attempt_price,
                    )
                if reconciled_result is not None:
                    result = reconciled_result
                    attempt_log.append(
                        {
                            "attempt": attempt_index + 1,
                            "submit_limit_price": attempt_price,
                            "status": "RECONCILED_MATCHED",
                            "order_id": str(result.get("exchange_order_id") or ""),
                            "shares": shares,
                            "request_exception_retryable": request_exception_failure,
                            "session_refresh": refresh_payload,
                            "reconciled_source": result.get("reconciled_source", ""),
                            **retry_context,
                        }
                    )
                    logger.warning(
                        "BracketExecutor: Phase 1 recovered ambiguous submission via reconciliation  "
                        "asset={} position_id={} attempt={} limit={:.4f} source={} refresh_ok={}",
                        event.asset,
                        position_id,
                        attempt_index + 1,
                        attempt_price,
                        result.get("reconciled_source", "unknown"),
                        refresh_payload.get("ok", False) if refresh_payload else "n/a",
                    )
                else:
                    stepdown_shares = 0.0
                    stepdown_order_id = ""
                    attempt_log.append(
                        {
                            "attempt": attempt_index + 1,
                            "submit_limit_price": attempt_price,
                            "status": "ORDER_FAILED",
                            "error": error_text,
                            **error_details,
                            "retryable": retryable,
                            "request_exception_retryable": request_exception_failure,
                            "session_refresh": refresh_payload,
                            **retry_context,
                        }
                    )
                    if (
                        entry_style == "FOLLOW_TAKER"
                        and not request_exception_failure
                        and self._should_retry_follow_taker_failure(error=error_text)
                    ):
                        stepdown_shares = self._phase1_stepdown_size(shares, min_shares)
                    if stepdown_shares > 0:
                        logger.info(
                            "BracketExecutor: Phase 1 retrying same-cycle with smaller block  "
                            "asset={} position_id={} attempt={} full_shares={:.3f} fallback_shares={:.3f} limit={:.4f}",
                            event.asset,
                            position_id,
                            attempt_index + 1,
                            shares,
                            stepdown_shares,
                            attempt_price,
                        )
                        try:
                            stepdown_result = await self._client.place_buy_order(
                                token_id=momentum_token_id,
                                price=attempt_price,
                                size=stepdown_shares,
                                entry_style=entry_style,
                                client_order_id=position_id,
                            )
                            stepdown_order_id = str(stepdown_result.get("exchange_order_id") or "")
                            filled, fill_price, fill_payload = await self._confirm_follow_taker_fill(
                                order_result=stepdown_result,
                                order_id=stepdown_order_id,
                                expected_size=stepdown_shares,
                                fallback_price=attempt_price,
                            )
                            attempt_log.append(
                                {
                                    "attempt": attempt_index + 1,
                                    "submit_limit_price": attempt_price,
                                    "order_id": stepdown_order_id,
                                    "status": "SIZE_STEPDOWN_FILLED" if filled else "SIZE_STEPDOWN_NOT_FILLED",
                                    "shares": stepdown_shares,
                                    "fill_payload": fill_payload,
                                    "stepdown": True,
                                    **retry_context,
                                }
                            )
                            if filled:
                                result = stepdown_result
                                order_id = stepdown_order_id
                                shares = stepdown_shares
                                phase1_filled = True
                                logger.info(
                                    "BracketExecutor: Phase 1 smaller-block fallback FILLED  "
                                    "asset={} position_id={} shares={:.3f} fill_price={:.4f}",
                                    event.asset,
                                    position_id,
                                    shares,
                                    fill_price,
                                )
                                break
                        except Exception as step_exc:
                            step_error_text = str(step_exc)
                            step_error_details = self._exception_diagnostics(step_exc)
                            attempt_log.append(
                                {
                                    "attempt": attempt_index + 1,
                                    "submit_limit_price": attempt_price,
                                    "order_id": stepdown_order_id,
                                    "status": "SIZE_STEPDOWN_FAILED",
                                    "error": step_error_text,
                                    "shares": stepdown_shares,
                                    "stepdown": True,
                                    **step_error_details,
                                    **retry_context,
                                }
                            )
                            logger.info(
                                "BracketExecutor: Phase 1 smaller-block fallback failed  "
                                "asset={} position_id={} attempt={} error={}",
                                event.asset,
                                position_id,
                                attempt_index + 1,
                                step_error_text,
                            )
                    if retryable:
                        logger.info(
                            "BracketExecutor: Phase 1 retrying after order failure  asset={} "
                            "position_id={} attempt={} limit={:.4f} error={} "
                            "exception_source={} cause_source={} refresh_ok={}",
                            event.asset,
                            position_id,
                            attempt_index + 1,
                            attempt_price,
                            error_text,
                            error_details["exception_source"],
                            error_details["cause_source"] or "-",
                            refresh_payload.get("ok", False) if refresh_payload else "n/a",
                        )
                        if retry_delay > 0 and not request_exception_failure:
                            await asyncio.sleep(retry_delay)
                        continue

                    logger.error("BracketExecutor: Phase 1 FAILED — {}", error_text)
                    self._audit({
                        "type": "PHASE1_ORDER_FAILED",
                        "position_id": position_id,
                        "event_id": event.event_id,
                        "asset": event.asset,
                        "entry_price": attempt_price,
                        "shares": shares,
                        "error": error_text,
                        **error_details,
                        "attempt_log": attempt_log,
                    })
                    self._cache_signal_attempt_summary(
                        event=event,
                        position_id=position_id,
                        phase="PHASE1_ORDER_FAILED",
                        shares=shares,
                        requested_price=attempt_price,
                    )
                    return False

            order_id = result.get("exchange_order_id", "")
            raw_status = str(result.get("status") or "").upper()
            logger.info(
                "BracketExecutor: Phase 1 placed  order_id={}  raw_status={} attempt={}",
                order_id, raw_status, attempt_index + 1,
            )

            # FOLLOW_TAKER = FOK. Submission success is not enough; explicitly
            # confirm that the order filled before we credit the position.
            if entry_style == "FOLLOW_TAKER":
                filled, fill_price, fill_payload = await self._confirm_follow_taker_fill(
                    order_result=result,
                    order_id=order_id,
                    expected_size=shares,
                    fallback_price=attempt_price,
                )
                if not filled:
                    attempt_status = self._normalized_status(fill_payload)
                    attempt_log.append(
                        {
                            "attempt": attempt_index + 1,
                            "submit_limit_price": attempt_price,
                            "order_id": order_id,
                            "status": attempt_status,
                            "shares": shares,
                            "fill_payload": fill_payload,
                            **retry_context,
                        }
                    )
                    if (
                        attempt_index < retry_budget
                        and self._should_retry_follow_taker_failure(
                            status=attempt_status,
                            error=str(fill_payload.get("error") or ""),
                        )
                    ):
                        logger.info(
                            "BracketExecutor: Phase 1 retrying after non-fill  asset={} "
                            "position_id={} attempt={} limit={:.4f} status={}",
                            event.asset,
                            position_id,
                            attempt_index + 1,
                            attempt_price,
                            attempt_status,
                        )
                        if retry_delay > 0:
                            await asyncio.sleep(retry_delay)
                        continue

                    logger.warning(
                        "BracketExecutor: Phase 1 not filled after submission  "
                        "asset={} status={} order_id={} position_id={}",
                        event.asset,
                        attempt_status,
                        order_id,
                        position_id,
                    )
                    self._audit({
                        "type": "PHASE1_NOT_FILLED",
                        "position_id": position_id,
                        "event_id": event.event_id,
                        "asset": event.asset,
                        "signal_price": signal_price,
                        "submit_limit_price": attempt_price,
                        "order_id": order_id,
                        "shares": shares,
                        "status": attempt_status,
                        "fill_payload": fill_payload,
                        "attempt_log": attempt_log,
                    })
                    self._cache_signal_attempt_summary(
                        event=event,
                        position_id=position_id,
                        phase="PHASE1_NOT_FILLED",
                        shares=shares,
                        requested_price=attempt_price,
                    )
                    return False

                entry_price = attempt_price
                phase1_filled = True
                break
            break

        if result is None or not phase1_filled:
            last_attempt = attempt_log[-1] if attempt_log else {}
            self._audit({
                "type": "PHASE1_NOT_FILLED",
                "position_id": position_id,
                "event_id": event.event_id,
                "asset": event.asset,
                "signal_price": signal_price,
                "submit_limit_price": entry_price,
                "status": str(last_attempt.get("status") or "RETRY_SKIPPED"),
                "shares": shares,
                "fill_payload": fill_payload,
                "attempt_log": attempt_log,
            })
            self._cache_signal_attempt_summary(
                event=event,
                position_id=position_id,
                phase="PHASE1_NOT_FILLED",
                shares=shares,
                requested_price=entry_price,
            )
            return False

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
            p1_initial_shares=shares,
            p1_price=entry_price,
            p1_shares=shares,
            p1_notional_usd=round(entry_price * shares, 4),
            p1_order_id=order_id,
            p2_token_id=opposite_token_id,
        )

        if entry_style == "FOLLOW_TAKER":
            pos.phase = BracketPhase.PHASE1_FILLED
            pos.p1_fill_price = fill_price
            pos.p1_filled_at = time.time()
            pos.p1_notional_usd = round(pos.p1_fill_price * shares, 4)
            await self._prime_token_sell_readiness(pos.p1_token_id)
            logger.info(
                "BracketExecutor: Phase 1 FILLED (FOLLOW_TAKER)  "
                "asset={}  fill_price={:.4f}  signal_price={:.4f} "
                "submit_limit={:.4f} shares={:.3f}  position_id={}",
                event.asset, pos.p1_fill_price, signal_price, entry_price, shares, position_id,
            )
            self._audit({
                "type": "PHASE1_FILLED",
                "position_id": position_id,
                "event_id": event.event_id,
                "asset": event.asset,
                "fill_price": pos.p1_fill_price,
                "signal_price": signal_price,
                "submit_limit_price": entry_price,
                "shares": shares,
                "style": "FOLLOW_TAKER",
                "execution_mode": self.execution_mode,
                "attempt_log": attempt_log,
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
                if pos.phase == BracketPhase.PHASE1_FILLED and self._cfg.phase2_enabled and not pos.hard_exit_reason:
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

        hard_exit_stop_price = float(self._cfg.hard_exit_stop_price)
        high_entry_cutoff = float(getattr(self._cfg, "hard_exit_high_entry_price", 0.0) or 0.0)
        high_entry_stop_price = float(
            getattr(self._cfg, "hard_exit_high_entry_stop_price", hard_exit_stop_price) or hard_exit_stop_price
        )
        if high_entry_cutoff > 0 and entry >= high_entry_cutoff:
            hard_exit_stop_price = max(hard_exit_stop_price, high_entry_stop_price)

        stop_trigger_price = hard_exit_stop_price + float(
            getattr(self._cfg, "hard_exit_trigger_buffer_cents", 0.0) or 0.0
        )
        stop_hit = mark <= stop_trigger_price
        final_window_loss = (
            seconds_to_close <= self._cfg.hard_exit_final_seconds
            and mark < entry
        )

        continuation_grace_active = (
            pos.entry_model == "continuation"
            and not (high_entry_cutoff > 0 and entry >= high_entry_cutoff)
            and seconds_since_fill < float(
                getattr(self._cfg, "continuation_hard_exit_grace_seconds", 0.0) or 0.0
            )
        )
        safe_arm_stop_protection = (
            bool(getattr(self._cfg, "safe_arm_suspend_stop", True))
            and pos.safe_opposite_price > 0
            and not pos.dipped_below_safe_price
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
        hard_exit_in_progress = (
            pos.hard_exit_last_attempt_ts > 0 or pos.hard_exit_filled_shares > 0
        )
        pos.hard_exit_last_attempt_ts = now_ts
        pos.hard_exit_reason = reason
        pos.hard_exit_attempted = True
        attempt_prices = self._build_hard_exit_attempt_prices(
            hard_exit_stop_price,
            include_emergency=hard_exit_in_progress,
        )
        sell_price = attempt_prices[-1]
        logger.info(
            "BracketExecutor: ⚠ HARD EXIT triggered  "
            "asset={}  reason={}  mark={:.4f}  entry={:.4f}  "
            "trigger_price={:.4f}  attempt_prices={}  seconds_to_close={:.1f}  position_id={}",
            pos.asset, reason, mark, entry, stop_trigger_price, attempt_prices, seconds_to_close, pos.position_id,
        )

        fill_payload: dict[str, object] = {}
        try:
            remaining_shares = float(pos.p1_shares)
            realized_pnl = float(pos.actual_pnl_usd or 0.0)
            total_exit_shares = float(pos.hard_exit_filled_shares or 0.0)
            weighted_exit_notional = float(pos.hard_exit_fill_price or 0.0) * total_exit_shares
            dust_shares = max(
                float(getattr(self._cfg, "hard_exit_dust_shares", 0.0) or 0.0),
                0.0,
            )
            last_error: Exception | None = None
            visible_position_remaining: float | None = None

            ensure_allowance = getattr(self._client, "ensure_token_sell_allowance", None)
            if callable(ensure_allowance):
                try:
                    await ensure_allowance(pos.p1_token_id)
                except Exception as exc:
                    logger.warning(
                        "BracketExecutor: token sell allowance refresh failed  asset={} "
                        "token={} error={} position_id={}",
                        pos.asset,
                        pos.p1_token_id,
                        exc,
                        pos.position_id,
                    )

            for idx, attempt_price in enumerate(attempt_prices):
                if remaining_shares <= 1e-6:
                    break

                if visible_position_remaining is None:
                    visible_position_remaining = await self._available_token_position_shares(
                        pos.p1_token_id
                    )
                visible_position_shares = visible_position_remaining
                sell_target_shares = remaining_shares
                if visible_position_shares is not None:
                    sell_target_shares = min(remaining_shares, visible_position_shares)
                    if sell_target_shares <= 1e-6:
                        # Token is in the positions API but balance shows 0 —
                        # the buy settled at the CLOB but the wallet credit is
                        # still propagating.  Wait for it instead of breaking
                        # immediately; without this wait, every retry attempt
                        # breaks here and the position rides to full loss.
                        logger.warning(
                            "BracketExecutor: hard exit token balance zero — waiting for credit  "
                            "asset={} reason={} remaining={:.3f} visible_position={:.3f} position_id={}",
                            pos.asset,
                            reason,
                            remaining_shares,
                            visible_position_shares,
                            pos.position_id,
                        )
                        waited_shares = await self._wait_for_visible_token_position(
                            pos.p1_token_id,
                            min_shares=min_shares,
                        )
                        if waited_shares is not None and waited_shares > 1e-6:
                            visible_position_remaining = waited_shares
                            sell_target_shares = min(remaining_shares, waited_shares)
                        else:
                            last_error = RuntimeError(
                                "hard exit token balance unavailable after visibility wait"
                            )
                            logger.warning(
                                "BracketExecutor: hard exit token still not spendable after wait  "
                                "asset={} reason={} position_id={}",
                                pos.asset,
                                reason,
                                pos.position_id,
                            )
                            break

                sell_context = await self._marketable_book_context(
                    token_id=pos.p1_token_id,
                    limit_price=attempt_price,
                    shares=sell_target_shares,
                    side="SELL",
                )
                executable_shares = self._round_shares(
                    min(
                        sell_target_shares,
                        float(sell_context.get("executable_shares") or 0.0),
                    )
                )
                if executable_shares <= 1e-6:
                    # The stop trigger stream can still be fresher than a
                    # follow-up orderbook fetch. If we already know the token
                    # position is visible/spendable, send a partial-capable
                    # sell anyway and let the exchange fill whatever is truly
                    # executable instead of silently riding to settlement.
                    if visible_position_shares is not None and sell_target_shares > 1e-6:
                        executable_shares = self._round_shares(sell_target_shares)
                        logger.warning(
                            "BracketExecutor: hard exit proceeding without prechecked depth  "
                            "asset={} reason={} attempt_price={:.4f} visible_position={:.3f} "
                            "best_bid_hint={:.4f} position_id={}",
                            pos.asset,
                            reason,
                            attempt_price,
                            visible_position_shares,
                            float(sell_context.get("best_price") or 0.0),
                            pos.position_id,
                        )
                    else:
                        last_error = RuntimeError(
                            f"hard exit no marketable depth at {attempt_price:.3f}"
                        )
                        continue

                result: dict[str, Any] | None = None
                step_lower_after_no_match = False
                for sell_attempt in range(2):
                    try:
                        result = await self._client.place_sell_order(
                            token_id=pos.p1_token_id,
                            price=attempt_price,
                            size=executable_shares,
                            entry_style="FOLLOW_TAKER_PARTIAL",
                            client_order_id=f"{pos.position_id}-hard-exit-{idx + 1}",
                        )
                        break
                    except Exception as exc:
                        if sell_attempt == 0 and self._is_allowance_failure(str(exc)):
                            if callable(ensure_allowance):
                                try:
                                    await ensure_allowance(pos.p1_token_id)
                                except Exception as refresh_exc:
                                    logger.warning(
                                        "BracketExecutor: token sell allowance retry failed  asset={} "
                                        "token={} error={} position_id={}",
                                        pos.asset,
                                        pos.p1_token_id,
                                        refresh_exc,
                                        pos.position_id,
                                    )
                            refresh_order_session = getattr(self._client, "refresh_live_order_session", None)
                            if callable(refresh_order_session):
                                try:
                                    await refresh_order_session()
                                except Exception as refresh_exc:
                                    logger.warning(
                                        "BracketExecutor: hard-exit session refresh failed  asset={} "
                                        "error={} position_id={}",
                                        pos.asset,
                                        refresh_exc,
                                        pos.position_id,
                                    )
                            refreshed_visible_shares = await self._wait_for_visible_token_position(
                                pos.p1_token_id,
                                min_shares=min(executable_shares, remaining_shares),
                            )
                            if refreshed_visible_shares is None or refreshed_visible_shares <= 1e-6:
                                last_error = RuntimeError(
                                    "hard exit token balance unavailable after allowance refresh"
                                )
                                break
                            visible_position_remaining = refreshed_visible_shares
                            executable_shares = self._round_shares(
                                min(executable_shares, refreshed_visible_shares)
                            )
                            if executable_shares <= 1e-6:
                                last_error = RuntimeError(
                                    "hard exit token balance unavailable after allowance refresh"
                                )
                                break
                            await asyncio.sleep(0.05)
                            continue
                        if self._is_fak_no_match_failure(str(exc)):
                            last_error = RuntimeError(
                                f"hard exit no match at {attempt_price:.3f}: {exc}"
                            )
                            step_lower_after_no_match = True
                            break
                        raise
                if result is None:
                    if step_lower_after_no_match:
                        if idx + 1 < len(attempt_prices):
                            logger.warning(
                                "BracketExecutor: hard exit no-match stepping lower  asset={} "
                                "reason={} attempt_price={:.4f} next_price={:.4f} remaining={:.3f} "
                                "position_id={}",
                                pos.asset,
                                reason,
                                attempt_price,
                                attempt_prices[idx + 1],
                                remaining_shares,
                                pos.position_id,
                            )
                            continue
                        raise last_error or RuntimeError("hard exit no match at floor")
                    raise last_error or RuntimeError("hard exit order placement returned no result")
                order_id = str(result.get("exchange_order_id") or "")
                if order_id:
                    pos.hard_exit_order_ids.append(order_id)
                filled, fill_price, fill_payload = await self._confirm_follow_taker_fill(
                    order_result=result,
                    order_id=order_id,
                    expected_size=executable_shares,
                    fallback_price=attempt_price,
                    allow_partial=True,
                )
                sell_price = attempt_price

                if filled:
                    filled_size = self._reported_filled_size(fill_payload, executable_shares)
                    entry_cost = entry * (1.0 + taker_fee(entry, category="crypto price"))
                    exit_proceeds = fill_price * (1.0 - taker_fee(fill_price, category="crypto price"))
                    chunk_pnl = round((exit_proceeds - entry_cost) * filled_size, 4)
                    realized_pnl = round(realized_pnl + chunk_pnl, 4)
                    remaining_shares = round(max(0.0, remaining_shares - filled_size), 6)
                    total_exit_shares = round(total_exit_shares + filled_size, 6)
                    weighted_exit_notional += fill_price * filled_size
                    if visible_position_remaining is not None:
                        visible_position_remaining = round(
                            max(0.0, visible_position_remaining - filled_size), 6
                        )
                    pos.p1_shares = remaining_shares
                    pos.hard_exit_filled_shares = total_exit_shares
                    pos.hard_exit_fill_price = round(weighted_exit_notional / total_exit_shares, 6)
                    pos.actual_pnl_usd = realized_pnl
                    logger.info(
                        "BracketExecutor: hard exit partial fill  asset={} reason={} "
                        "attempt_price={:.4f} fill_price={:.4f} filled={:.3f} remaining={:.3f} "
                        "position_id={}",
                        pos.asset,
                        reason,
                        attempt_price,
                        fill_price,
                        filled_size,
                        remaining_shares,
                        pos.position_id,
                    )
                    if remaining_shares <= dust_shares + 1e-6:
                        if remaining_shares > 1e-6:
                            logger.info(
                                "BracketExecutor: hard exit residual treated as dust  asset={} "
                                "reason={} residual={:.3f} dust_limit={:.3f} position_id={}",
                                pos.asset,
                                reason,
                                remaining_shares,
                                dust_shares,
                                pos.position_id,
                            )
                        remaining_shares = 0.0
                        pos.p1_shares = 0.0

                if remaining_shares <= 1e-6:
                    pos.phase = BracketPhase.HARD_EXITED
                    pos.closed_at = time.time()
                    pos.hard_exit_attempted = True
                    logger.info(
                        "BracketExecutor: HARD EXIT filled  "
                        "asset={}  avg_fill_price={:.4f}  reason={}  pnl=${:.4f}  position_id={}",
                        pos.asset,
                        pos.hard_exit_fill_price,
                        reason,
                        pos.actual_pnl_usd,
                        pos.position_id,
                    )
                    break

                last_error = RuntimeError(
                    f"hard exit remaining={remaining_shares:.3f} status={self._normalized_status(fill_payload)}"
                )
                if idx + 1 < len(attempt_prices):
                    logger.warning(
                        "BracketExecutor: hard exit retrying lower  asset={} reason={} "
                        "attempt_price={:.4f} next_price={:.4f} remaining={:.3f} position_id={}",
                        pos.asset,
                        reason,
                        attempt_price,
                        attempt_prices[idx + 1],
                        remaining_shares,
                        pos.position_id,
                    )

            if remaining_shares > 1e-6:
                raise last_error or RuntimeError("hard exit not filled")
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
        min_locked_profit = max(
            float(getattr(self._cfg, "phase2_min_locked_profit_per_share", 0.0) or 0.0),
            0.0,
        )
        profitable_y_ceiling = max_profitable_opposite_price(
            x_cost,
            min_net_margin=min_locked_profit,
            category="crypto price",
        )

        # Arm from the dynamic profitable ceiling implied by the actual Phase 1
        # fill. If the move later reaches the preferred target zone, tighten the
        # reclaim anchor downward to that better observed level.
        if pos.safe_opposite_price <= 0:
            if (
                profitable_y_ceiling > 0
                and y_price <= profitable_y_ceiling
                and net_margin >= min_locked_profit
            ):
                arm_price = y_price if y_price <= self._cfg.target_y_price else profitable_y_ceiling
                pos.safe_opposite_price = round(arm_price, 6)
                logger.info(
                    "BracketExecutor: Phase 2 armed  asset={}  safe_y={:.4f}  "
                    "profit_ceiling={:.4f}  x_cost={:.4f}  net_margin={:.4f}  position_id={}",
                    pos.asset, pos.safe_opposite_price, profitable_y_ceiling, x_cost, net_margin, pos.position_id,
                )
                self._audit({
                    "type": "PHASE2_SAFE_ARMED",
                    "position_id": pos.position_id,
                    "asset": pos.asset,
                    "safe_opposite_price": pos.safe_opposite_price,
                    "profit_ceiling": round(profitable_y_ceiling, 6),
                    "net_margin": round(net_margin, 6),
                })
            return

        reversal = self._cfg.phase2_reversal_threshold
        safe_price = pos.safe_opposite_price

        if (
            safe_price > self._cfg.target_y_price
            and y_price <= self._cfg.target_y_price
            and y_price < safe_price
        ):
            pos.safe_opposite_price = round(y_price, 6)
            safe_price = pos.safe_opposite_price
            logger.info(
                "BracketExecutor: Phase 2 safe tightened  asset={}  safe_y={:.4f}  "
                "profit_ceiling={:.4f}  x_cost={:.4f}  net_margin={:.4f}  position_id={}",
                pos.asset, safe_price, profitable_y_ceiling, x_cost, net_margin, pos.position_id,
            )
            self._audit({
                "type": "PHASE2_SAFE_TIGHTENED",
                "position_id": pos.position_id,
                "asset": pos.asset,
                "safe_opposite_price": safe_price,
                "profit_ceiling": round(profitable_y_ceiling, 6),
                "net_margin": round(net_margin, 6),
            })

        # Require continuation below the remembered safe level before we count
        # a reclaim as meaningful.
        breached_safe = pos.min_opposite_price <= safe_price - reversal
        if breached_safe and not pos.dipped_below_safe_price:
            logger.info(
                "BracketExecutor: Phase 2 continuation confirmed  asset={}  "
                "safe_y={:.4f}  floor_now={:.4f}  position_id={}",
                pos.asset, safe_price, pos.min_opposite_price, pos.position_id,
            )
            self._audit({
                "type": "PHASE2_SAFE_BREACHED",
                "position_id": pos.position_id,
                "asset": pos.asset,
                "safe_opposite_price": safe_price,
                "y_price": round(pos.min_opposite_price, 6),
            })
        pos.dipped_below_safe_price = bool(breached_safe)
        if y_price <= safe_price - reversal:
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
        if net_margin < min_locked_profit:
            return

        logger.info(
            "BracketExecutor: ▶ Phase 2 reclaim trigger!  "
            "asset={}  safe_y={:.4f}  y_floor={:.4f}  y_now={:.4f}  "
            "x_cost={:.4f}  net_margin={:.4f}  position_id={}",
            pos.asset, safe_price, pos.min_opposite_price, y_price,
            x_cost, net_margin, pos.position_id,
        )
        await self._place_phase2(pos, y_price, profitable_y_ceiling)

    async def _place_phase2(
        self, pos: BracketPosition, y_price: float, fok_ceiling: float
    ) -> None:
        """
        Place the Phase 2 (opposite_side) buy order.

        FIX #4 + #5: use FOLLOW_TAKER for Phase 2.
          - When the trigger fires we need to lock the bracket NOW before the
            opposite side bounces back.  A PASSIVE_LIMIT resting bid at the
            reversal price risks missing the fill window entirely.
          - FOLLOW_TAKER fills at current market price or fails immediately.
          - On success we mark BRACKET_COMPLETE immediately (no polling needed).

        FIX #6: one immediate FOK retry on rejection.
          - If the book moved between reclaim detection and order submission,
            retry once at y_price + phase2_fok_retry_slippage, capped at
            fok_ceiling (the profitable_y_ceiling from the caller).
        """
        pos.phase = BracketPhase.PHASE2_PENDING   # prevent re-entry next tick
        pos.p2_last_attempt_ts = time.time()

        entry_style = self._cfg.phase2_entry_style
        shares = pos.p1_shares   # symmetric sizing matches both legs

        # Build attempt price list: base price, then one retry at the FOK ceiling.
        base_price = round(min(max(y_price, 0.01), 0.99), 4)
        attempt_prices: list[float] = [base_price]
        fok_slippage = float(getattr(self._cfg, "phase2_fok_retry_slippage", 0.02) or 0.0)
        if entry_style == "FOLLOW_TAKER" and fok_slippage > 0:
            retry_price = round(min(y_price + fok_slippage, fok_ceiling, 0.99), 4)
            if retry_price >= base_price + 0.005:  # meaningful headroom (≥ half a cent)
                attempt_prices.append(retry_price)

        for attempt_idx, entry_price in enumerate(attempt_prices):
            is_retry = attempt_idx > 0

            logger.info(
                "BracketExecutor: placing Phase 2{}  asset={}  style={}  "
                "token={}  price={:.4f}  shares={:.1f}",
                " (FOK retry)" if is_retry else "",
                pos.asset, entry_style, pos.p2_token_id, entry_price, shares,
            )

            try:
                result = await self._client.place_buy_order(
                    token_id=pos.p2_token_id,
                    price=entry_price,
                    size=shares,
                    entry_style=entry_style,
                    client_order_id=f"{pos.position_id}-p2{'r' if is_retry else ''}",
                )
            except Exception as exc:
                exc_str = str(exc).lower()
                is_fok_rejection = (
                    "couldn't be fully filled" in exc_str or "fok orders" in exc_str
                )
                if not is_retry and is_fok_rejection and len(attempt_prices) > 1:
                    logger.info(
                        "BracketExecutor: Phase 2 FOK rejected at {:.4f} — "
                        "retrying at {:.4f}",
                        entry_price, attempt_prices[1],
                    )
                    continue
                # Non-FOK error, or last attempt — revert for 15s cooldown retry.
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
                    if not is_retry and len(attempt_prices) > 1:
                        logger.info(
                            "BracketExecutor: Phase 2 FOK not filled at {:.4f} — "
                            "retrying at {:.4f}",
                            entry_price, attempt_prices[1],
                        )
                        continue
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

            # Order placed (and filled if FOLLOW_TAKER) — exit the retry loop.
            break

        x_cost = pos.p1_fill_price or pos.p1_price
        guaranteed = round(
            1.0
            - x_cost * (1.0 + taker_fee(x_cost, category="crypto price"))
            - pos.p2_price * (1.0 + taker_fee(pos.p2_price, category="crypto price")),
            6,
        )

        logger.info(
            "BracketExecutor: Phase 2 PLACED — bracket LOCKED 🔒  "
            "asset={}  p1={:.4f}  p2={:.4f}  "
            "guaranteed_margin={:.4f}  order_id={}",
            pos.asset, x_cost, pos.p2_price, guaranteed, pos.p2_order_id,
        )
        self._audit({
            "type": "PHASE2_ORDER_PLACED",
            "position_id": pos.position_id,
            "asset": pos.asset,
            "entry_style": entry_style,
            "p2_token_id": pos.p2_token_id,
            "p2_price": pos.p2_price,
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
                pos.actual_pnl_usd = round(pos.actual_pnl_usd + (1.0 - cost) * pos.p1_shares, 4)
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

                pos.actual_pnl_usd = round(pos.actual_pnl_usd + pnl_per_share * pos.p1_shares, 4)
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
                "p1_shares": pos.p1_initial_shares,
                "p1_remaining_shares": pos.p1_shares,
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
            "hard_exit_attempted": pos.hard_exit_attempted,
            "hard_exit_fill_price": pos.hard_exit_fill_price,
            "hard_exit_filled_shares": pos.hard_exit_filled_shares,
            "cancelled": pos.phase == BracketPhase.CANCELLED,
            "actual_pnl_usd": pos.actual_pnl_usd,
            "p1_shares": pos.p1_initial_shares,
            "p1_remaining_shares": pos.p1_shares,
            "p1_fill_price": pos.p1_fill_price or pos.p1_price,
            "p2_fill_price": pos.p2_fill_price or pos.p2_price,
            "p1_order_id": pos.p1_order_id,
            "p2_order_id": pos.p2_order_id,
            "hard_exit_order_ids": list(pos.hard_exit_order_ids),
            "safe_opposite_price": pos.safe_opposite_price,
            "min_opposite_price": pos.min_opposite_price,
            "dipped_below_safe_price": pos.dipped_below_safe_price,
            "phase2_reclaim_seen": bool(
                pos.p2_last_attempt_ts > 0 or pos.p2_price > 0 or pos.p2_fill_price > 0
            ),
            "phase2_order_attempted": bool(pos.p2_last_attempt_ts > 0),
            "phase2_attempt_price": (
                pos.p2_price
                or (pos.safe_opposite_price if pos.p2_last_attempt_ts > 0 else 0.0)
            ),
        }
