"""
Bracket strategy — direction signal evaluator.

This module implements the Phase 1 entry signal for the paired bracket trade
on Polymarket 15-minute BTC/ETH binary markets.

Strategy recap:
  - x = momentum_side (YES if BTC going up, NO if going down), bought at ~58¢
  - y = opposite_side, watched until it falls to ~34¢ and reverses
  - Bracket: x*(1+fee_x) + y*(1+fee_y) < $1.00 → guaranteed profit at resolution
  - This module fires the signal for WHEN to buy x. No orders are placed here.

Signal logic (all gates must pass):
  1. Time gate:     >= time_gate_minutes remaining in the 15m window
  2. Asset move:    BTC/ETH has moved >= min_asset_move_pct from window open
  3. Price range:   momentum_side ask is in [entry_range_low, entry_range_high]
  4. Price sanity:  both YES and NO asks are non-zero
  5. Chop filter:   :00-second checkpoints show a clean directional move
  6. Lag gap:       GBM-implied fair value > actual price by >= lag_threshold
  7. Cooldown:      same side hasn't already fired a signal this window

Key insight — the "lag gap":
  BTC moves on Binance FIRST, Polymarket reprices AFTER. If BTC has moved
  up 0.5% but YES is only at 58¢ when the GBM model says it should be 65¢,
  there is a 7¢ lag gap. Entering YES at 58¢ when the market still needs to
  reprice gives better bracket geometry. We reuse _estimate_fair_p_up() from
  lag_signal.py (same GBM model, no duplication).

Post-signal observation:
  After a signal fires, every poll cycle continues to track:
    - bracket_margin: could we have locked a profitable bracket?
    - phase2_trigger: did the opposite side bottom and reverse?
    - momentum_side_peak: best unrealised P&L on the first leg
  These are written to JSONL logs for calibration before live trading.

Usage:
  evaluator = DirectionSignalEvaluator("BTC", cfg, cfg.btc)
  evaluator.on_window_open(window_ts, asset_price)
  # ... each poll cycle:
  evaluator.maybe_record_checkpoint(asset_price)
  evaluator.update_post_signal(yes_ask, no_ask, yes_ask_size, no_ask_size)
  signal = evaluator.evaluate(asset_price, yes_ask, no_ask, market_meta)

The evaluator itself never places orders. In shadow/live mode the surrounding
observer loop hands the emitted BracketSignalEvent to BracketExecutor so the
same signal path is used in every mode.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from loguru import logger  # type: ignore[import]
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from src.config import AssetVolConfig, CryptoDirectionConfig
from src.fees import max_profitable_opposite_price, taker_fee
from src.lag_signal import _estimate_fair_p_up
from src.models import BracketSignalEvent, PostSignalObservation, WindowCheckpoint
from src.window_report import WindowReportWriter


# ---------------------------------------------------------------------------
# Per-window internal state (not exposed outside this module)
# ---------------------------------------------------------------------------

@dataclass
class _WindowState:
    """
    Mutable state scoped to one 15-minute window for one asset.

    Reset completely on each window transition via on_window_open().
    Keeping this separate from BracketSignalEvent (the immutable log record)
    lets us mutate freely during the window without worrying about the audit log.
    """
    asset: str
    window_open_ts: int         # Unix ts of window start
    window_close_ts: int        # window_open_ts + 900
    asset_open: float           # BTC/ETH price at window open (or synthetic if mid-window)
    checkpoints: list[WindowCheckpoint] = field(default_factory=list)  # :00-second prices
    mid_window_start: bool = False  # True if bot started mid-window (baseline is uncertain)

    # Signal state
    signal_fired: bool = False
    signal_fired_side: str = ""     # "YES" or "NO" — prevents duplicate signals same side

    # :00-second deduplication — avoid recording the same second twice
    last_checkpoint_second: int = -1

    # Post-signal tracking (populated after first signal fires)
    observation: PostSignalObservation | None = None


# ---------------------------------------------------------------------------
# Gate result codes — used in evaluation log for debugging
# ---------------------------------------------------------------------------

GATE_TIME = "TIME_GATE_FAIL"
GATE_MOVE = "ASSET_MOVE_INSUFFICIENT"
GATE_RANGE = "PRICE_RANGE_FAIL"
GATE_STALE = "PRICES_STALE"
GATE_CHOP = "CHOP_FILTER_FAIL"
GATE_LAG = "LAG_GAP_INSUFFICIENT"
GATE_COOLDOWN = "ALREADY_FIRED_THIS_WINDOW"
GATE_NO_STATE = "NO_WINDOW_STATE"
GATE_FIRED = "SIGNAL_FIRED"


# ---------------------------------------------------------------------------
# DirectionSignalEvaluator — one instance per asset
# ---------------------------------------------------------------------------

class DirectionSignalEvaluator:
    """
    Evaluates the direction signal for a single asset (BTC or ETH).

    The evaluator is stateful within a 15-minute window. Call on_window_open()
    at the start of each new window to reset state. Call evaluate() every
    poll_interval_seconds to check all gates.

    All decisions (pass or fail) are logged to the evaluation JSONL so you
    can see exactly why the signal did or did not fire and calibrate gates.
    """

    def __init__(
        self,
        asset: str,
        cfg: CryptoDirectionConfig,
        asset_vol_cfg: AssetVolConfig,
    ) -> None:
        self.asset = asset
        self._cfg = cfg
        self._asset_vol_cfg = asset_vol_cfg

        # Volatility per second used by the GBM model.
        # _estimate_fair_p_up() expects vol_per_second = annual_vol / sqrt(seconds_per_year)
        # This converts our annual_vol_pct config value to the right unit.
        seconds_per_year = 365.0 * 24.0 * 3600.0
        self._vol_per_second: float = asset_vol_cfg.annual_vol_pct / math.sqrt(seconds_per_year)

        self._state: _WindowState | None = None
        self._last_event: BracketSignalEvent | None = None   # most recent signal event

        # Ensure log directories exist
        Path(cfg.signal_event_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.evaluation_log_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "DirectionSignalEvaluator init asset={} vol_per_second={:.8f} "
            "entry_range=[{}, {}] lag_threshold={} time_gate_minutes={} "
            "continuation={} cont_min_mins={} cont_min_move={} cont_min_chop={} "
            "cont_max_momentum={} cont_max_neg_lag={} cont_ignore_lag={} "
            "band_entry={} band_range=[{}, {}]",
            asset, self._vol_per_second,
            cfg.entry_range_low, cfg.entry_range_high,
            cfg.lag_threshold, cfg.time_gate_minutes,
            cfg.continuation_enabled,
            cfg.continuation_min_minutes_remaining,
            cfg.continuation_min_asset_move_pct,
            cfg.continuation_min_chop_score,
            cfg.continuation_max_momentum_price,
            cfg.continuation_max_negative_lag_gap,
            cfg.continuation_ignore_lag_veto,
            cfg.immediate_band_entry_enabled,
            cfg.immediate_band_entry_low,
            cfg.immediate_band_entry_high,
        )

    # ------------------------------------------------------------------ #
    # Window lifecycle
    # ------------------------------------------------------------------ #

    def on_window_open(
        self,
        window_open_ts: int,
        asset_price: float,
        mid_window: bool = False,
    ) -> None:
        """
        Reset state for a new 15-minute window.

        Called by the run loop when a window transition is detected.

        mid_window=True signals that the bot started mid-window and the
        asset_open price is a synthetic baseline (not the true window open).
        Signals fired in mid_window windows are marked in the log and should
        be excluded from calibration statistics.
        """
        self._state = _WindowState(
            asset=self.asset,
            window_open_ts=window_open_ts,
            window_close_ts=window_open_ts + self._cfg.window_duration_seconds,
            asset_open=asset_price,
            mid_window_start=mid_window,
            checkpoints=[WindowCheckpoint(ts=float(window_open_ts), price=asset_price)],
        )
        self._last_event = None

        logger.info(
            "Window opened asset={} window_ts={} asset_open={} mid_window={}",
            self.asset, window_open_ts, asset_price, mid_window,
        )

    def on_window_close(
        self,
        asset_close: float,
        yes_won: bool | None,
        outcome_source: str = "asset_price_proxy",
    ) -> None:
        """
        Record the window outcome and write post-signal observation to the log.

        Called by the run loop when a window transition is detected (the old
        window is closing as the new one opens). We use the final asset price
        from the RTDS feed as the settlement price proxy.

        yes_won:
          True  → BTC/ETH ended HIGHER than window open → YES resolves $1
          False → BTC/ETH ended LOWER  → NO resolves $1
          None  → Unknown (e.g., mid-window start; no reliable baseline)
        """
        if not self._state or not self._last_event or not self._state.observation:
            return

        obs = self._state.observation
        obs.asset_close_price = asset_close
        obs.outcome_source = outcome_source

        if yes_won is True:
            obs.outcome = "YES_WINS"
        elif yes_won is False:
            obs.outcome = "NO_WINS"
        else:
            obs.outcome = "UNKNOWN"

        # Estimate what P&L the early-exit path would have captured:
        # sell momentum_side at its peak price vs. what we paid (momentum_price at signal)
        entry_price = self._last_event.momentum_price
        peak_price = obs.momentum_side_peak
        obs.estimated_phase1_exit_pnl = round(
            (peak_price - entry_price) * self._required_shares(),
            4,
        )

        # CRITICAL: populate the observation on the in-memory event object so that
        # when the observer loop calls ev._last_event.model_dump() immediately after
        # ev.on_window_close(), the observation dict is present.  Previously this was
        # only written to the JSONL outcome entry (below), leaving _last_event.observation
        # as None and causing window_report to see no observation data.
        self._last_event.observation = obs

        # Append outcome record to the signal event log
        _append_jsonl(
            self._cfg.signal_event_log_path,
            {
                "type": "outcome",
                "event_id": self._last_event.event_id,
                "asset": self.asset,
                "window_close_ts": self._state.window_close_ts,
                "asset_close": asset_close,
                "outcome_source": outcome_source,
                "observation": obs.model_dump(),
            },
        )
        logger.info(
            "Window closed asset={} outcome={} source={} bracket_formed={} "
            "phase2_triggered={} phase1_exit_pnl={:.4f}",
            self.asset, obs.outcome, outcome_source, obs.bracket_would_have_formed,
            obs.phase2_would_have_triggered, obs.estimated_phase1_exit_pnl,
        )

    # ------------------------------------------------------------------ #
    # Checkpoint recording
    # ------------------------------------------------------------------ #

    def maybe_record_checkpoint(self, asset_price: float) -> None:
        """
        Record the asset price as a :00-second checkpoint if applicable.

        We only record when we're exactly on a :00-second boundary (any minute
        mark, e.g., 14:03:00, 14:04:00) and haven't recorded this second yet.

        These checkpoints are used by the chop filter to evaluate whether the
        asset has moved cleanly or choppily since window open.
        """
        if not self._state:
            return

        now_int = int(time.time())
        # Only at :00-second marks
        if now_int % 60 != 0:
            return
        # Deduplicate — don't record the same second twice
        if now_int == self._state.last_checkpoint_second:
            return

        self._state.checkpoints.append(
            WindowCheckpoint(ts=float(now_int), price=asset_price)
        )
        self._state.last_checkpoint_second = now_int

        # Keep at most 20 checkpoints (20 minutes of history — more than enough)
        if len(self._state.checkpoints) > 20:
            self._state.checkpoints = self._state.checkpoints[-20:]

    # ------------------------------------------------------------------ #
    # Post-signal tracking
    # ------------------------------------------------------------------ #

    def _required_shares(self) -> float:
        return max(round(self._cfg.phase1_shares, 1), self._cfg.min_bracket_shares)

    def update_post_signal(
        self,
        yes_ask: float,
        no_ask: float,
        yes_ask_size: float = 0.0,
        no_ask_size: float = 0.0,
    ) -> None:
        """
        Update post-signal observation metrics on each poll cycle.

        After a signal fires, we keep watching to answer:
          - Did the bracket become achievable (x + y + fees < $1.00)?
          - When did we first see a safe opposite-side price (y_safe)?
          - Did price extend below y_safe and later reclaim it on reversal?
          - What was the best unrealised gain on the momentum side?

        This data is the primary calibration input before live trading.
        """
        if not self._state or not self._state.signal_fired or not self._last_event:
            return
        if not self._state.observation:
            return

        obs = self._state.observation
        x_price = self._last_event.momentum_price  # what we "paid" for the first leg
        momentum_side = self._last_event.momentum_side

        # y is the opposite side — the one we're watching for the second leg.
        # We also track the top-of-book ask size so the paper report can tell
        # whether this would actually have been executable for the configured
        # live share size.
        y_price = no_ask if momentum_side == "YES" else yes_ask
        y_ask_size = no_ask_size if momentum_side == "YES" else yes_ask_size

        # Update momentum side peak and trough (for hard-exit loss modeling in paper P&L)
        momentum_now = yes_ask if momentum_side == "YES" else no_ask
        if momentum_now > obs.momentum_side_peak:
            obs.momentum_side_peak = momentum_now
        # Track lowest momentum-side price (models the hard-exit stop trigger)
        if 0 < momentum_now < obs.min_momentum_price:
            obs.min_momentum_price = momentum_now

        # Bracket margin: how much profit is locked if we buy y right now?
        # margin > 0 means we're profitable; use actual Polymarket fee curve.
        if y_price > 0:
            fee_x = taker_fee(x_price, category="crypto price")
            fee_y = taker_fee(y_price, category="crypto price")
            net_bracket = 1.0 - x_price * (1.0 + fee_x) - y_price * (1.0 + fee_y)
            min_locked_profit = float(
                getattr(self._cfg, "phase2_min_locked_profit_per_share", 0.0) or 0.0
            )
            profitable_y_ceiling = max_profitable_opposite_price(
                x_price,
                min_net_margin=min_locked_profit,
                category="crypto price",
            )

            if net_bracket > obs.peak_bracket_margin:
                obs.peak_bracket_margin = round(net_bracket, 6)

            if net_bracket > 0:
                obs.bracket_would_have_formed = True

            # Track the floor of y (lowest price seen after the signal).
            if y_price < obs.min_opposite_price:
                obs.min_opposite_price = y_price

            # y_safe discovery:
            # arm from the dynamic profitable ceiling implied by the actual
            # Phase 1 fill. If the move later reaches the preferred target zone,
            # tighten the reclaim anchor downward to that better observed price.
            if (
                obs.safe_opposite_price is None
                and profitable_y_ceiling > 0
                and y_price <= profitable_y_ceiling
                and net_bracket >= min_locked_profit
            ):
                arm_price = y_price if y_price <= self._cfg.target_y_price else profitable_y_ceiling
                obs.safe_opposite_price = round(arm_price, 6)
                logger.info(
                    "Safe opposite price armed asset={} safe_y={:.4f} "
                    "profit_ceiling={:.4f} bracket_margin={:.4f}",
                    self.asset, obs.safe_opposite_price, profitable_y_ceiling, net_bracket,
                )

            safe_price = obs.safe_opposite_price

            if (
                safe_price is not None
                and safe_price > self._cfg.target_y_price
                and y_price <= self._cfg.target_y_price
                and y_price < safe_price
            ):
                obs.safe_opposite_price = round(y_price, 6)
                safe_price = obs.safe_opposite_price
                logger.info(
                    "Safe opposite price tightened asset={} safe_y={:.4f} "
                    "profit_ceiling={:.4f} bracket_margin={:.4f}",
                    self.asset, safe_price, profitable_y_ceiling, net_bracket,
                )

            # Once y_safe exists, require the move to extend below it before
            # we credit a reclaim. This avoids buying the second leg the instant
            # the equation merely becomes safe.
            breached_safe = (
                safe_price is not None
                and obs.min_opposite_price <= safe_price - self._cfg.phase2_reversal_threshold
            )
            obs.dipped_below_safe_price = bool(breached_safe)

            # Phase 2 reclaim check:
            #   1. y_safe was armed
            #   2. price extended below y_safe
            #   3. the opposite ask later reclaimed y_safe on reversal
            #   4. the bracket is still profitable at the observed reclaim price
            if (
                safe_price is not None
                and obs.dipped_below_safe_price
                and y_price >= safe_price
                and y_price <= safe_price + self._cfg.phase2_reversal_threshold
                and net_bracket >= min_locked_profit
            ):
                required_shares = self._required_shares()
                phase2_would_fill = y_ask_size >= required_shares

                if not obs.phase2_reclaim_seen:
                    obs.phase2_reclaim_seen = True
                    obs.phase2_trigger_price = round(y_price, 6)
                    obs.phase2_trigger_ask_size = round(y_ask_size, 6)
                    obs.phase2_would_fill = phase2_would_fill
                    logger.info(
                        "Phase 2 reclaim seen asset={} safe_y={:.4f} y_now={:.4f} "
                        "y_floor={:.4f} ask_size={:.2f} required_shares={:.1f} "
                        "bracket_margin={:.4f}",
                        self.asset, safe_price, y_price, obs.min_opposite_price,
                        y_ask_size, required_shares, net_bracket,
                    )

                if phase2_would_fill and not obs.phase2_would_have_triggered:
                    obs.phase2_would_have_triggered = True
                    logger.info(
                        "Phase 2 would have triggered asset={} safe_y={:.4f} y_now={:.4f} "
                        "y_floor={:.4f} bracket_margin={:.4f}",
                        self.asset, safe_price, y_price, obs.min_opposite_price, net_bracket,
                    )

    # ------------------------------------------------------------------ #
    # Signal evaluation — the main entry point every poll cycle
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        asset_price: float,
        yes_ask: float,
        no_ask: float,
        market_meta: dict[str, Any],
    ) -> BracketSignalEvent | None:
        """
        Run all signal gates and return a BracketSignalEvent if all pass.

        Called every poll_interval_seconds by the run loop. Returns None (and
        logs the failure reason) if any gate fails. Returns a BracketSignalEvent
        (and writes to JSONL) if all gates pass.

        Parameters:
          asset_price : current BTC/ETH price from Binance RTDS feed
          yes_ask     : best ask price for the YES token (what you'd pay to buy)
          no_ask      : best ask price for the NO token
          market_meta : {market_id, yes_token_id, no_token_id, liquidity, volume}
        """
        # Run all gates sequentially — first failure short-circuits and logs
        gate = self._run_gates(asset_price, yes_ask, no_ask)
        _log_evaluation(
            self._cfg,
            self._asset_vol_cfg,
            self._vol_per_second,
            self.asset,
            self._state,
            gate,
            yes_ask,
            no_ask,
            asset_price,
            market_meta.get("yes_ask_size", 0.0),
            market_meta.get("no_ask_size", 0.0),
        )

        if gate["result"] != GATE_FIRED:
            return None

        # Build the signal event, update state, and write to log
        return self._build_and_emit_event(gate, yes_ask, no_ask, asset_price, market_meta)

    # ------------------------------------------------------------------ #
    # Internal: gate evaluation
    # ------------------------------------------------------------------ #

    def _run_gates(
        self,
        asset_price: float,
        yes_ask: float,
        no_ask: float,
    ) -> dict[str, Any]:
        """
        Run all signal gates in sequence. Returns a dict with at minimum:
          {"result": GATE_CODE, ...diagnostic fields...}

        Short-circuits on the first failing gate to keep logs readable.
        """
        # Precondition: window state must exist
        if not self._state:
            return {"result": GATE_NO_STATE}

        now = time.time()
        state = self._state

        # ---- Gate 0: Window settle ----
        # Skip the noisy first N seconds after window open. The AMM seeds
        # liquidity at extreme spreads and BTC flickers ±0.02% before real
        # market makers narrow the book. Waiting 60-90s for price stabilization
        # dramatically reduces false signals from opening noise.
        seconds_since_open = now - state.window_open_ts
        if seconds_since_open < self._cfg.window_settle_seconds:
            return {
                "result": "WINDOW_SETTLING",
                "seconds_since_open": round(seconds_since_open, 1),
                "window_settle_seconds": self._cfg.window_settle_seconds,
            }

        # ---- Gate 1: Time ----
        # Must have enough window left for Phase 2 to develop.
        minutes_remaining = (state.window_close_ts - now) / 60.0
        if minutes_remaining < self._cfg.time_gate_minutes:
            return {
                "result": GATE_TIME,
                "minutes_remaining": round(minutes_remaining, 2),
                "time_gate_minutes": self._cfg.time_gate_minutes,
            }

        # ---- Gate 2: Asset move ----
        # BTC/ETH must have moved meaningfully from window open.
        if state.asset_open <= 0:
            return {"result": GATE_MOVE, "reason": "asset_open_zero"}

        asset_move_pct = (asset_price - state.asset_open) / state.asset_open
        if abs(asset_move_pct) < self._asset_vol_cfg.min_asset_move_pct:
            return {
                "result": GATE_MOVE,
                "asset_move_pct": round(asset_move_pct, 6),
                "min_required": self._asset_vol_cfg.min_asset_move_pct,
            }

        # ---- Gate 3: Momentum side and price range ----
        # Determine which token is the "momentum side" based on direction.
        # Then check that it's in the target entry range (57-60¢).
        if asset_move_pct > 0:
            # BTC/ETH going UP → YES is the momentum side
            momentum_side = "YES"
            momentum_price = yes_ask
            opposite_price = no_ask
        else:
            # BTC/ETH going DOWN → NO is the momentum side
            momentum_side = "NO"
            momentum_price = no_ask
            opposite_price = yes_ask

        if not (self._cfg.entry_range_low <= momentum_price <= self._cfg.entry_range_high):
            return {
                "result": GATE_RANGE,
                "momentum_side": momentum_side,
                "momentum_price": round(momentum_price, 4),
                "entry_range": [self._cfg.entry_range_low, self._cfg.entry_range_high],
            }

        # ---- Gate 4: Price sanity ----
        # Both YES and NO prices must be available (non-zero means data is present).
        if yes_ask <= 0 or no_ask <= 0:
            return {
                "result": GATE_STALE,
                "yes_ask": yes_ask,
                "no_ask": no_ask,
            }

        # ---- Gate 5: Chop filter ----
        # The asset must have moved CLEANLY in one direction — not oscillated.
        # We evaluate the :00-second checkpoint history for directional consistency.
        # The current live price is included as a virtual endpoint so we get a real
        # score even before the first minute-mark checkpoint arrives.
        chop_score = self._compute_chop_score(asset_move_pct, asset_price)
        if chop_score < self._cfg.chop_min_score:
            return {
                "result": GATE_CHOP,
                "chop_score": round(chop_score, 3),
                "min_required": self._cfg.chop_min_score,
                "checkpoints_used": len(state.checkpoints[-self._cfg.chop_window:]),
            }

        # ---- Gate 6: Cooldown ----
        # Once a window has fired, the executor and report should treat later
        # evaluations as cooldown rather than emitting duplicate signals. This
        # applies before lag/continuation selection so continuation cannot
        # bypass the one-signal-per-window rule.
        if state.signal_fired:
            return {
                "result": GATE_COOLDOWN,
                "fired_side": state.signal_fired_side,
            }

        # ---- Gate 7: Lag gap ----
        # The GBM fair-value estimate for momentum_side must exceed the actual
        # Polymarket price by >= lag_threshold. This means the market hasn't
        # fully priced in the BTC/ETH move yet — we have a better entry.
        time_remaining_seconds = float(state.window_close_ts) - now
        implied_p_up = _estimate_fair_p_up(
            chainlink_price=asset_price,
            start_price=state.asset_open,
            time_remaining_seconds=time_remaining_seconds,
            realized_vol_per_second=self._vol_per_second,
        )
        # Convert to the momentum_side's perspective:
        # If YES is momentum, implied_momentum_price = P(BTC ends up) = implied_p_up
        # If NO is momentum, implied_momentum_price = P(BTC ends down) = 1 - implied_p_up
        implied_momentum_price = implied_p_up if momentum_side == "YES" else (1.0 - implied_p_up)
        lag_gap = implied_momentum_price - momentum_price

        if lag_gap < self._cfg.lag_threshold:
            if self._immediate_band_entry_ok(
                momentum_price=momentum_price,
            ):
                return {
                    "result": GATE_FIRED,
                    "entry_model": "band_touch",
                    "minutes_remaining": round(minutes_remaining, 2),
                    "asset_move_pct": round(asset_move_pct, 6),
                    "momentum_side": momentum_side,
                    "momentum_price": round(momentum_price, 4),
                    "opposite_price": round(opposite_price, 4),
                    "chop_score": round(chop_score, 3),
                    "lag_gap": round(lag_gap, 4),
                    "implied_momentum_price": round(implied_momentum_price, 4),
                }
            if self._continuation_entry_ok(
                asset_move_pct=asset_move_pct,
                minutes_remaining=minutes_remaining,
                momentum_price=momentum_price,
                opposite_price=opposite_price,
                chop_score=chop_score,
                lag_gap=lag_gap,
            ):
                return {
                    "result": GATE_FIRED,
                    "entry_model": "continuation",
                    "minutes_remaining": round(minutes_remaining, 2),
                    "asset_move_pct": round(asset_move_pct, 6),
                    "momentum_side": momentum_side,
                    "momentum_price": round(momentum_price, 4),
                    "opposite_price": round(opposite_price, 4),
                    "chop_score": round(chop_score, 3),
                    "lag_gap": round(lag_gap, 4),
                    "implied_momentum_price": round(implied_momentum_price, 4),
                }
            return {
                "result": GATE_LAG,
                "lag_gap": round(lag_gap, 4),
                "implied_momentum_price": round(implied_momentum_price, 4),
                "momentum_price": round(momentum_price, 4),
                "min_required": self._cfg.lag_threshold,
            }

        # ---- All gates passed ----
        return {
            "result": GATE_FIRED,
            "entry_model": "lag",
            "minutes_remaining": round(minutes_remaining, 2),
            "asset_move_pct": round(asset_move_pct, 6),
            "momentum_side": momentum_side,
            "momentum_price": round(momentum_price, 4),
            "opposite_price": round(opposite_price, 4),
            "chop_score": round(chop_score, 3),
            "lag_gap": round(lag_gap, 4),
            "implied_momentum_price": round(implied_momentum_price, 4),
        }

    def _compute_chop_score(self, asset_move_pct: float, current_price: float) -> float:
        """
        Score how cleanly directional the asset's move has been.

        Looks at the last `chop_window` :00-second checkpoints and counts how
        many consecutive steps moved in the same direction as the overall move.
        Counter-moves larger than chop_max_reversal_pct are penalised.

        The current live price is appended as a virtual endpoint so we always
        have at least one step (window-open → now), eliminating the need for
        a soft-pass placeholder. A real direction score is available immediately.

        Returns:
          1.0 = perfectly monotonic move in the right direction
          0.5 = no seeded checkpoint (shouldn't happen after window open)
          0.0 = all steps are reversals
        """
        state = self._state
        assert state is not None

        recent = list(state.checkpoints[-self._cfg.chop_window:])
        # Append current live price as a virtual endpoint for real-time scoring.
        # This means even before the first :00-second checkpoint arrives we
        # evaluate window-open → current as a real directional step.
        if current_price > 0:
            recent.append(WindowCheckpoint(ts=time.time(), price=current_price))

        if len(recent) < 2:
            # No seeded checkpoint at all — cannot assess direction.
            return 0.5

        direction = 1 if asset_move_pct > 0 else -1
        total_steps = len(recent) - 1
        clean_steps = 0

        for i in range(total_steps):
            step = recent[i + 1].price - recent[i].price
            step_pct = step / recent[i].price if recent[i].price > 0 else 0.0

            if step * direction > 0:
                # This step moved in the right direction
                clean_steps += 1
            elif abs(step_pct) > self._cfg.chop_max_reversal_pct:
                # This step is a meaningful reversal — penalise
                clean_steps -= 1
            # Small steps in the wrong direction (< chop_max_reversal_pct) are neutral

        return max(0.0, min(1.0, clean_steps / total_steps))

    def _continuation_entry_ok(
        self,
        *,
        asset_move_pct: float,
        minutes_remaining: float,
        momentum_price: float,
        opposite_price: float,
        chop_score: float,
        lag_gap: float,
    ) -> bool:
        """
        Allow a narrow continuation-style entry when classic lag is absent.

        This is deliberately strict so we only capture very clean early moves
        that still fit the bracket thesis, rather than broadly loosening the
        main signal band.
        """
        cfg = self._cfg
        if not cfg.continuation_enabled:
            return False
        if minutes_remaining < cfg.continuation_min_minutes_remaining:
            return False
        if abs(asset_move_pct) < cfg.continuation_min_asset_move_pct:
            return False
        if chop_score < cfg.continuation_min_chop_score:
            return False
        if momentum_price > cfg.continuation_max_momentum_price:
            return False
        if opposite_price > cfg.continuation_max_opposite_price:
            return False
        if (
            not cfg.continuation_ignore_lag_veto
            and lag_gap < -cfg.continuation_max_negative_lag_gap
        ):
            return False
        return True

    def _immediate_band_entry_ok(
        self,
        *,
        momentum_price: float,
    ) -> bool:
        """
        Aggressive thesis lane:

        If the momentum side is already trading inside the original 57-61c band,
        we buy immediately instead of waiting for lag confirmation.  This keeps
        the base move/time/chop gates intact but removes the lag veto.
        """
        cfg = self._cfg
        if not cfg.immediate_band_entry_enabled:
            return False
        return cfg.immediate_band_entry_low <= momentum_price <= cfg.immediate_band_entry_high

    # ------------------------------------------------------------------ #
    # Internal: signal event construction
    # ------------------------------------------------------------------ #

    def _build_and_emit_event(
        self,
        gate: dict[str, Any],
        yes_ask: float,
        no_ask: float,
        asset_price: float,
        market_meta: dict[str, Any],
    ) -> BracketSignalEvent:
        """
        Build a BracketSignalEvent, update window state, and write to JSONL log.

        This is only called when all gates have passed (GATE_FIRED).
        """
        assert self._state is not None
        state = self._state
        momentum_side = gate["momentum_side"]
        momentum_price = gate["momentum_price"]

        # Compute fee context using the actual Polymarket crypto fee curve
        fee_x = taker_fee(momentum_price, category="crypto price")
        fee_y = taker_fee(self._cfg.target_y_price, category="crypto price")
        # Net bracket margin if leg 2 enters at target_y_price:
        net_bracket_at_target = round(
            1.0
            - momentum_price * (1.0 + fee_x)
            - self._cfg.target_y_price * (1.0 + fee_y),
            6,
        )

        event = BracketSignalEvent(
            event_id=str(uuid4()),
            fired_at=datetime.now(timezone.utc),
            asset=self.asset,
            window_open_ts=state.window_open_ts,
            window_close_ts=state.window_close_ts,
            minutes_remaining=gate["minutes_remaining"],
            mid_window_start=state.mid_window_start,
            asset_open=state.asset_open,
            asset_current=asset_price,
            asset_move_pct=gate["asset_move_pct"],
            entry_model=str(gate.get("entry_model") or "lag"),
            momentum_side=momentum_side,
            momentum_price=momentum_price,
            opposite_price=gate["opposite_price"],
            momentum_ask_size=float(
                market_meta.get("yes_ask_size", 0.0)
                if momentum_side == "YES"
                else market_meta.get("no_ask_size", 0.0)
            ),
            opposite_ask_size=float(
                market_meta.get("no_ask_size", 0.0)
                if momentum_side == "YES"
                else market_meta.get("yes_ask_size", 0.0)
            ),
            required_shares=self._required_shares(),
            phase1_would_fill=(
                float(
                    market_meta.get("yes_ask_size", 0.0)
                    if momentum_side == "YES"
                    else market_meta.get("no_ask_size", 0.0)
                ) >= self._required_shares()
            ),
            implied_momentum_price=gate["implied_momentum_price"],
            lag_gap=gate["lag_gap"],
            chop_score=gate["chop_score"],
            checkpoints=list(state.checkpoints),
            fee_at_momentum_price=round(fee_x, 6),
            fee_at_target_y=round(fee_y, 6),
            net_bracket_at_target=net_bracket_at_target,
            market_id=market_meta.get("market_id", ""),
            yes_token_id=market_meta.get("yes_token_id", ""),
            no_token_id=market_meta.get("no_token_id", ""),
            market_liquidity=market_meta.get("liquidity", 0.0),
            market_volume=market_meta.get("volume", 0.0),
        )

        # Update window state so post-signal tracking and cooldown work
        state.signal_fired = True
        state.signal_fired_side = momentum_side
        state.observation = PostSignalObservation(
            event_id=event.event_id,
            momentum_side_peak=momentum_price,   # initialise to entry price
            min_opposite_price=gate["opposite_price"],
        )
        self._last_event = event

        # Write to the signal event log
        _append_jsonl(self._cfg.signal_event_log_path, {
            "type": "signal",
            **event.model_dump(mode="json"),
        })

        logger.info(
            "BRACKET SIGNAL FIRED | asset={} model={} side={} price={} lag_gap={:.3f} "
            "chop={:.2f} net_bracket_target={:.4f} mins_left={:.1f} "
            "market_id={}",
            self.asset, event.entry_model, momentum_side, momentum_price, gate["lag_gap"],
            gate["chop_score"], net_bracket_at_target, gate["minutes_remaining"],
            market_meta.get("market_id", "?"),
        )
        return event


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file (one JSON object per line)."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.warning("Failed to write to JSONL log path={} error={}", path, exc)


def _build_live_snapshot(
    cfg: CryptoDirectionConfig,
    asset_vol_cfg: AssetVolConfig,
    vol_per_second: float,
    state: _WindowState | None,
    yes_ask: float,
    no_ask: float,
    asset_price: float,
    yes_ask_size: float,
    no_ask_size: float,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "yes_ask_size": round(float(yes_ask_size or 0.0), 4),
        "no_ask_size": round(float(no_ask_size or 0.0), 4),
        "required_shares": round(max(cfg.phase1_shares, cfg.min_bracket_shares), 4),
        "time_gate_minutes_cfg": cfg.time_gate_minutes,
        "move_threshold_cfg": asset_vol_cfg.min_asset_move_pct,
        "entry_range_low_cfg": cfg.entry_range_low,
        "entry_range_high_cfg": cfg.entry_range_high,
        "chop_min_score_cfg": cfg.chop_min_score,
        "lag_threshold_cfg": cfg.lag_threshold,
    }

    if not state:
        return snapshot

    snapshot["window_open_ts"] = state.window_open_ts
    snapshot["asset_open"] = round(state.asset_open, 2)
    snapshot["minutes_remaining"] = round(
        (state.window_close_ts - time.time()) / 60.0,
        2,
    )
    snapshot["mid_window_start"] = state.mid_window_start
    snapshot["signal_already_fired"] = state.signal_fired

    if state.asset_open <= 0 or asset_price <= 0:
        return snapshot

    asset_move_pct = (asset_price - state.asset_open) / state.asset_open
    snapshot["asset_move_pct"] = round(asset_move_pct, 6)
    snapshot["move_pass_live"] = abs(asset_move_pct) >= asset_vol_cfg.min_asset_move_pct

    if asset_move_pct > 0:
        momentum_side = "YES"
        momentum_price = yes_ask
        opposite_price = no_ask
        momentum_ask_size = yes_ask_size
        opposite_ask_size = no_ask_size
    elif asset_move_pct < 0:
        momentum_side = "NO"
        momentum_price = no_ask
        opposite_price = yes_ask
        momentum_ask_size = no_ask_size
        opposite_ask_size = yes_ask_size
    else:
        momentum_side = ""
        momentum_price = 0.0
        opposite_price = 0.0
        momentum_ask_size = 0.0
        opposite_ask_size = 0.0

    if momentum_side:
        snapshot["momentum_side_live"] = momentum_side
        snapshot["momentum_price_live"] = round(momentum_price, 4)
        snapshot["opposite_price_live"] = round(opposite_price, 4)
        snapshot["momentum_ask_size_live"] = round(float(momentum_ask_size or 0.0), 4)
        snapshot["opposite_ask_size_live"] = round(float(opposite_ask_size or 0.0), 4)
        snapshot["phase1_would_fill_live"] = momentum_ask_size >= snapshot["required_shares"]
        snapshot["range_pass_live"] = cfg.entry_range_low <= momentum_price <= cfg.entry_range_high
        if momentum_price > cfg.entry_range_high:
            snapshot["range_side"] = "HIGH"
            snapshot["range_distance"] = round(momentum_price - cfg.entry_range_high, 4)
        elif 0 < momentum_price < cfg.entry_range_low:
            snapshot["range_side"] = "LOW"
            snapshot["range_distance"] = round(cfg.entry_range_low - momentum_price, 4)

    prices_pass = yes_ask > 0 and no_ask > 0
    snapshot["prices_pass_live"] = prices_pass
    chop_score = 0.0
    if momentum_side and prices_pass:
        recent = list(state.checkpoints[-cfg.chop_window:])
        if asset_price > 0:
            recent.append(WindowCheckpoint(ts=time.time(), price=asset_price))
        if len(recent) < 2:
            chop_score = 0.5
        else:
            direction = 1 if asset_move_pct > 0 else -1
            total_steps = len(recent) - 1
            clean_steps = 0
            for i in range(total_steps):
                step = recent[i + 1].price - recent[i].price
                step_pct = step / recent[i].price if recent[i].price > 0 else 0.0
                if step * direction > 0:
                    clean_steps += 1
                elif abs(step_pct) > cfg.chop_max_reversal_pct:
                    clean_steps -= 1
            chop_score = max(0.0, min(1.0, clean_steps / total_steps))
        snapshot["chop_score_live"] = round(chop_score, 4)
        snapshot["chop_pass_live"] = chop_score >= cfg.chop_min_score

        time_remaining_seconds = float(state.window_close_ts) - time.time()
        if time_remaining_seconds > 0:
            implied_p_up = _estimate_fair_p_up(
                chainlink_price=asset_price,
                start_price=state.asset_open,
                time_remaining_seconds=time_remaining_seconds,
                realized_vol_per_second=vol_per_second,
            )
            implied_momentum_price = implied_p_up if momentum_side == "YES" else (1.0 - implied_p_up)
            lag_gap = implied_momentum_price - momentum_price
            snapshot["implied_momentum_price_live"] = round(implied_momentum_price, 4)
            snapshot["lag_gap_live"] = round(lag_gap, 4)
            snapshot["lag_pass_live"] = lag_gap >= cfg.lag_threshold

    return snapshot


def _log_evaluation(
    cfg: CryptoDirectionConfig,
    asset_vol_cfg: AssetVolConfig,
    vol_per_second: float,
    asset: str,
    state: _WindowState | None,
    gate: dict[str, Any],
    yes_ask: float,
    no_ask: float,
    asset_price: float,
    yes_ask_size: float = 0.0,
    no_ask_size: float = 0.0,
) -> None:
    """
    Write a compact evaluation record to the evaluation JSONL log.

    One record per poll cycle per asset. Captures the full decision trail
    so you can replay exactly what the evaluator saw and why it decided
    what it did. The result field tells you which gate stopped the signal.
    """
    record: dict[str, Any] = {
        "ts": round(time.time(), 3),
        "asset": asset,
        "asset_price": round(asset_price, 2),
        "yes_ask": round(yes_ask, 4),
        "no_ask": round(no_ask, 4),
        "result": gate.get("result", "UNKNOWN"),
    }
    record.update(
        _build_live_snapshot(
            cfg,
            asset_vol_cfg,
            vol_per_second,
            state,
            yes_ask,
            no_ask,
            asset_price,
            yes_ask_size,
            no_ask_size,
        )
    )

    # Include all diagnostic fields from the gate result
    for k, v in gate.items():
        if k != "result":
            record[k] = v

    result = record["result"]
    if result == GATE_RANGE:
        low, high = cfg.entry_range_low, cfg.entry_range_high
        momentum_price = float(record.get("momentum_price", 0.0) or 0.0)
        if momentum_price > high:
            record["range_side"] = "HIGH"
            record["range_distance"] = round(momentum_price - high, 4)
        elif momentum_price < low:
            record["range_side"] = "LOW"
            record["range_distance"] = round(low - momentum_price, 4)
    elif result == GATE_MOVE:
        move = abs(float(record.get("asset_move_pct", 0.0) or 0.0))
        required = float(record.get("min_required", 0.0) or 0.0)
        if required > 0:
            record["move_shortfall"] = round(max(required - move, 0.0), 6)
    elif result == GATE_CHOP:
        score = float(record.get("chop_score", 0.0) or 0.0)
        required = float(record.get("min_required", 0.0) or 0.0)
        if required > 0:
            record["chop_shortfall"] = round(max(required - score, 0.0), 4)
    elif result == GATE_LAG:
        lag_gap = float(record.get("lag_gap", 0.0) or 0.0)
        required = float(record.get("min_required", 0.0) or 0.0)
        if required > 0:
            record["lag_shortfall"] = round(max(required - lag_gap, 0.0), 4)

    _append_jsonl(cfg.evaluation_log_path, record)


async def _resolve_market_outcome(
    *,
    client: Any,
    asset_slug_prefix: str,
    window_ts: int,
) -> dict[str, Any] | None:
    """
    Ask the market metadata for the closing window who won.

    This is the best available non-wallet source for post-close outcome
    labeling. If the market endpoint still doesn't expose a resolved binary
    winner, the caller should fall back to the Binance price proxy.
    """
    slug = f"{asset_slug_prefix}-{window_ts}"
    try:
        return await client.fetch_market_resolution(market_slug=slug)
    except Exception as exc:
        logger.debug("Market outcome lookup failed slug={} error={}", slug, exc)
        return None


# ---------------------------------------------------------------------------
# Main run loop — the top-level entry point
# ---------------------------------------------------------------------------

async def run_bracket_signal_observer(
    *,
    config: Any,   # AppConfig — typed as Any to avoid circular import at module level
    client: Any,   # PolymarketClient
    executor: Any = None,   # BracketExecutor | None — if provided, places real orders
) -> None:
    """
    Run the bracket strategy direction signal observer indefinitely.

    This is a standalone async task — it does NOT interact with the live
    engine, paper engine, or any existing strategy. It only reads prices
    and writes to JSONL log files.

    Start with:
      python main.py --observe-crypto

    Logs are written to:
      logs/crypto_signal_events.jsonl     ← signal fires + post-outcome records
      logs/crypto_signal_evaluations.jsonl ← compact per-cycle evaluation log

    Key log fields to watch:
      signal_events:    result="signal" / result="outcome"
      evaluations:      result="SIGNAL_FIRED" or gate fail reason
    """
    # Import here to avoid circular import at module level
    from src.crypto_market_watcher import CryptoMarketWatcher, current_window_ts

    cfg = config.crypto_direction

    if not cfg.enabled:
        logger.warning(
            "CryptoDirection observer is disabled in config "
            "(crypto_direction.enabled=false). Set to true to run."
        )
        return

    logger.info(
        "Starting bracket signal observer track_btc={} track_eth={} "
        "entry_range=[{}, {}] lag_threshold={} time_gate_minutes={} "
        "continuation={} cont_min_mins={} cont_min_move={} cont_min_chop={} "
        "cont_max_momentum={} cont_max_neg_lag={} cont_ignore_lag={} "
        "band_entry={} band_range=[{}, {}] "
        "phase1_chase={} phase1_retries={} retry_delay={}",
        cfg.track_btc, cfg.track_eth,
        cfg.entry_range_low, cfg.entry_range_high,
        cfg.lag_threshold, cfg.time_gate_minutes,
        cfg.continuation_enabled,
        cfg.continuation_min_minutes_remaining,
        cfg.continuation_min_asset_move_pct,
        cfg.continuation_min_chop_score,
        cfg.continuation_max_momentum_price,
        cfg.continuation_max_negative_lag_gap,
        cfg.continuation_ignore_lag_veto,
        cfg.immediate_band_entry_enabled,
        cfg.immediate_band_entry_low,
        cfg.immediate_band_entry_high,
        cfg.phase1_max_chase_cents,
        cfg.phase1_follow_taker_retry_attempts,
        cfg.phase1_follow_taker_retry_delay_seconds,
    )

    # ---- Startup diagnostics ----
    # Verify the fee model and GBM model produce sensible numbers before starting.
    _log_startup_diagnostics(cfg)

    # ---- Build per-asset evaluators ----
    evaluators: dict[str, DirectionSignalEvaluator] = {}
    if cfg.track_btc:
        evaluators["BTC"] = DirectionSignalEvaluator("BTC", cfg, cfg.btc)
    if cfg.track_eth:
        evaluators["ETH"] = DirectionSignalEvaluator("ETH", cfg, cfg.eth)

    if not evaluators:
        logger.error("No assets configured to track. Enable track_btc or track_eth.")
        return

    # ---- Build market watcher ----
    watcher = CryptoMarketWatcher(cfg, client)
    await watcher.start()

    # ---- Window report writer ----
    report_writer = WindowReportWriter(
        report_path=cfg.window_report_path,
        report_shares=cfg.report_shares,
        live_execution=(executor is not None),   # backward-compatible flag
        execution_mode=str(getattr(executor, "execution_mode", "observe")).lower(),
        window_duration_seconds=cfg.window_duration_seconds,
    )

    # ---- Initial window setup (likely starting mid-window) ----
    # We don't know the true window_open asset price, so we use the current
    # price as a synthetic baseline. Signals this window are flagged as
    # mid_window_start=True in the log.
    init_window_ts = current_window_ts(cfg.window_duration_seconds)
    await _initialise_window(watcher, evaluators, init_window_ts, mid_window=True)

    logger.info(
        "Observer running — watching {}-minute window cycles. Press Ctrl+C to stop.",
        int(cfg.window_duration_seconds / 60),
    )

    # ---- Main poll loop ----
    while True:
        try:
            # 1. Check for window transition (runs resolve_market if needed)
            transitioned_assets = await watcher.check_window_transition()

            if transitioned_assets:
                # A window just rolled over. Close the previous window for
                # each affected asset, then open the new one.
                new_ts = watcher.current_window_ts()

                for asset in transitioned_assets:
                    w = watcher.get_watcher(asset)
                    ev = evaluators.get(asset)
                    if not w or not ev:
                        continue

                    # Determine outcome: prefer actual market metadata for the
                    # closing window, then fall back to the asset-price proxy.
                    asset_price_now = w.asset_price()
                    ev_state = ev._state
                    yes_won: bool | None = None
                    outcome_source = "asset_price_proxy"
                    if ev_state:
                        resolution = await _resolve_market_outcome(
                            client=client,
                            asset_slug_prefix=ev._asset_vol_cfg.slug_prefix,
                            window_ts=ev_state.window_open_ts,
                        )
                        if resolution and resolution.get("resolved_yes") is not None:
                            yes_won = bool(resolution.get("resolved_yes"))
                            outcome_source = str(resolution.get("source") or "market_metadata")
                            logger.info(
                                "Window outcome sourced from market metadata asset={} window_ts={} source={} prices={}",
                                asset,
                                ev_state.window_open_ts,
                                outcome_source,
                                resolution.get("outcome_prices"),
                            )
                        elif ev_state.asset_open > 0:
                            yes_won = asset_price_now > ev_state.asset_open
                            outcome_source = "asset_price_proxy"
                            logger.info(
                                "Window outcome fell back to asset-price proxy asset={} window_ts={} asset_open={:.2f} asset_close={:.2f}",
                                asset,
                                ev_state.window_open_ts,
                                ev_state.asset_open,
                                asset_price_now,
                            )

                    # Snapshot state before on_window_close() clears it
                    ev_state_snap = ev._state
                    signal_fired_snap = ev_state_snap.signal_fired if ev_state_snap else False

                    # IMPORTANT ORDER: close window FIRST so the observation
                    # (phase2_would_have_triggered, bracket_would_have_formed, etc.)
                    # is populated on ev._last_event BEFORE we serialise it for the
                    # window report.  Previously record_window_close() was called first
                    # and always saw observation=None → Phase 2 was never credited in
                    # hyp_pnl and "Phase 2: Not triggered" always appeared.
                    ev.on_window_close(asset_price_now, yes_won, outcome_source=outcome_source)

                    execution_summary = None

                    # Settle any open bracket positions for this window
                    if executor is not None and ev_state_snap:
                        await executor.on_window_close(
                            window_ts=ev_state_snap.window_open_ts,
                            asset=asset,
                            yes_won=yes_won,
                        )
                        execution_summary = executor.take_window_summary(
                            window_ts=ev_state_snap.window_open_ts,
                            asset=asset,
                        )

                    # Now record the window — _last_event.observation is populated
                    if ev_state_snap:
                        last_event_dict = (
                            ev._last_event.model_dump()
                            if ev._last_event is not None
                            else None
                        )
                        report_writer.record_window_close(
                            window_ts=ev_state_snap.window_open_ts,
                            asset=asset,
                            asset_open=ev_state_snap.asset_open,
                            asset_close=asset_price_now,
                            yes_ask_final=w.yes_ask(),
                            no_ask_final=w.no_ask(),
                            last_signal_event=last_event_dict,
                            eval_log_path=cfg.evaluation_log_path,
                            signal_log_path=cfg.signal_event_log_path,
                            yes_won=yes_won,  # pass directly — more reliable than ask-price inference
                            outcome_source=outcome_source,
                            execution_summary=execution_summary,
                        )
                        if not signal_fired_snap and report_writer._windows:
                            gate = (
                                report_writer._windows[-1].dominant_gate_fail
                                or report_writer._windows[-1].primary_gate_fail
                            )
                            logger.info(
                                "No signal this window asset={} — blocked at: {}",
                                asset, gate or "NO_WINDOW_STATE",
                            )

                    ev.on_window_open(new_ts, asset_price_now, mid_window=False)

                # Also handle assets that didn't transition (e.g., market not resolved)
                for asset, ev in evaluators.items():
                    if asset not in transitioned_assets:
                        w = watcher.get_watcher(asset)
                        if w and ev._state and ev._state.window_open_ts != new_ts:
                            ev.on_window_open(new_ts, w.asset_price(), mid_window=False)

            # 2. Refresh YES/NO prices (throttled internally)
            await watcher.refresh_all_prices()

            # 3. Determine which assets to evaluate this cycle
            assets_to_evaluate = watcher.active_assets()

            # 4. Per-asset: record checkpoints, update post-signal, evaluate
            for asset in assets_to_evaluate:
                w = watcher.get_watcher(asset)
                ev = evaluators.get(asset)
                if not w or not ev:
                    continue

                asset_price = w.asset_price()
                yes_ask = w.yes_ask()
                no_ask = w.no_ask()
                yes_bid = w.yes_bid()
                no_bid = w.no_bid()

                # Record :00-second Chainlink-aligned price checkpoint
                ev.maybe_record_checkpoint(asset_price)

                # Update bracket tracking if a signal already fired this window
                yes_ask_size = w.yes_ask_size()
                no_ask_size = w.no_ask_size()
                ev.update_post_signal(yes_ask, no_ask, yes_ask_size, no_ask_size)

                # Phase 2 monitoring — check for bracket entry conditions
                if executor is not None:
                    await executor.tick(asset, yes_ask, no_ask, yes_bid=yes_bid, no_bid=no_bid)

                # Check if all signal gates pass
                if not w.is_ready():
                    readiness_reason = (
                        w.readiness_reason()
                        if hasattr(w, "readiness_reason")
                        else "watcher_not_ready"
                    )
                    _log_evaluation(
                        cfg, ev._asset_vol_cfg, ev._vol_per_second, asset, ev._state,
                        {"result": GATE_STALE, "reason": readiness_reason},
                        yes_ask, no_ask, asset_price, yes_ask_size, no_ask_size,
                    )
                    continue

                signal = ev.evaluate(
                    asset_price, yes_ask, no_ask, w.market_meta()
                )

                if signal:
                    if executor is not None:
                        submitted = await executor.on_signal(signal)
                        if submitted:
                            logger.info(
                                "Signal fired — Phase 1 order submitted mode={}. event_id={}",
                                str(getattr(executor, "execution_mode", "observe")).lower(),
                                signal.event_id,
                            )
                        else:
                            logger.info(
                                "Signal fired — executor did not open a position mode={}. event_id={}",
                                str(getattr(executor, "execution_mode", "observe")).lower(),
                                signal.event_id,
                            )
                    else:
                        logger.info(
                            "Signal logged — observation only, no orders placed. "
                            "event_id={}",
                            signal.event_id,
                        )

        except asyncio.CancelledError:
            logger.info("Bracket signal observer shutting down.")
            break
        except Exception as exc:
            # Log but don't crash — keep observing even if one cycle errors
            logger.warning("Observer poll cycle error: {}", exc)

        await asyncio.sleep(cfg.poll_interval_seconds)


async def _initialise_window(
    watcher: Any,
    evaluators: dict[str, "DirectionSignalEvaluator"],
    window_ts: int,
    mid_window: bool,
) -> None:
    """
    Set up initial window state for all evaluators at startup.

    We attempt to resolve markets and get initial prices before starting
    the main loop. If resolution fails, the evaluator will have no state
    and will log GATE_NO_STATE until the next window transition.
    """
    # Force a window transition to resolve markets
    # We do this by directly calling resolve_market on each watcher
    for asset, ev in evaluators.items():
        w = watcher.get_watcher(asset)
        if not w:
            continue

        # Resolve the market for this window
        await w.resolve_market(window_ts)

        # Get an initial price snapshot
        await w.refresh_prices()

        asset_price = w.asset_price()
        if asset_price <= 0:
            logger.warning(
                "No asset price available at startup asset={} "
                "— signal will not fire until Binance feed is live.",
                asset,
            )
            asset_price = 0.0

        ev.on_window_open(window_ts, asset_price, mid_window=mid_window)
        logger.info(
            "Initialised window asset={} window_ts={} asset_price={} mid_window={}",
            asset, window_ts, asset_price, mid_window,
        )

    # Set the watcher's last known window timestamp so check_window_transition
    # doesn't immediately fire a transition on the first loop iteration
    watcher._last_window_ts = window_ts


def _log_startup_diagnostics(cfg: CryptoDirectionConfig) -> None:
    """
    Log sanity checks on startup so you can verify the models are configured
    correctly before the first signal fires.

    Checks:
      - Fee values at key price points (should be < 2% each)
      - GBM implied probability for a typical BTC move
      - Net bracket margin at the target entry (should be positive)
    """
    import math

    # Fee check
    fee_58 = taker_fee(0.58, category="crypto price")
    fee_34 = taker_fee(0.34, category="crypto price")
    net_bracket_target = 1.0 - 0.58 * (1 + fee_58) - 0.34 * (1 + fee_34)
    logger.info(
        "Startup diagnostics | "
        "fee@0.58={:.4f} ({:.2f}%) fee@0.34={:.4f} ({:.2f}%) "
        "net_bracket(x=0.58,y=0.34)={:.4f}",
        fee_58, fee_58 * 100, fee_34, fee_34 * 100, net_bracket_target,
    )

    # GBM model check for BTC
    btc_vol_per_sec = cfg.btc.annual_vol_pct / math.sqrt(365.0 * 24.0 * 3600.0)
    p_up = _estimate_fair_p_up(
        chainlink_price=85000.0,   # synthetic: BTC moved up ~0.6% from $84,500
        start_price=84500.0,
        time_remaining_seconds=480.0,   # 8 minutes left
        realized_vol_per_second=btc_vol_per_sec,
    )
    logger.info(
        "GBM sanity check | BTC $84500→$85000 with 8min left → "
        "P(up)={:.4f} (should be > 0.65 for a strong signal)",
        p_up,
    )

    # Warn if net bracket at target is too thin to cover slippage
    if net_bracket_target <= 0:
        logger.warning(
            "WARNING: net_bracket(x=0.58, y=0.34) = {:.4f} is NEGATIVE — "
            "fees exceed the spread. Check fee model or adjust target_y_price.",
            net_bracket_target,
        )
