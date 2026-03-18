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
  evaluator.update_post_signal(yes_ask, no_ask)
  signal = evaluator.evaluate(asset_price, yes_ask, no_ask, market_meta)
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
from src.fees import taker_fee
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
            "entry_range=[{}, {}] lag_threshold={} time_gate_minutes={}",
            asset, self._vol_per_second,
            cfg.entry_range_low, cfg.entry_range_high,
            cfg.lag_threshold, cfg.time_gate_minutes,
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
            window_close_ts=window_open_ts + 900,
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
        obs.estimated_phase1_exit_pnl = round((peak_price - entry_price) * 10.0, 4)

        # Append outcome record to the signal event log
        _append_jsonl(
            self._cfg.signal_event_log_path,
            {
                "type": "outcome",
                "event_id": self._last_event.event_id,
                "asset": self.asset,
                "window_close_ts": self._state.window_close_ts,
                "asset_close": asset_close,
                "observation": obs.model_dump(),
            },
        )
        logger.info(
            "Window closed asset={} outcome={} bracket_formed={} "
            "phase2_triggered={} phase1_exit_pnl={:.4f}",
            self.asset, obs.outcome, obs.bracket_would_have_formed,
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

    def update_post_signal(self, yes_ask: float, no_ask: float) -> None:
        """
        Update post-signal observation metrics on each poll cycle.

        After a signal fires, we keep watching to answer:
          - Did the bracket become achievable (x + y + fees < $1.00)?
          - Did the opposite side bottom and start reversing (Phase 2 trigger)?
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

        # y is the opposite side — the one we're watching for the second leg
        y_price = no_ask if momentum_side == "YES" else yes_ask

        # Update momentum side peak (best unrealised gain on leg 1)
        momentum_now = yes_ask if momentum_side == "YES" else no_ask
        if momentum_now > obs.momentum_side_peak:
            obs.momentum_side_peak = momentum_now

        # Bracket margin: how much profit is locked if we buy y right now?
        # margin > 0 means we're profitable; use actual Polymarket fee curve.
        if y_price > 0:
            fee_x = taker_fee(x_price, category="crypto price")
            fee_y = taker_fee(y_price, category="crypto price")
            net_bracket = 1.0 - x_price * (1.0 + fee_x) - y_price * (1.0 + fee_y)

            if net_bracket > obs.peak_bracket_margin:
                obs.peak_bracket_margin = round(net_bracket, 6)

            if net_bracket > 0:
                obs.bracket_would_have_formed = True

            # Track the floor of y (lowest price seen — best possible entry for leg 2)
            if y_price < obs.min_opposite_price:
                obs.min_opposite_price = y_price

            # Phase 2 reversal check: has y bounced from its floor by >= threshold?
            # This is the trigger for buying the second leg in the real strategy.
            floor = obs.min_opposite_price
            if (
                floor < 999.0
                and y_price > floor + self._cfg.phase2_reversal_threshold
                and not obs.phase2_would_have_triggered
            ):
                obs.phase2_would_have_triggered = True
                obs.phase2_trigger_price = round(y_price, 6)
                logger.info(
                    "Phase 2 would have triggered asset={} y_floor={} y_now={} "
                    "bracket_margin={:.4f}",
                    self.asset, floor, y_price, obs.peak_bracket_margin,
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
        _log_evaluation(self._cfg, self.asset, self._state, gate, yes_ask, no_ask, asset_price)

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
        chop_score = self._compute_chop_score(asset_move_pct)
        if chop_score < self._cfg.chop_min_score:
            return {
                "result": GATE_CHOP,
                "chop_score": round(chop_score, 3),
                "min_required": self._cfg.chop_min_score,
                "checkpoints_used": len(state.checkpoints[-self._cfg.chop_window:]),
            }

        # ---- Gate 6: Lag gap ----
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
            return {
                "result": GATE_LAG,
                "lag_gap": round(lag_gap, 4),
                "implied_momentum_price": round(implied_momentum_price, 4),
                "momentum_price": round(momentum_price, 4),
                "min_required": self._cfg.lag_threshold,
            }

        # ---- Gate 7: Cooldown ----
        # Don't fire the same side twice in one window (prevents signal spam).
        if state.signal_fired and state.signal_fired_side == momentum_side:
            return {
                "result": GATE_COOLDOWN,
                "fired_side": state.signal_fired_side,
            }

        # ---- All gates passed ----
        return {
            "result": GATE_FIRED,
            "minutes_remaining": round(minutes_remaining, 2),
            "asset_move_pct": round(asset_move_pct, 6),
            "momentum_side": momentum_side,
            "momentum_price": round(momentum_price, 4),
            "opposite_price": round(opposite_price, 4),
            "chop_score": round(chop_score, 3),
            "lag_gap": round(lag_gap, 4),
            "implied_momentum_price": round(implied_momentum_price, 4),
        }

    def _compute_chop_score(self, asset_move_pct: float) -> float:
        """
        Score how cleanly directional the asset's move has been.

        Looks at the last `chop_window` :00-second checkpoints and counts how
        many consecutive steps moved in the same direction as the overall move.
        Counter-moves larger than chop_max_reversal_pct are penalised.

        Returns:
          1.0 = perfectly monotonic move in the right direction
          0.5 = insufficient checkpoint data (soft pass — only 1 checkpoint)
          0.0 = all steps are reversals
        """
        state = self._state
        assert state is not None

        recent = state.checkpoints[-self._cfg.chop_window:]
        if len(recent) < 2:
            # Not enough :00-second data yet — give a soft pass so early-window
            # signals aren't blocked purely by lack of checkpoint history.
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
            momentum_side=momentum_side,
            momentum_price=momentum_price,
            opposite_price=gate["opposite_price"],
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
            "BRACKET SIGNAL FIRED | asset={} side={} price={} lag_gap={:.3f} "
            "chop={:.2f} net_bracket_target={:.4f} mins_left={:.1f} "
            "market_id={}",
            self.asset, momentum_side, momentum_price, gate["lag_gap"],
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


def _log_evaluation(
    cfg: CryptoDirectionConfig,
    asset: str,
    state: _WindowState | None,
    gate: dict[str, Any],
    yes_ask: float,
    no_ask: float,
    asset_price: float,
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

    if state:
        record["window_open_ts"] = state.window_open_ts
        record["asset_open"] = round(state.asset_open, 2)
        record["minutes_remaining"] = round(
            (state.window_close_ts - time.time()) / 60.0, 2
        )
        asset_move_pct = (
            (asset_price - state.asset_open) / state.asset_open
            if state.asset_open > 0 else 0.0
        )
        record["asset_move_pct"] = round(asset_move_pct, 6)
        record["mid_window_start"] = state.mid_window_start
        record["signal_already_fired"] = state.signal_fired

    # Include all diagnostic fields from the gate result
    for k, v in gate.items():
        if k != "result":
            record[k] = v

    _append_jsonl(cfg.evaluation_log_path, record)


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
        "entry_range=[{}, {}] lag_threshold={} time_gate_minutes={}",
        cfg.track_btc, cfg.track_eth,
        cfg.entry_range_low, cfg.entry_range_high,
        cfg.lag_threshold, cfg.time_gate_minutes,
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
        hypothetical_bet_size=cfg.hypothetical_bet_size,
    )

    # ---- Initial window setup (likely starting mid-window) ----
    # We don't know the true window_open asset price, so we use the current
    # price as a synthetic baseline. Signals this window are flagged as
    # mid_window_start=True in the log.
    init_window_ts = current_window_ts()
    await _initialise_window(watcher, evaluators, init_window_ts, mid_window=True)

    logger.info(
        "Observer running — watching {} window cycles. Press Ctrl+C to stop.",
        "15-minute",
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

                    # Determine outcome: did YES win? (asset ended above open?)
                    asset_price_now = w.asset_price()
                    ev_state = ev._state
                    if ev_state and ev_state.asset_open > 0:
                        yes_won = asset_price_now > ev_state.asset_open
                    else:
                        yes_won = None

                    # Record the closing window in the rolling report
                    ev_state_snap = ev._state
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
                        )

                    # Settle any open bracket positions for this window
                    if executor is not None and ev_state_snap:
                        await executor.on_window_close(
                            window_ts=ev_state_snap.window_open_ts,
                            asset=asset,
                            yes_won=yes_won,
                        )

                    ev.on_window_close(asset_price_now, yes_won)
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

                # Record :00-second Chainlink-aligned price checkpoint
                ev.maybe_record_checkpoint(asset_price)

                # Update bracket tracking if a signal already fired this window
                ev.update_post_signal(yes_ask, no_ask)

                # Phase 2 monitoring — check for bracket entry conditions
                if executor is not None:
                    await executor.tick(asset, yes_ask, no_ask)

                # Check if all signal gates pass
                if not w.is_ready():
                    _log_evaluation(
                        cfg, asset, ev._state,
                        {"result": GATE_STALE, "reason": "watcher_not_ready"},
                        yes_ask, no_ask, asset_price,
                    )
                    continue

                signal = ev.evaluate(
                    asset_price, yes_ask, no_ask, w.market_meta()
                )

                if signal:
                    if executor is not None:
                        await executor.on_signal(signal)
                        logger.info(
                            "Signal fired — Phase 1 order submitted. event_id={}",
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
