"""
Unit tests for src/crypto_direction_signal.py

Coverage:
  - _compute_chop_score with various checkpoint patterns (monotonic, mixed, insufficient data)
  - All 7 signal gate failure modes (TIME, MOVE, RANGE, STALE, CHOP, LAG, COOLDOWN)
  - Full GATE_FIRED pass for both YES and NO momentum directions (symmetry check)
  - evaluate() happy-path: BracketSignalEvent construction and JSONL log output
  - Post-signal observation: bracket tracking, Phase 2 trigger, momentum peak
  - Checkpoint deduplication: same-second guard and 20-entry cap
  - Window lifecycle: reset on open, outcome write on close
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.config import AssetVolConfig, CryptoDirectionConfig
from src.crypto_direction_signal import (
    GATE_CHOP,
    GATE_COOLDOWN,
    GATE_FIRED,
    GATE_LAG,
    GATE_MOVE,
    GATE_NO_STATE,
    GATE_RANGE,
    GATE_STALE,
    GATE_TIME,
    DirectionSignalEvaluator,
)
from src.models import WindowCheckpoint


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

# Deterministic window timestamps (arbitrary round values)
WINDOW_OPEN_TS = 1_000_000       # window starts here
WINDOW_CLOSE_TS = WINDOW_OPEN_TS + 900  # +15 min
NOW_IN_WINDOW = WINDOW_OPEN_TS + 300    # 5 min in → 10 min remaining (> 9.0 gate)

# Representative asset prices
ASSET_OPEN = 84_000.0
ASSET_PRICE_UP = ASSET_OPEN * 1.005     # +0.5% (YES momentum, > min_asset_move 0.2%)
ASSET_PRICE_DOWN = ASSET_OPEN * 0.995   # -0.5% (NO  momentum)

# Typical YES/NO ask prices used across tests
YES_ASK_IN_RANGE = 0.58   # in [0.57, 0.60] — YES entry
NO_ASK_OPPOSITE = 0.41    # opposite side when YES is momentum
NO_ASK_IN_RANGE = 0.58    # in [0.57, 0.60] — NO entry (for down-move tests)
YES_ASK_OPPOSITE = 0.41   # opposite side when NO is momentum

SAMPLE_MARKET_META = {
    "market_id": "test-market-btc-15m",
    "yes_token_id": "yes-token-001",
    "no_token_id": "no-token-001",
    "liquidity": 5_000.0,
    "volume": 1_200.0,
    "yes_ask_size": 25.0,
    "no_ask_size": 25.0,
}


def make_config(tmp_path, **overrides) -> CryptoDirectionConfig:
    """Create a minimal CryptoDirectionConfig pointing logs at tmp_path."""
    defaults = dict(
        enabled=True,
        time_gate_minutes=9.0,
        entry_range_low=0.57,
        entry_range_high=0.60,
        chop_window=4,
        chop_max_reversal_pct=0.0005,
        chop_min_score=0.6,
        lag_threshold=0.04,
        target_y_price=0.34,
        phase2_reversal_threshold=0.01,
        signal_event_log_path=str(tmp_path / "signal_events.jsonl"),
        evaluation_log_path=str(tmp_path / "evaluations.jsonl"),
    )
    defaults.update(overrides)
    return CryptoDirectionConfig(**defaults)


def make_asset_vol_cfg(**overrides) -> AssetVolConfig:
    """Create a minimal AssetVolConfig for BTC."""
    defaults = dict(
        annual_vol_pct=0.69,
        min_asset_move_pct=0.002,
    )
    defaults.update(overrides)
    return AssetVolConfig(**defaults)


def make_evaluator(tmp_path, **cfg_overrides) -> DirectionSignalEvaluator:
    """Build a DirectionSignalEvaluator with a window already opened at WINDOW_OPEN_TS."""
    cfg_overrides = dict(cfg_overrides)
    asset_vol_overrides = cfg_overrides.pop("asset_vol_overrides", {})
    cfg = make_config(tmp_path, **cfg_overrides)
    asset_vol_cfg = make_asset_vol_cfg(**asset_vol_overrides)
    ev = DirectionSignalEvaluator("BTC", cfg, asset_vol_cfg)
    ev.on_window_open(WINDOW_OPEN_TS, ASSET_OPEN, mid_window=False)
    return ev


def _patch_time(t: float):
    """Fix time.time() to a constant for gate evaluation."""
    return patch("src.crypto_direction_signal.time.time", return_value=t)


def _patch_gbm(implied_p_up: float):
    """Fix _estimate_fair_p_up() so GBM lag tests are deterministic."""
    return patch(
        "src.crypto_direction_signal._estimate_fair_p_up",
        return_value=implied_p_up,
    )


def seed_clean_checkpoints_up(ev: DirectionSignalEvaluator, n: int = 4) -> None:
    """Replace state checkpoints with n monotonically rising entries (BTC going up)."""
    ev._state.checkpoints = [
        WindowCheckpoint(ts=float(WINDOW_OPEN_TS + i * 60), price=ASSET_OPEN + i * 50.0)
        for i in range(n)
    ]


def seed_clean_checkpoints_down(ev: DirectionSignalEvaluator, n: int = 4) -> None:
    """Replace state checkpoints with n monotonically falling entries (BTC going down)."""
    ev._state.checkpoints = [
        WindowCheckpoint(ts=float(WINDOW_OPEN_TS + i * 60), price=ASSET_OPEN - i * 50.0)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _compute_chop_score
# ---------------------------------------------------------------------------

class TestComputeChopScore:
    """Chop filter — score how cleanly directional the move has been."""

    def test_returns_soft_pass_with_one_checkpoint(self, tmp_path):
        """Fewer than 2 checkpoints → return 0.5 (soft pass, not enough data)."""
        ev = make_evaluator(tmp_path)
        # on_window_open seeds exactly 1 checkpoint
        ev._state.checkpoints = [WindowCheckpoint(ts=float(WINDOW_OPEN_TS), price=ASSET_OPEN)]
        assert ev._compute_chop_score(0.005, current_price=0.0) == 0.5

    def test_perfectly_monotonic_up_returns_one(self, tmp_path):
        """All steps in the up direction → 1.0."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)
        assert ev._compute_chop_score(asset_move_pct=0.003, current_price=0.0) == 1.0

    def test_perfectly_monotonic_down_returns_one(self, tmp_path):
        """All steps in the down direction → 1.0 (direction is aligned)."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_down(ev)
        assert ev._compute_chop_score(asset_move_pct=-0.003, current_price=0.0) == 1.0

    def test_all_meaningful_reversals_returns_zero(self, tmp_path):
        """Steps all go opposite to claimed direction (each big reversal) → 0.0."""
        ev = make_evaluator(tmp_path)
        # asset claims UP but every step drops ~$200 (far above 0.0005 reversal threshold)
        prices = [84_000.0, 83_800.0, 83_600.0, 83_400.0]
        ev._state.checkpoints = [
            WindowCheckpoint(ts=float(WINDOW_OPEN_TS + i * 60), price=p)
            for i, p in enumerate(prices)
        ]
        score = ev._compute_chop_score(asset_move_pct=0.003, current_price=0.0)  # UP direction
        assert score == 0.0

    def test_mixed_steps_partial_score(self, tmp_path):
        """2 clean steps + 1 large reversal → score between 0 and 1."""
        ev = make_evaluator(tmp_path)
        # Steps: up $200, down $400 (big reversal), up $300
        prices = [84_000.0, 84_200.0, 83_800.0, 84_100.0]
        ev._state.checkpoints = [
            WindowCheckpoint(ts=float(WINDOW_OPEN_TS + i * 60), price=p)
            for i, p in enumerate(prices)
        ]
        # clean_steps: +1 (up step) -1 (reversal) +1 (up step) = 1; total=3 → score=1/3
        score = ev._compute_chop_score(asset_move_pct=0.003, current_price=0.0)
        assert 0.0 < score < 1.0
        assert score == pytest.approx(1 / 3, abs=0.01)

    def test_score_always_clamped_to_0_1(self, tmp_path):
        """Score is always in [0.0, 1.0] even with extreme reversal patterns."""
        ev = make_evaluator(tmp_path)
        prices = [84_000.0, 83_000.0, 82_000.0, 81_000.0, 80_000.0]
        ev._state.checkpoints = [
            WindowCheckpoint(ts=float(WINDOW_OPEN_TS + i * 60), price=p)
            for i, p in enumerate(prices)
        ]
        score = ev._compute_chop_score(asset_move_pct=0.003, current_price=0.0)  # claims UP
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Gate failures — each gate tested in isolation
# ---------------------------------------------------------------------------

class TestGateFailures:
    """Verify each gate returns the correct failure code when its condition is violated."""

    def test_no_state_gate_when_window_not_opened(self, tmp_path):
        """evaluate() before on_window_open → GATE_NO_STATE."""
        cfg = make_config(tmp_path)
        ev = DirectionSignalEvaluator("BTC", cfg, make_asset_vol_cfg())
        # _state is None — window never opened
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            result = ev._run_gates(ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE)
        assert result["result"] == GATE_NO_STATE

    def test_time_gate_fail_when_window_nearly_over(self, tmp_path):
        """Signal blocked when fewer than time_gate_minutes remain."""
        ev = make_evaluator(tmp_path)
        # Only 2 minutes left (below 9.0 gate)
        near_end = WINDOW_CLOSE_TS - 120
        with _patch_time(near_end), _patch_gbm(0.70):
            result = ev._run_gates(ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE)
        assert result["result"] == GATE_TIME
        assert result["minutes_remaining"] == pytest.approx(2.0, abs=0.1)

    def test_asset_move_gate_fail_tiny_move(self, tmp_path):
        """Signal blocked when BTC barely moved from window open (<0.2%)."""
        ev = make_evaluator(tmp_path)
        tiny_price = ASSET_OPEN * 1.0001   # only 0.01% move
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            result = ev._run_gates(tiny_price, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE)
        assert result["result"] == GATE_MOVE

    def test_price_range_fail_too_low(self, tmp_path):
        """Signal blocked when momentum price is below entry_range_low (0.57)."""
        ev = make_evaluator(tmp_path)
        yes_ask_too_low = 0.50
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            result = ev._run_gates(ASSET_PRICE_UP, yes_ask_too_low, NO_ASK_OPPOSITE)
        assert result["result"] == GATE_RANGE
        assert result["momentum_side"] == "YES"

    def test_price_range_fail_too_high(self, tmp_path):
        """Signal blocked when momentum price is above entry_range_high (0.60)."""
        ev = make_evaluator(tmp_path)
        yes_ask_too_high = 0.65
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            result = ev._run_gates(ASSET_PRICE_UP, yes_ask_too_high, NO_ASK_OPPOSITE)
        assert result["result"] == GATE_RANGE

    def test_stale_no_ask_zero_blocked(self, tmp_path):
        """Signal blocked when NO ask is 0 (missing orderbook data)."""
        ev = make_evaluator(tmp_path)
        # YES is in range (gate 3 passes), NO is 0 → gate 4 (sanity) fails
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            result = ev._run_gates(ASSET_PRICE_UP, YES_ASK_IN_RANGE, 0.0)
        assert result["result"] == GATE_STALE

    def test_stale_yes_ask_zero_blocked(self, tmp_path):
        """Signal blocked when YES ask is 0 (gate 3 fails before gate 4, still blocked)."""
        ev = make_evaluator(tmp_path)
        # yes_ask=0 fails gate 3 (RANGE) because 0 not in [0.57, 0.60]
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            result = ev._run_gates(ASSET_PRICE_UP, 0.0, NO_ASK_OPPOSITE)
        # Blocked by some gate before GATE_FIRED (RANGE or STALE — doesn't matter which)
        assert result["result"] != GATE_FIRED

    def test_chop_filter_fail_choppy_price_action(self, tmp_path):
        """Signal blocked when price alternates direction (not a clean trend)."""
        ev = make_evaluator(tmp_path)
        # Alternating: up +$200, big drop −$400 (reversal), up +$300
        # → score ≈ 0.33 < chop_min_score=0.6
        prices = [84_000.0, 84_200.0, 83_800.0, 84_100.0]
        ev._state.checkpoints = [
            WindowCheckpoint(ts=float(WINDOW_OPEN_TS + i * 60), price=p)
            for i, p in enumerate(prices)
        ]
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            result = ev._run_gates(ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE)
        assert result["result"] == GATE_CHOP
        assert result["chop_score"] < 0.6

    def test_lag_gap_insufficient_blocked(self, tmp_path):
        """Signal blocked when GBM implied price barely exceeds actual price (<0.04 gap)."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)   # pass chop filter
        # GBM returns 0.60, yes_ask=0.58 → gap=0.02, below threshold 0.04
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.60):
            result = ev._run_gates(ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE)
        assert result["result"] == GATE_LAG
        assert result["lag_gap"] == pytest.approx(0.02, abs=0.001)
        assert result["lag_gap"] < 0.04

    def test_continuation_override_fires_when_lag_is_only_mildly_negative(self, tmp_path):
        """Clean in-range momentum can still fire through the continuation path."""
        ev = make_evaluator(
            tmp_path,
            time_gate_minutes=1.0,
            continuation_enabled=True,
            continuation_min_minutes_remaining=1.5,
            continuation_min_asset_move_pct=0.0004,
            continuation_min_chop_score=0.6,
            continuation_max_momentum_price=0.61,
            continuation_max_opposite_price=0.44,
            continuation_max_negative_lag_gap=0.04,
            lag_threshold=0.015,
        )
        seed_clean_checkpoints_up(ev, n=3)
        # implied 0.5797 vs yes ask 0.58 => lag_gap ~= -0.0003
        with _patch_time(WINDOW_OPEN_TS + 791), _patch_gbm(0.5797):
            result = ev._run_gates(ASSET_PRICE_UP, YES_ASK_IN_RANGE, 0.43)
        assert result["result"] == GATE_FIRED
        assert result["entry_model"] == "continuation"

    def test_continuation_override_still_blocks_when_lag_is_too_negative(self, tmp_path):
        """Continuation must not paper over a genuinely overextended market."""
        ev = make_evaluator(
            tmp_path,
            time_gate_minutes=1.0,
            continuation_enabled=True,
            continuation_min_minutes_remaining=1.5,
            continuation_min_asset_move_pct=0.0004,
            continuation_min_chop_score=0.6,
            continuation_max_momentum_price=0.61,
            continuation_max_opposite_price=0.44,
            continuation_max_negative_lag_gap=0.04,
            lag_threshold=0.015,
        )
        seed_clean_checkpoints_up(ev, n=3)
        # implied 0.52 vs yes ask 0.58 => lag_gap = -0.06, beyond continuation limit
        with _patch_time(WINDOW_OPEN_TS + 791), _patch_gbm(0.52):
            result = ev._run_gates(ASSET_PRICE_UP, YES_ASK_IN_RANGE, 0.43)
        assert result["result"] == GATE_LAG

    def test_continuation_can_ignore_lag_veto_in_experiment_mode(self, tmp_path):
        """Shadow experiment mode can take clean continuation entries despite very negative lag."""
        ev = make_evaluator(
            tmp_path,
            time_gate_minutes=1.0,
            continuation_enabled=True,
            continuation_min_minutes_remaining=1.5,
            continuation_min_asset_move_pct=0.0004,
            continuation_min_chop_score=0.6,
            continuation_max_momentum_price=0.61,
            continuation_max_opposite_price=0.44,
            continuation_max_negative_lag_gap=0.04,
            continuation_ignore_lag_veto=True,
            lag_threshold=0.015,
        )
        seed_clean_checkpoints_up(ev, n=3)
        with _patch_time(WINDOW_OPEN_TS + 791), _patch_gbm(0.52):
            result = ev._run_gates(ASSET_PRICE_UP, YES_ASK_IN_RANGE, 0.43)
        assert result["result"] == GATE_FIRED
        assert result["entry_model"] == "continuation"

    def test_cooldown_also_blocks_continuation_path(self, tmp_path):
        """Continuation must not bypass the one-signal-per-window cooldown."""
        ev = make_evaluator(
            tmp_path,
            time_gate_minutes=1.0,
            continuation_enabled=True,
            continuation_min_minutes_remaining=1.5,
            continuation_min_asset_move_pct=0.0004,
            continuation_min_chop_score=0.6,
            continuation_max_momentum_price=0.61,
            continuation_max_opposite_price=0.44,
            continuation_max_negative_lag_gap=0.04,
            continuation_ignore_lag_veto=True,
            lag_threshold=0.015,
        )
        seed_clean_checkpoints_up(ev, n=3)
        ev._state.signal_fired = True
        ev._state.signal_fired_side = "YES"
        with _patch_time(WINDOW_OPEN_TS + 791), _patch_gbm(0.52):
            result = ev._run_gates(ASSET_PRICE_UP, YES_ASK_IN_RANGE, 0.43)
        assert result["result"] == GATE_COOLDOWN

    def test_continuation_blocks_when_move_is_too_small_even_if_other_gates_pass(self, tmp_path):
        ev = make_evaluator(
            tmp_path,
            time_gate_minutes=1.0,
            continuation_enabled=True,
            continuation_min_minutes_remaining=1.5,
            continuation_min_asset_move_pct=0.0004,
            continuation_min_chop_score=0.6,
            continuation_max_momentum_price=0.61,
            continuation_max_opposite_price=0.44,
            continuation_max_negative_lag_gap=0.04,
            lag_threshold=0.015,
            asset_vol_overrides={"min_asset_move_pct": 0.0002},
        )
        ev._state.checkpoints = [
            WindowCheckpoint(ts=float(WINDOW_OPEN_TS + i * 60), price=ASSET_OPEN + i * 10.0)
            for i in range(3)
        ]
        tiny_move_price = ASSET_OPEN * 1.0003  # 0.03% move: above global gate, below continuation floor
        with _patch_time(WINDOW_OPEN_TS + 791), _patch_gbm(0.5797):
            result = ev._run_gates(tiny_move_price, YES_ASK_IN_RANGE, 0.43)
        assert result["result"] == GATE_LAG

    def test_cooldown_blocks_same_side_in_same_window(self, tmp_path):
        """Second YES signal in same window is blocked by cooldown gate."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)
        ev._state.signal_fired = True
        ev._state.signal_fired_side = "YES"
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            result = ev._run_gates(ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE)
        assert result["result"] == GATE_COOLDOWN
        assert result["fired_side"] == "YES"

    def test_cooldown_blocks_opposite_side_too(self, tmp_path):
        """Once a signal fires, the window is locked out even if direction flips."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_down(ev)
        ev._state.signal_fired = True
        ev._state.signal_fired_side = "YES"   # YES already fired
        # Now BTC going DOWN → momentum_side = "NO", different from "YES"
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.30):
            result = ev._run_gates(ASSET_PRICE_DOWN, YES_ASK_OPPOSITE, NO_ASK_IN_RANGE)
        assert result["result"] == GATE_COOLDOWN


# ---------------------------------------------------------------------------
# Successful signal: all gates pass (YES and NO symmetry)
# ---------------------------------------------------------------------------

class TestSignalFires:
    """Happy-path tests: all gates pass and signal is emitted."""

    def test_yes_signal_fires_on_up_move(self, tmp_path):
        """All gates pass for a YES signal when BTC moves up."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)
        # yes_ask=0.58, implied=0.70 → lag_gap=0.12 ≥ 0.04 ✓
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            result = ev._run_gates(ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE)
        assert result["result"] == GATE_FIRED
        assert result["momentum_side"] == "YES"
        assert result["momentum_price"] == YES_ASK_IN_RANGE
        assert result["opposite_price"] == NO_ASK_OPPOSITE
        assert result["lag_gap"] > 0.04
        assert result["chop_score"] == 1.0

    def test_no_signal_fires_on_down_move(self, tmp_path):
        """All gates pass for a NO signal when BTC moves down."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_down(ev)
        # P(up)=0.30 → implied_NO = 1-0.30 = 0.70; no_ask=0.58 → lag_gap=0.12 ✓
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.30):
            result = ev._run_gates(ASSET_PRICE_DOWN, YES_ASK_OPPOSITE, NO_ASK_IN_RANGE)
        assert result["result"] == GATE_FIRED
        assert result["momentum_side"] == "NO"
        assert result["momentum_price"] == NO_ASK_IN_RANGE
        assert result["opposite_price"] == YES_ASK_OPPOSITE

    def test_evaluate_returns_bracket_signal_event(self, tmp_path):
        """evaluate() returns a fully populated BracketSignalEvent on success."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            event = ev.evaluate(
                ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE, SAMPLE_MARKET_META
            )
        assert event is not None
        assert event.asset == "BTC"
        assert event.entry_model == "lag"
        assert event.momentum_side == "YES"
        assert event.momentum_price == YES_ASK_IN_RANGE
        assert event.market_id == SAMPLE_MARKET_META["market_id"]
        assert event.yes_token_id == SAMPLE_MARKET_META["yes_token_id"]
        assert event.lag_gap > 0.04
        assert event.chop_score == 1.0
        assert event.mid_window_start is False
        assert event.required_shares == pytest.approx(5.0)
        assert event.momentum_ask_size == pytest.approx(25.0)
        assert event.phase1_would_fill is True
        # State updated
        assert ev._state.signal_fired is True
        assert ev._state.signal_fired_side == "YES"
        assert ev._state.observation is not None
        assert ev._state.observation.event_id == event.event_id

    def test_evaluate_writes_to_jsonl_log(self, tmp_path):
        """A fired signal is written to the signal events JSONL log."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            event = ev.evaluate(
                ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE, SAMPLE_MARKET_META
            )
        log_path = tmp_path / "signal_events.jsonl"
        assert log_path.exists()
        content = log_path.read_text()
        assert '"type": "signal"' in content
        assert event.event_id in content

    def test_evaluate_returns_none_on_gate_failure(self, tmp_path):
        """evaluate() returns None and does NOT write a signal record when blocked."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)
        # Time gate failure: only 1 minute left
        near_end = WINDOW_CLOSE_TS - 60
        with _patch_time(near_end), _patch_gbm(0.70):
            result = ev.evaluate(
                ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE, SAMPLE_MARKET_META
            )
        assert result is None

    def test_evaluate_writes_evaluation_log_on_every_call(self, tmp_path):
        """The evaluation JSONL log receives a record regardless of pass or fail."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)
        near_end = WINDOW_CLOSE_TS - 60
        with _patch_time(near_end), _patch_gbm(0.70):
            ev.evaluate(ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE, SAMPLE_MARKET_META)
        eval_log = tmp_path / "evaluations.jsonl"
        assert eval_log.exists()
        content = eval_log.read_text()
        assert "TIME_GATE_FAIL" in content

    def test_net_bracket_at_target_uses_correct_fee_formula(self, tmp_path):
        """BracketSignalEvent.net_bracket_at_target matches the actual fee curve."""
        from src.fees import taker_fee
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            event = ev.evaluate(
                ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE, SAMPLE_MARKET_META
            )
        assert event is not None
        fee_x = taker_fee(YES_ASK_IN_RANGE, category="crypto price")
        fee_y = taker_fee(0.34, category="crypto price")   # target_y_price
        expected = 1.0 - YES_ASK_IN_RANGE * (1 + fee_x) - 0.34 * (1 + fee_y)
        assert event.net_bracket_at_target == pytest.approx(expected, abs=1e-5)

    def test_mid_window_flag_preserved_in_event(self, tmp_path):
        """Signals fired on a mid_window_start window carry the flag."""
        cfg = make_config(tmp_path)
        ev = DirectionSignalEvaluator("BTC", cfg, make_asset_vol_cfg())
        ev.on_window_open(WINDOW_OPEN_TS, ASSET_OPEN, mid_window=True)
        seed_clean_checkpoints_up(ev)
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            event = ev.evaluate(
                ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE, SAMPLE_MARKET_META
            )
        assert event is not None
        assert event.mid_window_start is True


# ---------------------------------------------------------------------------
# Post-signal observation tracking (update_post_signal)
# ---------------------------------------------------------------------------

class TestPostSignalTracking:
    """Post-signal bracket and Phase 2 observation metrics."""

    def _fire_yes_signal(self, tmp_path) -> DirectionSignalEvaluator:
        """Helper: fire a YES signal and return the evaluator ready for tracking."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            ev.evaluate(ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE, SAMPLE_MARKET_META)
        return ev

    def test_no_update_before_signal_fires(self, tmp_path):
        """update_post_signal is a no-op when no signal has fired."""
        ev = make_evaluator(tmp_path)
        ev.update_post_signal(yes_ask=0.62, no_ask=0.38)
        assert ev._state.observation is None

    def test_momentum_peak_tracks_highest_price(self, tmp_path):
        """momentum_side_peak updates whenever a higher price is seen."""
        ev = self._fire_yes_signal(tmp_path)
        obs = ev._state.observation

        ev.update_post_signal(yes_ask=0.68, no_ask=0.32)
        assert obs.momentum_side_peak == pytest.approx(0.68)

        ev.update_post_signal(yes_ask=0.65, no_ask=0.35)  # lower — should NOT update
        assert obs.momentum_side_peak == pytest.approx(0.68)

    def test_bracket_forms_when_equation_satisfied(self, tmp_path):
        """bracket_would_have_formed is set when 1 - x*(1+fee_x) - y*(1+fee_y) > 0."""
        ev = self._fire_yes_signal(tmp_path)
        obs = ev._state.observation
        # Entry was YES at 0.58. At y=0.34 the bracket is profitable (as verified by startup diagnostics).
        ev.update_post_signal(yes_ask=0.58, no_ask=0.34)
        assert obs.bracket_would_have_formed is True
        assert obs.peak_bracket_margin > 0

    def test_bracket_does_not_form_when_too_expensive(self, tmp_path):
        """bracket_would_have_formed stays False when y is too expensive."""
        ev = self._fire_yes_signal(tmp_path)
        obs = ev._state.observation
        # NO at 0.50 makes 0.58 + 0.50 + fees > 1.00 → no bracket
        ev.update_post_signal(yes_ask=0.60, no_ask=0.50)
        assert obs.bracket_would_have_formed is False

    def test_min_opposite_price_tracks_floor(self, tmp_path):
        """min_opposite_price tracks the lowest y price seen since signal."""
        ev = self._fire_yes_signal(tmp_path)
        obs = ev._state.observation

        ev.update_post_signal(yes_ask=0.60, no_ask=0.38)
        ev.update_post_signal(yes_ask=0.62, no_ask=0.34)  # new floor
        ev.update_post_signal(yes_ask=0.63, no_ask=0.36)  # bouncing up
        assert obs.min_opposite_price == pytest.approx(0.34)

    def test_safe_opposite_price_arms_at_first_safe_touch(self, tmp_path):
        """The first observed profitable safe price is remembered but not executed."""
        ev = self._fire_yes_signal(tmp_path)
        obs = ev._state.observation

        ev.update_post_signal(yes_ask=0.60, no_ask=0.34, yes_ask_size=25.0, no_ask_size=25.0)
        assert obs.safe_opposite_price == pytest.approx(0.34)
        assert obs.phase2_would_have_triggered is False
        assert obs.dipped_below_safe_price is False

    def test_phase2_triggers_only_after_safe_level_dips_and_reclaims(self, tmp_path):
        """Phase 2 requires y_safe discovery, extension below it, then reclaim back to it."""
        ev = self._fire_yes_signal(tmp_path)
        obs = ev._state.observation

        ev.update_post_signal(yes_ask=0.60, no_ask=0.34, yes_ask_size=25.0, no_ask_size=25.0)   # arm y_safe
        assert obs.phase2_would_have_triggered is False

        ev.update_post_signal(yes_ask=0.60, no_ask=0.32, yes_ask_size=25.0, no_ask_size=25.0)   # extension below safe
        assert obs.dipped_below_safe_price is True
        assert obs.phase2_would_have_triggered is False

        ev.update_post_signal(yes_ask=0.60, no_ask=0.345, yes_ask_size=25.0, no_ask_size=25.0)  # reclaim into safe band
        assert obs.phase2_reclaim_seen is True
        assert obs.phase2_would_fill is True
        assert obs.phase2_would_have_triggered is True
        assert obs.phase2_trigger_price == pytest.approx(0.345)

    def test_phase2_does_not_trigger_without_extension_below_safe(self, tmp_path):
        """Touching y_safe and bouncing around it is not enough on its own."""
        ev = self._fire_yes_signal(tmp_path)
        obs = ev._state.observation

        ev.update_post_signal(yes_ask=0.60, no_ask=0.34, yes_ask_size=25.0, no_ask_size=25.0)   # arm safe
        ev.update_post_signal(yes_ask=0.60, no_ask=0.345, yes_ask_size=25.0, no_ask_size=25.0)  # reclaim-like move without lower extension
        assert obs.phase2_would_have_triggered is False

    def test_phase2_does_not_trigger_if_reclaim_never_reaches_safe(self, tmp_path):
        """A rebound that stays below y_safe is still just continuation noise."""
        ev = self._fire_yes_signal(tmp_path)
        obs = ev._state.observation

        ev.update_post_signal(yes_ask=0.60, no_ask=0.34, yes_ask_size=25.0, no_ask_size=25.0)   # arm safe
        ev.update_post_signal(yes_ask=0.60, no_ask=0.31, yes_ask_size=25.0, no_ask_size=25.0)   # extension
        ev.update_post_signal(yes_ask=0.60, no_ask=0.335, yes_ask_size=25.0, no_ask_size=25.0)  # still below safe=0.34
        assert obs.phase2_would_have_triggered is False

    def test_phase2_does_not_trigger_twice(self, tmp_path):
        """Phase 2 trigger is one-shot; later moves don't overwrite the first reclaim."""
        ev = self._fire_yes_signal(tmp_path)
        obs = ev._state.observation

        ev.update_post_signal(yes_ask=0.60, no_ask=0.34, yes_ask_size=25.0, no_ask_size=25.0)   # arm safe
        ev.update_post_signal(yes_ask=0.60, no_ask=0.32, yes_ask_size=25.0, no_ask_size=25.0)   # extension
        ev.update_post_signal(yes_ask=0.60, no_ask=0.345, yes_ask_size=25.0, no_ask_size=25.0)  # first reclaim trigger
        first_trigger_price = obs.phase2_trigger_price
        assert obs.phase2_would_have_triggered is True

        ev.update_post_signal(yes_ask=0.60, no_ask=0.35, yes_ask_size=25.0, no_ask_size=25.0)   # more bouncing
        assert obs.phase2_trigger_price == first_trigger_price

    def test_phase2_reclaim_seen_but_not_executable_with_insufficient_size(self, tmp_path):
        """Price-only reclaim should not count as a live trigger if top ask size is too thin."""
        ev = self._fire_yes_signal(tmp_path)
        obs = ev._state.observation

        ev.update_post_signal(yes_ask=0.60, no_ask=0.34, yes_ask_size=25.0, no_ask_size=25.0)
        ev.update_post_signal(yes_ask=0.60, no_ask=0.32, yes_ask_size=25.0, no_ask_size=25.0)
        ev.update_post_signal(yes_ask=0.60, no_ask=0.345, yes_ask_size=25.0, no_ask_size=4.0)

        assert obs.phase2_reclaim_seen is True
        assert obs.phase2_would_fill is False
        assert obs.phase2_would_have_triggered is False
        assert obs.phase2_trigger_ask_size == pytest.approx(4.0)

    def test_opposite_tracked_correctly_for_no_momentum(self, tmp_path):
        """When NO is momentum, YES is the opposite side being tracked."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_down(ev)
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.30):
            ev.evaluate(ASSET_PRICE_DOWN, YES_ASK_OPPOSITE, NO_ASK_IN_RANGE, SAMPLE_MARKET_META)
        assert ev._last_event.momentum_side == "NO"

        # YES price falls → it's the opposite, its floor should be tracked
        ev.update_post_signal(yes_ask=0.38, no_ask=0.60)
        obs = ev._state.observation
        assert obs.min_opposite_price <= 0.38


# ---------------------------------------------------------------------------
# Checkpoint deduplication and capping
# ---------------------------------------------------------------------------

class TestCheckpointDeduplication:
    """Verify :00-second boundary logic, deduplication, and 20-entry cap."""

    def test_non_boundary_second_not_recorded(self, tmp_path):
        """maybe_record_checkpoint is a no-op when time is not at a :00-second mark."""
        ev = make_evaluator(tmp_path)
        initial_count = len(ev._state.checkpoints)
        # 1_000_045 % 60 = 45 — not a boundary
        with patch("src.crypto_direction_signal.time.time", return_value=1_000_045.0):
            ev.maybe_record_checkpoint(84_100.0)
        assert len(ev._state.checkpoints) == initial_count

    def test_boundary_second_recorded(self, tmp_path):
        """maybe_record_checkpoint adds an entry at a :00-second boundary."""
        ev = make_evaluator(tmp_path)
        initial_count = len(ev._state.checkpoints)
        # 1_000_080 % 60 == 0 — valid boundary
        with patch("src.crypto_direction_signal.time.time", return_value=1_000_080.0):
            ev.maybe_record_checkpoint(84_150.0)
        assert len(ev._state.checkpoints) == initial_count + 1
        assert ev._state.checkpoints[-1].price == 84_150.0
        assert ev._state.checkpoints[-1].ts == 1_000_080.0

    def test_same_second_recorded_only_once(self, tmp_path):
        """Calling maybe_record_checkpoint twice at the same second adds only one entry."""
        ev = make_evaluator(tmp_path)
        boundary_ts = 1_000_080.0
        with patch("src.crypto_direction_signal.time.time", return_value=boundary_ts):
            ev.maybe_record_checkpoint(84_150.0)
            ev.maybe_record_checkpoint(84_160.0)  # second call — deduplicated
        entries_at_ts = [c for c in ev._state.checkpoints if c.ts == boundary_ts]
        assert len(entries_at_ts) == 1
        assert entries_at_ts[0].price == 84_150.0  # first call wins

    def test_different_seconds_both_recorded(self, tmp_path):
        """Two different :00-second boundaries both get their own checkpoint."""
        ev = make_evaluator(tmp_path)
        initial_count = len(ev._state.checkpoints)
        with patch("src.crypto_direction_signal.time.time", return_value=1_000_080.0):
            ev.maybe_record_checkpoint(84_150.0)
        with patch("src.crypto_direction_signal.time.time", return_value=1_000_140.0):
            ev.maybe_record_checkpoint(84_200.0)
        assert len(ev._state.checkpoints) == initial_count + 2

    def test_checkpoint_list_capped_at_20(self, tmp_path):
        """Checkpoint list is trimmed to the last 20 entries to avoid unbounded growth."""
        ev = make_evaluator(tmp_path)
        # Manually pre-fill 25 checkpoints
        for i in range(25):
            ev._state.checkpoints.append(
                WindowCheckpoint(ts=float(WINDOW_OPEN_TS + i * 60), price=ASSET_OPEN + i)
            )
        # Use a timestamp that is divisible by 60.
        # WINDOW_OPEN_TS=1_000_000; 1_000_000 % 60 == 40, so we need an offset.
        # 1_000_020 = 16_667 × 60 → divisible by 60. Use it as the base for our boundary ts.
        new_ts = 1_001_520.0    # 1_000_020 + 25*60 = 1_001_520, and 1_001_520 % 60 == 0
        assert int(new_ts) % 60 == 0, "new_ts must be at a :00-second boundary"
        with patch("src.crypto_direction_signal.time.time", return_value=new_ts):
            ev.maybe_record_checkpoint(ASSET_OPEN + 99.0)
        assert len(ev._state.checkpoints) <= 20


# ---------------------------------------------------------------------------
# Window lifecycle
# ---------------------------------------------------------------------------

class TestWindowLifecycle:
    """Tests for on_window_open and on_window_close behaviour."""

    def test_on_window_open_resets_signal_state(self, tmp_path):
        """Opening a new window clears the previous signal/cooldown state."""
        ev = make_evaluator(tmp_path)
        ev._state.signal_fired = True
        ev._state.signal_fired_side = "YES"

        new_ts = WINDOW_OPEN_TS + 900
        ev.on_window_open(new_ts, 85_000.0)

        assert ev._state.signal_fired is False
        assert ev._state.signal_fired_side == ""
        assert ev._state.window_open_ts == new_ts
        assert ev._state.asset_open == 85_000.0
        assert ev._last_event is None

    def test_on_window_open_mid_window_flag_preserved(self, tmp_path):
        """mid_window=True is stored in state for later logging."""
        cfg = make_config(tmp_path)
        ev = DirectionSignalEvaluator("BTC", cfg, make_asset_vol_cfg())
        ev.on_window_open(WINDOW_OPEN_TS, ASSET_OPEN, mid_window=True)
        assert ev._state.mid_window_start is True

    def test_on_window_close_writes_outcome_after_signal(self, tmp_path):
        """on_window_close appends outcome record to JSONL when a signal fired."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            ev.evaluate(ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE, SAMPLE_MARKET_META)

        ev.on_window_close(asset_close=84_500.0, yes_won=True)

        log_path = tmp_path / "signal_events.jsonl"
        content = log_path.read_text()
        assert '"type": "outcome"' in content
        assert "YES_WINS" in content

    def test_on_window_close_with_no_win_writes_no_wins(self, tmp_path):
        """YES losses are logged as NO_WINS in the outcome record."""
        ev = make_evaluator(tmp_path)
        seed_clean_checkpoints_up(ev)
        with _patch_time(NOW_IN_WINDOW), _patch_gbm(0.70):
            ev.evaluate(ASSET_PRICE_UP, YES_ASK_IN_RANGE, NO_ASK_OPPOSITE, SAMPLE_MARKET_META)

        ev.on_window_close(asset_close=83_900.0, yes_won=False)
        content = (tmp_path / "signal_events.jsonl").read_text()
        assert "NO_WINS" in content

    def test_on_window_close_no_write_if_no_signal(self, tmp_path):
        """on_window_close is a no-op (no log write) when no signal fired."""
        ev = make_evaluator(tmp_path)
        ev.on_window_close(asset_close=84_000.0, yes_won=True)
        log_path = tmp_path / "signal_events.jsonl"
        if log_path.exists():
            assert '"type": "outcome"' not in log_path.read_text()

    def test_on_window_open_seeds_initial_checkpoint(self, tmp_path):
        """on_window_open always adds one checkpoint at the window open time."""
        cfg = make_config(tmp_path)
        ev = DirectionSignalEvaluator("BTC", cfg, make_asset_vol_cfg())
        ev.on_window_open(WINDOW_OPEN_TS, ASSET_OPEN)
        assert len(ev._state.checkpoints) == 1
        assert ev._state.checkpoints[0].ts == float(WINDOW_OPEN_TS)
        assert ev._state.checkpoints[0].price == ASSET_OPEN
