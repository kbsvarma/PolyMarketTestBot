"""
Window Report Writer — per-15-minute-cycle summary report.

Generates a human-readable Markdown report (logs/window_report.md) that
records every 15-minute window cycle: whether a signal fired, why it didn't
if not, and hypothetical PnL if bets had been placed.

One record per window, updated at every window transition. Designed to be
left open overnight and checked periodically.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Gate result display names
# ---------------------------------------------------------------------------

_GATE_LABELS: dict[str, str] = {
    # Keys must match constants in crypto_direction_signal.py exactly
    "WINDOW_SETTLING":          "Settling (AMM noise)",
    "TIME_GATE_FAIL":           "Too late in window",
    "ASSET_MOVE_INSUFFICIENT":  "BTC/ETH move too small",
    "PRICE_RANGE_FAIL":         "Price outside entry range",
    "PRICES_STALE":             "Orderbook prices stale",
    "CHOP_FILTER_FAIL":         "Choppy / no clean direction",
    "LAG_GAP_INSUFFICIENT":     "No GBM lag (market already priced in)",
    "ALREADY_FIRED_THIS_WINDOW": "Signal already fired this window",
    "SIGNAL_FIRED":             "✅ Signal fired",
    "NO_WINDOW_STATE":          "No window state",
}

_GATE_RANK: dict[str, int] = {
    # Higher rank = further through the gate sequence = better signal quality
    # Keys must match gate result strings from crypto_direction_signal.py
    "NO_WINDOW_STATE":          0,
    "WINDOW_SETTLING":          1,
    "TIME_GATE_FAIL":           2,
    "ASSET_MOVE_INSUFFICIENT":  3,
    "PRICE_RANGE_FAIL":         4,
    "PRICES_STALE":             5,
    "CHOP_FILTER_FAIL":         6,
    "LAG_GAP_INSUFFICIENT":     7,
    "ALREADY_FIRED_THIS_WINDOW": 8,
    "SIGNAL_FIRED":             9,
}


# ---------------------------------------------------------------------------
# Per-window summary record
# ---------------------------------------------------------------------------

@dataclass
class WindowSummary:
    window_ts: int              # Unix timestamp of window start
    window_close_ts: int        # window_ts + window_duration_seconds
    asset: str                  # "BTC" or "ETH"
    asset_open: float           # price at window open
    asset_close: float          # price at window close (last BTC/ETH price seen)
    yes_ask_final: float        # YES ask at window close (0.99 → YES likely won)
    no_ask_final: float         # NO ask at window close
    primary_gate_fail: str      # deepest gate reached (furthest in sequence)
    dominant_gate_fail: str     # most frequent blocker across the window
    dominant_gate_count: int    # number of eval cycles blocked there
    total_eval_cycles: int      # total eval cycles captured for the window
    gate_counts: dict           # {gate: count} — how often each gate fired
    near_miss_counts: dict      # compact near-miss diagnostics for this window
    counterfactual_counts: dict # single-guardrail loosening diagnostics
    signal_fired: bool          # did a Phase 1 signal fire?
    signal_side: str            # "YES" or "NO" if fired, else ""
    signal_entry_model: str     # "lag" or "continuation" when fired
    signal_price: float         # momentum-side entry ask price
    phase1_would_fill: bool = True
    phase1_ask_size: float = 0.0
    required_shares: float = 0.0
    safe_opposite_price: float = 0.0   # remembered y_safe level, if armed
    dipped_below_safe_price: bool = False
    phase2_reclaim_seen: bool = False
    phase2_triggered: bool = False     # did Phase 2 (bracket leg 2) trigger?
    phase2_price: float = 0.0          # opposite-side entry price for leg 2
    phase2_trigger_ask_size: float = 0.0
    phase2_would_fill: bool = False
    resolved_yes: Optional[bool] = None   # True=YES won, False=NO won, None=unclear
    share_count: float = 0.0              # configured or actual shares for this window
    hyp_pnl_per_share: float = 0.0        # PnL per share
    hyp_pnl_dollars: float = 0.0          # total PnL in dollars (share_count scaled)
    hard_exit_modeled: bool = False        # True if hard-exit stop (50¢) was modeled to have fired
    used_actual_execution: bool = False
    execution_mode: str = "observe"
    execution_phase: str = ""
    execution_phase1_filled: bool = False
    execution_phase2_filled: bool = False
    execution_p1_fill_price: float = 0.0
    execution_position_id: str = ""
    execution_p1_order_id: str = ""
    execution_p2_order_id: str = ""
    execution_hard_exit_order_ids: list[str] = field(default_factory=list)
    execution_phase2_attempted: bool = False
    execution_hard_exit_attempted: bool = False
    execution_hard_exit_reason: str = ""
    execution_hard_exit_fill_price: float = 0.0
    execution_hard_exit_filled_shares: float = 0.0
    outcome_source: str = "asset_price_proxy"

    @property
    def move_pct(self) -> float:
        if self.asset_open <= 0:
            return 0.0
        return (self.asset_close - self.asset_open) / self.asset_open

    @property
    def window_label(self) -> str:
        """Format: '9:15 PM – 9:30 PM ET'"""
        def _fmt(ts: int) -> str:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            # Convert UTC → ET (UTC-4 for EDT, UTC-5 for EST)
            # Use simple -4 offset (Eastern Daylight Time, March)
            et_hour = (dt.hour - 4) % 24
            suffix = "AM" if et_hour < 12 else "PM"
            h = et_hour % 12 or 12
            return f"{h}:{dt.minute:02d} {suffix}"
        return f"{_fmt(self.window_ts)} – {_fmt(self.window_close_ts)} ET"

    @property
    def date_label(self) -> str:
        dt = datetime.fromtimestamp(self.window_ts, tz=timezone.utc)
        et_hour = (dt.hour - 4) % 24
        et_date = dt.replace(hour=et_hour)
        return et_date.strftime("%b %d, %Y")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

class WindowReportWriter:
    """
    Maintains an in-memory list of per-window summaries and rewrites the
    Markdown report file on every window close.

    Usage (in the observer loop, on each window transition):
        writer.record_window_close(
            window_ts=..., asset=..., asset_open=..., asset_close=...,
            yes_ask_final=..., no_ask_final=...,
            last_signal_event=..., eval_log_path=...,
        )
    """

    def __init__(
        self,
        report_path: str,
        report_shares: float = 10.0,
        max_windows: int = 200,
        live_execution: bool = False,   # backward-compatible flag
        execution_mode: str = "observe",
        window_duration_seconds: int = 900,
    ) -> None:
        self._report_path = Path(report_path)
        self._share_count = report_shares
        self._max_windows = max_windows
        self._execution_mode = "live" if live_execution and execution_mode == "observe" else execution_mode
        self._live_execution = live_execution or execution_mode in {"shadow", "live"}
        self._window_duration_seconds = window_duration_seconds
        self._windows: list[WindowSummary] = []
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        # Seed with an empty report on startup
        self._write_report()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record_window_close(
        self,
        window_ts: int,
        asset: str,
        asset_open: float,
        asset_close: float,
        yes_ask_final: float,
        no_ask_final: float,
        last_signal_event: dict | None,
        eval_log_path: str,
        signal_log_path: str | None = None,
        yes_won: bool | None = None,
        outcome_source: str = "asset_price_proxy",
        execution_summary: dict | None = None,
    ) -> None:
        """
        Called when a window transition is detected. Reads the evaluation log
        for the closing window, builds a summary, appends it, and rewrites
        the report.

        yes_won: directly passed from the evaluator (asset_close > asset_open).
          Preferred over inferring from ask prices, which are unreliable at
          settlement time (both tokens go near 0, making inference ambiguous).
        """
        eval_records = self._scan_eval_log(eval_log_path, window_ts, asset)
        gate_counts = Counter(rec.get("result", "UNKNOWN") for rec in eval_records)

        # Determine the deepest gate reached (furthest in sequence)
        primary_gate = max(
            gate_counts.keys(),
            key=lambda g: _GATE_RANK.get(g, 0),
            default="NO_STATE",
        )
        dominant_gate = max(
            gate_counts.keys(),
            key=lambda g: gate_counts.get(g, 0),
            default="NO_STATE",
        )
        dominant_gate_count = int(gate_counts.get(dominant_gate, 0))
        total_eval_cycles = int(sum(gate_counts.values()))
        near_miss_counts = self._summarize_near_misses(eval_records)
        counterfactual_counts = self._summarize_counterfactuals(eval_records)

        # After a restart, the in-memory last_signal_event may be gone even when
        # the eval log already shows SIGNAL_FIRED for this window. Recover the
        # signal from the persisted signal log so the report doesn't falsely say
        # "no signal" for windows that really fired.
        if (
            last_signal_event is None
            and gate_counts.get("SIGNAL_FIRED", 0) > 0
            and signal_log_path
        ):
            last_signal_event = self._find_signal_event(signal_log_path, window_ts, asset)

        # Signal details from the last signal event (if any)
        signal_fired = False
        signal_side = ""
        signal_entry_model = ""
        signal_price = 0.0
        phase1_would_fill = True
        phase1_ask_size = 0.0
        required_shares = self._share_count
        safe_opposite_price = 0.0
        dipped_below_safe_price = False
        phase2_reclaim_seen = False
        phase2_triggered = False
        phase2_price = 0.0
        phase2_trigger_ask_size = 0.0
        phase2_would_fill = False

        if last_signal_event:
            signal_fired = True
            signal_side = last_signal_event.get("momentum_side", "")
            signal_entry_model = str(last_signal_event.get("entry_model") or "lag")
            signal_price = float(last_signal_event.get("momentum_price", 0.0))
            phase1_would_fill = bool(last_signal_event.get("phase1_would_fill", True))
            phase1_ask_size = float(last_signal_event.get("momentum_ask_size") or 0.0)
            required_shares = float(last_signal_event.get("required_shares") or self._share_count)
            # Check if phase2 was triggered (present in post-signal observations)
            obs = last_signal_event.get("observation") or {}
            safe_opposite_price = float(obs.get("safe_opposite_price") or 0.0)
            dipped_below_safe_price = bool(obs.get("dipped_below_safe_price"))
            phase2_reclaim_seen = bool(obs.get("phase2_reclaim_seen"))
            phase2_price = float(obs.get("phase2_trigger_price") or 0.0)
            phase2_trigger_ask_size = float(obs.get("phase2_trigger_ask_size") or 0.0)
            phase2_would_fill = bool(obs.get("phase2_would_fill"))
            if obs and obs.get("phase2_would_have_triggered"):
                phase2_triggered = True

        # Resolve outcome — prefer the directly-passed yes_won flag (most reliable).
        # Post-settlement ask prices are unreliable: after market close both tokens
        # drop to near-zero (no one quotes a settled market), so price-based inference
        # frequently returns wrong results.  The yes_won flag is computed directly
        # from asset_close > asset_open which is always available and correct.
        resolved_yes: Optional[bool] = None
        if yes_won is not None:
            resolved_yes = yes_won   # direct — always use this when available
        elif yes_ask_final >= 0.90:
            resolved_yes = True      # YES near 1.0 = YES won (pre-settlement snap)
        elif no_ask_final >= 0.90:
            resolved_yes = False     # NO near 1.0 = NO won (pre-settlement snap)
        elif yes_ask_final <= 0.10 and no_ask_final > yes_ask_final:
            resolved_yes = False     # YES near 0 AND NO higher → NO won
        elif no_ask_final <= 0.10 and yes_ask_final > no_ask_final:
            resolved_yes = True      # NO near 0 AND YES higher → YES won
        # else: both near 0 and can't tell — leave None (ambiguous post-settlement)

        # Pull observation fields (populated since timing-fix reorder)
        obs = last_signal_event.get("observation") or {} if last_signal_event else {}
        min_momentum_price: float = float(obs.get("min_momentum_price") or 999.0)

        # Compute hypothetical PnL
        hyp_pnl_per_share = 0.0
        hard_exit_modeled = False
        if signal_fired and not phase1_would_fill:
            hyp_pnl_per_share = 0.0
        elif signal_fired and signal_price > 0 and resolved_yes is not None:
            # Phase 1 entry: bet momentum_side at signal_price
            # Payout $1.00 if momentum direction wins, $0.00 if it loses
            fee = _taker_fee(signal_price)
            phase1_cost = signal_price + fee

            if phase2_triggered and phase2_price > 0:
                # Bracket complete: guaranteed profit regardless of outcome.
                # One side pays $1, the other $0 — net = 1 - total_cost.
                opp_fee = _taker_fee(phase2_price)
                total_cost = phase1_cost + phase2_price + opp_fee
                hyp_pnl_per_share = 1.0 - total_cost
            else:
                # Phase 1 only: depends on resolution
                momentum_won = (
                    (signal_side == "YES" and resolved_yes is True) or
                    (signal_side == "NO" and resolved_yes is False)
                )
                if momentum_won:
                    hyp_pnl_per_share = 1.0 - phase1_cost
                else:
                    # Lost — check if hard exit at 0.50 would have capped the loss.
                    # Hard exit fires when momentum mark drops to hard_exit_stop_price.
                    # In live mode bracket_executor sells at ~(mark - 2c); model at 0.50.
                    _HARD_EXIT_STOP = 0.50
                    if min_momentum_price <= _HARD_EXIT_STOP < signal_price:
                        # The price DID drop to 0.50 during the window — hard exit fires.
                        hard_exit_price = _HARD_EXIT_STOP
                        hard_exit_fee = _taker_fee(hard_exit_price)
                        # Net loss: paid phase1_cost, received hard_exit_price - exit_fee
                        hyp_pnl_per_share = (hard_exit_price - hard_exit_fee) - signal_price - fee
                        hard_exit_modeled = True
                    else:
                        hyp_pnl_per_share = -phase1_cost

        share_count = self._share_count
        hyp_pnl_dollars = hyp_pnl_per_share * share_count
        used_actual_execution = False
        execution_mode = self._execution_mode
        execution_phase = ""
        execution_phase1_filled = False
        execution_phase2_filled = False
        execution_p1_fill_price = 0.0
        execution_position_id = ""
        execution_p1_order_id = ""
        execution_p2_order_id = ""
        execution_hard_exit_order_ids: list[str] = []
        execution_phase2_attempted = False
        execution_hard_exit_attempted = False
        execution_hard_exit_reason = ""
        execution_hard_exit_fill_price = 0.0
        execution_hard_exit_filled_shares = 0.0

        if self._live_execution and execution_summary:
            used_actual_execution = True
            execution_mode = str(
                execution_summary.get("execution_mode")
                or ("live" if self._execution_mode == "observe" else self._execution_mode)
            )
            if execution_summary.get("signal_side"):
                signal_fired = True
                signal_side = str(execution_summary.get("signal_side") or signal_side)
            if execution_summary.get("signal_entry_model"):
                signal_entry_model = str(
                    execution_summary.get("signal_entry_model") or signal_entry_model
                )
            if execution_summary.get("signal_price"):
                signal_price = float(execution_summary.get("signal_price") or signal_price)
            execution_phase = str(execution_summary.get("phase") or "")
            execution_phase1_filled = bool(execution_summary.get("phase1_filled"))
            execution_phase2_filled = bool(execution_summary.get("phase2_filled"))
            execution_p1_fill_price = float(execution_summary.get("p1_fill_price") or 0.0)
            execution_position_id = str(execution_summary.get("position_id") or "")
            execution_p1_order_id = str(execution_summary.get("p1_order_id") or "")
            execution_p2_order_id = str(execution_summary.get("p2_order_id") or "")
            execution_hard_exit_order_ids = [
                str(order_id)
                for order_id in (execution_summary.get("hard_exit_order_ids") or [])
                if order_id
            ]
            execution_phase2_attempted = bool(execution_summary.get("phase2_order_attempted"))
            execution_hard_exit_attempted = bool(execution_summary.get("hard_exit_attempted"))
            execution_hard_exit_reason = str(execution_summary.get("hard_exit_reason") or "")
            execution_hard_exit_fill_price = float(
                execution_summary.get("hard_exit_fill_price") or 0.0
            )
            execution_hard_exit_filled_shares = float(
                execution_summary.get("hard_exit_filled_shares") or 0.0
            )
            if execution_summary.get("outcome_source"):
                outcome_source = str(execution_summary.get("outcome_source") or outcome_source)
            if "safe_opposite_price" in execution_summary:
                safe_opposite_price = float(execution_summary.get("safe_opposite_price") or 0.0)
            if "dipped_below_safe_price" in execution_summary:
                dipped_below_safe_price = bool(
                    execution_summary.get("dipped_below_safe_price")
                )
            elif safe_opposite_price > 0:
                min_exec_price = float(execution_summary.get("min_opposite_price") or 0.0)
                if min_exec_price > 0:
                    dipped_below_safe_price = min_exec_price < safe_opposite_price
            if "phase2_reclaim_seen" in execution_summary:
                phase2_reclaim_seen = bool(execution_summary.get("phase2_reclaim_seen"))
            if execution_summary.get("p2_fill_price"):
                phase2_price = float(execution_summary.get("p2_fill_price") or 0.0)
            phase2_triggered = execution_phase2_filled

            actual_pnl_dollars = float(execution_summary.get("actual_pnl_usd") or 0.0)
            shares = float(execution_summary.get("p1_shares") or 0.0)
            if shares > 0:
                share_count = shares
            hyp_pnl_dollars = actual_pnl_dollars
            hyp_pnl_per_share = (actual_pnl_dollars / shares) if shares > 0 else 0.0

        summary = WindowSummary(
            window_ts=window_ts,
            window_close_ts=window_ts + self._window_duration_seconds,
            asset=asset,
            asset_open=asset_open,
            asset_close=asset_close,
            yes_ask_final=yes_ask_final,
            no_ask_final=no_ask_final,
            primary_gate_fail=primary_gate,
            dominant_gate_fail=dominant_gate,
            dominant_gate_count=dominant_gate_count,
            total_eval_cycles=total_eval_cycles,
            gate_counts=dict(gate_counts),
            near_miss_counts=near_miss_counts,
            counterfactual_counts=counterfactual_counts,
            signal_fired=signal_fired,
            signal_side=signal_side,
            signal_entry_model=signal_entry_model,
            signal_price=signal_price,
            phase1_would_fill=phase1_would_fill,
            phase1_ask_size=phase1_ask_size,
            required_shares=required_shares,
            safe_opposite_price=safe_opposite_price,
            dipped_below_safe_price=dipped_below_safe_price,
            phase2_reclaim_seen=phase2_reclaim_seen,
            phase2_triggered=phase2_triggered,
            phase2_price=phase2_price,
            phase2_trigger_ask_size=phase2_trigger_ask_size,
            phase2_would_fill=phase2_would_fill,
            resolved_yes=resolved_yes,
            share_count=share_count,
            hyp_pnl_per_share=hyp_pnl_per_share,
            hyp_pnl_dollars=hyp_pnl_dollars,
            hard_exit_modeled=hard_exit_modeled,
            used_actual_execution=used_actual_execution,
            execution_mode=execution_mode,
            execution_phase=execution_phase,
            execution_phase1_filled=execution_phase1_filled,
            execution_phase2_filled=execution_phase2_filled,
            execution_p1_fill_price=execution_p1_fill_price,
            execution_position_id=execution_position_id,
            execution_p1_order_id=execution_p1_order_id,
            execution_p2_order_id=execution_p2_order_id,
            execution_hard_exit_order_ids=execution_hard_exit_order_ids,
            execution_phase2_attempted=execution_phase2_attempted,
            execution_hard_exit_attempted=execution_hard_exit_attempted,
            execution_hard_exit_reason=execution_hard_exit_reason,
            execution_hard_exit_fill_price=execution_hard_exit_fill_price,
            execution_hard_exit_filled_shares=execution_hard_exit_filled_shares,
            outcome_source=outcome_source,
        )

        self._windows.append(summary)
        # Keep cap to avoid unbounded memory
        if len(self._windows) > self._max_windows:
            self._windows = self._windows[-self._max_windows:]

        self._write_report()
        logger.info(
            "Window report updated window={} asset={} signal={} dominant_gate={} furthest_gate={} hyp_pnl=${:.2f}",
            summary.window_label, asset,
            "FIRED" if signal_fired else "none",
            dominant_gate,
            primary_gate,
            hyp_pnl_dollars,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _scan_eval_log(
        self,
        eval_log_path: str,
        window_ts: int,
        asset: str,
    ) -> list[dict]:
        """Read evaluation records for one asset/window."""
        records: list[dict] = []
        path = Path(eval_log_path)
        if not path.exists():
            return []
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if (
                            rec.get("asset") == asset
                            and rec.get("window_open_ts") == window_ts
                        ):
                            records.append(rec)
                    except Exception:
                        continue
        except Exception as exc:
            logger.debug("Could not scan eval log: {}", exc)
        return records

    def _find_signal_event(
        self,
        signal_log_path: str,
        window_ts: int,
        asset: str,
    ) -> dict | None:
        """Find the latest persisted signal event for one asset/window."""
        path = Path(signal_log_path)
        if not path.exists():
            return None
        match: dict | None = None
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if (
                        rec.get("type") == "signal"
                        and rec.get("asset") == asset
                        and rec.get("window_open_ts") == window_ts
                    ):
                        match = rec
        except Exception as exc:
            logger.debug("Could not scan signal log: {}", exc)
        return match

    def _summarize_near_misses(self, records: list[dict]) -> dict[str, int]:
        summary: Counter = Counter()
        for rec in records:
            result = rec.get("result")
            if result == "PRICE_RANGE_FAIL":
                range_side = rec.get("range_side")
                distance = float(rec.get("range_distance", 999.0) or 999.0)
                if range_side == "HIGH" and distance <= 0.02:
                    summary["price_high_within_2c"] += 1
                elif range_side == "LOW" and distance <= 0.02:
                    summary["price_low_within_2c"] += 1
            elif result == "ASSET_MOVE_INSUFFICIENT":
                shortfall = float(rec.get("move_shortfall", 999.0) or 999.0)
                required = float(rec.get("min_required", 0.0) or 0.0)
                if required > 0 and shortfall <= required * 0.25:
                    summary["move_within_25pct"] += 1
            elif result == "CHOP_FILTER_FAIL":
                shortfall = float(rec.get("chop_shortfall", 999.0) or 999.0)
                if shortfall <= 0.15:
                    summary["chop_within_0_15"] += 1
            elif result == "LAG_GAP_INSUFFICIENT":
                shortfall = float(rec.get("lag_shortfall", 999.0) or 999.0)
                required = float(rec.get("min_required", 0.0) or 0.0)
                if required > 0 and shortfall <= required * 0.25:
                    summary["lag_within_25pct"] += 1
        return dict(summary)

    def _summarize_counterfactuals(self, records: list[dict]) -> dict[str, int]:
        summary: Counter = Counter()
        for rec in records:
            if rec.get("result") in ("SIGNAL_FIRED", "ALREADY_FIRED_THIS_WINDOW"):
                continue
            if _would_fire_with_counterfactual(rec, range_relax_cents=0.02):
                summary["range_plus_2c_would_fire"] += 1
            if _would_fire_with_counterfactual(rec, move_relax_frac=0.25):
                summary["move_25pct_looser_would_fire"] += 1
            if _would_fire_with_counterfactual(rec, chop_relax=0.15):
                summary["chop_minus_0_15_would_fire"] += 1
            if _would_fire_with_counterfactual(rec, lag_relax_frac=0.25):
                summary["lag_25pct_looser_would_fire"] += 1
        return dict(summary)

    def _write_report(self) -> None:
        """Overwrite the Markdown report file with current state."""
        now_et = _now_et()
        total_windows = len(self._windows)
        windows_with_signal = sum(1 for w in self._windows if w.signal_fired)
        if self._live_execution:
            live_executable_entries = sum(
                1
                for w in self._windows
                if w.used_actual_execution and w.execution_phase1_filled
            )
            brackets_complete = sum(
                1 for w in self._windows
                if w.used_actual_execution and w.execution_phase2_filled
            )
            total_hyp_pnl = sum(
                w.hyp_pnl_dollars for w in self._windows if w.used_actual_execution
            )
            wins = sum(
                1
                for w in self._windows
                if w.used_actual_execution and w.execution_phase1_filled and w.hyp_pnl_dollars > 0
            )
            losses = sum(
                1
                for w in self._windows
                if w.used_actual_execution and w.execution_phase1_filled and w.hyp_pnl_dollars < 0
            )
        else:
            live_executable_entries = sum(
                1
                for w in self._windows
                if w.signal_fired and w.phase1_would_fill
            )
            brackets_complete = sum(1 for w in self._windows if w.phase2_triggered)
            total_hyp_pnl = sum(w.hyp_pnl_dollars for w in self._windows)
            wins = sum(
                1 for w in self._windows
                if w.signal_fired and w.phase1_would_fill and w.hyp_pnl_dollars > 0
            )
            losses = sum(
                1 for w in self._windows
                if w.signal_fired and w.phase1_would_fill and w.hyp_pnl_dollars < 0
            )

        lines: list[str] = []

        # ---- Header ----
        if self._execution_mode == "live":
            mode_line = "> **Mode:** 🔴 LIVE EXECUTION — real orders are being placed"
        elif self._execution_mode == "shadow":
            mode_line = "> **Mode:** 🟠 SHADOW-LIVE — same executor and order rules, simulated against live books"
        else:
            mode_line = "> **Mode:** Observation Only — no real money is being spent"
        lines += [
            "# 📊 Bracket Observer — Live Window Report",
            "",
            f"{mode_line}  ",
            f"> **Last Updated:** {now_et}  ",
            f"> **Configured Target Share Size:** {self._share_count:.1f} shares per signal (adaptive live sizing can trade smaller when depth is thin)  ",
            "> **Outcome Source:** Best available market metadata winner; falls back to Binance price proxy when unresolved  ",
            "",
            "---",
            "",
        ]

        # ---- Running totals ----
        pnl_color = "🟢" if total_hyp_pnl > 0 else ("🔴" if total_hyp_pnl < 0 else "⚪")
        pnl_label = "Executed PnL" if self._live_execution else "Hypothetical PnL"
        win_rate_str = f"{wins}/{windows_with_signal}" if windows_with_signal else "—"
        lines += [
            "## 💰 Running Totals",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Windows observed | **{total_windows}** |",
            f"| Signals fired | **{windows_with_signal}** ({_pct(windows_with_signal, total_windows)}) |",
            f"| Phase 1 live-executable | **{live_executable_entries}** |",
            f"| Brackets completed | **{brackets_complete}** |",
            f"| Win / Loss (Phase 1) | **{win_rate_str}** |",
            f"| {pnl_label} | {pnl_color} **${total_hyp_pnl:+.2f}** |",
            "",
            "---",
            "",
        ]

        # ---- Gate funnel stats ----
        if self._windows:
            all_gate_counts: Counter = Counter()
            all_near_misses: Counter = Counter()
            all_counterfactuals: Counter = Counter()
            for w in self._windows:
                all_gate_counts.update(w.gate_counts)
                all_near_misses.update(w.near_miss_counts)
                all_counterfactuals.update(w.counterfactual_counts)
            total_evals = sum(all_gate_counts.values()) or 1
            lines += [
                "## 🔬 Signal Gate Funnel (all windows)",
                "",
                "| Gate | Hits | % of evals |",
                "|------|------|-----------|",
            ]
            for gate in [
                "WINDOW_SETTLING", "TIME_GATE_FAIL", "ASSET_MOVE_INSUFFICIENT",
                "PRICE_RANGE_FAIL", "PRICES_STALE", "CHOP_FILTER_FAIL",
                "LAG_GAP_INSUFFICIENT", "ALREADY_FIRED_THIS_WINDOW", "SIGNAL_FIRED",
            ]:
                count = all_gate_counts.get(gate, 0)
                if count:
                    label = _GATE_LABELS.get(gate, gate)
                    lines.append(
                        f"| {label} | {count:,} | {count/total_evals*100:.1f}% |"
                    )
            if all_near_misses:
                lines += [
                    "",
                    "### Near Misses",
                    "",
                    "| Near miss | Count |",
                    "|-----------|-------|",
                ]
                near_labels = {
                    "price_high_within_2c": "Price just above range (<= +2c)",
                    "price_low_within_2c": "Price just below range (<= -2c)",
                    "move_within_25pct": "Asset move within 25% of threshold",
                    "chop_within_0_15": "Chop score within 0.15 of pass",
                    "lag_within_25pct": "Lag gap within 25% of threshold",
                }
                for key in [
                    "price_high_within_2c",
                    "price_low_within_2c",
                    "move_within_25pct",
                    "chop_within_0_15",
                    "lag_within_25pct",
                ]:
                    count = all_near_misses.get(key, 0)
                    if count:
                        lines.append(f"| {near_labels.get(key, key)} | {count:,} |")
            if all_counterfactuals:
                lines += [
                    "",
                    "### Single-Guardrail Counterfactual Fires",
                    "",
                    "_Eval cycles that would have fully passed if only this one guardrail were loosened, with the others kept unchanged._",
                    "",
                    "| Counterfactual | Count |",
                    "|----------------|-------|",
                ]
                counterfactual_labels = {
                    "range_plus_2c_would_fire": "Entry range widened by 2c",
                    "move_25pct_looser_would_fire": "Move threshold loosened by 25%",
                    "chop_minus_0_15_would_fire": "Chop minimum lowered by 0.15",
                    "lag_25pct_looser_would_fire": "Lag threshold loosened by 25%",
                }
                for key in [
                    "range_plus_2c_would_fire",
                    "move_25pct_looser_would_fire",
                    "chop_minus_0_15_would_fire",
                    "lag_25pct_looser_would_fire",
                ]:
                    count = all_counterfactuals.get(key, 0)
                    if count:
                        lines.append(f"| {counterfactual_labels.get(key, key)} | {count:,} |")
            lines += ["", "---", ""]

        # ---- Per-window log (newest first) ----
        lines += [
            "## 📋 Window Log (newest first)",
            "",
        ]

        if not self._windows:
            lines.append(
                f"*No windows completed yet. Waiting for first {int(self._window_duration_seconds / 60)}-minute cycle...*"
            )
            lines.append("")
        else:
            for w in reversed(self._windows):
                lines += _format_window(w)

        content = "\n".join(lines)
        try:
            self._report_path.write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write window report: {}", exc)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_window(w: WindowSummary) -> list[str]:
    """Format a single window as a Markdown section."""
    lines: list[str] = []
    pnl_label = (
        "Actual PnL" if (w.used_actual_execution and w.execution_mode == "live")
        else "Shadow PnL" if w.used_actual_execution
        else "Hypothetical PnL"
    )

    # --- Section heading ---
    signal_badge = "🔔 SIGNAL FIRED" if w.signal_fired else "— no signal"
    lines.append(f"### {w.date_label} · {w.window_label} · {w.asset} · {signal_badge}")
    lines.append("")

    # --- Price / move ---
    move_pct = w.move_pct * 100
    direction = "▲" if move_pct > 0 else ("▼" if move_pct < 0 else "—")
    move_str = f"{direction} {abs(move_pct):.3f}%"
    price_str = (
        f"${w.asset_open:,.2f} → ${w.asset_close:,.2f} ({move_str})"
        if w.asset_open > 0 else "*(mid-window start)*"
    )
    lines.append(f"- **{w.asset} price:** {price_str}")

    # --- Final market prices ---
    if w.yes_ask_final > 0 or w.no_ask_final > 0:
        resolution_hint = ""
        if w.resolved_yes is True:
            resolution_hint = "  → ✅ **YES won**"
        elif w.resolved_yes is False:
            resolution_hint = "  → ✅ **NO won**"
        lines.append(
            f"- **Final prices:** YES {w.yes_ask_final:.3f} / NO {w.no_ask_final:.3f}{resolution_hint}"
        )
    if w.outcome_source:
        lines.append(f"- **Outcome source:** `{w.outcome_source}`")

    # --- Gate result ---
    if w.signal_fired:
        lines.append(f"- **Signal:** {w.signal_side} @ {w.signal_price:.3f}")
        if w.signal_entry_model:
            lines.append(f"- **Signal path:** `{w.signal_entry_model}`")
        if w.used_actual_execution:
            if not w.execution_phase1_filled:
                lines.append("- **Phase 1 execution:** ❌ Signal fired but no live fill")
            else:
                lines.append(
                    f"- **Phase 1 execution:** ✅ {w.execution_mode.title()} fill @ "
                    f"{w.execution_p1_fill_price:.3f} for {w.share_count:.1f} shares"
                )
        elif w.execution_mode == "live":
            lines.append("- **Phase 1 execution:** ⚠ No confirmed live execution summary for this signal")
        elif not w.phase1_would_fill:
            lines.append(
                f"- **Phase 1 execution:** ⚠ Non-executable for live size — top ask "
                f"{w.phase1_ask_size:.1f} shares < required {w.required_shares:.1f}"
            )
        else:
            lines.append(
                f"- **Phase 1 execution:** ✅ Top ask supported {w.required_shares:.1f} shares "
                f"(saw {w.phase1_ask_size:.1f})"
            )
        if w.phase2_triggered:
            if w.safe_opposite_price > 0:
                lines.append(
                    f"- **Phase 2:** ✅ Safe level reclaimed — {_opp(w.signal_side)} "
                    f"armed @ {w.safe_opposite_price:.3f}, executed @ {w.phase2_price:.3f}"
                )
            else:
                lines.append(
                    f"- **Phase 2:** ✅ Bracket leg triggered — {_opp(w.signal_side)} @ {w.phase2_price:.3f}"
                )
        else:
            phase2_display_price = w.phase2_price or w.safe_opposite_price
            if w.used_actual_execution and w.execution_phase2_attempted and not w.execution_phase2_filled:
                if w.phase2_reclaim_seen and phase2_display_price > 0:
                    lines.append(
                        f"- **Phase 2:** ⚠ Reclaim seen @ {phase2_display_price:.3f}, but the live "
                        "Phase 2 order did not fill"
                    )
                else:
                    lines.append(
                        "- **Phase 2:** ⚠ Phase 2 order was attempted, but the live order did not fill"
                    )
            elif w.phase2_reclaim_seen and not w.phase2_would_fill:
                lines.append(
                    f"- **Phase 2:** ⚠ Reclaim seen @ {w.phase2_price:.3f}, but top ask "
                    f"{w.phase2_trigger_ask_size:.1f} shares < required {w.required_shares:.1f}"
                )
            elif w.safe_opposite_price > 0 and w.dipped_below_safe_price:
                lines.append(
                    f"- **Phase 2:** ⏳ Safe level armed @ {w.safe_opposite_price:.3f}, "
                    "extension confirmed, but reclaim never completed"
                )
            elif w.safe_opposite_price > 0:
                lines.append(
                    f"- **Phase 2:** ⏳ Safe level armed @ {w.safe_opposite_price:.3f}, "
                    "but price never extended lower first"
                )
            else:
                lines.append("- **Phase 2:** ⏳ Opposite side never reached the safe level")

        if w.used_actual_execution:
            lines.append(
                f"- **Execution:** mode=`{w.execution_mode}` phase=`{w.execution_phase}` "
                f"shares={w.share_count:.1f} "
                f"phase1_filled={w.execution_phase1_filled} "
                f"phase2_filled={w.execution_phase2_filled}"
            )
            if w.execution_phase == "HARD_EXITED":
                extra: list[str] = []
                if w.execution_hard_exit_reason:
                    extra.append(f"reason=`{w.execution_hard_exit_reason}`")
                if w.execution_hard_exit_fill_price > 0:
                    extra.append(f"fill={w.execution_hard_exit_fill_price:.3f}")
                if extra:
                    lines.append(f"- **Hard exit:** {' '.join(extra)}")
            elif w.execution_hard_exit_attempted and w.execution_hard_exit_fill_price <= 0:
                extra: list[str] = []
                if w.execution_hard_exit_reason:
                    extra.append(f"reason=`{w.execution_hard_exit_reason}`")
                lines.append(
                    "- **Hard exit:** attempted"
                    + (f" ({' '.join(extra)})" if extra else "")
                    + ", but no live sell filled before close"
                )
            elif w.execution_hard_exit_filled_shares > 0 and w.share_count - w.execution_hard_exit_filled_shares > 1e-6:
                lines.append(
                    f"- **Partial hard exit:** sold {w.execution_hard_exit_filled_shares:.2f} shares @ "
                    f"{w.execution_hard_exit_fill_price:.3f}; remaining "
                    f"{(w.share_count - w.execution_hard_exit_filled_shares):.2f} shares settled at close"
                )
            execution_refs: list[str] = []
            if w.execution_position_id:
                execution_refs.append(f"position=`{w.execution_position_id}`")
            if w.execution_p1_order_id:
                execution_refs.append(f"phase1_order=`{w.execution_p1_order_id}`")
            if w.execution_p2_order_id:
                execution_refs.append(f"phase2_order=`{w.execution_p2_order_id}`")
            if w.execution_hard_exit_order_ids:
                execution_refs.append(
                    "hard_exit_orders="
                    + ", ".join(f"`{order_id}`" for order_id in w.execution_hard_exit_order_ids)
                )
            if execution_refs:
                lines.append(f"- **Execution refs:** {' '.join(execution_refs)}")
    else:
        dominant_label = _GATE_LABELS.get(w.dominant_gate_fail, w.dominant_gate_fail)
        lines.append(f"- **Could not place bet:** {dominant_label}")

        if w.total_eval_cycles:
            details = (
                f"  - *(dominant blocker: `{w.dominant_gate_fail}` "
                f"{w.dominant_gate_count}/{w.total_eval_cycles} eval cycles"
            )
            if w.primary_gate_fail != w.dominant_gate_fail:
                details += f"; later blocker also seen: `{w.primary_gate_fail}`"
            details += ")*"
            lines.append(details)
        if w.near_miss_counts:
            parts: list[str] = []
            if w.near_miss_counts.get("price_high_within_2c"):
                parts.append(f"range high <= +2c: {w.near_miss_counts['price_high_within_2c']}")
            if w.near_miss_counts.get("price_low_within_2c"):
                parts.append(f"range low <= -2c: {w.near_miss_counts['price_low_within_2c']}")
            if w.near_miss_counts.get("move_within_25pct"):
                parts.append(f"move near-threshold: {w.near_miss_counts['move_within_25pct']}")
            if w.near_miss_counts.get("chop_within_0_15"):
                parts.append(f"chop near-pass: {w.near_miss_counts['chop_within_0_15']}")
            if w.near_miss_counts.get("lag_within_25pct"):
                parts.append(f"lag near-pass: {w.near_miss_counts['lag_within_25pct']}")
            if parts:
                lines.append(f"  - *(near misses: {'; '.join(parts)})*")
        if w.counterfactual_counts:
            parts = []
            if w.counterfactual_counts.get("range_plus_2c_would_fire"):
                parts.append(
                    f"range +2c => fire: {w.counterfactual_counts['range_plus_2c_would_fire']}"
                )
            if w.counterfactual_counts.get("move_25pct_looser_would_fire"):
                parts.append(
                    f"move -25% => fire: {w.counterfactual_counts['move_25pct_looser_would_fire']}"
                )
            if w.counterfactual_counts.get("chop_minus_0_15_would_fire"):
                parts.append(
                    f"chop -0.15 => fire: {w.counterfactual_counts['chop_minus_0_15_would_fire']}"
                )
            if w.counterfactual_counts.get("lag_25pct_looser_would_fire"):
                parts.append(
                    f"lag -25% => fire: {w.counterfactual_counts['lag_25pct_looser_would_fire']}"
                )
            if parts:
                lines.append(f"  - *(single-guardrail counterfactuals: {'; '.join(parts)})*")

    # --- PnL ---
    trade_live_executable = (
        w.execution_phase1_filled if w.used_actual_execution else w.phase1_would_fill
    )

    if w.signal_fired and not trade_live_executable:
        lines.append(f"- **{pnl_label}:** $0.00 (signal not executable for the configured live size)")
    elif w.signal_fired and w.resolved_yes is not None:
        pnl_icon = "🟢" if w.hyp_pnl_dollars > 0 else "🔴"
        hard_exit_note = (
            " ⚡ hard-exit capped (sold @ 50¢)"
            if w.hard_exit_modeled and not w.used_actual_execution
            else ""
        )
        share_suffix = f" × {w.share_count:.1f} shares"
        lines.append(
            f"- **{pnl_label}:** {pnl_icon} ${w.hyp_pnl_dollars:+.2f} "
            f"({w.hyp_pnl_per_share:+.4f}/share{share_suffix}){hard_exit_note}"
        )
    elif w.signal_fired:
        lines.append(f"- **{pnl_label}:** ⏳ Pending resolution")
    else:
        lines.append("- **Hypothetical PnL:** $0.00 (no bet placed)")

    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def _taker_fee(p: float) -> float:
    """Polymarket taker fee: p * 0.25 * (p*(1-p))^2"""
    if p <= 0 or p >= 1:
        return 0.0
    return p * 0.25 * (p * (1 - p)) ** 2


def _pct(n: int, d: int) -> str:
    if d == 0:
        return "—"
    return f"{n/d*100:.0f}%"


def _opp(side: str) -> str:
    return "NO" if side == "YES" else "YES"


def _would_fire_with_counterfactual(
    rec: dict,
    *,
    range_relax_cents: float = 0.0,
    move_relax_frac: float = 0.0,
    chop_relax: float = 0.0,
    lag_relax_frac: float = 0.0,
) -> bool:
    try:
        minutes_remaining = float(rec.get("minutes_remaining") or 0.0)
        time_gate = float(rec.get("time_gate_minutes_cfg") or 0.0)
        move = abs(float(rec.get("asset_move_pct") or 0.0))
        move_threshold = float(rec.get("move_threshold_cfg") or 0.0)
        momentum_price = float(rec.get("momentum_price_live") or 0.0)
        low = float(rec.get("entry_range_low_cfg") or 0.0)
        high = float(rec.get("entry_range_high_cfg") or 0.0)
        prices_pass = bool(rec.get("prices_pass_live"))
        chop_score = float(rec.get("chop_score_live") or 0.0)
        chop_min = float(rec.get("chop_min_score_cfg") or 0.0)
        lag_gap = float(rec.get("lag_gap_live") or -999.0)
        lag_threshold = float(rec.get("lag_threshold_cfg") or 0.0)
        signal_already_fired = bool(rec.get("signal_already_fired"))
    except (TypeError, ValueError):
        return False

    if signal_already_fired:
        return False
    if minutes_remaining < time_gate:
        return False
    if move < move_threshold * (1.0 - move_relax_frac):
        return False
    if momentum_price <= 0:
        return False
    if not (low - range_relax_cents <= momentum_price <= high + range_relax_cents):
        return False
    if not prices_pass:
        return False
    if chop_score < max(chop_min - chop_relax, 0.0):
        return False
    if lag_gap < lag_threshold * (1.0 - lag_relax_frac):
        return False
    return True


def _now_et() -> str:
    """Current time formatted as ET (EDT = UTC-4)."""
    from datetime import timedelta
    now_utc = datetime.now(timezone.utc)
    et = now_utc - timedelta(hours=4)
    return et.strftime("%Y-%m-%d %I:%M %p ET")
