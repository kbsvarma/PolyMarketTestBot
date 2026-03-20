from __future__ import annotations

import json

from src.window_report import WindowReportWriter


def test_report_header_uses_share_sizing(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=False,
        window_duration_seconds=300,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Configured Target Share Size:** 10.0 shares per signal" in content
    assert "Hypothetical Bet Size" not in content


def test_report_formats_hypothetical_pnl_with_share_count(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    eval_path.write_text("", encoding="utf-8")
    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=False,
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=84_500.0,
        yes_ask_final=0.99,
        no_ask_final=0.01,
        last_signal_event={
            "momentum_side": "YES",
            "momentum_price": 0.58,
            "observation": {
                "safe_opposite_price": 0.35,
                "dipped_below_safe_price": True,
                "phase2_would_have_triggered": True,
                "phase2_trigger_price": 0.35,
                "min_momentum_price": 0.58,
            },
        },
        eval_log_path=str(eval_path),
        yes_won=True,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "/share × 10.0 shares" in content
    assert "/share × $10" not in content


def test_report_zeroes_pnl_when_signal_would_not_fill_live_size(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    eval_path.write_text("", encoding="utf-8")
    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=False,
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=84_500.0,
        yes_ask_final=0.99,
        no_ask_final=0.01,
        last_signal_event={
            "momentum_side": "YES",
            "momentum_price": 0.58,
            "momentum_ask_size": 4.0,
            "required_shares": 10.0,
            "phase1_would_fill": False,
            "observation": {},
        },
        eval_log_path=str(eval_path),
        yes_won=True,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Phase 1 execution:** ⚠ Non-executable for live size" in content
    assert "Hypothetical PnL:** $0.00 (signal not executable for the configured live size)" in content


def test_live_report_uses_actual_share_count_from_execution_summary(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    eval_path.write_text("", encoding="utf-8")
    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=True,
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=84_500.0,
        yes_ask_final=0.99,
        no_ask_final=0.01,
        last_signal_event={
            "momentum_side": "YES",
            "momentum_price": 0.58,
            "observation": {
                "safe_opposite_price": 0.35,
                "dipped_below_safe_price": True,
                "phase2_would_have_triggered": True,
                "phase2_trigger_price": 0.35,
                "min_momentum_price": 0.58,
            },
        },
        eval_log_path=str(eval_path),
        yes_won=True,
        outcome_source="market_outcome_prices",
        execution_summary={
            "phase": "BRACKET_SETTLED",
            "execution_mode": "shadow",
            "phase1_filled": True,
            "phase2_filled": True,
            "actual_pnl_usd": 1.23,
            "p1_shares": 7.0,
            "p1_fill_price": 0.59,
            "p2_fill_price": 0.35,
            "safe_opposite_price": 0.35,
        },
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Shadow PnL" in content
    assert "/share × 7.0 shares" in content
    assert "Phase 1 execution:** ✅ Shadow fill @ 0.590 for 7.0 shares" in content
    assert "Execution:** mode=`shadow` phase=`BRACKET_SETTLED` shares=7.0" in content
    assert "Outcome source:** `market_outcome_prices`" in content


def test_shadow_report_uses_failed_execution_summary_instead_of_snapshot(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    eval_path.write_text("", encoding="utf-8")
    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=True,
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=84_500.0,
        yes_ask_final=0.99,
        no_ask_final=0.01,
        last_signal_event={
            "momentum_side": "YES",
            "momentum_price": 0.58,
            "momentum_ask_size": 50.0,
            "required_shares": 10.0,
            "phase1_would_fill": True,
            "observation": {
                "safe_opposite_price": 0.35,
                "dipped_below_safe_price": True,
                "phase2_would_have_triggered": True,
                "phase2_trigger_price": 0.35,
                "phase2_would_fill": True,
                "min_momentum_price": 0.58,
            },
        },
        eval_log_path=str(eval_path),
        yes_won=True,
        execution_summary={
            "phase": "PHASE1_NOT_FILLED",
            "execution_mode": "shadow",
            "phase1_filled": False,
            "phase2_filled": False,
            "actual_pnl_usd": 0.0,
            "p1_shares": 10.0,
        },
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Phase 1 execution:** ❌ Signal fired but no live fill" in content
    assert "Phase 2:** ⏳ Safe level armed" in content or "Phase 2:** ✅" not in content
    assert "Shadow PnL:** $0.00" in content
    assert "Execution:** mode=`shadow` phase=`PHASE1_NOT_FILLED` shares=10.0 phase1_filled=False phase2_filled=False" in content


def test_live_report_does_not_count_failed_entry_as_actual_pnl(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    eval_path.write_text("", encoding="utf-8")
    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=True,
        execution_mode="live",
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=84_500.0,
        yes_ask_final=0.99,
        no_ask_final=0.01,
        last_signal_event={
            "momentum_side": "YES",
            "momentum_price": 0.58,
            "momentum_ask_size": 50.0,
            "required_shares": 10.0,
            "phase1_would_fill": True,
            "observation": {
                "safe_opposite_price": 0.35,
                "dipped_below_safe_price": True,
                "phase2_would_have_triggered": True,
                "phase2_trigger_price": 0.35,
                "phase2_would_fill": True,
                "min_momentum_price": 0.58,
            },
        },
        eval_log_path=str(eval_path),
        yes_won=True,
        execution_summary={
            "phase": "PHASE1_ORDER_FAILED",
            "execution_mode": "live",
            "phase1_filled": False,
            "phase2_filled": False,
            "actual_pnl_usd": 0.0,
            "p1_shares": 10.0,
        },
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Executed PnL | ⚪ **$+0.00**" in content
    assert "| Phase 1 live-executable | **0** |" in content
    assert "Actual PnL:** $0.00" in content
    assert "Phase 1 execution:** ❌ Signal fired but no live fill" in content
    assert "Execution:** mode=`live` phase=`PHASE1_ORDER_FAILED` shares=10.0 phase1_filled=False phase2_filled=False" in content


def test_report_recovers_signal_from_signal_log_when_memory_is_missing(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    signal_path = tmp_path / "signals.jsonl"
    eval_rows = [
        {
            "asset": "BTC",
            "window_open_ts": 1_700_000_000,
            "result": "SIGNAL_FIRED",
        }
    ]
    signal_rows = [
        {
            "type": "signal",
            "event_id": "evt-1",
            "asset": "BTC",
            "window_open_ts": 1_700_000_000,
            "window_close_ts": 1_700_000_300,
            "momentum_side": "YES",
            "momentum_price": 0.58,
            "momentum_ask_size": 25.0,
            "required_shares": 10.0,
            "phase1_would_fill": True,
            "observation": {
                "safe_opposite_price": 0.34,
                "dipped_below_safe_price": True,
                "phase2_would_have_triggered": True,
                "phase2_trigger_price": 0.35,
                "phase2_would_fill": True,
                "min_momentum_price": 0.58,
            },
        }
    ]
    eval_path.write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n", encoding="utf-8")
    signal_path.write_text("\n".join(json.dumps(r) for r in signal_rows) + "\n", encoding="utf-8")

    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=False,
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=84_500.0,
        yes_ask_final=0.99,
        no_ask_final=0.01,
        last_signal_event=None,
        eval_log_path=str(eval_path),
        signal_log_path=str(signal_path),
        yes_won=True,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "SIGNAL FIRED" in content
    assert "Signal:** YES @ 0.580" in content


def test_report_shows_signal_entry_model(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    eval_path.write_text("", encoding="utf-8")
    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=False,
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=84_500.0,
        yes_ask_final=0.99,
        no_ask_final=0.01,
        last_signal_event={
            "momentum_side": "YES",
            "entry_model": "continuation",
            "momentum_price": 0.58,
            "observation": {
                "min_momentum_price": 0.58,
            },
        },
        eval_log_path=str(eval_path),
        yes_won=True,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Signal path:** `continuation`" in content


def test_report_shows_hard_exit_reason_and_fill_price(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    eval_path.write_text("", encoding="utf-8")
    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=True,
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=84_100.0,
        yes_ask_final=0.50,
        no_ask_final=0.50,
        last_signal_event={
            "momentum_side": "YES",
            "momentum_price": 0.61,
            "observation": {"min_momentum_price": 0.59},
        },
        eval_log_path=str(eval_path),
        yes_won=False,
        execution_summary={
            "phase": "HARD_EXITED",
            "execution_mode": "shadow",
            "phase1_filled": True,
            "phase2_filled": False,
            "actual_pnl_usd": -0.21,
            "p1_shares": 10.0,
            "p1_fill_price": 0.61,
            "hard_exit_reason": "FINAL_30S_LOSS",
            "hard_exit_fill_price": 0.599,
        },
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Hard exit:** reason=`FINAL_30S_LOSS` fill=0.599" in content


def test_live_report_shows_phase2_reclaim_attempt_failure_and_execution_refs(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    eval_path.write_text("", encoding="utf-8")
    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=True,
        execution_mode="live",
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=83_900.0,
        yes_ask_final=0.45,
        no_ask_final=0.55,
        last_signal_event={
            "momentum_side": "NO",
            "entry_model": "lag",
            "momentum_price": 0.58,
            "observation": {
                "safe_opposite_price": 0.31,
                "dipped_below_safe_price": True,
                "phase2_reclaim_seen": True,
                "phase2_trigger_price": 0.31,
                "min_momentum_price": 0.52,
            },
        },
        eval_log_path=str(eval_path),
        yes_won=False,
        execution_summary={
            "phase": "HARD_EXITED",
            "execution_mode": "live",
            "position_id": "pos-1",
            "p1_order_id": "order-p1",
            "phase1_filled": True,
            "phase2_filled": False,
            "phase2_reclaim_seen": True,
            "phase2_order_attempted": True,
            "safe_opposite_price": 0.31,
            "actual_pnl_usd": -1.09,
            "p1_shares": 10.0,
            "p1_fill_price": 0.60,
            "hard_exit_reason": "STOP_50C",
            "hard_exit_fill_price": 0.50,
            "hard_exit_order_ids": ["order-exit-1"],
        },
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Phase 2:** ⚠ Reclaim seen @ 0.310, but the live Phase 2 order did not fill" in content
    assert "Execution refs:** position=`pos-1` phase1_order=`order-p1` hard_exit_orders=`order-exit-1`" in content


def test_report_prefers_executed_signal_metadata_over_later_snapshot(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    eval_path.write_text("", encoding="utf-8")
    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=True,
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=83_900.0,
        yes_ask_final=0.45,
        no_ask_final=0.55,
        last_signal_event={
            "momentum_side": "YES",
            "entry_model": "continuation",
            "momentum_price": 0.60,
            "observation": {
                "safe_opposite_price": 0.34,
                "dipped_below_safe_price": False,
                "phase2_would_have_triggered": False,
                "min_momentum_price": 0.47,
            },
        },
        eval_log_path=str(eval_path),
        yes_won=False,
        execution_summary={
            "phase": "HARD_EXITED",
            "execution_mode": "shadow",
            "signal_side": "NO",
            "signal_entry_model": "continuation",
            "signal_price": 0.58,
            "phase1_filled": True,
            "phase2_filled": False,
            "actual_pnl_usd": -1.28,
            "p1_shares": 10.0,
            "p1_fill_price": 0.58,
            "safe_opposite_price": 0.34,
            "min_opposite_price": 0.33,
            "hard_exit_reason": "STOP_50C",
            "hard_exit_fill_price": 0.46,
        },
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Signal:** NO @ 0.580" in content
    assert "Phase 1 execution:** ✅ Shadow fill @ 0.580 for 10.0 shares" in content
    assert "Phase 2:** ⏳ Safe level armed @ 0.340, extension confirmed, but reclaim never completed" in content
    assert "Hard exit:** reason=`STOP_50C` fill=0.460" in content
    assert "hard-exit capped (sold @ 50¢)" not in content


def test_report_calls_out_unfilled_hard_exit_attempt_before_close(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    eval_path.write_text("", encoding="utf-8")
    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=True,
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=83_900.0,
        yes_ask_final=0.45,
        no_ask_final=0.55,
        last_signal_event={
            "momentum_side": "YES",
            "entry_model": "continuation",
            "momentum_price": 0.60,
            "observation": {
                "safe_opposite_price": 0.39,
                "dipped_below_safe_price": True,
                "phase2_reclaim_seen": False,
            },
        },
        eval_log_path=str(eval_path),
        yes_won=False,
        execution_summary={
            "phase": "PHASE1_ONLY_CLOSED",
            "execution_mode": "live",
            "phase1_filled": True,
            "phase2_filled": False,
            "phase2_order_attempted": False,
            "hard_exit_attempted": True,
            "hard_exit_reason": "STOP_50C",
            "hard_exit_fill_price": 0.0,
            "actual_pnl_usd": -6.05,
            "p1_shares": 10.0,
            "p1_fill_price": 0.60,
        },
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Hard exit:** attempted (reason=`STOP_50C`), but no live sell filled before close" in content


def test_report_uses_dominant_blocker_and_near_misses_for_no_signal(tmp_path) -> None:
    report_path = tmp_path / "window_report.md"
    eval_path = tmp_path / "evaluations.jsonl"
    rows = [
        {
            "asset": "BTC",
            "window_open_ts": 1_700_000_000,
            "result": "PRICE_RANGE_FAIL",
            "momentum_price": 0.64,
            "entry_range": [0.57, 0.63],
            "range_side": "HIGH",
            "range_distance": 0.01,
            "minutes_remaining": 1.8,
            "time_gate_minutes_cfg": 1.0,
            "asset_move_pct": 0.0005,
            "move_threshold_cfg": 0.0002,
            "momentum_price_live": 0.64,
            "entry_range_low_cfg": 0.57,
            "entry_range_high_cfg": 0.63,
            "prices_pass_live": True,
            "chop_score_live": 0.8,
            "chop_min_score_cfg": 0.51,
            "lag_gap_live": 0.03,
            "lag_threshold_cfg": 0.015,
            "signal_already_fired": False,
        },
        {
            "asset": "BTC",
            "window_open_ts": 1_700_000_000,
            "result": "PRICE_RANGE_FAIL",
            "momentum_price": 0.65,
            "entry_range": [0.57, 0.63],
            "range_side": "HIGH",
            "range_distance": 0.02,
            "minutes_remaining": 1.8,
            "time_gate_minutes_cfg": 1.0,
            "asset_move_pct": 0.0005,
            "move_threshold_cfg": 0.0002,
            "momentum_price_live": 0.65,
            "entry_range_low_cfg": 0.57,
            "entry_range_high_cfg": 0.63,
            "prices_pass_live": True,
            "chop_score_live": 0.8,
            "chop_min_score_cfg": 0.51,
            "lag_gap_live": 0.03,
            "lag_threshold_cfg": 0.015,
            "signal_already_fired": False,
        },
        {
            "asset": "BTC",
            "window_open_ts": 1_700_000_000,
            "result": "LAG_GAP_INSUFFICIENT",
            "lag_gap": 0.012,
            "min_required": 0.015,
            "lag_shortfall": 0.003,
            "minutes_remaining": 1.8,
            "time_gate_minutes_cfg": 1.0,
            "asset_move_pct": 0.0005,
            "move_threshold_cfg": 0.0002,
            "momentum_price_live": 0.6,
            "entry_range_low_cfg": 0.57,
            "entry_range_high_cfg": 0.63,
            "prices_pass_live": True,
            "chop_score_live": 0.8,
            "chop_min_score_cfg": 0.51,
            "lag_gap_live": 0.012,
            "lag_threshold_cfg": 0.015,
            "signal_already_fired": False,
        },
    ]
    eval_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    writer = WindowReportWriter(
        report_path=str(report_path),
        report_shares=10.0,
        live_execution=False,
        window_duration_seconds=300,
    )

    writer.record_window_close(
        window_ts=1_700_000_000,
        asset="BTC",
        asset_open=84_000.0,
        asset_close=84_100.0,
        yes_ask_final=0.55,
        no_ask_final=0.45,
        last_signal_event=None,
        eval_log_path=str(eval_path),
        yes_won=True,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Could not place bet:** Price outside entry range" in content
    assert "dominant blocker: `PRICE_RANGE_FAIL` 2/3 eval cycles; later blocker also seen: `LAG_GAP_INSUFFICIENT`" in content
    assert "near misses: range high <= +2c: 2; lag near-pass: 1" in content
    assert "single-guardrail counterfactuals: range +2c => fire: 2; lag -25% => fire: 1" in content
    assert "Entry range widened by 2c | 2" in content
    assert "Lag threshold loosened by 25% | 1" in content
