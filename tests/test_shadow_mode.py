from __future__ import annotations

import json
from pathlib import Path

from main import _build_shadow_config, _write_shadow_cycle_artifacts
from src.config import load_config
from src.models import DecisionAction, EntryStyle, TradeDecision
from src.state import AppStateStore


def _config():
    root = Path(__file__).resolve().parent.parent
    return load_config(root / "config.live_smoke.yaml", root / ".env.example")


def _decision(*, action: DecisionAction, allowed: bool, strategy_name: str = "wallet_follow") -> TradeDecision:
    return TradeDecision(
        strategy_name=strategy_name,
        allowed=allowed,
        action=action,
        reason_code="TEST",
        human_readable_reason="test",
        local_decision_id="d1",
        wallet_address="0xabc",
        market_id="m1",
        token_id="t1",
        entry_style=EntryStyle.FOLLOW_TAKER,
        category="politics",
        scaled_notional=3.0,
        source_price=0.5,
        executable_price=0.51,
        cluster_confirmed=False,
        hedge_suspicion_score=0.0,
        context={},
    )


def test_build_shadow_config_relaxes_live_config() -> None:
    config = _config()

    shadow = _build_shadow_config(config)

    assert shadow.mode.value == "PAPER"
    assert shadow.live.enable_multi_entry_style_live is True
    assert shadow.strategies.enable_event_driven_official is False
    assert shadow.strategies.resolution_window_live_enabled is True
    assert shadow.risk.max_entry_drift_pct >= config.risk.max_entry_drift_pct
    assert shadow.risk.max_spread_pct >= config.risk.max_spread_pct


def test_write_shadow_cycle_artifacts_records_summary_and_comparisons(tmp_path: Path) -> None:
    (tmp_path / "data" / "shadow").mkdir(parents=True, exist_ok=True)
    state = AppStateStore(tmp_path / "data" / "shadow" / "app_state.json")
    state.write(
        {
            "paper_summary": {
                "net_pnl_total": 4.2,
                "realized_pnl_total": 1.0,
                "unrealized_pnl_total": 3.2,
                "open_positions": 2,
                "open_notional": 6.0,
            }
        }
    )

    actual = [_decision(action=DecisionAction.LIVE_COPY, allowed=True)]
    shadow = [_decision(action=DecisionAction.PAPER_COPY, allowed=True, strategy_name="resolution_window")]

    _write_shadow_cycle_artifacts(
        root=tmp_path,
        cycle_ts="2026-03-15T04:00:00+00:00",
        actual_decisions=actual,
        shadow_decisions=shadow,
        shadow_state=state,
    )

    rows = (tmp_path / "data" / "shadow_live_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    payloads = [json.loads(row) for row in rows]
    assert any(payload["actual"] is not None for payload in payloads)
    assert any(payload["shadow"] is not None for payload in payloads)

    summary = json.loads((tmp_path / "data" / "shadow_live_summary.json").read_text(encoding="utf-8"))
    assert summary["shadow_net_pnl_total"] == 4.2
    assert summary["shadow_open_positions"] == 2
