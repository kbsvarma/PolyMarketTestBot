from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.models import DecisionAction, DetectionEvent, EntryStyle, TradeDecision
from src.state import AppStateStore
from src.strategy import StrategyEngine


def test_paper_decision_trace_written(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    data_dir = tmp_path
    state = AppStateStore(data_dir / "app_state.json")
    state.write({"paper_run_enabled": True})
    engine = StrategyEngine(config, data_dir, state)

    detection = DetectionEvent(
        event_key="evt",
        wallet_address="0xabc",
        market_title="Test",
        market_slug="test",
        market_id="market-1",
        token_id="token-1",
        side="BUY",
        price=0.5,
        size=10,
        notional=5,
        transaction_hash="tx",
        detection_latency_seconds=5,
        category="crypto price",
    )
    decision = TradeDecision(
        allowed=True,
        action=DecisionAction.PAPER_COPY,
        reason_code="OK",
        human_readable_reason="Risk checks passed.",
        wallet_address="0xabc",
        market_id="market-1",
        token_id="token-1",
        entry_style=EntryStyle.FOLLOW_TAKER,
        category="crypto price",
        scaled_notional=5.0,
        source_price=0.5,
        executable_price=0.5,
        cluster_confirmed=False,
        hedge_suspicion_score=0.1,
        context={"fill": {"fillable": True}, "risk_context": {"foo": "bar"}},
    )

    engine._write_decision_trace(  # noqa: SLF001
        detection,
        decision,
        False,
        [{"entry_style": "FOLLOW_TAKER", "allowed": True}],
        "SUCCESS",
        "SUCCESS",
        detection.source_quality,
    )
    rows = (data_dir / "paper_decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[-1])
    assert payload["wallet_address"] == "0xabc"
    assert payload["final_action"] == "PAPER_COPY"
    assert payload["freshness_state"] == "FRESH"
    assert payload["discovery_state"] == "SUCCESS"
    assert payload["source_quality"] == "REAL_PUBLIC_DATA"
    assert payload["trust_level"] == "STRONG"
    assert payload["fallback_used"] is False
