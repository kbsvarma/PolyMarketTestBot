from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.reporting import ReportWriter
from src.state import AppStateStore


def test_daily_summary_contains_paper_quality_fields(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "PAPER",
            "approved_wallets": {"paper_wallets": ["0xabc"], "research_wallets": ["0xabc"], "live_wallets": []},
            "last_cycle_detection_count": 4,
        }
    )
    (tmp_path / "wallet_discovery_diagnostics.json").write_text(
        json.dumps(
            {
                "state": "SUCCESS",
                "diagnostics": {"discovery_state": "SUCCESS", "fallback_used": False},
                "reason": "ok",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "wallet_scoring_diagnostics.json").write_text(
        json.dumps(
            {
                "state": "SUCCESS",
                "source_quality": "REAL_PUBLIC_DATA",
                "diagnostics": {"rejected_count": 1, "synthetic_wallet_count": 0},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "paper_decision_trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"final_action": "PAPER_COPY", "reason_code": "OK", "source_quality": "REAL_PUBLIC_DATA", "risk_allowed": True}),
                json.dumps({"final_action": "SKIP", "reason_code": "WIDE_SPREAD", "source_quality": "DEGRADED_PUBLIC_DATA", "risk_allowed": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    writer = ReportWriter(config, tmp_path, state)
    writer.write_daily_summary([], [])

    payload = json.loads((tmp_path / "daily_summary.json").read_text(encoding="utf-8"))
    assert payload["discovery_state"] == "SUCCESS"
    assert payload["scoring_state"] == "SUCCESS"
    assert "source_quality_summary" in payload
    assert "paper_readiness" in payload
    assert "fallback_in_use" in payload
    assert "approved_wallet_count" in payload
    assert "rejected_wallet_count" in payload
    assert "synthetic_wallet_count" in payload


def test_research_snapshot_preserves_existing_daily_summary_fields(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write({"mode": "RESEARCH", "approved_wallets": {"paper_wallets": [], "research_wallets": [], "live_wallets": []}})

    (tmp_path / "wallet_discovery_diagnostics.json").write_text(
        json.dumps({"state": "SUCCESS", "diagnostics": {"discovery_state": "SUCCESS", "fallback_used": False}, "reason": "ok"}),
        encoding="utf-8",
    )
    (tmp_path / "wallet_scoring_diagnostics.json").write_text(
        json.dumps({"state": "SUCCESS", "source_quality": "REAL_PUBLIC_DATA", "diagnostics": {"rejected_count": 0, "synthetic_wallet_count": 0}}),
        encoding="utf-8",
    )
    (tmp_path / "daily_summary.json").write_text(
        json.dumps({"decision_count": 7, "approved_decisions": 2, "skipped_decisions": 5, "skip_reason_distribution": {"A": 1}}),
        encoding="utf-8",
    )

    writer = ReportWriter(config, tmp_path, state)
    writer.write_research_snapshot([], [], [])

    payload = json.loads((tmp_path / "daily_summary.json").read_text(encoding="utf-8"))
    assert payload["decision_count"] == 7
    assert payload["approved_decisions"] == 2
    assert payload["skipped_decisions"] == 5
    assert payload["skip_reason_distribution"] == {"A": 1}
