from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.reporting import ReportWriter
from src.state import AppStateStore


def test_current_run_artifacts_agree_on_truth(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "RESEARCH",
            "system_status": "RESEARCH",
            "status": "RESEARCH",
            "approved_wallets": {
                "paper_wallets": [],
                "research_wallets": ["0xabc", "0xdef"],
                "live_wallets": [],
            },
            "last_cycle_detection_count": 0,
            "wallet_discovery_state": "SUCCESS",
            "wallet_scoring_state": "SUCCESS",
            "wallet_discovery_source_quality": "REAL_PUBLIC_DATA",
            "wallet_scoring_source_quality": "REAL_PUBLIC_DATA",
        }
    )
    (tmp_path / "wallet_discovery_diagnostics.json").write_text(
        json.dumps(
            {
                "state": "SUCCESS",
                "source_quality": "REAL_PUBLIC_DATA",
                "reason": "Real public wallet discovery succeeded.",
                "diagnostics": {
                    "discovery_state": "SUCCESS",
                    "fallback_used": False,
                    "wallet_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "wallet_scoring_diagnostics.json").write_text(
        json.dumps(
            {
                "state": "SUCCESS",
                "source_quality": "REAL_PUBLIC_DATA",
                "diagnostics": {
                    "wallet_count": 2,
                    "scored_count": 2,
                    "rejected_count": 0,
                    "synthetic_wallet_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    writer = ReportWriter(config, tmp_path, state)
    writer.write_daily_summary([], [])

    daily = json.loads((tmp_path / "daily_summary.json").read_text(encoding="utf-8"))
    quality = json.loads((tmp_path / "paper_quality_summary.json").read_text(encoding="utf-8"))
    app_state = json.loads((tmp_path / "app_state.json").read_text(encoding="utf-8"))

    assert daily["mode"] == quality["mode"] == "RESEARCH"
    assert daily["discovery_state"] == quality["discovery_state"] == app_state["wallet_discovery_state"] == "SUCCESS"
    assert daily["scoring_state"] == quality["scoring_state"] == app_state["wallet_scoring_state"] == "SUCCESS"
    assert daily["fallback_in_use"] is quality["fallback_in_use"] is False
    assert daily["trust_level"] == quality["trust_level"] == "TRUSTWORTHY"
    assert daily["paper_readiness"] == quality["paper_readiness"] == "NOT_TRUSTWORTHY"
    assert daily["approved_decisions_total"] == 0
    assert quality["total_approved_decisions"] == 0


def test_daily_summary_state_view_stays_current_run_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "LIVE",
            "system_status": "LIVE_READY",
            "status": "LIVE_READY",
            "paper_summary": {"approved_decisions_total": 99},
            "approved_wallets": {"paper_wallets": [], "research_wallets": [], "live_wallets": []},
        }
    )
    (tmp_path / "wallet_discovery_diagnostics.json").write_text(
        json.dumps(
            {
                "state": "SYNTHETIC_FALLBACK_USED",
                "source_quality": "SYNTHETIC_FALLBACK",
                "diagnostics": {
                    "discovery_state": "SYNTHETIC_FALLBACK_USED",
                    "fallback_used": True,
                    "wallet_count": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "wallet_scoring_diagnostics.json").write_text(
        json.dumps(
            {
                "state": "SUCCESS",
                "source_quality": "SYNTHETIC_FALLBACK",
                "diagnostics": {
                    "wallet_count": 3,
                    "scored_count": 3,
                    "rejected_count": 0,
                    "synthetic_wallet_count": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    writer = ReportWriter(config, tmp_path, state)
    writer.write_daily_summary([], [])
    daily = json.loads((tmp_path / "daily_summary.json").read_text(encoding="utf-8"))

    assert daily["mode"] == "RESEARCH"
    assert daily["state"]["mode"] == "RESEARCH"
    assert daily["state"]["system_status"] == "RESEARCH"
    assert "paper_summary" not in daily["state"]
    assert daily["approved_decisions_total"] == 0
