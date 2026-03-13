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
    assert "trust_level" in payload
    assert "decision_count_total" in payload
    assert "approved_decisions_trustworthy" in payload
    assert "approved_decisions_degraded" in payload
    assert "warnings" in payload
    assert "approved_wallet_count" in payload
    assert "rejected_wallet_count" in payload
    assert "synthetic_wallet_count" in payload
    assert "skip_reason_distribution" in payload

    paper_quality = json.loads((tmp_path / "paper_quality_summary.json").read_text(encoding="utf-8"))
    assert paper_quality["mode"] == "PAPER"
    assert "generated_at" in paper_quality
    assert paper_quality["total_detected_source_trades"] == 4
    assert paper_quality["total_candidate_signals"] == 0
    assert paper_quality["total_scored_wallets"] == 1
    assert paper_quality["total_approved_wallets"] == 1
    assert paper_quality["total_rejected_wallets"] == 1
    assert "trust_level" in paper_quality
    assert "total_approved_decisions_trustworthy" in paper_quality
    assert "total_approved_decisions_degraded" in paper_quality
    assert "warnings" in paper_quality
    assert "trust_level_summary" in paper_quality


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
    writer = ReportWriter(config, tmp_path, state)
    writer.write_daily_summary([], [])
    writer.write_research_snapshot([], [], [])

    payload = json.loads((tmp_path / "daily_summary.json").read_text(encoding="utf-8"))
    assert payload["decision_count_total"] == 0
    assert payload["approved_decisions_total"] == 0
    assert payload["skipped_decisions_total"] == 0
    assert payload["skip_reason_distribution"] == {}
    research_snapshot = json.loads((tmp_path / "research_snapshot.json").read_text(encoding="utf-8"))
    assert research_snapshot["mode"] == "RESEARCH"


def test_paper_quality_summary_downgrades_empty_scoring(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "PAPER",
            "approved_wallets": {"paper_wallets": [], "research_wallets": [], "live_wallets": []},
            "last_cycle_detection_count": 5,
        }
    )
    (tmp_path / "wallet_discovery_diagnostics.json").write_text(
        json.dumps({"state": "SYNTHETIC_FALLBACK_USED", "diagnostics": {"discovery_state": "SYNTHETIC_FALLBACK_USED", "fallback_used": True, "wallet_count": 3}}),
        encoding="utf-8",
    )
    (tmp_path / "wallet_scoring_diagnostics.json").write_text(
        json.dumps({"state": "EMPTY", "source_quality": "SYNTHETIC_FALLBACK", "diagnostics": {"scored_count": 0, "rejected_count": 0, "synthetic_wallet_count": 3}}),
        encoding="utf-8",
    )
    writer = ReportWriter(config, tmp_path, state)
    payload = writer.write_paper_quality_summary([])
    assert payload["paper_readiness"] == "NOT_TRUSTWORTHY"
    assert payload["trust_level"] == "NOT_TRUSTWORTHY"
    assert payload["total_approved_decisions_trustworthy"] == 0
    assert payload["fallback_in_use"] is True
    assert payload["source_quality_summary"]["SYNTHETIC_FALLBACK"] == 1.0
    assert payload["validation_mode"] == "DEV_ONLY"


def test_paper_quality_summary_reflects_current_decisions_consistently(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "PAPER",
            "approved_wallets": {"paper_wallets": ["0xabc"], "research_wallets": ["0xabc"], "live_wallets": []},
            "last_cycle_detection_count": 2,
        }
    )
    (tmp_path / "wallet_discovery_diagnostics.json").write_text(
        json.dumps({"state": "SUCCESS", "diagnostics": {"discovery_state": "SUCCESS", "fallback_used": False, "wallet_count": 1}, "reason": "ok"}),
        encoding="utf-8",
    )
    (tmp_path / "wallet_scoring_diagnostics.json").write_text(
        json.dumps({"state": "SUCCESS", "source_quality": "REAL_PUBLIC_DATA", "diagnostics": {"scored_count": 1, "rejected_count": 0, "synthetic_wallet_count": 0}}),
        encoding="utf-8",
    )

    writer = ReportWriter(config, tmp_path, state)
    payload = writer.write_paper_quality_summary([])
    assert payload["mode"] == "PAPER"
    assert payload["fallback_in_use"] is False
    assert payload["total_discovered_wallets"] == 1
    assert payload["total_scored_wallets"] == 1
    assert payload["total_approved_wallets"] == 1
    assert payload["total_rejected_wallets"] == 0
    assert payload["total_detected_source_trades"] == 2
    assert payload["source_quality_summary"]["REAL_PUBLIC_DATA"] == 1.0


def test_daily_summary_is_current_run_only_not_stale_merge(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "RESEARCH",
            "approved_wallets": {"paper_wallets": [], "research_wallets": [], "live_wallets": []},
            "last_cycle_detection_count": 0,
        }
    )
    (tmp_path / "wallet_discovery_diagnostics.json").write_text(
        json.dumps({"state": "SYNTHETIC_FALLBACK_USED", "diagnostics": {"discovery_state": "SYNTHETIC_FALLBACK_USED", "fallback_used": True, "wallet_count": 3}}),
        encoding="utf-8",
    )
    (tmp_path / "wallet_scoring_diagnostics.json").write_text(
        json.dumps({"state": "SUCCESS", "source_quality": "SYNTHETIC_FALLBACK", "diagnostics": {"scored_count": 3, "rejected_count": 0, "synthetic_wallet_count": 3}}),
        encoding="utf-8",
    )
    (tmp_path / "daily_summary.json").write_text(
        json.dumps({"approved_decisions_total": 99, "skipped_decisions_total": 1, "mode": "LIVE"}),
        encoding="utf-8",
    )
    writer = ReportWriter(config, tmp_path, state)
    writer.write_daily_summary([], [])
    payload = json.loads((tmp_path / "daily_summary.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "RESEARCH"
    assert payload["approved_decisions_total"] == 0
    assert payload["skipped_decisions_total"] == 0
    assert payload["fallback_in_use"] is True
    assert payload["trust_level"] == "NOT_TRUSTWORTHY"


def test_daily_summary_uses_configured_run_mode_not_stale_state_mode(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "LIVE",
            "system_status": "LIVE_READY",
            "approved_wallets": {"paper_wallets": [], "research_wallets": [], "live_wallets": []},
            "last_cycle_detection_count": 0,
        }
    )
    (tmp_path / "wallet_discovery_diagnostics.json").write_text(
        json.dumps({"state": "SYNTHETIC_FALLBACK_USED", "diagnostics": {"discovery_state": "SYNTHETIC_FALLBACK_USED", "fallback_used": True, "wallet_count": 3}}),
        encoding="utf-8",
    )
    (tmp_path / "wallet_scoring_diagnostics.json").write_text(
        json.dumps({"state": "SUCCESS", "source_quality": "SYNTHETIC_FALLBACK", "diagnostics": {"scored_count": 3, "rejected_count": 0, "synthetic_wallet_count": 3}}),
        encoding="utf-8",
    )
    writer = ReportWriter(config, tmp_path, state)
    writer.write_daily_summary([], [])
    payload = json.loads((tmp_path / "daily_summary.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "RESEARCH"
    assert payload["state"]["mode"] == "RESEARCH"
    assert payload["state"]["system_status"] == "RESEARCH"
    assert "paper_summary" not in payload["state"]
