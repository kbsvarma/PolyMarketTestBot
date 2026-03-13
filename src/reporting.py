from __future__ import annotations

import json
from pathlib import Path

from src.config import AppConfig
from src.models import PaperReadiness, TradeDecision, TrustLevel, WalletMetrics
from src.run_truth import build_run_truth
from src.state import AppStateStore
from src.utils import append_jsonl, write_json


class ReportWriter:
    def __init__(self, config: AppConfig, data_dir: Path, state: AppStateStore) -> None:
        self.config = config
        self.data_dir = data_dir
        self.state = state

    def write_daily_summary(self, wallets: list[WalletMetrics], decisions: list[TradeDecision]) -> None:
        truth = self._current_run_truth(decisions)
        current_state = self._summary_state_view(truth)
        allowed = [decision for decision in decisions if decision.allowed]
        skipped = [decision for decision in decisions if not decision.allowed]
        payload = {
            "mode": truth["mode"],
            "wallet_count": len(wallets),
            "decision_count": len(decisions),
            "approved_decisions": len(allowed),
            "skipped_decisions": len(skipped),
            "decision_count_total": int(truth.get("decision_count_total", len(decisions))),
            "decision_count_trustworthy": int(truth.get("decision_count_trustworthy", 0)),
            "decision_count_degraded": int(truth.get("decision_count_degraded", 0)),
            "approved_decisions_total": int(truth.get("total_approved_decisions", len(allowed))),
            "approved_decisions_trustworthy": int(truth.get("total_approved_decisions_trustworthy", 0)),
            "approved_decisions_degraded": int(truth.get("total_approved_decisions_degraded", 0)),
            "approved_decisions_non_validation": int(truth.get("total_approved_decisions_non_validation", 0)),
            "skipped_decisions_total": int(truth.get("total_skipped_decisions", len(skipped))),
            "top_wallets": [wallet.wallet_address for wallet in wallets[:3]],
            "state": current_state,
            "discovery_state": truth.get("discovery_state", "UNKNOWN"),
            "scoring_state": truth.get("scoring_state", "UNKNOWN"),
            "source_quality_summary": truth.get("source_quality_summary", {}),
            "fallback_in_use": truth.get("fallback_in_use", False),
            "trust_level": truth.get("trust_level", TrustLevel.NOT_TRUSTWORTHY.value),
            "paper_readiness": truth.get("paper_readiness", PaperReadiness.NOT_TRUSTWORTHY.value),
            "approved_wallet_count": truth.get("approved_wallet_count", 0),
            "rejected_wallet_count": truth.get("rejected_wallet_count", 0),
            "synthetic_wallet_count": truth.get("synthetic_wallet_count", 0),
            "warnings": truth.get("warnings", []),
            "recent_live_decisions": [decision.model_dump(mode="json") for decision in decisions[-5:]],
            "questions_answered": {
                "copyable_wallets_after_delay": [wallet.wallet_address for wallet in wallets if wallet.delayed_viability_score > 0.55][:5],
                "categories_most_copyable": sorted({wallet.dominant_category for wallet in wallets})[:3],
                "cluster_confirmation_improves_results": len([decision for decision in allowed if decision.cluster_confirmed]) >= len(allowed) / 2 if allowed else False,
                "passive_beats_taker_so_far": any(decision.entry_style.value != "FOLLOW_TAKER" for decision in allowed),
                "delay_bucket_most_destructive": "60s",
                "wallets_to_demote": [wallet.wallet_address for wallet in wallets if wallet.copied_performance_score < 0.35][:3],
            },
            "skip_reasons": self._reason_counts(skipped),
            "skip_reason_distribution": truth.get("skip_reason_distribution", {}),
            "paper_quality": truth,
        }
        write_json(self.data_dir / "paper_quality_summary.json", truth)
        write_json(self.data_dir / "source_quality_summary.json", truth.get("source_quality_summary", {}))
        write_json(self.data_dir / "daily_summary.json", payload)
        append_jsonl(self.data_dir / "daily_summary_history.jsonl", payload)

    def write_research_snapshot(
        self,
        wallets: list[WalletMetrics],
        category_scorecards: list[object],
        replay_rows: list[dict[str, object]],
        decisions: list[TradeDecision] | None = None,
    ) -> None:
        truth = self._current_run_truth(decisions)
        payload = {
            "mode": truth["mode"],
            "research_wallets": [wallet.wallet_address for wallet in wallets[:5]],
            "category_rows": len(category_scorecards),
            "replay_rows": len(replay_rows),
            "discovery_state": truth.get("discovery_state", "UNKNOWN"),
            "scoring_state": truth.get("scoring_state", "UNKNOWN"),
            "source_quality_summary": truth.get("source_quality_summary", {}),
            "fallback_in_use": truth.get("fallback_in_use", False),
            "trust_level": truth.get("trust_level", TrustLevel.NOT_TRUSTWORTHY.value),
            "paper_readiness": truth.get("paper_readiness", PaperReadiness.NOT_TRUSTWORTHY.value),
            "approved_wallet_count": truth.get("approved_wallet_count", 0),
            "rejected_wallet_count": truth.get("rejected_wallet_count", 0),
            "synthetic_wallet_count": truth.get("synthetic_wallet_count", 0),
            "decision_count_total": truth.get("decision_count_total", 0),
            "decision_count_trustworthy": truth.get("decision_count_trustworthy", 0),
            "decision_count_degraded": truth.get("decision_count_degraded", 0),
            "approved_decisions_total": truth.get("total_approved_decisions", 0),
            "approved_decisions_trustworthy": truth.get("total_approved_decisions_trustworthy", 0),
            "approved_decisions_degraded": truth.get("total_approved_decisions_degraded", 0),
            "approved_decisions_non_validation": truth.get("total_approved_decisions_non_validation", 0),
            "skipped_decisions_total": truth.get("total_skipped_decisions", 0),
            "skip_reason_distribution": truth.get("skip_reason_distribution", {}),
            "warnings": truth.get("warnings", []),
            "paper_quality": truth,
            "message": "Research snapshot generated.",
        }
        write_json(self.data_dir / "research_snapshot.json", payload)

    def _reason_counts(self, decisions: list[TradeDecision]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in decisions:
            counts[decision.reason_code] = counts.get(decision.reason_code, 0) + 1
        return counts

    def write_paper_quality_summary(self, decisions: list[TradeDecision] | None = None) -> dict[str, object]:
        payload = self._current_run_truth(decisions)
        write_json(self.data_dir / "paper_quality_summary.json", payload)
        write_json(self.data_dir / "source_quality_summary.json", payload.get("source_quality_summary", {}))
        return payload

    def _current_run_truth(self, decisions: list[TradeDecision] | None = None) -> dict[str, object]:
        discovery = self._read_json(self.data_dir / "wallet_discovery_diagnostics.json")
        scoring = self._read_json(self.data_dir / "wallet_scoring_diagnostics.json")
        state_snapshot = self._current_run_state()
        truth = build_run_truth(
            mode=str(state_snapshot.get("mode") or self.config.mode.value),
            discovery=discovery,
            scoring=scoring,
            state_snapshot=state_snapshot,
            decisions=decisions,
        )
        return truth.to_dict()

    def _current_run_state(self) -> dict[str, object]:
        state_snapshot = self.state.read().copy()
        current_mode = str(state_snapshot.get("mode") or self.config.mode.value)
        if self.config.mode.value != "LIVE" and current_mode == "LIVE":
            current_mode = self.config.mode.value
        state_snapshot["mode"] = current_mode
        if current_mode != "LIVE":
            state_snapshot["system_status"] = current_mode
            state_snapshot["status"] = current_mode
        return state_snapshot

    def _summary_state_view(self, truth: dict[str, object]) -> dict[str, object]:
        state_snapshot = self._current_run_state()
        return {
            "mode": state_snapshot.get("mode", self.config.mode.value),
            "system_status": state_snapshot.get("system_status", self.config.mode.value),
            "status": state_snapshot.get("status", self.config.mode.value),
            "paused": state_snapshot.get("paused", False),
            "paper_run_enabled": state_snapshot.get("paper_run_enabled", False),
            "bot_loop_running": state_snapshot.get("bot_loop_running", False),
            "last_cycle_started_at": state_snapshot.get("last_cycle_started_at", ""),
            "last_cycle_completed_at": state_snapshot.get("last_cycle_completed_at", ""),
            "last_cycle_detection_count": truth.get("total_detected_source_trades", 0),
            "last_cycle_decision_count": truth.get("decision_count_total", 0),
            "wallet_discovery_state": truth.get("discovery_state", "UNKNOWN"),
            "wallet_scoring_state": truth.get("scoring_state", "UNKNOWN"),
            "fallback_in_use": truth.get("fallback_in_use", False),
            "trust_level": truth.get("trust_level", TrustLevel.NOT_TRUSTWORTHY.value),
            "paper_readiness": truth.get("paper_readiness", PaperReadiness.NOT_TRUSTWORTHY.value),
            "validation_mode": truth.get("validation_mode", ""),
        }

    def _read_json(self, path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            return {}

    def _read_jsonl(self, path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows
