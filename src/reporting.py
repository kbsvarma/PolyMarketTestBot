from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from src.config import AppConfig
from src.models import PaperReadiness, SourceQuality, TradeDecision, TrustLevel, WalletMetrics
from src.paper_quality import (
    classify_paper_readiness,
    classify_trust_level,
    summarize_source_quality_from_truth,
)
from src.state import AppStateStore
from src.utils import write_json


class ReportWriter:
    def __init__(self, config: AppConfig, data_dir: Path, state: AppStateStore) -> None:
        self.config = config
        self.data_dir = data_dir
        self.state = state

    def write_daily_summary(self, wallets: list[WalletMetrics], decisions: list[TradeDecision]) -> None:
        truth = self._current_run_truth(decisions)
        allowed = [decision for decision in decisions if decision.allowed]
        skipped = [decision for decision in decisions if not decision.allowed]
        payload = {
            "mode": self.config.mode.value,
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
            "state": self.state.read(),
            "discovery_state": truth.get("current_discovery_state", "UNKNOWN"),
            "scoring_state": truth.get("current_scoring_state", "UNKNOWN"),
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

    def write_research_snapshot(
        self,
        wallets: list[WalletMetrics],
        category_scorecards: list[object],
        replay_rows: list[dict[str, object]],
        decisions: list[TradeDecision] | None = None,
    ) -> None:
        truth = self._current_run_truth(decisions)
        payload = {
            "mode": self.config.mode.value,
            "research_wallets": [wallet.wallet_address for wallet in wallets[:5]],
            "category_rows": len(category_scorecards),
            "replay_rows": len(replay_rows),
            "discovery_state": truth.get("current_discovery_state", "UNKNOWN"),
            "scoring_state": truth.get("current_scoring_state", "UNKNOWN"),
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
        write_json(self.data_dir / "daily_summary.json", payload)

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
        traces = self._read_jsonl(self.data_dir / "paper_decision_trace.jsonl")
        state_snapshot = self.state.read()
        approved_wallets = state_snapshot.get("approved_wallets", {})

        if decisions is not None:
            signal_count = len(decisions)
            skip_count = len([decision for decision in decisions if decision.action.value == "SKIP"])
            entered_count = len([decision for decision in decisions if decision.action.value == "PAPER_COPY"])
            source_quality_values = [str(decision.context.get("source_quality") or "") for decision in decisions]
            skip_reason_distribution = Counter(decision.reason_code for decision in decisions if decision.action.value == "SKIP")
            trust_level_summary = Counter(
                str(decision.context.get("trust_level") or "UNKNOWN") for decision in decisions
            )
            eligible_count = len([decision for decision in decisions if decision.allowed])
            trustworthy_approvals = len(
                [
                    decision
                    for decision in decisions
                    if bool(decision.context.get("counts_as_trustworthy_approval"))
                ]
            )
            degraded_approvals = entered_count - trustworthy_approvals
        else:
            signal_count = len(traces)
            skip_count = len([trace for trace in traces if trace.get("final_action") == "SKIP"])
            entered_count = len([trace for trace in traces if trace.get("final_action") == "PAPER_COPY"])
            source_quality_values = [str(trace.get("source_quality") or "") for trace in traces]
            skip_reason_distribution = Counter(str(trace.get("reason_code") or "UNKNOWN") for trace in traces if trace.get("final_action") == "SKIP")
            trust_level_summary = Counter(str(trace.get("trust_level") or "UNKNOWN") for trace in traces)
            eligible_count = len([trace for trace in traces if trace.get("risk_allowed")])
            trustworthy_approvals = len([trace for trace in traces if bool(trace.get("counts_as_trustworthy_approval"))])
            degraded_approvals = entered_count - trustworthy_approvals

        scoring_source_quality = str(scoring.get("source_quality") or SourceQuality.DEGRADED_PUBLIC_DATA.value)
        scoring_diagnostics = scoring.get("diagnostics", {})
        discovery_diagnostics = discovery.get("diagnostics", {})
        approved_wallets_list = approved_wallets.get("paper_wallets", [])
        synthetic_wallet_count = int(scoring_diagnostics.get("synthetic_wallet_count", 0))
        total_scored_wallets = max(int(scoring_diagnostics.get("scored_count", 0)), len(approved_wallets_list))
        current_discovery_state = str(discovery_diagnostics.get("discovery_state") or discovery.get("state") or "UNKNOWN")
        current_scoring_state = str(scoring.get("state") or "UNKNOWN")
        fallback_in_use = bool(discovery_diagnostics.get("fallback_used")) or current_discovery_state == "SYNTHETIC_FALLBACK_USED"
        source_quality_summary = summarize_source_quality_from_truth(
            values=source_quality_values,
            discovery_state=current_discovery_state,
            scoring_state=current_scoring_state,
            fallback_in_use=fallback_in_use,
            synthetic_wallet_count=synthetic_wallet_count,
            total_scored_wallets=total_scored_wallets,
            dominant_hint=scoring_source_quality,
        )
        dominant_source_quality = str(source_quality_summary.get("dominant_source_quality") or scoring_source_quality)
        readiness = classify_paper_readiness(
            discovery_state=current_discovery_state,
            scoring_state=current_scoring_state,
            approved_wallet_count=len(approved_wallets_list),
            candidate_signal_count=signal_count,
            real_data_signal_pct=float(source_quality_summary.get("REAL_PUBLIC_DATA", 0.0)),
            fallback_signal_pct=float(source_quality_summary.get("SYNTHETIC_FALLBACK", 0.0)),
            degraded_signal_pct=float(source_quality_summary.get("DEGRADED_PUBLIC_DATA", 0.0)),
            fallback_in_use=fallback_in_use,
            approved_wallet_source_quality=scoring_source_quality,
        )
        trust_level = classify_trust_level(
            source_quality=dominant_source_quality,
            discovery_state=current_discovery_state,
            scoring_state=current_scoring_state,
            fallback_in_use=fallback_in_use,
        )
        warnings: list[str] = []
        if fallback_in_use:
            warnings.append("Synthetic fallback is active; paper output is not trustworthy for live-readiness decisions.")
        if current_scoring_state == "EMPTY":
            warnings.append("Wallet scoring is empty; trustworthy paper approvals are blocked.")
        if trustworthy_approvals > 0 and trust_level != TrustLevel.TRUSTWORTHY.value:
            warnings.append("Trust-qualified approvals were downgraded because source quality or scoring state is degraded.")
        if current_scoring_state == "EMPTY" and trustworthy_approvals > 0:
            trustworthy_approvals = 0
            warnings.append("Invariant repaired: empty scoring cannot produce trustworthy approvals.")
        non_validation_approvals = entered_count if fallback_in_use else max(0, degraded_approvals)
        funnel = {
            "detected": int(state_snapshot.get("last_cycle_detection_count", signal_count)),
            "candidates": signal_count,
            "eligible": eligible_count,
            "approved": entered_count,
            "skipped": skip_count,
            "entered": entered_count,
        }
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode.value,
            "trust_level": trust_level,
            "paper_readiness": readiness.value if isinstance(readiness, PaperReadiness) else str(readiness),
            "total_discovered_wallets": int(discovery_diagnostics.get("wallet_count", 0)),
            "total_scored_wallets": total_scored_wallets,
            "total_approved_wallets": len(approved_wallets_list),
            "total_rejected_wallets": int(scoring_diagnostics.get("rejected_count", 0)),
            "total_detected_source_trades": int(state_snapshot.get("last_cycle_detection_count", 0)),
            "total_candidate_signals": signal_count,
            "total_approved_decisions": entered_count,
            "total_approved_decisions_trustworthy": trustworthy_approvals,
            "total_approved_decisions_degraded": degraded_approvals,
            "total_approved_decisions_non_validation": non_validation_approvals,
            "total_skipped_decisions": skip_count,
            "total_skipped_signals": skip_count,
            "skip_reason_distribution": dict(skip_reason_distribution),
            "trust_level_summary": dict(trust_level_summary),
            "source_quality_summary": source_quality_summary,
            "approved_wallet_count": len(approved_wallets_list),
            "rejected_wallet_count": int(scoring_diagnostics.get("rejected_count", 0)),
            "synthetic_wallet_count": synthetic_wallet_count,
            "current_discovery_state": current_discovery_state,
            "current_scoring_state": current_scoring_state,
            "funnel": funnel,
            "discovery_reason": discovery.get("reason", ""),
            "scoring_diagnostics": scoring_diagnostics,
            "fallback_in_use": fallback_in_use,
            "dominant_source_quality": dominant_source_quality,
            "decision_count": signal_count,
            "decision_count_total": signal_count,
            "decision_count_trustworthy": trustworthy_approvals,
            "decision_count_degraded": signal_count - trustworthy_approvals,
            "approved_decisions": entered_count,
            "skipped_decisions": skip_count,
            "warnings": warnings,
            "notes": [
                "This artifact represents the current run truth-state only.",
                "Synthetic fallback is development continuity, not validation-grade evidence." if fallback_in_use else "",
            ],
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
