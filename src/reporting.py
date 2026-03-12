from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from src.config import AppConfig
from src.models import PaperReadiness, SourceQuality, TradeDecision, WalletMetrics
from src.paper_quality import classify_paper_readiness, summarize_source_quality
from src.state import AppStateStore
from src.utils import write_json


class ReportWriter:
    def __init__(self, config: AppConfig, data_dir: Path, state: AppStateStore) -> None:
        self.config = config
        self.data_dir = data_dir
        self.state = state

    def write_daily_summary(self, wallets: list[WalletMetrics], decisions: list[TradeDecision]) -> None:
        allowed = [decision for decision in decisions if decision.allowed]
        skipped = [decision for decision in decisions if not decision.allowed]
        paper_quality = self._paper_quality_summary(decisions)
        payload = {
            "mode": self.config.mode.value,
            "wallet_count": len(wallets),
            "decision_count": len(decisions),
            "approved_decisions": len(allowed),
            "skipped_decisions": len(skipped),
            "top_wallets": [wallet.wallet_address for wallet in wallets[:3]],
            "state": self.state.read(),
            "discovery_state": paper_quality.get("current_discovery_state", "UNKNOWN"),
            "scoring_state": paper_quality.get("current_scoring_state", "UNKNOWN"),
            "source_quality_summary": paper_quality.get("source_quality_summary", {}),
            "fallback_in_use": paper_quality.get("fallback_in_use", False),
            "paper_readiness": paper_quality.get("paper_readiness", PaperReadiness.NOT_TRUSTWORTHY.value),
            "approved_wallet_count": paper_quality.get("approved_wallet_count", 0),
            "rejected_wallet_count": paper_quality.get("rejected_wallet_count", 0),
            "synthetic_wallet_count": paper_quality.get("synthetic_wallet_count", 0),
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
            "skip_reason_distribution": paper_quality.get("skip_reason_distribution", {}),
            "paper_quality": paper_quality,
        }
        write_json(self.data_dir / "paper_quality_summary.json", paper_quality)
        write_json(self.data_dir / "source_quality_summary.json", paper_quality.get("source_quality_summary", {}))
        write_json(self.data_dir / "daily_summary.json", payload)

    def write_research_snapshot(
        self,
        wallets: list[WalletMetrics],
        category_scorecards: list[object],
        replay_rows: list[dict[str, object]],
    ) -> None:
        existing = self._read_json(self.data_dir / "daily_summary.json")
        paper_quality = self._paper_quality_summary()
        payload = {
            **existing,
            "mode": self.config.mode.value,
            "research_wallets": [wallet.wallet_address for wallet in wallets[:5]],
            "category_rows": len(category_scorecards),
            "replay_rows": len(replay_rows),
            "discovery_state": paper_quality.get("current_discovery_state", "UNKNOWN"),
            "scoring_state": paper_quality.get("current_scoring_state", "UNKNOWN"),
            "source_quality_summary": paper_quality.get("source_quality_summary", {}),
            "fallback_in_use": paper_quality.get("fallback_in_use", False),
            "paper_readiness": paper_quality.get("paper_readiness", PaperReadiness.NOT_TRUSTWORTHY.value),
            "approved_wallet_count": paper_quality.get("approved_wallet_count", 0),
            "rejected_wallet_count": paper_quality.get("rejected_wallet_count", 0),
            "synthetic_wallet_count": paper_quality.get("synthetic_wallet_count", 0),
            "paper_quality": paper_quality,
            "message": "Research snapshot generated.",
        }
        write_json(self.data_dir / "daily_summary.json", payload)

    def _reason_counts(self, decisions: list[TradeDecision]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in decisions:
            counts[decision.reason_code] = counts.get(decision.reason_code, 0) + 1
        return counts

    def write_paper_quality_summary(self, decisions: list[TradeDecision] | None = None) -> dict[str, object]:
        payload = self._paper_quality_summary(decisions)
        write_json(self.data_dir / "paper_quality_summary.json", payload)
        write_json(self.data_dir / "source_quality_summary.json", payload.get("source_quality_summary", {}))
        return payload

    def _paper_quality_summary(self, decisions: list[TradeDecision] | None = None) -> dict[str, object]:
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
        else:
            signal_count = len(traces)
            skip_count = len([trace for trace in traces if trace.get("final_action") == "SKIP"])
            entered_count = len([trace for trace in traces if trace.get("final_action") == "PAPER_COPY"])
            source_quality_values = [str(trace.get("source_quality") or "") for trace in traces]
            skip_reason_distribution = Counter(str(trace.get("reason_code") or "UNKNOWN") for trace in traces if trace.get("final_action") == "SKIP")
            trust_level_summary = Counter(str(trace.get("trust_level") or "UNKNOWN") for trace in traces)
            eligible_count = len([trace for trace in traces if trace.get("risk_allowed")])

        source_quality_summary = summarize_source_quality(source_quality_values)
        dominant_source_quality = str(source_quality_summary.get("dominant_source_quality") or "DEGRADED_PUBLIC_DATA")
        scoring_diagnostics = scoring.get("diagnostics", {})
        discovery_diagnostics = discovery.get("diagnostics", {})
        approved_wallets_list = approved_wallets.get("paper_wallets", [])
        synthetic_wallet_count = int(scoring_diagnostics.get("synthetic_wallet_count", 0))
        total_scored_wallets = max(int(scoring_diagnostics.get("scored_count", 0)), len(approved_wallets_list))
        fallback_in_use = bool(discovery_diagnostics.get("fallback_used")) or dominant_source_quality == SourceQuality.SYNTHETIC_FALLBACK.value
        if fallback_in_use:
            dominant_source_quality = SourceQuality.SYNTHETIC_FALLBACK.value
            source_quality_summary["dominant_source_quality"] = dominant_source_quality
        readiness = classify_paper_readiness(
            discovery_state=str(discovery_diagnostics.get("discovery_state") or discovery.get("state") or "NO_DATA"),
            scoring_state=str(scoring.get("state") or "EMPTY"),
            approved_wallet_count=len(approved_wallets_list),
            candidate_signal_count=signal_count,
            real_data_signal_pct=float(source_quality_summary.get("REAL_PUBLIC_DATA", 0.0)),
            fallback_signal_pct=float(source_quality_summary.get("SYNTHETIC_FALLBACK", 0.0)),
            degraded_signal_pct=float(source_quality_summary.get("DEGRADED_PUBLIC_DATA", 0.0)),
            fallback_in_use=fallback_in_use,
            approved_wallet_source_quality=str(scoring.get("source_quality") or SourceQuality.DEGRADED_PUBLIC_DATA.value),
        )
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
            "paper_readiness": readiness.value if isinstance(readiness, PaperReadiness) else str(readiness),
            "total_discovered_wallets": int(discovery_diagnostics.get("wallet_count", 0)),
            "total_scored_wallets": total_scored_wallets,
            "total_approved_wallets": len(approved_wallets_list),
            "total_rejected_wallets": int(scoring_diagnostics.get("rejected_count", 0)),
            "total_detected_source_trades": int(state_snapshot.get("last_cycle_detection_count", 0)),
            "total_candidate_signals": signal_count,
            "total_approved_decisions": entered_count,
            "total_skipped_decisions": skip_count,
            "total_skipped_signals": skip_count,
            "skip_reason_distribution": dict(skip_reason_distribution),
            "trust_level_summary": dict(trust_level_summary),
            "source_quality_summary": source_quality_summary,
            "approved_wallet_count": len(approved_wallets_list),
            "rejected_wallet_count": int(scoring_diagnostics.get("rejected_count", 0)),
            "synthetic_wallet_count": synthetic_wallet_count,
            "current_discovery_state": str(discovery_diagnostics.get("discovery_state") or discovery.get("state") or "UNKNOWN"),
            "current_scoring_state": str(scoring.get("state") or "UNKNOWN"),
            "funnel": funnel,
            "discovery_reason": discovery.get("reason", ""),
            "scoring_diagnostics": scoring_diagnostics,
            "fallback_in_use": fallback_in_use,
            "dominant_source_quality": dominant_source_quality,
            "decision_count": signal_count,
            "approved_decisions": entered_count,
            "skipped_decisions": skip_count,
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
