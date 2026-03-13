from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.models import (
    PaperReadiness,
    SourceQuality,
    TrustLevel,
    TradeDecision,
    ValidationMode,
)
from src.paper_quality import (
    classify_paper_readiness,
    classify_trust_level,
    summarize_source_quality_from_truth,
)


@dataclass
class RunTruthState:
    generated_at: str
    mode: str
    discovery_state: str
    scoring_state: str
    source_quality: str
    dominant_source_quality: str
    fallback_in_use: bool
    trust_level: str
    paper_readiness: str
    validation_mode: str
    total_discovered_wallets: int
    total_scored_wallets: int
    total_approved_wallets: int
    total_rejected_wallets: int
    synthetic_wallet_count: int
    total_detected_source_trades: int
    total_candidate_signals: int
    total_approved_decisions: int
    total_approved_decisions_trustworthy: int
    total_approved_decisions_degraded: int
    total_approved_decisions_non_validation: int
    total_skipped_decisions: int
    skip_reason_distribution: dict[str, int]
    warnings: list[str]
    notes: list[str]
    source_quality_summary: dict[str, float | str]
    trust_level_summary: dict[str, int]
    funnel: dict[str, int]
    approved_wallet_count: int
    rejected_wallet_count: int
    discovery_reason: str
    scoring_diagnostics: dict[str, Any]
    decision_count_total: int
    decision_count_trustworthy: int
    decision_count_degraded: int
    current_discovery_state: str
    current_scoring_state: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def build_run_truth(
    *,
    mode: str,
    discovery: dict[str, Any],
    scoring: dict[str, Any],
    state_snapshot: dict[str, Any],
    decisions: list[TradeDecision] | None = None,
) -> RunTruthState:
    approved_wallets = state_snapshot.get("approved_wallets", {}) if isinstance(state_snapshot, dict) else {}
    approved_wallets_list = approved_wallets.get("paper_wallets", []) if isinstance(approved_wallets, dict) else []
    scoring_diagnostics = scoring.get("diagnostics", {}) if isinstance(scoring, dict) else {}
    discovery_diagnostics = discovery.get("diagnostics", {}) if isinstance(discovery, dict) else {}

    current_discovery_state = str(discovery_diagnostics.get("discovery_state") or discovery.get("state") or "UNKNOWN")
    current_scoring_state = str(scoring.get("state") or "UNKNOWN")
    scoring_source_quality = str(scoring.get("source_quality") or SourceQuality.DEGRADED_PUBLIC_DATA.value)
    fallback_in_use = bool(discovery_diagnostics.get("fallback_used")) or current_discovery_state == "SYNTHETIC_FALLBACK_USED"
    synthetic_wallet_count = int(scoring_diagnostics.get("synthetic_wallet_count", 0))
    total_scored_wallets = max(int(scoring_diagnostics.get("scored_count", 0)), len(approved_wallets_list))

    if decisions is not None:
        signal_count = len(decisions)
        skip_count = len([decision for decision in decisions if decision.action.value == "SKIP"])
        entered_count = len([decision for decision in decisions if decision.action.value == "PAPER_COPY"])
        source_quality_values = [str(decision.context.get("source_quality") or "") for decision in decisions]
        skip_reason_distribution = Counter(decision.reason_code for decision in decisions if decision.action.value == "SKIP")
        trust_level_summary = Counter(str(decision.context.get("trust_level") or "UNKNOWN") for decision in decisions)
        eligible_count = len([decision for decision in decisions if decision.allowed])
        trustworthy_approvals = len(
            [decision for decision in decisions if bool(decision.context.get("counts_as_trustworthy_approval"))]
        )
    else:
        signal_count = 0
        skip_count = 0
        entered_count = 0
        source_quality_values = []
        skip_reason_distribution = Counter()
        trust_level_summary = Counter()
        eligible_count = 0
        trustworthy_approvals = 0

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
    trust_level = classify_trust_level(
        source_quality=dominant_source_quality,
        discovery_state=current_discovery_state,
        scoring_state=current_scoring_state,
        fallback_in_use=fallback_in_use,
    )
    paper_readiness = classify_paper_readiness(
        discovery_state=current_discovery_state,
        scoring_state=current_scoring_state,
        approved_wallet_count=len(approved_wallets_list),
        candidate_signal_count=signal_count,
        real_data_signal_pct=float(source_quality_summary.get(SourceQuality.REAL_PUBLIC_DATA.value, 0.0)),
        fallback_signal_pct=float(source_quality_summary.get(SourceQuality.SYNTHETIC_FALLBACK.value, 0.0)),
        degraded_signal_pct=float(source_quality_summary.get(SourceQuality.DEGRADED_PUBLIC_DATA.value, 0.0)),
        fallback_in_use=fallback_in_use,
        approved_wallet_source_quality=scoring_source_quality,
    )
    warnings: list[str] = []
    if fallback_in_use:
        warnings.append("Synthetic fallback is active; this run is development continuity, not validation-grade evidence.")
    if current_scoring_state == "EMPTY":
        warnings.append("Wallet scoring is empty; trustworthy paper approvals are blocked.")

    degraded_approvals = max(0, entered_count - trustworthy_approvals)
    non_validation_approvals = entered_count if fallback_in_use else 0
    if fallback_in_use and trustworthy_approvals > 0:
        trustworthy_approvals = 0
        degraded_approvals = 0
        non_validation_approvals = entered_count
        warnings.append("Invariant repaired: synthetic fallback cannot produce trustworthy approvals.")
    if current_scoring_state == "EMPTY" and trustworthy_approvals > 0:
        trustworthy_approvals = 0
        warnings.append("Invariant repaired: empty scoring cannot produce trustworthy approvals.")

    if fallback_in_use:
        validation_mode = ValidationMode.DEV_ONLY.value
    elif trust_level == TrustLevel.TRUSTWORTHY.value:
        validation_mode = ValidationMode.VALIDATION_GRADE.value
    else:
        validation_mode = ValidationMode.DEGRADED_VALIDATION.value

    return RunTruthState(
        generated_at=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        discovery_state=current_discovery_state,
        scoring_state=current_scoring_state,
        source_quality=scoring_source_quality,
        dominant_source_quality=dominant_source_quality,
        fallback_in_use=fallback_in_use,
        trust_level=trust_level,
        paper_readiness=paper_readiness.value if isinstance(paper_readiness, PaperReadiness) else str(paper_readiness),
        validation_mode=validation_mode,
        total_discovered_wallets=int(discovery_diagnostics.get("wallet_count", 0)),
        total_scored_wallets=total_scored_wallets,
        total_approved_wallets=len(approved_wallets_list),
        total_rejected_wallets=int(scoring_diagnostics.get("rejected_count", 0)),
        synthetic_wallet_count=synthetic_wallet_count,
        total_detected_source_trades=int(state_snapshot.get("last_cycle_detection_count", 0)),
        total_candidate_signals=signal_count,
        total_approved_decisions=entered_count,
        total_approved_decisions_trustworthy=trustworthy_approvals,
        total_approved_decisions_degraded=degraded_approvals,
        total_approved_decisions_non_validation=non_validation_approvals,
        total_skipped_decisions=skip_count,
        skip_reason_distribution=dict(skip_reason_distribution),
        warnings=warnings,
        notes=[
            "This artifact represents the current run truth-state only.",
            "Synthetic fallback is development continuity, not validation-grade evidence." if fallback_in_use else "",
        ],
        source_quality_summary=source_quality_summary,
        trust_level_summary=dict(trust_level_summary),
        funnel={
            "detected": int(state_snapshot.get("last_cycle_detection_count", signal_count)),
            "candidates": signal_count,
            "eligible": eligible_count,
            "approved": entered_count,
            "skipped": skip_count,
            "entered": entered_count,
        },
        approved_wallet_count=len(approved_wallets_list),
        rejected_wallet_count=int(scoring_diagnostics.get("rejected_count", 0)),
        discovery_reason=str(discovery.get("reason", "")),
        scoring_diagnostics=scoring_diagnostics,
        decision_count_total=signal_count,
        decision_count_trustworthy=trustworthy_approvals,
        decision_count_degraded=signal_count - trustworthy_approvals,
        current_discovery_state=current_discovery_state,
        current_scoring_state=current_scoring_state,
    )
