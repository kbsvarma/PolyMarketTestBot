from __future__ import annotations

from src.models import DiscoveryState, PaperReadiness, SourceQuality
from src.source_quality import dominant_source_quality


def classify_trust_level(
    *,
    source_quality: str,
    discovery_state: str,
    scoring_state: str,
    fallback_in_use: bool,
) -> str:
    if fallback_in_use or source_quality == SourceQuality.SYNTHETIC_FALLBACK.value:
        return PaperReadiness.NOT_TRUSTWORTHY.value
    if discovery_state in {
        DiscoveryState.NO_DATA.value,
        DiscoveryState.FETCH_FAILED.value,
        DiscoveryState.MALFORMED_RESPONSE.value,
        DiscoveryState.FILTERED_TO_ZERO.value,
        DiscoveryState.SYNTHETIC_FALLBACK_USED.value,
    }:
        return PaperReadiness.NOT_TRUSTWORTHY.value
    if source_quality == SourceQuality.DEGRADED_PUBLIC_DATA.value or scoring_state not in {"SUCCESS", "PARTIAL_SUCCESS"}:
        return PaperReadiness.DEGRADED.value
    return PaperReadiness.STRONG.value


def classify_paper_readiness(
    *,
    discovery_state: str,
    scoring_state: str,
    approved_wallet_count: int,
    candidate_signal_count: int,
    real_data_signal_pct: float,
    fallback_signal_pct: float,
    degraded_signal_pct: float = 0.0,
    fallback_in_use: bool = False,
    approved_wallet_source_quality: str = SourceQuality.DEGRADED_PUBLIC_DATA.value,
) -> PaperReadiness:
    trust_level = classify_trust_level(
        source_quality=approved_wallet_source_quality,
        discovery_state=discovery_state,
        scoring_state=scoring_state,
        fallback_in_use=fallback_in_use,
    )
    if trust_level == PaperReadiness.NOT_TRUSTWORTHY.value:
        return PaperReadiness.NOT_TRUSTWORTHY
    if approved_wallet_count == 0 or candidate_signal_count == 0:
        return PaperReadiness.NOT_TRUSTWORTHY
    if scoring_state not in {"SUCCESS", "PARTIAL_SUCCESS"}:
        return PaperReadiness.DEGRADED
    if real_data_signal_pct < 0.6 or fallback_signal_pct > 0.0:
        return PaperReadiness.NOT_TRUSTWORTHY
    if degraded_signal_pct > 0.4:
        return PaperReadiness.DEGRADED
    return PaperReadiness.STRONG


def summarize_source_quality(values: list[str]) -> dict[str, float | str]:
    total = len(values)
    if total == 0:
        return {
            SourceQuality.REAL_PUBLIC_DATA.value: 0.0,
            SourceQuality.DEGRADED_PUBLIC_DATA.value: 0.0,
            SourceQuality.SYNTHETIC_FALLBACK.value: 0.0,
            "dominant_source_quality": SourceQuality.DEGRADED_PUBLIC_DATA.value,
        }
    counts = {quality: 0 for quality in SourceQuality}
    for value in values:
        if value in counts:
            counts[SourceQuality(value)] += 1
    summary = {quality.value: round(count / total, 4) for quality, count in counts.items()}
    summary["dominant_source_quality"] = dominant_source_quality(values).value
    return summary
