from __future__ import annotations

from src.models import DiscoveryState, PaperReadiness, SourceQuality


def classify_paper_readiness(
    *,
    discovery_state: str,
    scoring_state: str,
    approved_wallet_count: int,
    candidate_signal_count: int,
    real_data_signal_pct: float,
    fallback_signal_pct: float,
) -> PaperReadiness:
    if approved_wallet_count == 0 or candidate_signal_count == 0:
        return PaperReadiness.NOT_TRUSTWORTHY
    if discovery_state != DiscoveryState.SUCCESS.value or scoring_state not in {"SUCCESS", "PARTIAL_SUCCESS"}:
        return PaperReadiness.DEGRADED
    if real_data_signal_pct < 0.6 or fallback_signal_pct > 0.2:
        return PaperReadiness.DEGRADED
    return PaperReadiness.STRONG


def summarize_source_quality(values: list[str]) -> dict[str, float]:
    total = len(values)
    if total == 0:
        return {
            SourceQuality.REAL_PUBLIC_DATA.value: 0.0,
            SourceQuality.DEGRADED_PUBLIC_DATA.value: 0.0,
            SourceQuality.SYNTHETIC_FALLBACK.value: 0.0,
        }
    counts = {quality: 0 for quality in SourceQuality}
    for value in values:
        if value in counts:
            counts[SourceQuality(value)] += 1
    return {quality.value: round(count / total, 4) for quality, count in counts.items()}
