from __future__ import annotations

from src.models import DiscoveryState, PaperReadiness, SourceQuality, TrustLevel
from src.source_quality import dominant_source_quality


def classify_trust_level(
    *,
    source_quality: str,
    discovery_state: str,
    scoring_state: str,
    fallback_in_use: bool,
) -> str:
    if fallback_in_use or source_quality == SourceQuality.SYNTHETIC_FALLBACK.value:
        return TrustLevel.NOT_TRUSTWORTHY.value
    if discovery_state in {
        DiscoveryState.NO_DATA.value,
        DiscoveryState.FETCH_FAILED.value,
        DiscoveryState.MALFORMED_RESPONSE.value,
        DiscoveryState.FILTERED_TO_ZERO.value,
        DiscoveryState.SYNTHETIC_FALLBACK_USED.value,
    }:
        return TrustLevel.NOT_TRUSTWORTHY.value
    if source_quality == SourceQuality.DEGRADED_PUBLIC_DATA.value or scoring_state not in {"SUCCESS", "PARTIAL_SUCCESS"}:
        return TrustLevel.DEGRADED.value
    return TrustLevel.TRUSTWORTHY.value


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
    if trust_level == TrustLevel.NOT_TRUSTWORTHY.value:
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


def counts_as_trustworthy_approval(
    *,
    final_action: str,
    trust_level: str,
    fallback_in_use: bool,
    scoring_state: str,
) -> bool:
    return (
        final_action == "PAPER_COPY"
        and trust_level == TrustLevel.TRUSTWORTHY.value
        and not fallback_in_use
        and scoring_state in {"SUCCESS", "PARTIAL_SUCCESS"}
    )


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


def summarize_source_quality_from_truth(
    *,
    values: list[str],
    discovery_state: str,
    scoring_state: str,
    fallback_in_use: bool,
    synthetic_wallet_count: int,
    total_scored_wallets: int,
    dominant_hint: str | None = None,
) -> dict[str, float | str]:
    if values:
        summary = summarize_source_quality(values)
        if fallback_in_use:
            summary[SourceQuality.REAL_PUBLIC_DATA.value] = 0.0
            summary[SourceQuality.DEGRADED_PUBLIC_DATA.value] = 0.0
            summary[SourceQuality.SYNTHETIC_FALLBACK.value] = 1.0
            summary["dominant_source_quality"] = SourceQuality.SYNTHETIC_FALLBACK.value
        return summary

    if fallback_in_use or discovery_state == DiscoveryState.SYNTHETIC_FALLBACK_USED.value:
        return {
            SourceQuality.REAL_PUBLIC_DATA.value: 0.0,
            SourceQuality.DEGRADED_PUBLIC_DATA.value: 0.0,
            SourceQuality.SYNTHETIC_FALLBACK.value: 1.0,
            "dominant_source_quality": SourceQuality.SYNTHETIC_FALLBACK.value,
        }

    if synthetic_wallet_count > 0 and total_scored_wallets > 0:
        synthetic_pct = round(synthetic_wallet_count / max(total_scored_wallets, 1), 4)
        degraded_pct = round(1.0 - synthetic_pct, 4)
        dominant = (
            SourceQuality.SYNTHETIC_FALLBACK.value
            if synthetic_pct >= degraded_pct
            else SourceQuality.DEGRADED_PUBLIC_DATA.value
        )
        return {
            SourceQuality.REAL_PUBLIC_DATA.value: 0.0,
            SourceQuality.DEGRADED_PUBLIC_DATA.value: degraded_pct,
            SourceQuality.SYNTHETIC_FALLBACK.value: synthetic_pct,
            "dominant_source_quality": dominant,
        }

    if scoring_state == "SUCCESS" and dominant_hint == SourceQuality.REAL_PUBLIC_DATA.value:
        return {
            SourceQuality.REAL_PUBLIC_DATA.value: 1.0,
            SourceQuality.DEGRADED_PUBLIC_DATA.value: 0.0,
            SourceQuality.SYNTHETIC_FALLBACK.value: 0.0,
            "dominant_source_quality": SourceQuality.REAL_PUBLIC_DATA.value,
        }

    dominant = dominant_hint or SourceQuality.DEGRADED_PUBLIC_DATA.value
    if dominant not in {quality.value for quality in SourceQuality}:
        dominant = SourceQuality.DEGRADED_PUBLIC_DATA.value

    summary = {
        SourceQuality.REAL_PUBLIC_DATA.value: 0.0,
        SourceQuality.DEGRADED_PUBLIC_DATA.value: 0.0,
        SourceQuality.SYNTHETIC_FALLBACK.value: 0.0,
    }
    summary[dominant] = 1.0
    summary["dominant_source_quality"] = dominant
    return summary
