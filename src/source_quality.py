from __future__ import annotations

from src.models import DiscoveryState, SourceQuality


def quality_from_discovery_state(state: DiscoveryState, fallback_used: bool = False) -> SourceQuality:
    if fallback_used:
        return SourceQuality.SYNTHETIC_FALLBACK
    if state == DiscoveryState.SUCCESS:
        return SourceQuality.REAL_PUBLIC_DATA
    if state in {
        DiscoveryState.NO_DATA,
        DiscoveryState.FETCH_FAILED,
        DiscoveryState.MALFORMED_RESPONSE,
        DiscoveryState.FILTERED_TO_ZERO,
        DiscoveryState.SYNTHETIC_FALLBACK_USED,
    }:
        return SourceQuality.DEGRADED_PUBLIC_DATA
    return SourceQuality.DEGRADED_PUBLIC_DATA


def quality_rank(source_quality: SourceQuality) -> int:
    return {
        SourceQuality.REAL_PUBLIC_DATA: 3,
        SourceQuality.DEGRADED_PUBLIC_DATA: 2,
        SourceQuality.SYNTHETIC_FALLBACK: 1,
    }[source_quality]


def dominant_source_quality(values: list[str]) -> SourceQuality:
    ranked = [SourceQuality(value) for value in values if value in {quality.value for quality in SourceQuality}]
    if not ranked:
        return SourceQuality.DEGRADED_PUBLIC_DATA
    counts = {quality: ranked.count(quality) for quality in SourceQuality}
    return max(counts, key=lambda quality: (counts[quality], quality_rank(quality)))
