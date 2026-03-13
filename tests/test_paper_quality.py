from src.models import PaperReadiness, TrustLevel
from src.paper_quality import (
    classify_paper_readiness,
    classify_trust_level,
    counts_as_trustworthy_approval,
    summarize_source_quality,
    summarize_source_quality_from_truth,
)


def test_paper_readiness_classification() -> None:
    readiness = classify_paper_readiness(
        discovery_state="SUCCESS",
        scoring_state="SUCCESS",
        approved_wallet_count=3,
        candidate_signal_count=10,
        real_data_signal_pct=0.8,
        fallback_signal_pct=0.0,
        degraded_signal_pct=0.0,
        fallback_in_use=False,
        approved_wallet_source_quality="REAL_PUBLIC_DATA",
    )
    assert readiness == PaperReadiness.STRONG

    degraded = classify_paper_readiness(
        discovery_state="NO_DATA",
        scoring_state="EMPTY",
        approved_wallet_count=0,
        candidate_signal_count=0,
        real_data_signal_pct=0.0,
        fallback_signal_pct=1.0,
        degraded_signal_pct=0.0,
        fallback_in_use=True,
        approved_wallet_source_quality="SYNTHETIC_FALLBACK",
    )
    assert degraded == PaperReadiness.NOT_TRUSTWORTHY


def test_source_quality_summary() -> None:
    summary = summarize_source_quality(["REAL_PUBLIC_DATA", "REAL_PUBLIC_DATA", "DEGRADED_PUBLIC_DATA"])
    assert summary["REAL_PUBLIC_DATA"] == 0.6667
    assert summary["SYNTHETIC_FALLBACK"] == 0.0
    assert summary["dominant_source_quality"] == "REAL_PUBLIC_DATA"


def test_source_quality_summary_derives_truth_without_decisions() -> None:
    summary = summarize_source_quality_from_truth(
        values=[],
        discovery_state="SYNTHETIC_FALLBACK_USED",
        scoring_state="EMPTY",
        fallback_in_use=True,
        synthetic_wallet_count=3,
        total_scored_wallets=3,
    )
    assert summary["SYNTHETIC_FALLBACK"] == 1.0
    assert summary["dominant_source_quality"] == "SYNTHETIC_FALLBACK"


def test_trust_level_is_conservative_under_fallback() -> None:
    assert (
        classify_trust_level(
            source_quality="SYNTHETIC_FALLBACK",
            discovery_state="SYNTHETIC_FALLBACK_USED",
            scoring_state="SUCCESS",
            fallback_in_use=True,
        )
        == TrustLevel.NOT_TRUSTWORTHY.value
    )


def test_source_quality_summary_marks_synthetic_as_dominant_without_decisions() -> None:
    summary = summarize_source_quality_from_truth(
        values=[],
        discovery_state="SYNTHETIC_FALLBACK_USED",
        scoring_state="SUCCESS",
        fallback_in_use=True,
        synthetic_wallet_count=3,
        total_scored_wallets=3,
        dominant_hint="SYNTHETIC_FALLBACK",
    )
    assert summary["dominant_source_quality"] == "SYNTHETIC_FALLBACK"
    assert summary["SYNTHETIC_FALLBACK"] == 1.0


def test_counts_as_trustworthy_approval_is_conservative() -> None:
    assert (
        counts_as_trustworthy_approval(
            final_action="PAPER_COPY",
            trust_level=TrustLevel.TRUSTWORTHY.value,
            fallback_in_use=False,
            scoring_state="SUCCESS",
        )
        is True
    )
    assert (
        counts_as_trustworthy_approval(
            final_action="PAPER_COPY",
            trust_level=TrustLevel.NOT_TRUSTWORTHY.value,
            fallback_in_use=True,
            scoring_state="SUCCESS",
        )
        is False
    )
