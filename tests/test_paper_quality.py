from src.models import PaperReadiness
from src.paper_quality import classify_paper_readiness, summarize_source_quality


def test_paper_readiness_classification() -> None:
    readiness = classify_paper_readiness(
        discovery_state="SUCCESS",
        scoring_state="SUCCESS",
        approved_wallet_count=3,
        candidate_signal_count=10,
        real_data_signal_pct=0.8,
        fallback_signal_pct=0.0,
    )
    assert readiness == PaperReadiness.STRONG

    degraded = classify_paper_readiness(
        discovery_state="NO_DATA",
        scoring_state="EMPTY",
        approved_wallet_count=0,
        candidate_signal_count=0,
        real_data_signal_pct=0.0,
        fallback_signal_pct=1.0,
    )
    assert degraded == PaperReadiness.NOT_TRUSTWORTHY


def test_source_quality_summary() -> None:
    summary = summarize_source_quality(["REAL_PUBLIC_DATA", "REAL_PUBLIC_DATA", "DEGRADED_PUBLIC_DATA"])
    assert summary["REAL_PUBLIC_DATA"] == 0.6667
    assert summary["SYNTHETIC_FALLBACK"] == 0.0
