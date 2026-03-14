from __future__ import annotations

from src.models import DecisionAction, EntryStyle, TrustLevel, ValidationMode
from src.models import TradeDecision
from src.run_truth import build_run_truth


def _paper_copy_decision(*, trust_level: str, counts_as_trustworthy_approval: bool) -> TradeDecision:
    return TradeDecision(
        allowed=True,
        action=DecisionAction.PAPER_COPY,
        reason_code="OK",
        human_readable_reason="eligible",
        local_decision_id="decision-1",
        wallet_address="0xabc",
        market_id="market-1",
        token_id="token-1",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        category="politics",
        scaled_notional=25.0,
        source_price=0.51,
        executable_price=0.52,
        cluster_confirmed=True,
        hedge_suspicion_score=0.0,
        context={
            "trust_level": trust_level,
            "counts_as_trustworthy_approval": counts_as_trustworthy_approval,
            "source_quality": "SYNTHETIC_FALLBACK" if trust_level == TrustLevel.NOT_TRUSTWORTHY.value else "REAL_PUBLIC_DATA",
        },
    )


def test_run_truth_repairs_synthetic_fallback_trustworthy_approvals() -> None:
    truth = build_run_truth(
        mode="RESEARCH",
        discovery={
            "state": "SYNTHETIC_FALLBACK_USED",
            "diagnostics": {"discovery_state": "SYNTHETIC_FALLBACK_USED", "fallback_used": True, "wallet_count": 3},
        },
        scoring={
            "state": "SUCCESS",
            "source_quality": "SYNTHETIC_FALLBACK",
            "diagnostics": {"scored_count": 3, "rejected_count": 0, "synthetic_wallet_count": 3},
        },
        state_snapshot={
            "approved_wallets": {"paper_wallets": ["0xabc"], "research_wallets": ["0xabc"], "live_wallets": []},
            "last_cycle_detection_count": 1,
        },
        decisions=[_paper_copy_decision(trust_level=TrustLevel.TRUSTWORTHY.value, counts_as_trustworthy_approval=True)],
    )

    assert truth.fallback_in_use is True
    assert truth.trust_level == TrustLevel.NOT_TRUSTWORTHY.value
    assert truth.validation_mode == ValidationMode.DEV_ONLY.value
    assert truth.total_approved_decisions == 1
    assert truth.total_approved_decisions_trustworthy == 0
    assert truth.total_approved_decisions_non_validation == 1
    assert any("synthetic fallback cannot produce trustworthy approvals" in warning.lower() for warning in truth.warnings)


def test_run_truth_repairs_empty_scoring_trustworthy_approvals() -> None:
    truth = build_run_truth(
        mode="RESEARCH",
        discovery={
            "state": "SUCCESS",
            "diagnostics": {"discovery_state": "SUCCESS", "fallback_used": False, "wallet_count": 1},
        },
        scoring={
            "state": "EMPTY",
            "source_quality": "REAL_PUBLIC_DATA",
            "diagnostics": {"scored_count": 0, "rejected_count": 0, "synthetic_wallet_count": 0},
        },
        state_snapshot={
            "approved_wallets": {"paper_wallets": [], "research_wallets": [], "live_wallets": []},
            "last_cycle_detection_count": 1,
        },
        decisions=[_paper_copy_decision(trust_level=TrustLevel.TRUSTWORTHY.value, counts_as_trustworthy_approval=True)],
    )

    assert truth.scoring_state == "EMPTY"
    assert truth.total_approved_decisions == 1
    assert truth.total_approved_decisions_trustworthy == 0
    assert truth.paper_readiness == "NOT_TRUSTWORTHY"
    assert any("empty scoring cannot produce trustworthy approvals" in warning.lower() for warning in truth.warnings)
