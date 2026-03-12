from src.reconciliation import reconcile_live_state


def test_reconciliation_detects_position_mismatch() -> None:
    summary = reconcile_live_state(
        local_positions=[{"market_id": "m1", "token_id": "t1", "side": "BUY", "quantity": 1.0, "closed": False}],
        exchange_positions=[],
        local_orders=[],
        exchange_orders=[],
    )
    assert not summary.clean
    assert summary.severity == "SEVERE"
