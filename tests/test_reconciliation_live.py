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


def test_reconciliation_normalizes_live_open_order_payload() -> None:
    summary = reconcile_live_state(
        local_positions=[],
        exchange_positions=[],
        local_orders=[
            {
                "exchange_order_id": "ex1",
                "client_order_id": "c1",
                "lifecycle_status": "UNKNOWN",
                "last_exchange_status": "LIVE",
                "filled_size": 0.0,
                "remaining_size": 5.769231,
                "terminal_state": False,
                "linked_position_id": "",
            }
        ],
        exchange_orders=[
            {
                "id": "ex1",
                "status": "LIVE",
                "filled_size": 0.0,
                "remaining_size": 0.0,
                "original_size": "5.769231",
                "size_matched": "0",
            }
        ],
    )
    assert summary.clean
    assert summary.severity == "NONE"
