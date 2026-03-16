from datetime import datetime, timedelta, timezone

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


def test_reconciliation_ignores_local_linked_position_for_open_order_match() -> None:
    summary = reconcile_live_state(
        local_positions=[],
        exchange_positions=[],
        local_orders=[
            {
                "exchange_order_id": "ex1",
                "client_order_id": "c1",
                "lifecycle_status": "RESTING",
                "last_exchange_status": "LIVE",
                "filled_size": 0.0,
                "remaining_size": 5.0,
                "terminal_state": False,
                "linked_position_id": "pos-1",
            }
        ],
        exchange_orders=[
            {
                "id": "ex1",
                "status": "LIVE",
                "filled_size": 0.0,
                "remaining_size": 5.0,
            }
        ],
    )
    assert summary.clean
    assert summary.severity == "NONE"


def test_reconciliation_ignores_terminal_matched_and_market_resolved_orders() -> None:
    summary = reconcile_live_state(
        local_positions=[],
        exchange_positions=[],
        local_orders=[
            {
                "exchange_order_id": "ex-filled",
                "client_order_id": "c-filled",
                "lifecycle_status": "UNKNOWN",
                "last_exchange_status": "MATCHED",
                "filled_size": 5.0,
                "remaining_size": 0.0,
                "terminal_state": True,
                "linked_position_id": "p1",
            },
            {
                "exchange_order_id": "ex-cancelled",
                "client_order_id": "c-cancelled",
                "lifecycle_status": "UNKNOWN",
                "last_exchange_status": "CANCELED_MARKET_RESOLVED",
                "filled_size": 0.0,
                "remaining_size": 0.0,
                "terminal_state": True,
                "linked_position_id": "",
            },
        ],
        exchange_orders=[],
    )
    assert summary.clean
    assert summary.severity == "NONE"


def test_reconciliation_tolerates_recent_local_position_exchange_lag() -> None:
    now = datetime.now(timezone.utc)
    summary = reconcile_live_state(
        local_positions=[
            {
                "market_id": "m1",
                "token_id": "t1",
                "side": "BUY",
                "quantity": 5.0,
                "closed": False,
                "opened_at": now.isoformat(),
            }
        ],
        exchange_positions=[],
        local_orders=[],
        exchange_orders=[],
    )
    assert summary.clean
    assert summary.severity == "NONE"


def test_reconciliation_tolerates_recent_delayed_order_exchange_lag() -> None:
    now = datetime.now(timezone.utc)
    summary = reconcile_live_state(
        local_positions=[],
        exchange_positions=[],
        local_orders=[
            {
                "exchange_order_id": "ex1",
                "client_order_id": "c1",
                "lifecycle_status": "DELAYED",
                "filled_size": 0.0,
                "remaining_size": 5.0,
                "terminal_state": False,
                "submitted_at": now.isoformat(),
                "last_update_at": now.isoformat(),
            }
        ],
        exchange_orders=[],
    )
    assert summary.clean
    assert summary.severity == "NONE"


def test_reconciliation_flags_old_missing_local_position() -> None:
    old = datetime.now(timezone.utc) - timedelta(minutes=5)
    summary = reconcile_live_state(
        local_positions=[
            {
                "market_id": "m1",
                "token_id": "t1",
                "side": "BUY",
                "quantity": 5.0,
                "closed": False,
                "opened_at": old.isoformat(),
            }
        ],
        exchange_positions=[],
        local_orders=[],
        exchange_orders=[],
    )
    assert not summary.clean
    assert summary.severity == "SEVERE"


def test_reconciliation_uses_remaining_size_for_open_local_positions() -> None:
    summary = reconcile_live_state(
        local_positions=[
            {
                "market_id": "m1",
                "token_id": "t1",
                "side": "BUY",
                "quantity": 5.0,
                "remaining_size": 4.9632,
                "closed": False,
            }
        ],
        exchange_positions=[
            {
                "conditionId": "m1",
                "asset": "t1",
                "side": "BUY",
                "size": 4.9632,
            }
        ],
        local_orders=[],
        exchange_orders=[],
    )
    assert summary.clean
    assert summary.severity == "NONE"


def test_reconciliation_ignores_zero_sized_unclosed_local_positions() -> None:
    summary = reconcile_live_state(
        local_positions=[
            {
                "market_id": "m1",
                "token_id": "t1",
                "side": "BUY",
                "quantity": 0.0,
                "remaining_size": 0.0,
                "closed": False,
            }
        ],
        exchange_positions=[],
        local_orders=[],
        exchange_orders=[],
    )
    assert summary.clean
    assert summary.severity == "NONE"
