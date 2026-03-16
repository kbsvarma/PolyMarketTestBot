from pathlib import Path

from src.live_orders import LiveOrderStore
from src.models import Mode, OrderLifecycleStatus
from src.positions import PositionStore
from src.utils import write_json


def test_load_normalizes_live_status_from_legacy_unknown(tmp_path: Path) -> None:
    path = tmp_path / "live_orders.json"
    write_json(
        path,
        [
            {
                "local_decision_id": "d1",
                "local_order_id": "o1",
                "client_order_id": "c1",
                "exchange_order_id": "ex1",
                "market_id": "m1",
                "token_id": "t1",
                "side": "BUY",
                "intended_price": 0.52,
                "intended_size": 5.769231,
                "entry_style": "PASSIVE_LIMIT",
                "created_at": "2026-03-14T22:27:17.501725Z",
                "submitted_at": "2026-03-14T22:27:17.502926Z",
                "last_exchange_status": "LIVE",
                "lifecycle_status": "UNKNOWN",
                "filled_size": 0.0,
                "average_fill_price": 0.0,
                "remaining_size": 5.769231,
                "cancel_requested": False,
                "cancel_confirmed": False,
                "terminal_state": False,
                "linked_position_id": "",
                "audit_log_ref": "",
                "raw_exchange_response_refs": [],
                "repriced_once": False,
                "timeout_at": "2026-03-14T22:27:37.501741Z",
                "is_exit": False,
                "linked_parent_order_id": "",
                "last_update_at": "2026-03-14T22:27:18.290180Z",
            }
        ],
    )

    orders = LiveOrderStore(path).load()

    assert len(orders) == 1
    assert orders[0].lifecycle_status == OrderLifecycleStatus.RESTING
    assert orders[0].terminal_state is False


def test_load_normalizes_matched_status_from_legacy_unknown(tmp_path: Path) -> None:
    path = tmp_path / "live_orders.json"
    write_json(
        path,
        [
            {
                "local_decision_id": "d1",
                "local_order_id": "o1",
                "client_order_id": "c1",
                "exchange_order_id": "ex1",
                "market_id": "m1",
                "token_id": "t1",
                "side": "BUY",
                "intended_price": 0.45,
                "intended_size": 5.0,
                "entry_style": "PASSIVE_LIMIT",
                "created_at": "2026-03-14T22:27:17.501725Z",
                "submitted_at": "2026-03-14T22:27:17.502926Z",
                "last_exchange_status": "MATCHED",
                "lifecycle_status": "UNKNOWN",
                "filled_size": 5.0,
                "average_fill_price": 0.45,
                "remaining_size": 0.0,
                "cancel_requested": False,
                "cancel_confirmed": False,
                "terminal_state": False,
                "linked_position_id": "",
                "audit_log_ref": "",
                "raw_exchange_response_refs": [],
                "repriced_once": False,
                "timeout_at": "2026-03-14T22:27:37.501741Z",
                "is_exit": False,
                "linked_parent_order_id": "",
                "last_update_at": "2026-03-14T22:27:18.290180Z",
            }
        ],
    )

    orders = LiveOrderStore(path).load()

    assert len(orders) == 1
    assert orders[0].lifecycle_status == OrderLifecycleStatus.FILLED
    assert orders[0].terminal_state is True


def test_load_dedupes_duplicate_local_order_rows(tmp_path: Path) -> None:
    path = tmp_path / "live_orders.json"
    write_json(
        path,
        [
            {
                "local_decision_id": "d1",
                "local_order_id": "o1",
                "client_order_id": "c1-old",
                "exchange_order_id": "ex-old",
                "market_id": "m1",
                "token_id": "t1",
                "side": "BUY",
                "intended_price": 0.52,
                "intended_size": 5.0,
                "entry_style": "PASSIVE_LIMIT",
                "created_at": "2026-03-16T02:00:00Z",
                "submitted_at": "2026-03-16T02:00:10Z",
                "last_exchange_status": "LIVE",
                "lifecycle_status": "RESTING",
                "filled_size": 0.0,
                "average_fill_price": 0.0,
                "remaining_size": 5.0,
                "raw_exchange_response_refs": ["old"],
                "last_update_at": "2026-03-16T02:00:20Z",
            },
            {
                "local_decision_id": "d1",
                "local_order_id": "o1",
                "client_order_id": "c1-new",
                "exchange_order_id": "ex-new",
                "market_id": "m1",
                "token_id": "t1",
                "side": "BUY",
                "intended_price": 0.53,
                "intended_size": 5.0,
                "entry_style": "PASSIVE_LIMIT",
                "created_at": "2026-03-16T02:00:00Z",
                "submitted_at": "2026-03-16T02:01:10Z",
                "last_exchange_status": "MATCHED",
                "lifecycle_status": "FILLED",
                "filled_size": 5.0,
                "average_fill_price": 0.53,
                "remaining_size": 0.0,
                "raw_exchange_response_refs": ["new"],
                "last_update_at": "2026-03-16T02:01:20Z",
            },
        ],
    )

    orders = LiveOrderStore(path).load()

    assert len(orders) == 1
    assert orders[0].exchange_order_id == "ex-new"
    assert orders[0].filled_size == 5.0
    assert orders[0].raw_exchange_response_refs == ["old", "new"]


def test_position_store_dedupes_duplicate_position_rows(tmp_path: Path) -> None:
    path = tmp_path / "positions.json"
    write_json(
        path,
        {
            "paper": [],
            "live": [
                {
                    "position_id": "p1",
                    "mode": "LIVE",
                    "wallet_address": "0xabc",
                    "market_id": "m1",
                    "token_id": "t1",
                    "category": "politics",
                    "entry_style": "PASSIVE_LIMIT",
                    "entry_price": 0.5,
                    "current_mark_price": 0.5,
                    "quantity": 0.0,
                    "notional": 0.0,
                    "fees_paid": 0.0,
                    "source_trade_timestamp": "2026-03-16T02:00:00Z",
                    "opened_at": "2026-03-16T02:00:00Z",
                    "closed": False,
                    "remaining_size": 0.0,
                    "entry_order_ids": ["old-order"],
                    "last_reconciled_at": "2026-03-16T02:00:30Z",
                },
                {
                    "position_id": "p1",
                    "mode": "LIVE",
                    "wallet_address": "0xabc",
                    "market_id": "m1",
                    "token_id": "t1",
                    "category": "politics",
                    "entry_style": "PASSIVE_LIMIT",
                    "entry_price": 0.5,
                    "current_mark_price": 0.48,
                    "quantity": 5.0,
                    "notional": 2.5,
                    "fees_paid": 0.0,
                    "source_trade_timestamp": "2026-03-16T02:00:00Z",
                    "opened_at": "2026-03-16T02:00:00Z",
                    "closed": False,
                    "remaining_size": 5.0,
                    "entry_order_ids": ["new-order"],
                    "last_reconciled_at": "2026-03-16T02:01:30Z",
                },
            ],
        },
    )

    positions = PositionStore(path).positions_for_mode(Mode.LIVE)

    assert len(positions) == 1
    assert positions[0].remaining_size == 5.0
    assert positions[0].entry_order_ids == ["old-order", "new-order"]


def test_position_store_normalizes_exit_data_unavailable_positions(tmp_path: Path) -> None:
    path = tmp_path / "positions.json"
    write_json(
        path,
        {
            "paper": [],
            "live": [
                {
                    "position_id": "p1",
                    "mode": "LIVE",
                    "wallet_address": "0xabc",
                    "market_id": "m1",
                    "token_id": "t1",
                    "category": "crypto price",
                    "entry_style": "PASSIVE_LIMIT",
                    "entry_price": 0.5,
                    "current_mark_price": 0.2,
                    "quantity": 5.0,
                    "notional": 2.5,
                    "fees_paid": 0.0,
                    "source_trade_timestamp": "2026-03-16T02:00:00Z",
                    "opened_at": "2026-03-16T02:00:00Z",
                    "closed": False,
                    "remaining_size": 5.0,
                    "exit_state": "EXIT_DATA_UNAVAILABLE",
                    "source_exit_following_enabled": True,
                }
            ],
        },
    )

    positions = PositionStore(path).positions_for_mode(Mode.LIVE)

    assert len(positions) == 1
    assert positions[0].exit_state == "EXIT_DATA_UNAVAILABLE"
    assert positions[0].source_exit_following_enabled is False


def test_position_store_keeps_quarantine_when_duplicate_rows_disagree(tmp_path: Path) -> None:
    path = tmp_path / "positions.json"
    write_json(
        path,
        {
            "paper": [],
            "live": [
                {
                    "position_id": "p1",
                    "mode": "LIVE",
                    "wallet_address": "0xabc",
                    "market_id": "m1",
                    "token_id": "t1",
                    "category": "crypto price",
                    "entry_style": "PASSIVE_LIMIT",
                    "entry_price": 0.5,
                    "current_mark_price": 0.2,
                    "quantity": 5.0,
                    "notional": 2.5,
                    "fees_paid": 0.0,
                    "source_trade_timestamp": "2026-03-16T02:00:00Z",
                    "opened_at": "2026-03-16T02:00:00Z",
                    "closed": False,
                    "remaining_size": 5.0,
                    "exit_state": "EXIT_DATA_UNAVAILABLE",
                    "source_exit_following_enabled": False,
                    "last_reconciled_at": "2026-03-16T02:00:30Z",
                },
                {
                    "position_id": "p1",
                    "mode": "LIVE",
                    "wallet_address": "0xabc",
                    "market_id": "m1",
                    "token_id": "t1",
                    "category": "crypto price",
                    "entry_style": "PASSIVE_LIMIT",
                    "entry_price": 0.5,
                    "current_mark_price": 0.22,
                    "quantity": 5.0,
                    "notional": 2.5,
                    "fees_paid": 0.0,
                    "source_trade_timestamp": "2026-03-16T02:00:00Z",
                    "opened_at": "2026-03-16T02:00:00Z",
                    "closed": False,
                    "remaining_size": 5.0,
                    "exit_state": "OPEN",
                    "source_exit_following_enabled": True,
                    "last_reconciled_at": "2026-03-16T02:01:30Z",
                },
            ],
        },
    )

    positions = PositionStore(path).positions_for_mode(Mode.LIVE)

    assert len(positions) == 1
    assert positions[0].exit_state == "EXIT_DATA_UNAVAILABLE"
    assert positions[0].source_exit_following_enabled is False
