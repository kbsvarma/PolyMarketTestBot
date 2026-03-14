from pathlib import Path

from src.live_orders import LiveOrderStore
from src.models import OrderLifecycleStatus
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
