from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.audit import AuditLogger
from src.config import load_config
from src.live_order_manager import LiveOrderManager
from src.models import DecisionAction, EntryStyle, LiveOrder, OrderLifecycleStatus, OrderbookLevel, OrderbookSnapshot, TradeDecision
from src.polymarket_client import PolymarketClient


def build_decision() -> TradeDecision:
    return TradeDecision(
        allowed=True,
        action=DecisionAction.LIVE_COPY,
        reason_code="OK",
        human_readable_reason="ok",
        local_decision_id="decision-1",
        wallet_address="0xabc",
        market_id="m1",
        token_id="t1",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        category="politics",
        scaled_notional=5.0,
        source_price=0.5,
        executable_price=0.5,
        cluster_confirmed=True,
        hedge_suspicion_score=0.1,
    )


def test_no_position_before_fill() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    manager = LiveOrderManager(config, root / "data", PolymarketClient(config), AuditLogger(root / "data" / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())
    positions = []
    result = manager.apply_fill_to_position(order, build_decision(), positions)
    assert result is None
    assert positions == []


def test_partial_fill_creates_partial_position() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    manager = LiveOrderManager(config, root / "data", PolymarketClient(config), AuditLogger(root / "data" / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())
    order.filled_size = 3.0
    order.average_fill_price = 0.5
    positions = []
    result = manager.apply_fill_to_position(order, build_decision(), positions)
    assert result is not None
    assert result.entry_size == 3.0
    assert result.remaining_size == 3.0


def test_partial_fill_can_reconstruct_position_without_current_decision() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    manager = LiveOrderManager(config, root / "data", PolymarketClient(config), AuditLogger(root / "data" / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())
    order.filled_size = 5.0
    order.average_fill_price = 0.46
    positions = []

    result = manager.apply_fill_to_position(order, None, positions)

    assert result is not None
    assert result.wallet_address == "0xabc"
    assert result.category == "politics"
    assert result.strategy_name == "wallet_follow"
    assert result.entry_size == 5.0
    assert result.entry_price_actual == 0.46
    assert order.linked_position_id == result.position_id


def test_paired_decision_metadata_survives_into_live_position() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    manager = LiveOrderManager(config, root / "data", PolymarketClient(config), AuditLogger(root / "data" / "live_audit.jsonl"))
    decision = build_decision().model_copy(
        update={
            "strategy_name": "paired_binary_arb",
            "thesis_type": "paired_arb",
            "bundle_id": "bundle-1",
            "bundle_role": "paired_yes",
            "paired_token_id": "t2",
        }
    )
    order = manager.create_entry_order(decision)
    order.filled_size = 5.0
    order.average_fill_price = 0.48
    positions = []

    result = manager.apply_fill_to_position(order, decision, positions)

    assert result is not None
    assert result.thesis_type == "paired_arb"
    assert result.bundle_id == "bundle-1"
    assert result.bundle_role == "paired_yes"
    assert result.paired_token_id == "t2"


def test_timeout_cancels_unfilled_order() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    manager = LiveOrderManager(config, root / "data", PolymarketClient(config), AuditLogger(root / "data" / "live_audit.jsonl"))
    order = LiveOrder(
        local_decision_id="d1",
        local_order_id="o1",
        client_order_id="c1",
        market_id="m1",
        token_id="t1",
        side="BUY",
        intended_price=0.5,
        intended_size=5,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        lifecycle_status=OrderLifecycleStatus.RESTING,
        timeout_at=datetime.now(timezone.utc),
    )
    result = __import__("asyncio").run(manager.handle_timeout(order))
    assert result in {"CANCELLED", "TIMED_OUT", "REPRICE_READY"}


def test_passive_entry_timeout_uses_longer_rest_window() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    manager = LiveOrderManager(config, root / "data", PolymarketClient(config), AuditLogger(root / "data" / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())
    assert order.timeout_at is not None
    remaining_seconds = (order.timeout_at - datetime.now(timezone.utc)).total_seconds()
    assert remaining_seconds >= 55


def test_handle_timeout_returns_reprice_ready_for_passive_order(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    client = PolymarketClient(config)
    manager = LiveOrderManager(config, tmp_path, client, AuditLogger(tmp_path / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())
    order.exchange_order_id = "ex1"
    order.lifecycle_status = OrderLifecycleStatus.RESTING
    order.timeout_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    async def _cancel(order_id: str):
        assert order_id == "ex1"
        return {"id": order_id, "status": "CANCELLED"}

    client.cancel_order = _cancel  # type: ignore[method-assign]

    result = __import__("asyncio").run(manager.handle_timeout(order))

    assert result == "REPRICE_READY"
    assert order.cancel_confirmed is True
    assert order.terminal_state is True


def test_prepare_reprice_resets_order_and_updates_price(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    client = PolymarketClient(config)
    manager = LiveOrderManager(config, tmp_path, client, AuditLogger(tmp_path / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())
    order.intended_price = 0.45
    order.exchange_order_id = "ex1"
    order.cancel_requested = True
    order.cancel_confirmed = True
    order.terminal_state = True
    order.lifecycle_status = OrderLifecycleStatus.CANCELLED
    old_client_order_id = order.client_order_id
    orderbook = OrderbookSnapshot(
        token_id="t1",
        bids=[OrderbookLevel(price=0.45, size=10)],
        asks=[OrderbookLevel(price=0.47, size=10)],
    )

    changed = manager.prepare_reprice(order, orderbook, tradable=True, drift_ok=True)

    assert changed is True
    assert order.intended_price == 0.46
    assert order.client_order_id != old_client_order_id
    assert order.reprice_attempts == 1
    assert order.repriced_once is True
    assert order.cancel_requested is False
    assert order.cancel_confirmed is False
    assert order.terminal_state is False
    assert order.lifecycle_status == OrderLifecycleStatus.CREATED


def test_failed_submit_marks_order_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    client = PolymarketClient(config)
    manager = LiveOrderManager(config, tmp_path, client, AuditLogger(tmp_path / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())

    async def _fail(**kwargs):
        raise RuntimeError("invalid signature")

    client.place_buy_order = _fail  # type: ignore[method-assign]

    try:
        __import__("asyncio").run(manager.submit_order(order, side="BUY"))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "invalid signature" in str(exc)

    assert order.lifecycle_status == OrderLifecycleStatus.REJECTED
    assert order.terminal_state is True
    assert order.last_exchange_status == "REJECTED"


def test_refresh_order_maps_live_status_to_resting_and_uses_raw_size(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    client = PolymarketClient(config)
    manager = LiveOrderManager(config, tmp_path, client, AuditLogger(tmp_path / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())
    order.exchange_order_id = "ex1"
    order.intended_size = 5.769231

    async def _status(order_id: str):
        assert order_id == "ex1"
        return {
            "exchange_order_id": "ex1",
            "status": "LIVE",
            "filled_size": 0.0,
            "average_fill_price": 0.0,
            "remaining_size": 0.0,
            "raw": {"original_size": "5.76", "size_matched": "0"},
        }

    client.get_order_status = _status  # type: ignore[method-assign]

    __import__("asyncio").run(manager.refresh_order(order))

    assert order.lifecycle_status == OrderLifecycleStatus.RESTING
    assert order.last_exchange_status == "LIVE"
    assert order.remaining_size == 5.76
    assert order.terminal_state is False


def test_refresh_order_maps_canceled_status_to_terminal_cancelled(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    client = PolymarketClient(config)
    manager = LiveOrderManager(config, tmp_path, client, AuditLogger(tmp_path / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())
    order.exchange_order_id = "ex1"
    order.intended_size = 5.769231

    async def _status(order_id: str):
        assert order_id == "ex1"
        return {
            "exchange_order_id": "ex1",
            "status": "CANCELED",
            "filled_size": 0.0,
            "average_fill_price": 0.0,
            "remaining_size": 5.76,
            "raw": {"original_size": "5.76", "size_matched": "0"},
        }

    client.get_order_status = _status  # type: ignore[method-assign]

    __import__("asyncio").run(manager.refresh_order(order))

    assert order.lifecycle_status == OrderLifecycleStatus.CANCELLED
    assert order.last_exchange_status == "CANCELED"
    assert order.remaining_size == 0.0
    assert order.terminal_state is True


def test_refresh_order_maps_matched_status_to_terminal_filled(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    client = PolymarketClient(config)
    manager = LiveOrderManager(config, tmp_path, client, AuditLogger(tmp_path / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())
    order.exchange_order_id = "ex1"

    async def _status(order_id: str):
        assert order_id == "ex1"
        return {
            "exchange_order_id": "ex1",
            "status": "MATCHED",
            "filled_size": 5.0,
            "average_fill_price": 0.45,
            "remaining_size": 0.0,
            "raw": {"original_size": "5", "size_matched": "5", "price": "0.45"},
        }

    client.get_order_status = _status  # type: ignore[method-assign]

    __import__("asyncio").run(manager.refresh_order(order))

    assert order.lifecycle_status == OrderLifecycleStatus.FILLED
    assert order.last_exchange_status == "MATCHED"
    assert order.filled_size == 5.0
    assert order.average_fill_price == 0.45
    assert order.remaining_size == 0.0
    assert order.terminal_state is True


def test_cancel_open_order_marks_terminal_and_records_cancel(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    client = PolymarketClient(config)
    manager = LiveOrderManager(config, tmp_path, client, AuditLogger(tmp_path / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())
    order.exchange_order_id = "ex1"
    order.lifecycle_status = OrderLifecycleStatus.RESTING
    order.remaining_size = order.intended_size

    async def _cancel(order_id: str):
        assert order_id == "ex1"
        return {"id": order_id, "status": "CANCELLED"}

    client.cancel_order = _cancel  # type: ignore[method-assign]

    __import__("asyncio").run(manager.cancel_open_order(order))

    assert order.cancel_requested is True
    assert order.cancel_confirmed is True
    assert order.lifecycle_status == OrderLifecycleStatus.CANCELLED
    assert order.terminal_state is True
    assert order.remaining_size == 0.0


def test_mark_cancelled_by_reconciliation_marks_terminal(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    client = PolymarketClient(config)
    manager = LiveOrderManager(config, tmp_path, client, AuditLogger(tmp_path / "live_audit.jsonl"))
    order = manager.create_entry_order(build_decision())
    order.exchange_order_id = "ex1"
    order.lifecycle_status = OrderLifecycleStatus.RESTING
    order.remaining_size = order.intended_size

    manager.mark_cancelled_by_reconciliation(order)

    assert order.cancel_requested is True
    assert order.cancel_confirmed is True
    assert order.lifecycle_status == OrderLifecycleStatus.CANCELLED
    assert order.terminal_state is True
    assert order.remaining_size == 0.0
