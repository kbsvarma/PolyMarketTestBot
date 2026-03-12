from pathlib import Path

from src.audit import AuditLogger
from src.config import load_config
from src.live_order_manager import LiveOrderManager
from src.models import DecisionAction, EntryStyle, LiveOrder, OrderLifecycleStatus, TradeDecision
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
        timeout_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    result = __import__("asyncio").run(manager.handle_timeout(order))
    assert result in {"CANCELLED", "TIMED_OUT"}
