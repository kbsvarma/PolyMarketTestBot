from pathlib import Path
import asyncio
from datetime import datetime, timezone

from src.config import load_config
from src.geoblock import GeoblockChecker
from src.live_engine import LiveTradingEngine
from src.health import HealthMonitor
from src.models import EntryStyle, LiveOrder, Mode, Position, TradeDecision, DecisionAction
from src.models import HealthComponent, HealthState
from src.models import OrderLifecycleStatus, OrderbookLevel, OrderbookSnapshot
from src.state import AppStateStore


def _engine(tmp_path: Path) -> LiveTradingEngine:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config = config.model_copy(update={"mode": Mode.LIVE})
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    return LiveTradingEngine(config, tmp_path, AppStateStore(tmp_path / "app_state.json"), GeoblockChecker(config))


def _decision(notional: float) -> TradeDecision:
    return TradeDecision(
        allowed=True,
        action=DecisionAction.LIVE_COPY,
        reason_code="OK",
        human_readable_reason="ok",
        local_decision_id="d1",
        wallet_address="0xabc",
        market_id="m1",
        token_id="t1",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        category="politics",
        scaled_notional=notional,
        source_price=0.5,
        executable_price=0.5,
        cluster_confirmed=True,
        hedge_suspicion_score=0.0,
    )


def _decision_with_id(notional: float, decision_id: str, market_id: str, token_id: str) -> TradeDecision:
    return _decision(notional).model_copy(
        update={
            "local_decision_id": decision_id,
            "market_id": market_id,
            "token_id": token_id,
        }
    )


def test_operator_cap_blocks_trade_over_max_trade(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.config.env = engine.config.env.model_copy(update={"operator_live_max_trade_usd": 5.0})

    reason = engine._operator_entry_block_reason(_decision(6.0), [], [])

    assert reason is not None
    assert "OPERATOR_MAX_TRADE_USD" in reason


def test_effective_live_decision_applies_exchange_min_notional_floor(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    decision = _decision(0.6).model_copy(update={"executable_price": 0.12})

    effective = engine._effective_live_decision(decision)

    assert effective.scaled_notional == 1.0
    assert effective.context["minimum_live_order_notional_usd"] == 1.0
    assert effective.context["minimum_live_order_size_applied"] is True


def test_operator_cap_blocks_when_max_positions_reached(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.config.env = engine.config.env.model_copy(update={"operator_live_max_positions": 1})
    position = Position(
        position_id="p1",
        mode=Mode.LIVE,
        wallet_address="0xabc",
        market_id="m1",
        token_id="t1",
        category="politics",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        entry_price=0.5,
        current_mark_price=0.5,
        quantity=10,
        notional=5,
        fees_paid=0.0,
        source_trade_timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        remaining_size=10,
    )

    reason = engine._operator_entry_block_reason(_decision(1.0), [position], [])

    assert reason is not None
    assert "OPERATOR_MAX_POSITIONS" in reason


def test_operator_cap_ignores_quarantined_exit_data_unavailable_positions_for_slot_count(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.config.env = engine.config.env.model_copy(update={"operator_live_max_positions": 1})
    position = Position(
        position_id="p1",
        mode=Mode.LIVE,
        wallet_address="0xabc",
        market_id="m1",
        token_id="t1",
        category="crypto price",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        entry_price=0.5,
        current_mark_price=0.01,
        quantity=10,
        notional=5,
        fees_paid=0.0,
        source_trade_timestamp=datetime.now(timezone.utc),
        remaining_size=10,
        exit_state="EXIT_DATA_UNAVAILABLE",
        source_exit_following_enabled=False,
    )

    reason = engine._operator_entry_block_reason(_decision(1.0), [position], [])

    assert reason is None


def test_operator_cap_blocks_when_session_exposure_exceeded(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.config.env = engine.config.env.model_copy(update={"operator_live_session_max_usd": 30.0})
    order = LiveOrder(
        local_decision_id="d0",
        local_order_id="o0",
        client_order_id="c0",
        market_id="m0",
        token_id="t0",
        side="BUY",
        intended_price=0.5,
        intended_size=50,
        remaining_size=50,
        entry_style=EntryStyle.PASSIVE_LIMIT,
    )

    reason = engine._operator_entry_block_reason(_decision(6.0), [], [order])

    assert reason is not None
    assert "OPERATOR_SESSION_MAX_USD" in reason


def test_live_engine_submits_order_when_health_is_degraded_but_not_unhealthy(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.state.write(
        {
            "paused": False,
            "kill_switch": False,
            "manual_live_enable": True,
            "manual_resume_required": False,
            "live_readiness_last_result": {"ready": True, "checks": []},
            "reconciliation_clean": True,
        }
    )
    engine.health_monitor = HealthMonitor(tmp_path / "health_status.json")
    engine.geoblock.live_trading_allowed = lambda: type("Geo", (), {"eligible": True, "detail": "ok"})()  # type: ignore[method-assign]

    async def _refresh_live_status():
        engine.state.update_system_status(
            paused=False,
            manual_resume_required=False,
            live_readiness_last_result={"ready": True, "checks": []},
            reconciliation_clean=True,
        )

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate(
            [HealthComponent(name="market_ws", state=HealthState.DEGRADED, detail="lagging stream")]
        )

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _submit(order, side: str):
        order.last_exchange_status = "LIVE"
        order.last_update_at = datetime.now(timezone.utc)
        return {"exchange_order_id": "ex-1", "status": "LIVE"}

    async def _manage_existing_orders(live_orders, live_positions, decisions):
        return None

    async def _apply_live_exits(live_positions, live_orders):
        return None

    async def _reconcile():
        return type("Recon", (), {"clean": True, "severity": "NONE", "model_dump": lambda self, mode="json": {"clean": True}})()

    engine.refresh_live_status = _refresh_live_status  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.order_manager.submit_order = _submit  # type: ignore[method-assign]
    engine._manage_existing_orders = _manage_existing_orders  # type: ignore[method-assign]
    engine._apply_live_exits = _apply_live_exits  # type: ignore[method-assign]
    engine.reconcile = _reconcile  # type: ignore[method-assign]

    asyncio.run(engine.handle_decisions([_decision(1.0)]))

    orders = engine.orders.load()
    assert len(orders) == 1
    assert orders[0].local_decision_id == "d1"
    assert orders[0].intended_size == 5.0


def test_entries_last_hour_ignores_rejected_orders(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    rejected = LiveOrder(
        local_decision_id="d-rejected",
        local_order_id="o-rejected",
        client_order_id="c-rejected",
        market_id="m-rejected",
        token_id="t-rejected",
        side="BUY",
        intended_price=0.5,
        intended_size=5.0,
        remaining_size=5.0,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        submitted_at=now,
        lifecycle_status=OrderLifecycleStatus.REJECTED,
        terminal_state=True,
    )
    live = LiveOrder(
        local_decision_id="d-live",
        local_order_id="o-live",
        client_order_id="c-live",
        market_id="m-live",
        token_id="t-live",
        side="BUY",
        intended_price=0.5,
        intended_size=5.0,
        remaining_size=5.0,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        submitted_at=now,
        lifecycle_status=OrderLifecycleStatus.RESTING,
    )

    assert engine._entries_last_hour([rejected, live]) == 1


def test_entries_last_hour_dedupes_duplicate_local_order_ids(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    first = LiveOrder(
        local_decision_id="d-1",
        local_order_id="o-shared",
        client_order_id="c-old",
        market_id="m1",
        token_id="t1",
        side="BUY",
        intended_price=0.5,
        intended_size=5.0,
        remaining_size=5.0,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        submitted_at=now,
        lifecycle_status=OrderLifecycleStatus.RESTING,
    )
    latest = first.model_copy(update={"client_order_id": "c-new", "last_update_at": now})

    assert engine._entries_last_hour([first, latest]) == 1


def test_live_engine_skips_tradability_lookup_errors_without_crashing(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.state.write(
        {
            "paused": False,
            "kill_switch": False,
            "manual_live_enable": True,
            "manual_resume_required": False,
            "live_readiness_last_result": {"ready": True, "checks": []},
            "reconciliation_clean": True,
        }
    )
    engine.health_monitor = HealthMonitor(tmp_path / "health_status.json")
    engine.geoblock.live_trading_allowed = lambda: type("Geo", (), {"eligible": True, "detail": "ok"})()  # type: ignore[method-assign]

    async def _refresh_live_status():
        engine.state.update_system_status(
            paused=False,
            manual_resume_required=False,
            live_readiness_last_result={"ready": True, "checks": []},
            reconciliation_clean=True,
        )

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate(
            [HealthComponent(name="market_ws", state=HealthState.DEGRADED, detail="lagging stream")]
        )

    async def _tradability(_market_id: str, _token_id: str):
        raise RuntimeError("tradability metadata missing")

    async def _manage_existing_orders(live_orders, live_positions, decisions):
        return None

    async def _apply_live_exits(live_positions, live_orders):
        return None

    async def _reconcile():
        return type("Recon", (), {"clean": True, "severity": "NONE", "model_dump": lambda self, mode="json": {"clean": True}})()

    engine.refresh_live_status = _refresh_live_status  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine._manage_existing_orders = _manage_existing_orders  # type: ignore[method-assign]
    engine._apply_live_exits = _apply_live_exits  # type: ignore[method-assign]
    engine.reconcile = _reconcile  # type: ignore[method-assign]

    asyncio.run(engine.handle_decisions([_decision(1.0)]))

    orders = engine.orders.load()
    assert orders == []
    assert engine.state.read()["paused"] is False


def test_live_engine_reuses_cached_tradability_when_lookup_fails(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.state.write(
        {
            "paused": False,
            "kill_switch": False,
            "manual_live_enable": True,
            "manual_resume_required": False,
            "live_readiness_last_result": {"ready": True, "checks": []},
            "reconciliation_clean": True,
        }
    )
    engine.health_monitor = HealthMonitor(tmp_path / "health_status.json")
    engine.geoblock.live_trading_allowed = lambda: type("Geo", (), {"eligible": True, "detail": "ok"})()  # type: ignore[method-assign]

    async def _refresh_live_status():
        engine.state.update_system_status(
            paused=False,
            manual_resume_required=False,
            live_readiness_last_result={"ready": True, "checks": []},
            reconciliation_clean=True,
        )

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate(
            [HealthComponent(name="market_ws", state=HealthState.DEGRADED, detail="lagging stream")]
        )

    async def _tradability(_market_id: str, _token_id: str):
        raise RuntimeError("tradability metadata missing")

    async def _submit(order, side: str):
        order.last_exchange_status = "LIVE"
        order.last_update_at = datetime.now(timezone.utc)
        return {"exchange_order_id": "ex-1", "status": "LIVE"}

    async def _manage_existing_orders(live_orders, live_positions, decisions):
        return None

    async def _apply_live_exits(live_positions, live_orders):
        return None

    async def _reconcile():
        return type("Recon", (), {"clean": True, "severity": "NONE", "model_dump": lambda self, mode="json": {"clean": True}})()

    engine.refresh_live_status = _refresh_live_status  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.order_manager.submit_order = _submit  # type: ignore[method-assign]
    engine._manage_existing_orders = _manage_existing_orders  # type: ignore[method-assign]
    engine._apply_live_exits = _apply_live_exits  # type: ignore[method-assign]
    engine.reconcile = _reconcile  # type: ignore[method-assign]

    decision = _decision(1.0).model_copy(
        update={
            "context": {
                "tradability": {
                    "market_id": "m1",
                    "token_id": "t1",
                    "tradable": True,
                    "orderbook_enabled": True,
                }
            }
        }
    )

    asyncio.run(engine.handle_decisions([decision]))

    orders = engine.orders.load()
    assert len(orders) == 1
    assert orders[0].local_decision_id == "d1"


def test_live_engine_rate_limit_applies_to_actual_created_orders(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.config.risk.max_new_entries_per_hour = 1
    engine.state.write(
        {
            "paused": False,
            "kill_switch": False,
            "manual_live_enable": True,
            "manual_resume_required": False,
            "live_readiness_last_result": {"ready": True, "checks": []},
            "reconciliation_clean": True,
        }
    )
    engine.health_monitor = HealthMonitor(tmp_path / "health_status.json")
    engine.geoblock.live_trading_allowed = lambda: type("Geo", (), {"eligible": True, "detail": "ok"})()  # type: ignore[method-assign]

    async def _refresh_live_status():
        engine.state.update_system_status(
            paused=False,
            manual_resume_required=False,
            live_readiness_last_result={"ready": True, "checks": []},
            reconciliation_clean=True,
        )

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate(
            [HealthComponent(name="market_ws", state=HealthState.DEGRADED, detail="lagging stream")]
        )

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _submit(order, side: str):
        order.last_exchange_status = "LIVE"
        order.last_update_at = datetime.now(timezone.utc)
        return {"exchange_order_id": f"ex-{order.local_order_id}", "status": "LIVE"}

    async def _manage_existing_orders(live_orders, live_positions, decisions):
        return None

    async def _apply_live_exits(live_positions, live_orders):
        return None

    async def _reconcile():
        return type("Recon", (), {"clean": True, "severity": "NONE", "model_dump": lambda self, mode="json": {"clean": True}})()

    engine.refresh_live_status = _refresh_live_status  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.order_manager.submit_order = _submit  # type: ignore[method-assign]
    engine._manage_existing_orders = _manage_existing_orders  # type: ignore[method-assign]
    engine._apply_live_exits = _apply_live_exits  # type: ignore[method-assign]
    engine.reconcile = _reconcile  # type: ignore[method-assign]

    asyncio.run(
        engine.handle_decisions(
            [
                _decision_with_id(1.0, "d1", "m1", "t1"),
                _decision_with_id(1.0, "d2", "m2", "t2"),
            ]
        )
    )

    orders = engine.orders.load()
    assert len(orders) == 1
    assert orders[0].local_decision_id == "d1"


def test_live_engine_continues_after_rejected_submit_and_allows_next_order(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.config.risk.max_new_entries_per_hour = 1
    engine.state.write(
        {
            "paused": False,
            "kill_switch": False,
            "manual_live_enable": True,
            "manual_resume_required": False,
            "live_readiness_last_result": {"ready": True, "checks": []},
            "reconciliation_clean": True,
        }
    )
    engine.health_monitor = HealthMonitor(tmp_path / "health_status.json")
    engine.geoblock.live_trading_allowed = lambda: type("Geo", (), {"eligible": True, "detail": "ok"})()  # type: ignore[method-assign]

    async def _refresh_live_status():
        engine.state.update_system_status(
            paused=False,
            manual_resume_required=False,
            live_readiness_last_result={"ready": True, "checks": []},
            reconciliation_clean=True,
        )

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate(
            [HealthComponent(name="market_ws", state=HealthState.HEALTHY, detail="ok")]
        )

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _submit(order, side: str):
        if order.local_decision_id == "d1":
            order.last_exchange_status = "REJECTED"
            order.lifecycle_status = OrderLifecycleStatus.REJECTED
            order.terminal_state = True
            order.last_update_at = datetime.now(timezone.utc)
            raise RuntimeError("exchange rejected")
        order.exchange_order_id = "ex-2"
        order.last_exchange_status = "LIVE"
        order.lifecycle_status = OrderLifecycleStatus.RESTING
        order.last_update_at = datetime.now(timezone.utc)
        return {"exchange_order_id": "ex-2", "status": "LIVE"}

    async def _manage_existing_orders(live_orders, live_positions, decisions):
        return None

    async def _apply_live_exits(live_positions, live_orders):
        return None

    async def _reconcile():
        return type("Recon", (), {"clean": True, "severity": "NONE", "model_dump": lambda self, mode="json": {"clean": True}})()

    engine.refresh_live_status = _refresh_live_status  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.order_manager.submit_order = _submit  # type: ignore[method-assign]
    engine._manage_existing_orders = _manage_existing_orders  # type: ignore[method-assign]
    engine._apply_live_exits = _apply_live_exits  # type: ignore[method-assign]
    engine.reconcile = _reconcile  # type: ignore[method-assign]

    asyncio.run(
        engine.handle_decisions(
            [
                _decision_with_id(1.0, "d1", "m1", "t1"),
                _decision_with_id(1.0, "d2", "m2", "t2"),
            ]
        )
    )

    orders = engine.orders.load()
    assert len(orders) == 2
    assert orders[0].lifecycle_status == OrderLifecycleStatus.REJECTED
    assert orders[1].lifecycle_status == OrderLifecycleStatus.RESTING
    assert engine.state.read()["paused"] is False


def test_manage_existing_orders_reprices_without_current_cycle_decision(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    order = LiveOrder(
        local_decision_id="older-decision",
        local_order_id="o1",
        client_order_id="c1",
        exchange_order_id="ex1",
        market_id="m1",
        token_id="t1",
        side="BUY",
        intended_price=0.45,
        intended_size=5.0,
        remaining_size=5.0,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        lifecycle_status=OrderLifecycleStatus.RESTING,
    )
    live_orders = [order]

    async def _refresh(order_obj):
        order_obj.lifecycle_status = OrderLifecycleStatus.RESTING
        order_obj.last_exchange_status = "LIVE"
        return {"status": "LIVE"}

    async def _handle_timeout(order_obj):
        order_obj.cancel_requested = True
        order_obj.cancel_confirmed = True
        order_obj.terminal_state = True
        order_obj.lifecycle_status = OrderLifecycleStatus.CANCELLED
        return "REPRICE_READY"

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(_token_id: str):
        return OrderbookSnapshot(
            token_id="t1",
            bids=[OrderbookLevel(price=0.45, size=10)],
            asks=[OrderbookLevel(price=0.47, size=10)],
        )

    submitted: list[tuple[str, float]] = []

    async def _submit(order_obj, side: str):
        submitted.append((side, order_obj.intended_price))
        order_obj.last_exchange_status = "LIVE"
        order_obj.lifecycle_status = OrderLifecycleStatus.RESTING
        order_obj.terminal_state = False
        return {"exchange_order_id": "ex2", "status": "LIVE"}

    engine.order_manager.refresh_order = _refresh  # type: ignore[method-assign]
    engine.order_manager.handle_timeout = _handle_timeout  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.client.get_orderbook = _orderbook  # type: ignore[method-assign]
    engine.order_manager.submit_order = _submit  # type: ignore[method-assign]

    asyncio.run(engine._manage_existing_orders(live_orders, [], []))

    assert submitted == [("BUY", 0.46)]
    assert live_orders[0].reprice_attempts == 1
