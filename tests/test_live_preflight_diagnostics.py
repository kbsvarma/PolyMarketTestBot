from __future__ import annotations

import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone

from src.geoblock import GeoblockChecker
from src.config import load_config
from src.live_engine import LiveTradingEngine
from src.live_orders import LiveOrderStore
from src.positions import PositionStore
from src.models import EntryStyle, LiveOrder, Mode, OrderLifecycleStatus, Position
from src.models import HealthComponent, HealthState
from src.state import AppStateStore
from src.utils import read_json


def test_live_startup_validation_exposes_diagnostics(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    async def _health():
        return type("Health", (), {"ok": True, "detail": "ok"})()  # pragma: no cover

    async def _balance():
        return {"balance": "0"}

    async def _allowance():
        return {
            "available": 0.0,
            "sufficient": False,
            "query_visible": True,
            "raw_summary": {"top_level_balance": 0.0, "allowance_entry_count": 0},
            "sdk_signature_type": 1,
            "configured_signature_type": None,
        }

    async def _wallet_balances():
        return {"wallet_address": "0xabc", "usdc": 0.0, "usdce": 5.0, "total_stablecoins": 5.0}

    async def _open_orders():
        return []

    async def _positions():
        return []

    async def _refresh_markets():
        return {}

    async def _stream_watchlist(token_ids: list[str]):
        return {}

    async def _reconcile():
        return type("Recon", (), {"clean": True, "severity": "CLEAN", "model_dump": lambda self, mode="json": {"clean": True}})()

    engine.client.health_check = _health  # type: ignore[method-assign]
    engine.client.get_balance = _balance  # type: ignore[method-assign]
    engine.client.get_allowance = _allowance  # type: ignore[method-assign]
    engine.client.get_wallet_stablecoin_balances = _wallet_balances  # type: ignore[method-assign]
    engine.client.get_open_orders = _open_orders  # type: ignore[method-assign]
    engine.client.get_positions = _positions  # type: ignore[method-assign]
    engine.client.live_order_capable = lambda: (True, "ok")  # type: ignore[method-assign]
    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.reconcile = _reconcile  # type: ignore[method-assign]

    checks = asyncio.run(engine.startup_validation())
    assert checks["allowance_visible"] is True
    assert checks["allowance_available"] == 0.0
    assert checks["allowance_raw_summary"] == {"top_level_balance": 0.0, "allowance_entry_count": 0}
    assert checks["allowance_sdk_signature_type"] == 1
    assert checks["allowance_configured_signature_type"] is None
    assert checks["positions_visible"] is True
    assert checks["positions_count"] == 0
    assert checks["positions_query_error"] == ""


def test_refresh_live_status_warms_ws_before_collecting_health(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    async def _startup_validation():
        engine.market_data.ws.connected = True
        engine.market_data.ws.last_event_at = datetime.now(timezone.utc)
        return {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_visible": True,
            "allowance_sufficient": True,
            "allowance_available": 5.0,
            "allowance_raw_summary": {},
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 5.0,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "0 orders visible",
            "positions_visible": True,
            "positions_count": 0,
            "positions_query_error": "",
            "positions_detail": "0 positions visible",
            "tradability_ok": True,
            "tradability_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "NONE",
        }

    async def _collect_health(reconciliation_override=None):
        state = HealthState.HEALTHY if engine.market_data.ws.connected else HealthState.DEGRADED
        return engine.health_monitor.aggregate([HealthComponent(name="market_ws", state=state, detail="ok")])

    async def _positions():
        return []

    async def _wallet_balances():
        return {"usdc": 0.0, "usdce": 5.0, "total_stablecoins": 5.0}

    async def _reconcile():
        return type("Recon", (), {"clean": True, "severity": "NONE", "model_dump": lambda self, mode="json": {"clean": True}})()

    engine.startup_validation = _startup_validation  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.client.get_positions = _positions  # type: ignore[method-assign]
    engine.client.get_wallet_stablecoin_balances = _wallet_balances  # type: ignore[method-assign]
    engine.reconcile = _reconcile  # type: ignore[method-assign]

    asyncio.run(engine.refresh_live_status())
    payload = state.read()
    assert payload["live_readiness_last_result"]["ready"] is True


def test_refresh_live_status_quarantines_exit_data_unavailable_positions_without_new_decisions(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    position = Position(
        position_id="p1",
        mode=Mode.LIVE,
        wallet_address="0xabc",
        market_id="m1",
        token_id="t1",
        category="crypto price",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        entry_price=0.5,
        current_mark_price=0.2,
        quantity=5.0,
        notional=2.5,
        fees_paid=0.0,
        source_trade_timestamp=datetime.now(timezone.utc),
        opened_at=datetime.now(timezone.utc),
        remaining_size=5.0,
        source_exit_following_enabled=True,
        exit_state="EXIT_DATA_UNAVAILABLE",
    )
    PositionStore(tmp_path / "positions.json").save_positions([], [position])

    async def _startup_validation():
        return {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_visible": True,
            "allowance_sufficient": True,
            "allowance_available": 5.0,
            "allowance_raw_summary": {},
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 5.0,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "0 orders visible",
            "positions_visible": True,
            "positions_count": 1,
            "positions_query_error": "",
            "positions_detail": "1 positions visible",
            "tradability_ok": True,
            "tradability_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "NONE",
            "reconciliation_summary": {"clean": True},
        }

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate(
            [HealthComponent(name="market_ws", state=HealthState.HEALTHY, detail="ok")]
        )

    async def _positions():
        return []

    async def _wallet_balances():
        return {"usdc": 0.0, "usdce": 5.0, "total_stablecoins": 5.0}

    async def _noop(*args, **kwargs):
        return False

    engine.startup_validation = _startup_validation  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.client.get_positions = _positions  # type: ignore[method-assign]
    engine.client.get_wallet_stablecoin_balances = _wallet_balances  # type: ignore[method-assign]
    engine._sync_existing_open_orders = _noop  # type: ignore[method-assign]
    engine._repair_closed_live_positions_from_exit_orders = _noop  # type: ignore[method-assign]
    engine._repair_missing_live_positions = _noop  # type: ignore[method-assign]

    asyncio.run(engine.refresh_live_status())

    repaired_positions = PositionStore(tmp_path / "positions.json").positions_for_mode(Mode.LIVE)
    assert repaired_positions[0].source_exit_following_enabled is False


def test_refresh_live_status_clears_stale_pause_and_unresolved_ids_when_orders_are_resting(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "paused": True,
            "pause_reason": "Live readiness gate failed.",
            "manual_live_enable": True,
            "manual_resume_required": True,
            "system_status": "PAUSED",
            "unresolved_live_order_ids": ["legacy-order"],
        }
    )
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    order = LiveOrder(
        local_decision_id="decision-1",
        local_order_id="order-1",
        client_order_id="client-1",
        exchange_order_id="ex1",
        market_id="m1",
        token_id="t1",
        side="BUY",
        intended_price=0.52,
        intended_size=5.769231,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        lifecycle_status=OrderLifecycleStatus.RESTING,
        last_exchange_status="LIVE",
        remaining_size=5.769231,
        terminal_state=False,
    )
    LiveOrderStore(tmp_path / "live_orders.json").save([order])

    async def _startup_validation():
        return {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_visible": True,
            "allowance_sufficient": True,
            "allowance_available": 5.0,
            "allowance_raw_summary": {},
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 5.0,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "1 orders visible",
            "positions_visible": True,
            "positions_count": 0,
            "positions_query_error": "",
            "positions_detail": "0 positions visible",
            "tradability_ok": True,
            "tradability_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "NONE",
            "reconciliation_summary": {"clean": True, "severity": "NONE", "issues": []},
        }

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate([HealthComponent(name="market_ws", state=HealthState.HEALTHY, detail="ok")])

    async def _positions():
        return []

    async def _wallet_balances():
        return {"usdc": 0.0, "usdce": 5.0, "total_stablecoins": 5.0}

    engine.startup_validation = _startup_validation  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.client.get_positions = _positions  # type: ignore[method-assign]
    engine.client.get_wallet_stablecoin_balances = _wallet_balances  # type: ignore[method-assign]

    asyncio.run(engine.refresh_live_status())

    payload = state.read()
    assert payload["paused"] is False
    assert payload["pause_reason"] == ""
    assert payload["unresolved_live_order_ids"] == []
    assert payload["live_readiness_last_result"]["ready"] is True


def test_refresh_live_status_clears_soft_exit_pause_when_current_truth_is_clean(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "paused": True,
            "pause_reason": "Live exit ambiguity for pos-1: Unable to fetch real orderbook in LIVE mode: 404",
            "manual_live_enable": True,
            "manual_resume_required": True,
            "system_status": "PAUSED",
        }
    )
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    async def _startup_validation():
        return {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_visible": True,
            "allowance_sufficient": True,
            "allowance_available": 5.0,
            "allowance_raw_summary": {},
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 5.0,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "0 orders visible",
            "positions_visible": True,
            "positions_count": 1,
            "positions_query_error": "",
            "positions_detail": "1 positions visible",
            "tradability_ok": True,
            "tradability_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "NONE",
            "reconciliation_summary": {"clean": True, "severity": "NONE", "issues": []},
        }

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate([HealthComponent(name="market_ws", state=HealthState.HEALTHY, detail="ok")])

    async def _positions():
        return [{"market_id": "m1", "token_id": "t1", "size": 5.0, "side": "BUY"}]

    async def _wallet_balances():
        return {"usdc": 0.0, "usdce": 5.0, "total_stablecoins": 5.0}

    engine.startup_validation = _startup_validation  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.client.get_positions = _positions  # type: ignore[method-assign]
    engine.client.get_wallet_stablecoin_balances = _wallet_balances  # type: ignore[method-assign]

    asyncio.run(engine.refresh_live_status())

    payload = state.read()
    assert payload["paused"] is False
    assert payload["pause_reason"] == ""
    assert payload["manual_resume_required"] is False
    assert payload["live_readiness_last_result"]["ready"] is True


def test_stale_readiness_pause_does_not_block_fresh_clean_preflight(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "paused": True,
            "pause_reason": "Live readiness gate failed.",
            "manual_live_enable": True,
        }
    )
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    async def _startup_validation():
        return {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_visible": True,
            "allowance_sufficient": True,
            "allowance_available": 5.0,
            "allowance_raw_summary": {},
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 5.0,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "1 orders visible",
            "positions_visible": True,
            "positions_count": 0,
            "positions_query_error": "",
            "positions_detail": "0 positions visible",
            "tradability_ok": True,
            "tradability_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "NONE",
            "reconciliation_summary": {"clean": True, "severity": "NONE", "issues": []},
        }

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate([HealthComponent(name="market_ws", state=HealthState.HEALTHY, detail="ok")])

    async def _positions():
        return []

    async def _wallet_balances():
        return {"usdc": 0.0, "usdce": 5.0, "total_stablecoins": 5.0}

    engine.startup_validation = _startup_validation  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.client.get_positions = _positions  # type: ignore[method-assign]
    engine.client.get_wallet_stablecoin_balances = _wallet_balances  # type: ignore[method-assign]

    asyncio.run(engine.refresh_live_status())

    payload = state.read()
    assert payload["live_readiness_last_result"]["ready"] is True


def test_refresh_live_status_reuses_startup_reconciliation_truth_for_health(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    async def _startup_validation():
        return {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_visible": True,
            "allowance_sufficient": True,
            "allowance_available": 5.0,
            "allowance_raw_summary": {},
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 5.0,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "1 orders visible",
            "positions_visible": True,
            "positions_count": 0,
            "positions_query_error": "",
            "positions_detail": "0 positions visible",
            "tradability_ok": True,
            "tradability_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "NONE",
            "reconciliation_summary": {"clean": True, "severity": "NONE", "issues": []},
        }

    async def _health_check():
        return type("Health", (), {"ok": True, "detail": "ok"})()

    async def _balance():
        return {"balance": "0"}

    async def _allowance():
        return {"available": 5.0, "sufficient": True, "query_visible": True}

    async def _heartbeat():
        return {"timestamp": datetime.now(timezone.utc).isoformat()}

    async def _positions():
        return []

    async def _wallet_balances():
        return {"usdc": 0.0, "usdce": 5.0, "total_stablecoins": 5.0}

    async def _open_orders():
        return [{"id": "ex1"}]

    engine.startup_validation = _startup_validation  # type: ignore[method-assign]
    engine.client.health_check = _health_check  # type: ignore[method-assign]
    engine.client.get_balance = _balance  # type: ignore[method-assign]
    engine.client.get_allowance = _allowance  # type: ignore[method-assign]
    engine.client.send_heartbeat = _heartbeat  # type: ignore[method-assign]
    engine.client.get_positions = _positions  # type: ignore[method-assign]
    engine.client.get_wallet_stablecoin_balances = _wallet_balances  # type: ignore[method-assign]
    engine.client.get_open_orders = _open_orders  # type: ignore[method-assign]
    engine.client.live_order_capable = lambda: (True, "ok")  # type: ignore[method-assign]
    engine.market_data.ws.connected = True
    engine.market_data.ws.last_event_at = datetime.now(timezone.utc)

    async def _reconcile_should_not_matter():
        return type("Recon", (), {"clean": False, "severity": "SEVERE", "model_dump": lambda self, mode="json": {"clean": False}})()

    engine.reconcile = _reconcile_should_not_matter  # type: ignore[method-assign]

    asyncio.run(engine.refresh_live_status())

    payload = state.read()
    assert payload["reconciliation_clean"] is True
    assert payload["live_readiness_last_result"]["ready"] is True
    failed = [check for check in payload["live_readiness_last_result"]["checks"] if not check["passed"]]
    assert failed == []


def test_refresh_live_status_marks_missing_open_order_cancelled(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    order = LiveOrder(
        local_decision_id="operator-smoke-1",
        local_order_id="o1",
        client_order_id="c1",
        exchange_order_id="ex1",
        market_id="m1",
        token_id="t1",
        side="BUY",
        intended_price=0.52,
        intended_size=5.769231,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        lifecycle_status=OrderLifecycleStatus.RESTING,
        last_exchange_status="LIVE",
        remaining_size=5.769231,
        terminal_state=False,
    )
    LiveOrderStore(tmp_path / "live_orders.json").save([order])

    async def _open_orders():
        return []

    async def _get_order_status(_order_id: str):
        raise RuntimeError("order not found")

    async def _startup_validation():
        return {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_visible": True,
            "allowance_sufficient": True,
            "allowance_available": 5.0,
            "allowance_raw_summary": {},
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 5.0,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "0 orders visible",
            "positions_visible": True,
            "positions_count": 0,
            "positions_query_error": "",
            "positions_detail": "0 positions visible",
            "tradability_ok": True,
            "tradability_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "NONE",
            "reconciliation_summary": {"clean": True, "severity": "NONE", "issues": []},
        }

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate([HealthComponent(name="market_ws", state=HealthState.HEALTHY, detail="ok")])

    async def _positions():
        return []

    async def _wallet_balances():
        return {"usdc": 0.0, "usdce": 5.0, "total_stablecoins": 5.0}

    engine.client.get_open_orders = _open_orders  # type: ignore[method-assign]
    engine.client.get_order_status = _get_order_status  # type: ignore[method-assign]
    engine.startup_validation = _startup_validation  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.client.get_positions = _positions  # type: ignore[method-assign]
    engine.client.get_wallet_stablecoin_balances = _wallet_balances  # type: ignore[method-assign]

    asyncio.run(engine.refresh_live_status())

    orders = LiveOrderStore(tmp_path / "live_orders.json").load()
    assert orders[0].lifecycle_status == OrderLifecycleStatus.CANCELLED
    assert orders[0].terminal_state is True


def test_refresh_live_status_reactivates_terminal_order_when_exchange_reports_it_open(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "paused": True,
            "pause_reason": "Live readiness gate failed.",
            "manual_live_enable": True,
            "manual_resume_required": True,
            "system_status": "PAUSED",
        }
    )
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    stale_order = LiveOrder(
        local_decision_id="decision-1",
        local_order_id="order-1",
        client_order_id="client-1",
        exchange_order_id="ex1",
        strategy_name="wallet_follow",
        market_id="m1",
        token_id="t1",
        side="BUY",
        intended_price=0.47,
        intended_size=5.0,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        lifecycle_status=OrderLifecycleStatus.FILLED,
        last_exchange_status="MATCHED",
        filled_size=5.0,
        average_fill_price=0.47,
        remaining_size=0.0,
        terminal_state=True,
    )
    LiveOrderStore(tmp_path / "live_orders.json").save([stale_order])

    async def _open_orders():
        return [
            {
                "exchange_order_id": "ex1",
                "client_order_id": "client-1",
                "status": "LIVE",
                "market_id": "m1",
                "token_id": "t1",
                "side": "BUY",
                "size": 5.0,
                "filled_size": 0.0,
                "remaining_size": 5.0,
                "price": 0.47,
            }
        ]

    async def _startup_validation():
        return {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_visible": True,
            "allowance_sufficient": True,
            "allowance_available": 5.0,
            "allowance_raw_summary": {},
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 5.0,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "1 orders visible",
            "positions_visible": True,
            "positions_count": 0,
            "positions_query_error": "",
            "positions_detail": "0 positions visible",
            "tradability_ok": True,
            "tradability_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "NONE",
            "reconciliation_summary": {"clean": True, "severity": "NONE", "issues": []},
        }

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate([HealthComponent(name="market_ws", state=HealthState.HEALTHY, detail="ok")])

    async def _positions():
        return []

    async def _wallet_balances():
        return {"usdc": 0.0, "usdce": 5.0, "total_stablecoins": 5.0}

    engine.client.get_open_orders = _open_orders  # type: ignore[method-assign]
    engine.startup_validation = _startup_validation  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.client.get_positions = _positions  # type: ignore[method-assign]
    engine.client.get_wallet_stablecoin_balances = _wallet_balances  # type: ignore[method-assign]

    asyncio.run(engine.refresh_live_status())

    repaired_orders = LiveOrderStore(tmp_path / "live_orders.json").load()
    assert repaired_orders[0].lifecycle_status == OrderLifecycleStatus.RESTING
    assert repaired_orders[0].terminal_state is False
    assert repaired_orders[0].remaining_size == 5.0


def test_refresh_live_status_repairs_missing_live_positions_from_exchange(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "paused": True,
            "pause_reason": "Live readiness gate failed.",
            "manual_live_enable": True,
            "manual_resume_required": True,
            "system_status": "PAUSED",
        }
    )
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    legacy_order = LiveOrder(
        local_decision_id="decision-1",
        local_order_id="order-1",
        client_order_id="client-1",
        exchange_order_id="ex1",
        strategy_name="wallet_follow",
        market_id="m1",
        token_id="t1",
        side="BUY",
        intended_price=0.45,
        intended_size=5.0,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        lifecycle_status=OrderLifecycleStatus.CANCELLED,
        last_exchange_status="CANCELLED",
        remaining_size=0.0,
        terminal_state=True,
    )
    LiveOrderStore(tmp_path / "live_orders.json").save([legacy_order])

    exchange_positions = [
        {
            "conditionId": "m1",
            "asset": "t1",
            "size": 5.0,
            "side": "BUY",
            "initialValue": 2.25,
            "currentValue": 2.35,
        }
    ]

    async def _startup_validation():
        return {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_visible": True,
            "allowance_sufficient": True,
            "allowance_available": 5.0,
            "allowance_raw_summary": {},
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 5.0,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "0 orders visible",
            "positions_visible": True,
            "positions_count": 1,
            "positions_query_error": "",
            "positions_detail": "1 positions visible",
            "tradability_ok": True,
            "tradability_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "NONE",
            "reconciliation_summary": {"clean": True, "severity": "NONE", "issues": []},
        }

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate([HealthComponent(name="market_ws", state=HealthState.HEALTHY, detail="ok")])

    async def _positions():
        return exchange_positions

    async def _wallet_balances():
        return {"usdc": 0.0, "usdce": 5.0, "total_stablecoins": 5.0}

    async def _market_metadata(market_id: str, token_id: str = "", market_slug: str = "", outcome: str = ""):
        assert market_id == "m1"
        assert token_id == "t1"
        return {"market_id": "m1", "token_id": "t1", "category": "politics"}

    engine.startup_validation = _startup_validation  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.client.get_positions = _positions  # type: ignore[method-assign]
    engine.client.get_wallet_stablecoin_balances = _wallet_balances  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _market_metadata  # type: ignore[method-assign]

    asyncio.run(engine.refresh_live_status())

    payload = state.read()
    assert payload["paused"] is False
    assert payload["manual_resume_required"] is False


def test_repair_missing_live_positions_does_not_consume_currently_open_order(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    active_order = LiveOrder(
        local_decision_id="decision-1",
        local_order_id="order-1",
        client_order_id="client-1",
        exchange_order_id="ex1",
        strategy_name="wallet_follow",
        market_id="m1",
        token_id="t1",
        side="BUY",
        intended_price=0.47,
        intended_size=5.0,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        lifecycle_status=OrderLifecycleStatus.RESTING,
        last_exchange_status="LIVE",
        remaining_size=5.0,
        terminal_state=False,
    )
    LiveOrderStore(tmp_path / "live_orders.json").save([active_order])

    exchange_positions = [
        {
            "conditionId": "m1",
            "asset": "t1",
            "size": 5.0,
            "side": "BUY",
            "initialValue": 2.35,
            "currentValue": 2.35,
        }
    ]
    exchange_open_orders = [
        {
            "exchange_order_id": "ex1",
            "client_order_id": "client-1",
            "status": "LIVE",
            "market_id": "m1",
            "token_id": "t1",
            "side": "BUY",
            "size": 5.0,
            "filled_size": 0.0,
            "remaining_size": 5.0,
            "price": 0.47,
        }
    ]

    async def _market_metadata(market_id: str, token_id: str = "", market_slug: str = "", outcome: str = ""):
        return {"market_id": market_id, "token_id": token_id, "category": "politics"}

    engine.market_data.fetch_market_metadata = _market_metadata  # type: ignore[method-assign]

    live_orders = LiveOrderStore(tmp_path / "live_orders.json").load()
    live_positions: list[Position] = []
    asyncio.run(
        engine._repair_missing_live_positions(
            live_positions=live_positions,
            live_orders=live_orders,
            exchange_positions=exchange_positions,
            exchange_open_orders=exchange_open_orders,
        )
    )

    repaired_orders = LiveOrderStore(tmp_path / "live_orders.json").load()
    assert repaired_orders[0].lifecycle_status == OrderLifecycleStatus.RESTING
    assert repaired_orders[0].terminal_state is False
    assert repaired_orders[0].filled_size == 0.0
    assert repaired_orders[0].linked_position_id == ""
    assert len(live_positions) == 1


def test_repair_missing_live_positions_syncs_existing_quantity_from_exchange(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    position = Position(
        position_id="pos-1",
        mode=Mode.LIVE,
        strategy_name="wallet_follow",
        wallet_address="0xabc",
        source_wallet="0xabc",
        market_id="m1",
        token_id="t1",
        category="crypto price",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        entry_price=0.48,
        current_mark_price=0.48,
        quantity=5.0,
        notional=2.4,
        fees_paid=0.0,
        source_trade_timestamp=datetime.now(timezone.utc),
        entry_reason="OK",
        cluster_confirmed=False,
        hedge_suspicion_score=0.0,
        side="BUY",
        entry_order_ids=["entry-order"],
        entry_price_estimated=0.48,
        entry_price_actual=0.48,
        stop_loss_rule="10pct",
        take_profit_rule="15pct",
        time_stop_rule="48h",
        source_exit_following_enabled=True,
        exit_state="OPEN",
        last_reconciled_at=datetime.now(timezone.utc),
        entry_time=datetime.now(timezone.utc),
        entry_size=5.0,
        exited_size=0.0,
        remaining_size=5.0,
    )
    live_positions = [position]

    asyncio.run(
        engine._repair_missing_live_positions(
            live_positions=live_positions,
            live_orders=[],
            exchange_positions=[
                {
                    "conditionId": "m1",
                    "asset": "t1",
                    "size": 4.9632,
                    "side": "BUY",
                    "initialValue": 2.382336,
                    "currentValue": 2.382336,
                }
            ],
            exchange_open_orders=[],
        )
    )

    assert live_positions[0].quantity == 4.9632
    assert live_positions[0].entry_size == 4.9632
    assert live_positions[0].remaining_size == 4.9632


def test_repair_missing_live_positions_closes_stale_local_position_absent_from_exchange(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    stale_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    position = Position(
        position_id="pos-1",
        mode=Mode.LIVE,
        strategy_name="wallet_follow",
        wallet_address="0xabc",
        source_wallet="0xabc",
        market_id="m1",
        token_id="t1",
        category="politics",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        entry_price=0.47,
        current_mark_price=0.47,
        quantity=5.0,
        notional=2.35,
        fees_paid=0.0,
        source_trade_timestamp=stale_time,
        entry_reason="OK",
        cluster_confirmed=False,
        hedge_suspicion_score=0.0,
        side="BUY",
        entry_order_ids=["entry-order"],
        entry_price_estimated=0.47,
        entry_price_actual=0.47,
        stop_loss_rule="10pct",
        take_profit_rule="15pct",
        time_stop_rule="48h",
        source_exit_following_enabled=True,
        exit_state="OPEN",
        last_reconciled_at=stale_time,
        entry_time=stale_time,
        entry_size=5.0,
        exited_size=0.0,
        remaining_size=5.0,
        opened_at=stale_time,
    )
    live_positions = [position]

    asyncio.run(
        engine._repair_missing_live_positions(
            live_positions=live_positions,
            live_orders=[],
            exchange_positions=[
                {
                    "conditionId": "m2",
                    "asset": "t2",
                    "size": 5.0,
                    "side": "BUY",
                    "initialValue": 2.5,
                    "currentValue": 2.5,
                }
            ],
            exchange_open_orders=[],
        )
    )

    assert live_positions[0].closed is True
    assert live_positions[0].remaining_size == 0.0
    assert live_positions[0].exit_reason == "EXCHANGE_POSITION_MISSING"


def test_repair_missing_live_positions_closes_duplicate_local_slice_when_exchange_has_single_net_position(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    older = datetime.now(timezone.utc) - timedelta(minutes=20)
    newer = datetime.now(timezone.utc) - timedelta(minutes=5)
    first = Position(
        position_id="pos-1",
        mode=Mode.LIVE,
        strategy_name="wallet_follow",
        wallet_address="0xabc",
        source_wallet="0xabc",
        market_id="m1",
        token_id="t1",
        category="crypto price",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        entry_price=0.48,
        current_mark_price=0.48,
        quantity=5.0,
        notional=2.4,
        fees_paid=0.0,
        source_trade_timestamp=older,
        entry_reason="OK",
        cluster_confirmed=False,
        hedge_suspicion_score=0.0,
        side="BUY",
        entry_order_ids=["entry-order-1"],
        entry_price_estimated=0.48,
        entry_price_actual=0.48,
        stop_loss_rule="10pct",
        take_profit_rule="15pct",
        time_stop_rule="48h",
        source_exit_following_enabled=True,
        exit_state="OPEN",
        last_reconciled_at=older,
        entry_time=older,
        entry_size=5.0,
        exited_size=0.0,
        remaining_size=5.0,
        opened_at=older,
    )
    second = first.model_copy(
        update={
            "position_id": "pos-2",
            "wallet_address": "0xdef",
            "source_wallet": "0xdef",
            "entry_order_ids": ["entry-order-2"],
            "source_trade_timestamp": newer,
            "entry_time": newer,
            "opened_at": newer,
            "last_reconciled_at": newer,
        }
    )
    live_positions = [first, second]

    asyncio.run(
        engine._repair_missing_live_positions(
            live_positions=live_positions,
            live_orders=[],
            exchange_positions=[
                {
                    "conditionId": "m1",
                    "asset": "t1",
                    "size": 5.0,
                    "side": "BUY",
                    "initialValue": 2.4,
                    "currentValue": 2.4,
                }
            ],
            exchange_open_orders=[],
        )
    )

    assert live_positions[0].closed is False
    assert live_positions[0].remaining_size == 5.0
    assert live_positions[1].closed is True
    assert live_positions[1].remaining_size == 0.0
    assert live_positions[1].exit_reason == "EXCHANGE_NET_POSITION_REDUCED"


def test_apply_live_exits_does_not_pause_on_missing_live_orderbook(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write({"paused": False, "manual_resume_required": False, "manual_live_enable": True})
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    position = Position(
        position_id="pos-1",
        mode=config.mode,
        strategy_name="wallet_follow",
        wallet_address="0xabc",
        source_wallet="0xabc",
        market_id="m1",
        token_id="t1",
        category="politics",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        entry_price=0.45,
        current_mark_price=0.45,
        quantity=5.0,
        notional=2.25,
        fees_paid=0.0,
        source_trade_timestamp=datetime.now(timezone.utc),
        entry_reason="entry",
        cluster_confirmed=False,
        hedge_suspicion_score=0.0,
        side="BUY",
        entry_order_ids=[],
        entry_price_estimated=0.45,
        entry_price_actual=0.45,
        stop_loss_rule="10pct",
        take_profit_rule="15pct",
        time_stop_rule="48h",
        source_exit_following_enabled=True,
        exit_state="OPEN",
        last_reconciled_at=datetime.now(timezone.utc),
        entry_time=datetime.now(timezone.utc),
        entry_size=5.0,
        exited_size=0.0,
        remaining_size=5.0,
    )

    async def _missing_orderbook(token_id: str):
        raise RuntimeError("Unable to fetch real orderbook in LIVE mode: 404")

    engine.client.get_orderbook = _missing_orderbook  # type: ignore[method-assign]

    asyncio.run(engine._apply_live_exits([position], []))

    payload = state.read()
    assert payload["paused"] is False
    assert position.exit_state == "EXIT_DATA_UNAVAILABLE"
    assert position.source_exit_following_enabled is False


def test_refresh_live_status_keeps_legacy_cancelled_partial_terminal_after_repair(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    legacy_order = LiveOrder(
        local_decision_id="decision-1",
        local_order_id="order-1",
        client_order_id="client-1",
        exchange_order_id="ex1",
        strategy_name="wallet_follow",
        market_id="m1",
        token_id="t1",
        side="BUY",
        intended_price=0.46,
        intended_size=5.0,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        lifecycle_status=OrderLifecycleStatus.CANCELLED,
        last_exchange_status="CANCELLED",
        remaining_size=0.0,
        filled_size=0.0,
        cancel_requested=True,
        cancel_confirmed=True,
        terminal_state=True,
    )
    LiveOrderStore(tmp_path / "live_orders.json").save([legacy_order])

    exchange_positions = [
        {
            "conditionId": "m1",
            "asset": "t1",
            "size": 4.99603,
            "side": "BUY",
            "initialValue": 0.2997618,
            "currentValue": 0.2997618,
        }
    ]

    async def _market_metadata(market_id: str, token_id: str = "", market_slug: str = "", outcome: str = ""):
        return {"market_id": market_id, "token_id": token_id, "category": "crypto price"}

    engine.client.get_positions = lambda: None  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _market_metadata  # type: ignore[method-assign]

    asyncio.run(
        engine._repair_missing_live_positions(
            live_positions=[],
            live_orders=LiveOrderStore(tmp_path / "live_orders.json").load(),
            exchange_positions=exchange_positions,
        )
    )

    repaired_orders = LiveOrderStore(tmp_path / "live_orders.json").load()
    assert repaired_orders[0].lifecycle_status == OrderLifecycleStatus.PARTIALLY_FILLED
    assert repaired_orders[0].terminal_state is True
    assert repaired_orders[0].remaining_size == 0.0


def test_refresh_live_status_closes_position_when_exit_order_is_already_filled(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    state.write({"paused": True, "pause_reason": "Live readiness gate failed.", "manual_live_enable": True})
    engine = LiveTradingEngine(config, tmp_path, state, GeoblockChecker(config))

    (tmp_path / "positions.json").write_text(
        __import__("json").dumps(
            {
                "paper": [],
                "live": [
                    {
                        "position_id": "pos-1",
                        "mode": "LIVE",
                        "strategy_name": "wallet_follow",
                        "wallet_address": "exchange-repaired",
                        "source_wallet": "exchange-repaired",
                        "market_id": "m1",
                        "token_id": "t1",
                        "category": "entertainment / pop culture",
                        "entry_style": "PASSIVE_LIMIT",
                        "entry_price": 0.45,
                        "current_mark_price": 0.45,
                        "quantity": 5.0,
                        "notional": 2.25,
                        "fees_paid": 0.0,
                        "source_trade_timestamp": datetime.now(timezone.utc).isoformat(),
                        "opened_at": datetime.now(timezone.utc).isoformat(),
                        "realized_pnl": 0.0,
                        "unrealized_pnl": 0.0,
                        "entry_reason": "LIVE_POSITION_REPAIRED",
                        "exit_reason": "",
                        "cluster_confirmed": False,
                        "hedge_suspicion_score": 0.0,
                        "closed": False,
                        "side": "BUY",
                        "entry_order_ids": ["entry-order"],
                        "entry_price_estimated": 0.45,
                        "entry_price_actual": 0.45,
                        "stop_loss_rule": "10pct",
                        "take_profit_rule": "15pct",
                        "time_stop_rule": "48h",
                        "source_exit_following_enabled": True,
                        "exit_state": "EXIT_ORDER_OPEN",
                        "last_reconciled_at": datetime.now(timezone.utc).isoformat(),
                        "closed_at": None,
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                        "entry_size": 5.0,
                        "exited_size": 0.0,
                        "remaining_size": 5.0,
                        "exit_order_ids": ["exit-order"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    filled_exit = LiveOrder(
        local_decision_id="exit-decision",
        local_order_id="exit-order",
        client_order_id="exit-client",
        exchange_order_id="ex-exit",
        strategy_name="wallet_follow",
        market_id="m1",
        token_id="t1",
        side="SELL",
        intended_price=0.01,
        intended_size=5.0,
        entry_style=EntryStyle.PASSIVE_LIMIT,
        lifecycle_status=OrderLifecycleStatus.FILLED,
        last_exchange_status="MATCHED",
        filled_size=5.0,
        average_fill_price=0.01,
        remaining_size=0.0,
        terminal_state=True,
        is_exit=True,
        linked_position_id="pos-1",
    )
    LiveOrderStore(tmp_path / "live_orders.json").save([filled_exit])

    async def _startup_validation():
        return {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_visible": True,
            "allowance_sufficient": True,
            "allowance_available": 5.0,
            "allowance_raw_summary": {},
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 5.0,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "0 orders visible",
            "positions_visible": True,
            "positions_count": 0,
            "positions_query_error": "",
            "positions_detail": "0 positions visible",
            "tradability_ok": True,
            "tradability_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "NONE",
            "reconciliation_summary": {"clean": True, "severity": "NONE", "issues": []},
        }

    async def _collect_health(reconciliation_override=None):
        return engine.health_monitor.aggregate([HealthComponent(name="market_ws", state=HealthState.HEALTHY, detail="ok")])

    async def _positions():
        return []

    async def _wallet_balances():
        return {"usdc": 0.0, "usdce": 5.0, "total_stablecoins": 5.0}

    engine.startup_validation = _startup_validation  # type: ignore[method-assign]
    engine.collect_health = _collect_health  # type: ignore[method-assign]
    engine.client.get_positions = _positions  # type: ignore[method-assign]
    engine.client.get_wallet_stablecoin_balances = _wallet_balances  # type: ignore[method-assign]

    asyncio.run(engine.refresh_live_status())

    repaired_positions = read_json(tmp_path / "positions.json", {"paper": [], "live": []})["live"]
    assert repaired_positions[0]["closed"] is True
    assert repaired_positions[0]["remaining_size"] == 0.0
    assert repaired_positions[0]["exit_state"] == "CLOSED"
