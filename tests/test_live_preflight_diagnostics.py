from __future__ import annotations

import asyncio
from pathlib import Path
from datetime import datetime, timezone

from src.geoblock import GeoblockChecker
from src.config import load_config
from src.live_engine import LiveTradingEngine
from src.live_orders import LiveOrderStore
from src.models import EntryStyle, LiveOrder, OrderLifecycleStatus
from src.models import HealthComponent, HealthState
from src.state import AppStateStore


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

    async def _collect_health():
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

    async def _collect_health():
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
