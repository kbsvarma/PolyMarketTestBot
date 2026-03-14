from pathlib import Path

from src.config import load_config
from src.health import HealthMonitor
from src.live_readiness import build_readiness_result
from src.models import HealthComponent, HealthState
from src.state import AppStateStore


def test_live_readiness_requires_manual_enable(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    config = config.model_copy(update={"mode": "LIVE"})
    store = AppStateStore(tmp_path / "app_state.json")
    store.write({"mode": "LIVE", "manual_live_enable": False, "kill_switch": False, "paused": False})
    health = HealthMonitor(tmp_path / "health_status.json").aggregate(
        [HealthComponent(name="auth", state=HealthState.HEALTHY, detail="ok")]
    )
    readiness = build_readiness_result(
        config,
        store,
        health,
        {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_sufficient": True,
            "allowance_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "ok",
            "positions_visible": True,
            "positions_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "ok",
            "tradability_ok": True,
            "tradability_detail": "ok",
        },
    )
    assert not readiness.ready


def test_live_readiness_requires_wallet_balance_buffer(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    config = config.model_copy(update={"mode": "LIVE"})
    store = AppStateStore(tmp_path / "app_state.json")
    store.write({"mode": "LIVE", "manual_live_enable": True, "kill_switch": False, "paused": False})
    health = HealthMonitor(tmp_path / "health_status.json").aggregate(
        [HealthComponent(name="auth", state=HealthState.HEALTHY, detail="ok")]
    )
    readiness = build_readiness_result(
        config,
        store,
        health,
        {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_sufficient": True,
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 0.5,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "ok",
            "positions_visible": True,
            "positions_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "ok",
            "tradability_ok": True,
            "tradability_detail": "ok",
        },
    )
    assert not readiness.ready
    assert any(check.name == "wallet_balance_sufficient" and not check.passed for check in readiness.checks)


def test_live_readiness_uses_config_manual_enable_as_confirmation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    store = AppStateStore(tmp_path / "app_state.json")
    store.write({"mode": "LIVE", "manual_live_enable": False, "kill_switch": False, "paused": False})
    health = HealthMonitor(tmp_path / "health_status.json").aggregate(
        [HealthComponent(name="auth", state=HealthState.HEALTHY, detail="ok")]
    )
    readiness = build_readiness_result(
        config,
        store,
        health,
        {
            "auth_valid": True,
            "auth_detail": "ok",
            "balance_visible": True,
            "balance_detail": "ok",
            "allowance_sufficient": True,
            "allowance_detail": "ok",
            "wallet_balance_visible": True,
            "wallet_balance_detail": "visible",
            "wallet_total_stablecoins": 5.0,
            "live_order_capable": True,
            "live_order_capable_detail": "ok",
            "open_orders_visible": True,
            "open_orders_detail": "ok",
            "positions_visible": True,
            "positions_detail": "ok",
            "rest_ok": True,
            "rest_detail": "ok",
            "reconciliation_clean": True,
            "reconciliation_detail": "ok",
            "tradability_ok": True,
            "tradability_detail": "ok",
        },
    )
    assert any(check.name == "manual_live_confirmation" and check.passed for check in readiness.checks)
