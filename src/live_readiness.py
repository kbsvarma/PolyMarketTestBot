from __future__ import annotations

from src.config import AppConfig
from src.models import HealthState, LiveReadinessResult, ReadinessCheck, SystemHealth
from src.state import AppStateStore


def build_readiness_result(config: AppConfig, state: AppStateStore, health: SystemHealth, client_checks: dict[str, bool | str]) -> LiveReadinessResult:
    current = state.read()
    mode_value = config.mode.value if hasattr(config.mode, "value") else str(config.mode)
    checks = [
        ReadinessCheck(name="auth_valid", passed=bool(client_checks.get("auth_valid")), detail=str(client_checks.get("auth_detail", ""))),
        ReadinessCheck(name="balance_visible", passed=bool(client_checks.get("balance_visible")), detail=str(client_checks.get("balance_detail", ""))),
        ReadinessCheck(name="allowance_sufficient", passed=bool(client_checks.get("allowance_sufficient")), detail=str(client_checks.get("allowance_detail", ""))),
        ReadinessCheck(name="open_orders_query", passed=bool(client_checks.get("open_orders_visible")), detail=str(client_checks.get("open_orders_detail", ""))),
        ReadinessCheck(name="positions_query", passed=bool(client_checks.get("positions_visible")), detail=str(client_checks.get("positions_detail", ""))),
        ReadinessCheck(name="market_ws_healthy", passed=health.overall == HealthState.HEALTHY, detail=health.summary),
        ReadinessCheck(name="rest_health_ok", passed=bool(client_checks.get("rest_ok")), detail=str(client_checks.get("rest_detail", ""))),
        ReadinessCheck(name="reconciliation_clean", passed=bool(client_checks.get("reconciliation_clean")), detail=str(client_checks.get("reconciliation_detail", ""))),
        ReadinessCheck(name="kill_switch_off", passed=not current.get("kill_switch", config.live.global_kill_switch), detail="Kill switch must be off."),
        ReadinessCheck(name="manual_live_confirmation", passed=bool(current.get("manual_live_enable", config.live.manual_live_enable)), detail="Manual live enable required."),
        ReadinessCheck(name="mode_live", passed=mode_value == "LIVE", detail="Config mode must be LIVE."),
        ReadinessCheck(name="allowed_categories_configured", passed=bool(config.live.selected_categories), detail="At least one live category required."),
        ReadinessCheck(name="preferred_live_entry_style", passed=bool(config.entry_styles.preferred_live_entry_style), detail="Preferred live entry style required."),
        ReadinessCheck(name="no_unresolved_pause_reason", passed=not current.get("paused", False), detail=str(current.get("pause_reason", ""))),
        ReadinessCheck(name="market_tradability_lookup", passed=bool(client_checks.get("tradability_ok")), detail=str(client_checks.get("tradability_detail", ""))),
    ]
    return LiveReadinessResult(ready=all(check.passed for check in checks), checks=checks)
