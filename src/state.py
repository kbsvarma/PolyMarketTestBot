from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models import SystemStatus
from src.utils import read_json, write_json


DEFAULT_STATE: dict[str, Any] = {
    "mode": "RESEARCH",
    "paused": False,
    "pause_reason": "",
    "system_status": SystemStatus.INIT.value,
    "kill_switch": False,
    "manual_live_enable": False,
    "manual_resume_required": False,
    "live_health_state": "DEGRADED",
    "live_readiness_last_result": {},
    "last_transition_reason": "",
    "watched_market_tradability_cache": {},
    "unresolved_live_order_ids": [],
    "allowance_visible": False,
    "allowance_sufficient": False,
    "heartbeat_ok": False,
    "balance_visible": False,
    "reconciliation_clean": False,
    "paper_bankroll_override": 0.0,
    "paper_trade_notional_override": 0.0,
    "paper_run_enabled": False,
    "paper_summary": {},
}


class AppStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> dict[str, Any]:
        current = read_json(self.path, DEFAULT_STATE.copy())  # type: ignore[return-value]
        merged = DEFAULT_STATE.copy()
        merged.update(current)
        return merged

    def write(self, payload: dict[str, Any]) -> None:
        write_json(self.path, payload)

    def update_system_status(self, **kwargs: Any) -> None:
        payload = self.read()
        payload.update(kwargs)
        self.write(payload)

    def set_wallets(self, approved_wallets: object, scorecards: object) -> None:
        payload = self.read()
        payload["approved_wallets"] = approved_wallets.model_dump() if hasattr(approved_wallets, "model_dump") else approved_wallets
        payload["wallet_scorecards_loaded"] = len(scorecards)
        self.write(payload)

    def pause(self, reason: str) -> None:
        payload = self.read()
        payload["paused"] = True
        payload["pause_reason"] = reason
        payload["manual_resume_required"] = True
        payload["system_status"] = SystemStatus.PAUSED.value
        self.write(payload)

    def clear_pause(self) -> None:
        payload = self.read()
        payload["paused"] = False
        payload["pause_reason"] = ""
        payload["manual_resume_required"] = False
        self.write(payload)
