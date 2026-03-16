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
    "allowance_available": 0.0,
    "allowance_sdk_signature_type": None,
    "allowance_configured_signature_type": None,
    "heartbeat_ok": False,
    "balance_visible": False,
    "reconciliation_clean": False,
    "paper_bankroll_override": 0.0,
    "paper_trade_notional_override": 0.0,
    "paper_run_enabled": False,
    "paper_summary": {},
    "wallet_discovery_state": "UNKNOWN",
    "wallet_discovery_reason": "",
    "wallet_discovery_source_quality": "DEGRADED_PUBLIC_DATA",
    "wallet_scoring_state": "UNKNOWN",
    "wallet_scoring_source_quality": "DEGRADED_PUBLIC_DATA",
}


class AppStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> dict[str, Any]:
        current = read_json(self.path, DEFAULT_STATE.copy())  # type: ignore[return-value]
        merged = DEFAULT_STATE.copy()
        merged.update(current)
        if not bool(merged.get("paused", False)) and not str(merged.get("pause_reason", "")):
            merged["manual_resume_required"] = False
        readiness = merged.get("live_readiness_last_result", {}) or {}
        system_status = str(merged.get("system_status", SystemStatus.INIT.value))
        if system_status == SystemStatus.LIVE_READY.value and not readiness.get("ready", False):
            merged["system_status"] = SystemStatus.DEGRADED.value
            merged["status"] = SystemStatus.DEGRADED.value
            merged["last_transition_reason"] = "Invalid LIVE_READY state downgraded because readiness is unresolved."
        return merged

    def write(self, payload: dict[str, Any]) -> None:
        write_json(self.path, payload)

    def update_system_status(self, **kwargs: Any) -> None:
        payload = self.read()
        payload.update(kwargs)
        if not bool(payload.get("paused", False)) and not str(payload.get("pause_reason", "")):
            payload["manual_resume_required"] = False
        readiness = payload.get("live_readiness_last_result", {}) or {}
        if payload.get("system_status") == SystemStatus.LIVE_READY.value and (
            not isinstance(readiness, dict) or not readiness or not readiness.get("ready", False)
        ):
            payload["system_status"] = SystemStatus.DEGRADED.value
            payload["status"] = SystemStatus.DEGRADED.value
            payload["last_transition_reason"] = "LIVE_READY rejected because readiness result is empty or false."
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
