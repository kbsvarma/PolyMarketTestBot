from __future__ import annotations

from src.models import SystemStatus
from src.state import AppStateStore


ALLOWED_TRANSITIONS: dict[SystemStatus, set[SystemStatus]] = {
    SystemStatus.INIT: {SystemStatus.RESEARCH, SystemStatus.PAPER, SystemStatus.LIVE_READY, SystemStatus.PAUSED},
    SystemStatus.RESEARCH: {SystemStatus.PAPER, SystemStatus.PAUSED, SystemStatus.DEGRADED},
    SystemStatus.PAPER: {SystemStatus.RESEARCH, SystemStatus.LIVE_READY, SystemStatus.PAUSED, SystemStatus.DEGRADED},
    SystemStatus.LIVE_READY: {SystemStatus.LIVE_ACTIVE, SystemStatus.PAUSED, SystemStatus.DEGRADED, SystemStatus.RECONCILING},
    SystemStatus.LIVE_ACTIVE: {SystemStatus.RECONCILING, SystemStatus.PAUSED, SystemStatus.DEGRADED},
    SystemStatus.RECONCILING: {SystemStatus.LIVE_ACTIVE, SystemStatus.PAUSED, SystemStatus.DEGRADED},
    SystemStatus.DEGRADED: {SystemStatus.PAUSED, SystemStatus.RECONCILING, SystemStatus.RESEARCH, SystemStatus.PAPER},
    SystemStatus.PAUSED: {SystemStatus.RESEARCH, SystemStatus.PAPER, SystemStatus.LIVE_READY},
}


class SystemStateMachine:
    def __init__(self, store: AppStateStore) -> None:
        self.store = store

    def transition(self, new_status: SystemStatus, reason: str) -> None:
        current_raw = self.store.read().get("system_status", SystemStatus.INIT.value)
        current = SystemStatus(current_raw)
        if new_status not in ALLOWED_TRANSITIONS.get(current, set()):
            raise ValueError(f"Invalid state transition: {current.value} -> {new_status.value}")
        self.store.update_system_status(system_status=new_status.value, status=new_status.value, last_transition_reason=reason)
