from pathlib import Path

import pytest

from src.state import AppStateStore
from src.state_machine import SystemStateMachine
from src.models import SystemStatus


def test_live_state_machine_transitions() -> None:
    root = Path(__file__).resolve().parent.parent
    store = AppStateStore(root / "data" / "app_state.json")
    store.write({"mode": "LIVE", "system_status": "INIT", "paused": False, "live_readiness_last_result": {"ready": True}})
    machine = SystemStateMachine(store)
    machine.transition(SystemStatus.LIVE_READY, "ready")
    assert store.read()["system_status"] == "LIVE_READY"


def test_live_state_machine_rejects_live_ready_without_readiness() -> None:
    root = Path(__file__).resolve().parent.parent
    store = AppStateStore(root / "data" / "app_state.json")
    store.write({"mode": "LIVE", "system_status": "INIT", "paused": False, "live_readiness_last_result": {}})
    machine = SystemStateMachine(store)
    with pytest.raises(ValueError, match="Cannot transition to LIVE_READY"):
        machine.transition(SystemStatus.LIVE_READY, "ready")
