from pathlib import Path

from src.state import AppStateStore
from src.state_machine import SystemStateMachine
from src.models import SystemStatus


def test_live_state_machine_transitions() -> None:
    root = Path(__file__).resolve().parent.parent
    store = AppStateStore(root / "data" / "app_state.json")
    store.write({"mode": "LIVE", "system_status": "INIT", "paused": False})
    machine = SystemStateMachine(store)
    machine.transition(SystemStatus.LIVE_READY, "ready")
    assert store.read()["system_status"] == "LIVE_READY"
