from __future__ import annotations

from pathlib import Path

import pytest

from src.models import SystemStatus
from src.state import AppStateStore
from src.state_machine import SystemStateMachine


def test_state_store_downgrades_invalid_live_ready(tmp_path: Path) -> None:
    store = AppStateStore(tmp_path / "app_state.json")
    store.write(
        {
            "system_status": SystemStatus.LIVE_READY.value,
            "live_readiness_last_result": {},
        }
    )
    payload = store.read()
    assert payload["system_status"] == SystemStatus.DEGRADED.value


def test_state_machine_blocks_live_ready_without_readiness(tmp_path: Path) -> None:
    store = AppStateStore(tmp_path / "app_state.json")
    store.write(
        {
            "system_status": SystemStatus.PAUSED.value,
            "live_readiness_last_result": {},
        }
    )
    machine = SystemStateMachine(store)
    with pytest.raises(ValueError, match="Cannot transition to LIVE_READY"):
        machine.transition(SystemStatus.LIVE_READY, "test")


def test_state_machine_blocks_live_ready_with_empty_readiness_dict(tmp_path: Path) -> None:
    store = AppStateStore(tmp_path / "app_state.json")
    store.write({"system_status": SystemStatus.PAUSED.value, "live_readiness_last_result": {}})
    machine = SystemStateMachine(store)
    with pytest.raises(ValueError, match="Cannot transition to LIVE_READY"):
        machine.transition(SystemStatus.LIVE_READY, "test")


def test_state_machine_allows_live_ready_with_positive_readiness(tmp_path: Path) -> None:
    store = AppStateStore(tmp_path / "app_state.json")
    store.write(
        {
            "system_status": SystemStatus.PAUSED.value,
            "live_readiness_last_result": {"ready": True},
        }
    )
    machine = SystemStateMachine(store)
    machine.transition(SystemStatus.LIVE_READY, "test")
    assert store.read()["system_status"] == SystemStatus.LIVE_READY.value
