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


def test_state_machine_coerces_legacy_live_status_for_live_ready_transition(tmp_path: Path) -> None:
    store = AppStateStore(tmp_path / "app_state.json")
    store.write(
        {
            "system_status": "LIVE",
            "live_readiness_last_result": {"ready": True},
        }
    )
    machine = SystemStateMachine(store)
    machine.transition(SystemStatus.LIVE_READY, "test")
    assert store.read()["system_status"] == SystemStatus.LIVE_READY.value


def test_state_store_clears_stale_manual_resume_when_not_paused(tmp_path: Path) -> None:
    store = AppStateStore(tmp_path / "app_state.json")
    store.write(
        {
            "paused": False,
            "pause_reason": "",
            "manual_resume_required": True,
        }
    )
    payload = store.read()
    assert payload["manual_resume_required"] is False
