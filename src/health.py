from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.models import HealthComponent, HealthState, SystemHealth
from src.utils import write_json


class HealthMonitor:
    def __init__(self, path: Path) -> None:
        self.path = path

    def aggregate(self, components: list[HealthComponent]) -> SystemHealth:
        if any(component.state == HealthState.UNHEALTHY for component in components):
            overall = HealthState.UNHEALTHY
        elif any(component.state == HealthState.DEGRADED for component in components):
            overall = HealthState.DEGRADED
        else:
            overall = HealthState.HEALTHY
        summary = ", ".join(f"{component.name}:{component.state.value}" for component in components)
        health = SystemHealth(overall=overall, components=components, summary=summary, checked_at=datetime.now(timezone.utc))
        write_json(self.path, health.model_dump(mode="json"))
        return health
