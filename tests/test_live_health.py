from pathlib import Path

from src.health import HealthMonitor
from src.models import HealthComponent, HealthState


def test_health_aggregation_degraded() -> None:
    root = Path(__file__).resolve().parent.parent
    monitor = HealthMonitor(root / "data" / "health_status.json")
    health = monitor.aggregate(
        [
            HealthComponent(name="auth", state=HealthState.HEALTHY, detail="ok"),
            HealthComponent(name="ws", state=HealthState.DEGRADED, detail="slow"),
        ]
    )
    assert health.overall == HealthState.DEGRADED
