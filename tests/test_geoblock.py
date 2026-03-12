from pathlib import Path

from src.config import load_config
from src.geoblock import GeoblockChecker


def test_geoblock_eligible_by_default() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    status = GeoblockChecker(config).preflight_status()
    assert status.eligible
