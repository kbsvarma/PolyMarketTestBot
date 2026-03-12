from __future__ import annotations

from pathlib import Path

from src.config import AppConfig
from src.logger import logger


class AlertManager:
    def __init__(self, config: AppConfig, log_dir: Path) -> None:
        self.config = config
        self.log_dir = log_dir

    def emit_health_alerts(self, state: dict) -> None:
        if self.config.alerts.enable_console_alerts and state.get("paused"):
            logger.warning("System paused: {}", state.get("pause_reason", "unknown"))
