from __future__ import annotations

import asyncio

from src.config import AppConfig


class AppScheduler:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    async def ticks(self):
        while True:
            await asyncio.sleep(self.config.runtime.polling_interval_seconds)
            yield True
