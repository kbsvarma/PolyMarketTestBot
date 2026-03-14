from __future__ import annotations

import asyncio
from pathlib import Path

from src.config import load_config
from src.market_data import MarketDataService


def test_live_market_refresh_uses_wider_limit(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    service = MarketDataService(config, tmp_path)
    seen: dict[str, int] = {}

    async def _fetch_markets(limit: int = 100):
        seen["limit"] = limit
        return []

    service.client.fetch_markets = _fetch_markets  # type: ignore[method-assign]
    asyncio.run(service.refresh_markets())
    assert seen["limit"] == 500
