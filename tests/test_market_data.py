from __future__ import annotations

import asyncio
from pathlib import Path

from src.config import load_config
from src.market_data import MarketDataService
from src.models import OrderbookLevel, OrderbookSnapshot


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
    assert seen["limit"] == config.strategies.strategy_market_page_size


def test_live_tradability_falls_back_to_orderbook_depth(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    service = MarketDataService(config, tmp_path)

    async def _get_market_tradability(market_id: str, token_id: str):
        raise RuntimeError("metadata miss")

    async def _orderbook(token_id: str):
        return OrderbookSnapshot(
            token_id=token_id,
            bids=[OrderbookLevel(price=0.49, size=100.0)],
            asks=[OrderbookLevel(price=0.51, size=120.0)],
        )

    service.client.get_market_tradability = _get_market_tradability  # type: ignore[method-assign]
    service.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    tradability = asyncio.run(service.get_tradability("market-1", "token-1"))

    assert tradability["tradable"] is True
    assert tradability["orderbook_enabled"] is True
    assert tradability["derived_from_orderbook"] is True
