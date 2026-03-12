from __future__ import annotations

from pathlib import Path

from src.config import AppConfig
from src.market_ws import MarketWSClient
from src.models import MarketInfo, OrderbookSnapshot
from src.polymarket_client import PolymarketClient


class MarketDataService:
    def __init__(self, config: AppConfig, data_dir: Path) -> None:
        self.config = config
        self.data_dir = data_dir
        self.client = PolymarketClient(config)
        self.ws = MarketWSClient()
        self.market_cache: dict[str, MarketInfo] = {}

    async def refresh_markets(self) -> dict[str, MarketInfo]:
        markets = await self.client.fetch_markets(limit=100)
        self.market_cache = {market.market_id: market for market in markets}
        return self.market_cache

    async def fetch_orderbook(self, token_id: str) -> OrderbookSnapshot:
        return await self.client.get_orderbook(token_id)

    async def fetch_market_metadata(self, market_id: str) -> dict[str, object]:
        if market_id not in self.market_cache:
            await self.refresh_markets()
        market = self.market_cache.get(market_id)
        if market:
            return market.model_dump()
        if self.config.mode.value == "LIVE":
            raise RuntimeError(f"Missing market metadata for {market_id} in LIVE mode.")
        return {"market_id": market_id, "title": market_id, "category": "unknown", "tradable": False}

    async def stream_watchlist(self, token_ids: list[str]) -> dict[str, dict]:
        if token_ids != self.ws.watched_token_ids or not self.ws.latest_messages:
            await self.ws.connect(token_ids)
        else:
            await self.ws.refresh()
        if self.config.mode.value == "LIVE" and not self.ws.connected:
            raise RuntimeError(f"Market websocket unhealthy: {self.ws.last_error}")
        return self.ws.latest_messages

    async def get_tradability(self, market_id: str, token_id: str) -> dict[str, object]:
        return await self.client.get_market_tradability(market_id, token_id)
