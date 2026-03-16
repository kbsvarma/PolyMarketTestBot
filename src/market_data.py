from __future__ import annotations

from datetime import datetime, timezone
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
        self.token_cache: dict[str, MarketInfo] = {}

    def _market_rank(self, market: MarketInfo) -> tuple[int, float, float]:
        end_at = None
        if market.end_date_iso:
            try:
                end_at = datetime.fromisoformat(market.end_date_iso.replace("Z", "+00:00"))
            except ValueError:
                end_at = None
        now = datetime.now(timezone.utc)
        hours_to_resolution = 9_999_999.0
        if end_at is not None and end_at > now:
            hours_to_resolution = (end_at - now).total_seconds() / 3600.0
        active_rank = 0 if (market.active and not market.closed) else 1
        return (active_rank, hours_to_resolution, -float(market.liquidity or 0.0))

    async def refresh_markets(self) -> dict[str, MarketInfo]:
        page_size = self.config.strategies.strategy_market_page_size
        max_pages = (
            self.config.strategies.strategy_market_max_pages_live
            if self.config.mode.value == "LIVE"
            else self.config.strategies.strategy_market_max_pages_research
        )
        markets_by_token: dict[str, MarketInfo] = {}
        for page in range(max_pages):
            offset = page * page_size
            try:
                rows = await self.client.fetch_markets(limit=page_size, offset=offset, active_only=True)
            except TypeError:
                rows = await self.client.fetch_markets(limit=page_size)
            if not rows:
                break
            for market in rows:
                if market.token_id and market.token_id not in markets_by_token:
                    markets_by_token[market.token_id] = market
        markets = sorted(markets_by_token.values(), key=self._market_rank)
        self.market_cache = {market.market_id: market for market in markets}
        self.token_cache = {market.token_id: market for market in markets}
        return self.market_cache

    async def fetch_orderbook(self, token_id: str) -> OrderbookSnapshot:
        return await self.client.get_orderbook(token_id)

    async def fetch_market_metadata(self, market_id: str, token_id: str = "", market_slug: str = "", outcome: str = "") -> dict[str, object]:
        if market_id not in self.market_cache and (not token_id or token_id not in self.token_cache):
            await self.refresh_markets()
        market = self.market_cache.get(market_id)
        if market is None and token_id:
            market = self.token_cache.get(token_id)
        if market is None:
            try:
                market = await self.client.fetch_market_lookup(market_id, token_id, market_slug=market_slug, outcome=outcome)
            except TypeError:
                market = await self.client.fetch_market_lookup(market_id, token_id)
            if market is not None:
                self.market_cache[market.market_id] = market
                self.token_cache[market.token_id] = market
        if market:
            return market.model_dump()
        if self.config.mode.value == "LIVE":
            raise RuntimeError(f"Missing market metadata for {market_id}/{token_id or 'unknown-token'} in LIVE mode.")
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
        try:
            return await self.client.get_market_tradability(market_id, token_id)
        except RuntimeError:
            if token_id and token_id not in {"unknown-token", "unknown", "None"}:
                try:
                    orderbook = await self.fetch_orderbook(token_id)
                except Exception:
                    raise
                has_depth = bool(orderbook.bids or orderbook.asks)
                market = self.market_cache.get(market_id) or self.token_cache.get(token_id)
                if market is None:
                    try:
                        market = await self.client.fetch_market_lookup(market_id, token_id)
                    except Exception:
                        market = None
                return {
                    "market_id": market_id,
                    "token_id": token_id,
                    "tradable": has_depth,
                    "orderbook_enabled": has_depth,
                    "category": getattr(market, "category", "unknown"),
                    "title": getattr(market, "title", market_id),
                    "liquidity": float(getattr(market, "liquidity", 0.0) or 0.0),
                    "derived_from_orderbook": True,
                }
            raise
