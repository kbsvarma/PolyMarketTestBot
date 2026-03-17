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
        self._cache_last_refreshed_at: datetime | None = None
        self._CACHE_REFRESH_INTERVAL_SECONDS = 300  # 5 minutes

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
        self._cache_last_refreshed_at = datetime.now(timezone.utc)
        return self.market_cache

    async def fetch_orderbook(self, token_id: str) -> OrderbookSnapshot:
        return await self.client.get_orderbook(token_id)

    async def fetch_market_metadata(self, market_id: str, token_id: str = "", market_slug: str = "", outcome: str = "") -> dict[str, object]:
        # Quick cache check first.
        market = self.market_cache.get(market_id)
        if market is None and token_id:
            market = self.token_cache.get(token_id)
        if market:
            return market.model_dump()

        # Only refresh the full market list if the cache is stale (older than ~5 min).
        # Avoids expensive bulk fetches every cycle for short-term markets that will
        # never appear in the standard Gamma /markets endpoint (5-min BTC/ETH markets etc).
        now = datetime.now(timezone.utc)
        cache_age_s = float("inf") if self._cache_last_refreshed_at is None else (now - self._cache_last_refreshed_at).total_seconds()
        if cache_age_s > self._CACHE_REFRESH_INTERVAL_SECONDS:
            await self.refresh_markets()
            market = self.market_cache.get(market_id)
            if market is None and token_id:
                market = self.token_cache.get(token_id)
            if market:
                return market.model_dump()

        # Try a direct API lookup — but only when token_id is NOT already provided.
        # When both market_id AND token_id are present (e.g. momentum signals), the stub
        # below is sufficient: the orderbook and tradability checks will gate the trade.
        # Skipping the API call avoids 1-2s latency per signal which causes strategy timeouts.
        if market_id and not token_id:
            try:
                looked_up = await self.client.fetch_market_lookup(market_id, token_id, market_slug=market_slug, outcome=outcome)
            except TypeError:
                try:
                    looked_up = await self.client.fetch_market_lookup(market_id, token_id)
                except Exception:
                    looked_up = None
            except Exception:
                looked_up = None
            if looked_up is not None:
                self.market_cache[looked_up.market_id] = looked_up
                self.token_cache[looked_up.token_id] = looked_up
                return looked_up.model_dump()

        # Market not found in cache or API (e.g. short-term markets not surfaced by Gamma endpoint).
        # Return a minimal stub so other gates (orderbook depth, tradability) can decide.
        # Expired/closed markets will be naturally blocked by empty orderbooks.
        return {
            "market_id": market_id,
            "token_id": token_id or "",
            "title": market_id,
            "category": "unknown",
            "active": True,
            "tradable": bool(token_id),
        }

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
