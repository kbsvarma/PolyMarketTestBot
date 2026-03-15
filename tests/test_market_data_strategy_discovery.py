from __future__ import annotations

import asyncio
from pathlib import Path

from src.config import load_config
from src.market_data import MarketDataService
from src.models import MarketInfo


def _service(tmp_path: Path) -> MarketDataService:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    return MarketDataService(config, tmp_path)


def test_refresh_markets_pages_active_markets_for_strategy_discovery(tmp_path: Path) -> None:
    service = _service(tmp_path)
    calls: list[tuple[int, int, bool]] = []

    async def _fetch_markets(limit: int = 100, offset: int = 0, active_only: bool = False):
        calls.append((limit, offset, active_only))
        if offset == 0:
            return [
                MarketInfo(market_id="m1", token_id="t1", title="First", slug="first", category="politics"),
                MarketInfo(market_id="m2", token_id="t2", title="Second", slug="second", category="politics"),
            ]
        if offset == limit:
            return [
                MarketInfo(market_id="m3", token_id="t3", title="Third", slug="third", category="politics"),
            ]
        return []

    service.client.fetch_markets = _fetch_markets  # type: ignore[method-assign]

    markets = asyncio.run(service.refresh_markets())

    assert set(markets.keys()) == {"m1", "m2", "m3"}
    assert calls[0][2] is True
    assert calls[1][1] == calls[1][0]


def test_refresh_markets_prioritizes_near_resolution_active_markets(tmp_path: Path) -> None:
    service = _service(tmp_path)

    async def _fetch_markets(limit: int = 100, offset: int = 0, active_only: bool = False):
        if offset > 0:
            return []
        return [
            MarketInfo(
                market_id="far",
                token_id="t-far",
                title="Far",
                slug="far",
                category="politics",
                end_date_iso="2027-12-31T00:00:00Z",
                liquidity=5000.0,
            ),
            MarketInfo(
                market_id="near",
                token_id="t-near",
                title="Near",
                slug="near",
                category="politics",
                end_date_iso="2026-03-16T00:00:00Z",
                liquidity=1000.0,
            ),
        ]

    service.client.fetch_markets = _fetch_markets  # type: ignore[method-assign]

    asyncio.run(service.refresh_markets())

    ordered_ids = list(service.market_cache.keys())
    assert ordered_ids[0] == "near"
