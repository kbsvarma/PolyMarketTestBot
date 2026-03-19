from __future__ import annotations

import asyncio

from src.config import CryptoDirectionConfig
from src.crypto_market_watcher import AssetWatcher, CryptoMarketWatcher, current_window_ts
from src.models import MarketInfo, OrderbookLevel, OrderbookSnapshot


class _FakeLookupClient:
    def __init__(self, mode: str = "both") -> None:
        self.mode = mode
        self.lookup_calls: list[tuple[str, str]] = []

    async def fetch_market_lookup(self, market_id: str, market_slug: str, outcome: str) -> MarketInfo:
        self.lookup_calls.append((market_slug, outcome))
        if self.mode == "missing_both":
            raise RuntimeError("missing")
        if self.mode == "missing_down" and outcome == "Down":
            raise RuntimeError("missing down")
        if self.mode == "missing_up" and outcome == "Up":
            raise RuntimeError("missing up")
        token = "yes-token" if outcome == "Up" else "no-token"
        return MarketInfo(
            market_id="market-1",
            token_id=token,
            title="Bitcoin Up or Down",
            slug=market_slug,
            category="crypto",
            outcome_name=outcome,
            liquidity=1000.0,
            volume=500.0,
        )

    async def get_orderbook(self, token_id: str) -> OrderbookSnapshot:
        price = 0.58 if token_id == "yes-token" else 0.42
        return OrderbookSnapshot(
            token_id=token_id,
            bids=[],
            asks=[OrderbookLevel(price=price, size=25.0)],
        )


def test_asset_watcher_requires_both_tokens_to_resolve_market() -> None:
    cfg = CryptoDirectionConfig(market_resolve_retry_seconds=5.0)
    watcher = AssetWatcher("BTC", cfg.btc, _FakeLookupClient(mode="missing_down"), cfg)

    resolved = asyncio.run(watcher.resolve_market(current_window_ts(cfg.window_duration_seconds)))

    assert resolved is False
    assert watcher.market_meta()["market_id"] == ""
    assert watcher.market_meta()["yes_token_id"] == ""
    assert watcher.market_meta()["no_token_id"] == ""
    assert watcher.yes_ask() == 0.0
    assert watcher.no_ask() == 0.0


def test_asset_watcher_retries_unresolved_market_during_same_window() -> None:
    cfg = CryptoDirectionConfig(market_resolve_retry_seconds=0.0)
    client = _FakeLookupClient(mode="missing_down")
    watcher = AssetWatcher("BTC", cfg.btc, client, cfg)
    window_ts = current_window_ts(cfg.window_duration_seconds)

    first = asyncio.run(watcher.resolve_market(window_ts))
    assert first is False

    client.mode = "both"
    asyncio.run(watcher.refresh_prices())

    assert watcher.market_meta()["market_id"] == "market-1"
    assert watcher.market_meta()["yes_token_id"] == "yes-token"
    assert watcher.market_meta()["no_token_id"] == "no-token"
    assert watcher.yes_ask() == 0.58
    assert watcher.no_ask() == 0.42


def test_window_transition_rolls_window_even_if_market_unresolved() -> None:
    cfg = CryptoDirectionConfig(track_btc=True, track_eth=False)
    market_watcher = CryptoMarketWatcher(cfg, _FakeLookupClient(mode="missing_both"))
    market_watcher._last_window_ts = current_window_ts(cfg.window_duration_seconds) - cfg.window_duration_seconds

    transitioned = asyncio.run(market_watcher.check_window_transition())

    assert transitioned == ["BTC"]


def test_asset_watcher_readiness_reason_is_specific() -> None:
    cfg = CryptoDirectionConfig()
    watcher = AssetWatcher("BTC", cfg.btc, _FakeLookupClient(mode="both"), cfg)

    assert watcher.readiness_reason() == "market_unresolved"

    asyncio.run(watcher.resolve_market(current_window_ts(cfg.window_duration_seconds)))
    assert watcher.readiness_reason() == "asset_feed_stale"
