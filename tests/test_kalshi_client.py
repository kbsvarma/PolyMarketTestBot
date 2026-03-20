from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.config import apply_crypto_direction_profile, load_config
from src.kalshi_client import KalshiClient


ROOT = Path(__file__).resolve().parents[1]


def _pem_private_key() -> str:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem.decode("utf-8")


def _kalshi_config():
    config = apply_crypto_direction_profile(load_config(ROOT / "config.yaml"), "kalshi_btc_15m")
    env = config.env.model_copy(
        update={
            "kalshi_api_key_id": "test-key",
            "kalshi_private_key": _pem_private_key(),
            "live_trading_enabled": True,
        }
    )
    return config.model_copy(update={"env": env})


def test_fetch_market_lookup_resolves_kalshi_window_market() -> None:
    config = _kalshi_config()
    window_open = int(datetime(2026, 3, 20, 2, 30, tzinfo=timezone.utc).timestamp())
    market = {
        "ticker": "KXBTC15M-26MAR192245-45",
        "title": "BTC price up in next 15 mins?",
        "status": "active",
        "open_time": "2026-03-20T02:30:00Z",
        "close_time": "2026-03-20T02:45:00Z",
        "yes_ask_dollars": "0.5900",
        "no_ask_dollars": "0.4200",
        "liquidity_dollars": "1000.0000",
        "volume_fp": "500.00",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/trade-api/v2/markets"
        assert request.url.params["series_ticker"] == "KXBTC15M"
        return httpx.Response(200, json={"markets": [market]})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.elections.kalshi.com",
        ) as http_client:
            client = KalshiClient(config, http_client=http_client)
            yes_info = await client.fetch_market_lookup(
                market_id="",
                market_slug=f"KXBTC15M-{window_open}",
                outcome="Up",
            )
            no_info = await client.fetch_market_lookup(
                market_id="",
                market_slug=f"KXBTC15M-{window_open}",
                outcome="Down",
            )
            assert yes_info is not None
            assert no_info is not None
            assert yes_info.market_id == market["ticker"]
            assert yes_info.token_id.endswith(":yes")
            assert no_info.token_id.endswith(":no")
            assert yes_info.slug == market["ticker"]

    asyncio.run(run())


def test_get_orderbook_builds_implied_asks_from_opposite_bids() -> None:
    config = _kalshi_config()
    ticker = "KXBTC15M-26MAR192245-45"
    market = {
        "ticker": ticker,
        "title": "BTC price up in next 15 mins?",
        "status": "active",
        "yes_ask_dollars": "0.4000",
        "yes_ask_size_fp": "4.00",
        "yes_bid_dollars": "0.3900",
        "yes_bid_size_fp": "7.00",
        "no_ask_dollars": "0.6100",
        "no_ask_size_fp": "7.00",
        "no_bid_dollars": "0.6000",
        "no_bid_size_fp": "4.00",
    }
    orderbook = {
        "orderbook_fp": {
            "yes_dollars": [["0.1200", "2.00"], ["0.3900", "7.00"]],
            "no_dollars": [["0.2500", "3.00"], ["0.6000", "4.00"]],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/orderbook"):
            return httpx.Response(200, json=orderbook)
        return httpx.Response(200, json={"market": market})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.elections.kalshi.com",
        ) as http_client:
            client = KalshiClient(config, http_client=http_client)
            snapshot = await client.get_orderbook(f"kalshi:{ticker}:yes")
            assert snapshot.bids[0].price == 0.39
            assert snapshot.bids[0].size == 7.0
            assert snapshot.asks[0].price == 0.4
            assert snapshot.asks[0].size == 4.0

    asyncio.run(run())


def test_fetch_market_resolution_refreshes_cached_window_market() -> None:
    config = _kalshi_config()
    window_open = int(datetime(2026, 3, 20, 2, 30, tzinfo=timezone.utc).timestamp())
    active_market = {
        "ticker": "KXBTC15M-26MAR192245-45",
        "title": "BTC price up in next 15 mins?",
        "status": "active",
        "result": "",
        "open_time": "2026-03-20T02:30:00Z",
        "close_time": "2026-03-20T02:45:00Z",
    }
    finalized_market = dict(active_market)
    finalized_market.update({"status": "finalized", "result": "yes"})
    call_count = {"markets": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["markets"] += 1
        payload = active_market if call_count["markets"] == 1 else finalized_market
        return httpx.Response(200, json={"markets": [payload]})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.elections.kalshi.com",
        ) as http_client:
            client = KalshiClient(config, http_client=http_client)
            info = await client.fetch_market_lookup("", market_slug=f"KXBTC15M-{window_open}", outcome="Up")
            assert info is not None
            resolution = await client.fetch_market_resolution(market_slug=f"KXBTC15M-{window_open}")
            assert resolution is not None
            assert resolution["resolved_yes"] is True
            assert call_count["markets"] == 2

    asyncio.run(run())


def test_place_buy_order_signs_and_normalizes_response() -> None:
    config = _kalshi_config()
    ticker = "KXBTC15M-26MAR192245-45"
    market = {
        "ticker": ticker,
        "title": "BTC price up in next 15 mins?",
        "status": "active",
        "fractional_trading_enabled": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"market": market})
        assert request.method == "POST"
        assert request.url.path == "/trade-api/v2/portfolio/orders"
        assert request.headers["KALSHI-ACCESS-KEY"] == "test-key"
        assert request.headers["KALSHI-ACCESS-SIGNATURE"]
        assert request.headers["KALSHI-ACCESS-TIMESTAMP"]
        body = json.loads(request.content.decode("utf-8"))
        assert body["ticker"] == ticker
        assert body["side"] == "yes"
        assert body["action"] == "buy"
        assert body["time_in_force"] == "fill_or_kill"
        assert body["yes_price_dollars"] == "0.5800"
        assert body["count"] == 10
        assert body["count_fp"] == "10.00"
        return httpx.Response(
            201,
            json={
                "order": {
                    "order_id": "kal-1",
                    "client_order_id": body["client_order_id"],
                    "ticker": ticker,
                    "side": "yes",
                    "action": "buy",
                    "type": "limit",
                    "status": "executed",
                    "yes_price_dollars": "0.5800",
                    "no_price_dollars": "0.4200",
                    "fill_count_fp": "10.00",
                    "remaining_count_fp": "0.00",
                    "initial_count_fp": "10.00",
                    "taker_fill_cost_dollars": "0.5800",
                    "maker_fill_cost_dollars": "0.0000",
                    "taker_fees_dollars": "0.0200",
                    "maker_fees_dollars": "0.0000",
                }
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.elections.kalshi.com",
        ) as http_client:
            client = KalshiClient(config, http_client=http_client)
            result = await client.place_buy_order(
                token_id=f"kalshi:{ticker}:yes",
                price=0.58,
                size=10.0,
                entry_style="FOLLOW_TAKER",
                client_order_id="client-1",
            )
            assert result["exchange_order_id"] == "kal-1"
            assert result["status"] == "FILLED"
            assert result["filled_size"] == 10.0
            assert result["average_fill_price"] == 0.58

    asyncio.run(run())
