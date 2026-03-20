from __future__ import annotations

import asyncio
from pathlib import Path

from src.config import load_config
from src.polymarket_client import PolymarketClient


def _config():
    root = Path(__file__).resolve().parent.parent
    return load_config(root / "config.yaml", root / ".env.example")


def test_infer_binary_market_winner_from_outcome_prices() -> None:
    client = PolymarketClient(_config())

    result = client._infer_binary_market_winner(
        {
            "outcomes": '["Up","Down"]',
            "outcomePrices": "[\"1\",\"0\"]",
            "closed": True,
            "active": False,
            "acceptingOrders": False,
            "umaResolutionStatus": "resolved",
        }
    )

    assert result["resolved_yes"] is True
    assert result["source"] == "market_outcome_prices"


def test_fetch_market_resolution_uses_slug_endpoint_and_parses_winner() -> None:
    client = PolymarketClient(_config())

    async def _fake_get_json(url: str, params=None):
        assert "markets/slug/btc-updown-5m-123" in url
        return {
            "conditionId": "mkt-1",
            "slug": "btc-updown-5m-123",
            "outcomes": '["Up","Down"]',
            "outcomePrices": "[\"0\",\"1\"]",
            "closed": True,
            "active": False,
            "acceptingOrders": False,
        }

    client._get_json = _fake_get_json  # type: ignore[method-assign]

    result = asyncio.run(client.fetch_market_resolution(market_slug="btc-updown-5m-123"))

    assert result is not None
    assert result["market_slug"] == "btc-updown-5m-123"
    assert result["resolved_yes"] is False
    assert result["source"] == "market_outcome_prices"
