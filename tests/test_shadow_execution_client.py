from __future__ import annotations

import asyncio

from src.models import OrderbookLevel, OrderbookSnapshot
from src.shadow_execution_client import ShadowExecutionClient


class _BookClient:
    def __init__(self, asks: list[tuple[float, float]], bids: list[tuple[float, float]]) -> None:
        self._snapshot = OrderbookSnapshot(
            token_id="token-1",
            asks=[OrderbookLevel(price=p, size=s) for p, s in asks],
            bids=[OrderbookLevel(price=p, size=s) for p, s in bids],
        )

    async def get_orderbook(self, token_id: str) -> OrderbookSnapshot:
        return self._snapshot


def test_shadow_follow_taker_buy_matches_when_book_has_size() -> None:
    client = ShadowExecutionClient(
        _BookClient(
            asks=[(0.58, 6.0), (0.59, 5.0)],
            bids=[(0.57, 10.0)],
        )
    )

    result = asyncio.run(
        client.place_buy_order(
            token_id="token-1",
            price=0.59,
            size=10.0,
            entry_style="FOLLOW_TAKER",
            client_order_id="buy-1",
        )
    )

    assert result["status"] == "MATCHED"
    assert result["filled_size"] == 10.0
    assert result["average_fill_price"] > 0.58


def test_shadow_follow_taker_buy_cancels_when_book_is_too_thin() -> None:
    client = ShadowExecutionClient(
        _BookClient(
            asks=[(0.58, 3.0), (0.59, 2.0)],
            bids=[(0.57, 10.0)],
        )
    )

    result = asyncio.run(
        client.place_buy_order(
            token_id="token-1",
            price=0.59,
            size=10.0,
            entry_style="FOLLOW_TAKER",
            client_order_id="buy-2",
        )
    )

    assert result["status"] == "CANCELLED"
    assert result["filled_size"] == 0.0
    assert result["raw"]["miss_reason"] == "insufficient_marketable_depth"


def test_shadow_follow_taker_buy_reports_best_price_outside_limit() -> None:
    client = ShadowExecutionClient(
        _BookClient(
            asks=[(0.61, 20.0), (0.62, 20.0)],
            bids=[(0.57, 10.0)],
        )
    )

    result = asyncio.run(
        client.place_buy_order(
            token_id="token-1",
            price=0.60,
            size=10.0,
            entry_style="FOLLOW_TAKER",
            client_order_id="buy-3",
        )
    )

    assert result["status"] == "CANCELLED"
    assert result["filled_size"] == 0.0
    assert result["raw"]["best_book_price"] == 0.61
    assert result["raw"]["miss_reason"] == "best_price_above_limit"
