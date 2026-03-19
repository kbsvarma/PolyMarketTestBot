from __future__ import annotations

import itertools
from typing import Any


class ShadowExecutionClient:
    """
    Dry-run order transport for the crypto bracket strategy.

    This class mirrors the small subset of the Polymarket client API that the
    BracketExecutor uses, but it never touches the wallet. Instead, every
    order is evaluated against a fresh orderbook snapshot at submission time.

    The goal is not to predict queue dynamics perfectly. The goal is to answer
    the same question live FOK logic asks:
      "Could this order have crossed the book and filled right now?"
    """

    execution_mode = "shadow"

    def __init__(self, market_client: Any) -> None:
        self._client = market_client
        self._seq = itertools.count(1)
        self._orders: dict[str, dict[str, Any]] = {}

    async def place_buy_order(
        self,
        token_id: str,
        price: float,
        size: float,
        entry_style: str,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.place_order(
            token_id=token_id,
            price=price,
            size=size,
            side="BUY",
            entry_style=entry_style,
            client_order_id=client_order_id,
        )

    async def place_sell_order(
        self,
        token_id: str,
        price: float,
        size: float,
        entry_style: str,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.place_order(
            token_id=token_id,
            price=price,
            size=size,
            side="SELL",
            entry_style=entry_style,
            client_order_id=client_order_id,
        )

    async def place_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        entry_style: str,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        order_id = f"shadow-{next(self._seq)}"
        if entry_style == "FOLLOW_TAKER":
            payload = await self._simulate_fok_order(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
                order_id=order_id,
                client_order_id=client_order_id,
            )
        else:
            payload = {
                "exchange_order_id": order_id,
                "status": "LIVE",
                "filled_size": 0.0,
                "average_fill_price": 0.0,
                "remaining_size": float(size),
                "client_order_id": client_order_id or "",
                "price": float(price),
                "size": float(size),
                "side": side,
                "token_id": token_id,
                "raw": {
                    "mode": "shadow",
                    "reason": "PASSIVE_LIMIT_NOT_SIMULATED",
                },
            }
        self._orders[order_id] = dict(payload)
        return payload

    async def _simulate_fok_order(
        self,
        *,
        token_id: str,
        price: float,
        size: float,
        side: str,
        order_id: str,
        client_order_id: str | None,
    ) -> dict[str, Any]:
        orderbook = await self._client.get_orderbook(token_id)
        levels = orderbook.asks if side == "BUY" else orderbook.bids
        best_book_price = float(levels[0].price) if levels else 0.0

        remaining = float(size)
        spent = 0.0
        filled = 0.0
        marketable_depth = 0.0

        for level in levels:
            level_price = float(level.price)
            level_size = float(level.size)
            marketable = level_price <= price if side == "BUY" else level_price >= price
            if not marketable:
                break
            marketable_depth += level_size
            take = min(remaining, level_size)
            if take <= 0:
                continue
            spent += take * level_price
            filled += take
            remaining -= take
            if remaining <= 1e-9:
                break

        matched = remaining <= 1e-9
        average_fill_price = (spent / filled) if filled > 0 else 0.0
        if matched:
            miss_reason = ""
        elif not levels:
            miss_reason = "no_book"
        elif best_book_price <= 0:
            miss_reason = "no_book"
        elif side == "BUY" and best_book_price > price:
            miss_reason = "best_price_above_limit"
        elif side == "SELL" and best_book_price < price:
            miss_reason = "best_price_below_limit"
        else:
            miss_reason = "insufficient_marketable_depth"

        return {
            "exchange_order_id": order_id,
            "status": "MATCHED" if matched else "CANCELLED",
            "filled_size": round(filled if matched else 0.0, 6),
            "average_fill_price": round(average_fill_price if matched else 0.0, 6),
            "remaining_size": round(0.0 if matched else float(size), 6),
            "client_order_id": client_order_id or "",
            "price": float(price),
            "size": float(size),
            "side": side,
            "token_id": token_id,
            "raw": {
                "mode": "shadow",
                "requested_price": float(price),
                "requested_size": float(size),
                "filled_size": round(filled if matched else 0.0, 6),
                "average_fill_price": round(average_fill_price if matched else 0.0, 6),
                "best_book_price": round(best_book_price, 6),
                "marketable_depth": round(marketable_depth, 6),
                "miss_reason": miss_reason,
            },
        }

    async def get_order_status(self, exchange_order_id: str) -> dict[str, Any]:
        order = self._orders.get(exchange_order_id)
        if order is None:
            raise RuntimeError(f"Unknown shadow order: {exchange_order_id}")
        return dict(order)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        order = self._orders.get(order_id)
        if order is None:
            raise RuntimeError(f"Unknown shadow order: {order_id}")
        order["status"] = "CANCELLED"
        order["remaining_size"] = float(order.get("remaining_size") or 0.0)
        return {
            "exchange_order_id": order_id,
            "status": "CANCELLED",
            "raw": {"mode": "shadow"},
        }
