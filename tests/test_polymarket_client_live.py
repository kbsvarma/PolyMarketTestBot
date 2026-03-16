from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.config import load_config
import src.polymarket_client as polymarket_client_module
from src.polymarket_client import PolymarketClient


class _StubSdkClient:
    def __init__(self, result: dict[str, object]) -> None:
        self._result = result
        self.builder = type("Builder", (), {"sig_type": 0})()

    def create_order(self, order_args: object) -> dict[str, object]:
        return {"signed": True}

    def post_order(self, signed_order: object, order_type: object) -> dict[str, object]:
        return self._result

    def get_order(self, order_id: str) -> dict[str, object]:
        return {"id": order_id, "status": "RESTING"}

    def cancel(self, order_id: str) -> dict[str, object]:
        return {"id": order_id, "status": "CANCELLED"}

    def get_orders(self) -> list[dict[str, object]]:
        return []


def _live_config():
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    config = config.model_copy(update={"mode": "LIVE"})
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    return config


def test_live_order_capable_false_when_sdk_missing_methods() -> None:
    client = PolymarketClient(_live_config())
    client._sdk_client = object()
    ok, detail = client.live_order_capable()
    assert not ok
    assert "missing required live order methods" in detail.lower()


def test_place_order_requires_exchange_order_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyOrderArgs:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _DummyOrderType:
        GTC = "GTC"

    client = PolymarketClient(_live_config())
    client._sdk_client = _StubSdkClient({"status": "SUBMITTED"})
    monkeypatch.setattr(polymarket_client_module, "OrderArgs", _DummyOrderArgs)
    monkeypatch.setattr(polymarket_client_module, "OrderType", _DummyOrderType)
    with pytest.raises(RuntimeError, match="exchange_order_id"):
        asyncio.run(client.place_buy_order("token-1", 0.51, 5.0, "PASSIVE_LIMIT", client_order_id="cid-1"))


def test_place_order_retries_with_preferred_signature_type(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyOrderArgs:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _DummyOrderType:
        GTC = "GTC"

    class _RetrySdkClient:
        def __init__(self) -> None:
            self.builder = type("Builder", (), {"sig_type": 0})()

        def create_order(self, order_args: object) -> dict[str, object]:
            return {"signed": True, "sig_type": self.builder.sig_type}

        def post_order(self, signed_order: object, order_type: object) -> dict[str, object]:
            if self.builder.sig_type == 0:
                raise RuntimeError("invalid signature")
            return {"orderID": "order-1", "status": "SUBMITTED"}

        def get_order(self, order_id: str) -> dict[str, object]:
            return {"id": order_id, "status": "RESTING"}

        def cancel(self, order_id: str) -> dict[str, object]:
            return {"id": order_id, "status": "CANCELLED"}

        def get_orders(self) -> list[dict[str, object]]:
            return []

    client = PolymarketClient(_live_config())
    client._sdk_client = _RetrySdkClient()
    client._preferred_signature_type = 1
    monkeypatch.setattr(polymarket_client_module, "OrderArgs", _DummyOrderArgs)
    monkeypatch.setattr(polymarket_client_module, "OrderType", _DummyOrderType)

    result = asyncio.run(client.place_buy_order("token-1", 0.51, 5.0, "PASSIVE_LIMIT", client_order_id="cid-1"))

    assert result["exchange_order_id"] == "order-1"
    assert result["signature_type_used"] == 1


def test_get_open_orders_normalizes_remaining_size_from_original_size() -> None:
    class _OrderSdkClient:
        def __init__(self) -> None:
            self.builder = type("Builder", (), {"sig_type": 0})()

        def get_orders(self) -> list[dict[str, object]]:
            return [
                {
                    "id": "order-1",
                    "status": "LIVE",
                    "market": "m1",
                    "asset_id": "t1",
                    "side": "BUY",
                    "original_size": "5.76",
                    "size_matched": "0",
                    "price": "0.52",
                }
            ]

    client = PolymarketClient(_live_config())
    client._sdk_client = _OrderSdkClient()

    result = asyncio.run(client.get_open_orders())

    assert result[0]["remaining_size"] == 5.76
    assert result[0]["size"] == 5.76


def test_get_order_status_uses_size_matched_and_price_for_matched_orders() -> None:
    class _OrderSdkClient:
        def __init__(self) -> None:
            self.builder = type("Builder", (), {"sig_type": 0})()

        def get_order(self, order_id: str) -> dict[str, object]:
            return {
                "id": order_id,
                "status": "MATCHED",
                "original_size": "5",
                "size_matched": "5",
                "price": "0.45",
            }

    client = PolymarketClient(_live_config())
    client._sdk_client = _OrderSdkClient()

    result = asyncio.run(client.get_order_status("order-1"))

    assert result["status"] == "MATCHED"
    assert result["filled_size"] == 5.0
    assert result["average_fill_price"] == 0.45
    assert result["remaining_size"] == 0.0
