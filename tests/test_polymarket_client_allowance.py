from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from src.config import load_config
from src.polymarket_client import PolymarketClient


class _AllowanceSdkClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.builder = type("Builder", (), {"sig_type": 0})()

    def get_balance_allowance(self, params: object = None) -> dict[str, object]:
        return self.payload


class _SignatureAwareAllowanceSdkClient:
    def __init__(self) -> None:
        self.builder = type("Builder", (), {"sig_type": 0})()

    def get_balance_allowance(self, params: object = None) -> dict[str, object]:
        if getattr(params, "signature_type", None) == 1:
            return {"available": "4.25", "allowances": {"proxy": "4.25"}}
        return {"balance": "0", "allowances": {"a": "0"}}


class _FakeAssetType:
    COLLATERAL = "COLLATERAL"


class _FakeBalanceAllowanceParams:
    def __init__(self, asset_type: object = None, token_id: str | None = None, signature_type: int = -1) -> None:
        self.asset_type = asset_type
        self.token_id = token_id
        self.signature_type = signature_type


def _live_config():
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    return config


def test_allowance_parses_nested_allowance_map() -> None:
    client = PolymarketClient(_live_config())
    client._sdk_client = _AllowanceSdkClient(
        {
            "balance": "0",
            "allowances": {
                "a": "0",
                "b": "12.5",
            },
        }
    )
    payload = asyncio.run(client.get_allowance())
    assert payload["query_visible"] is True
    assert payload["available"] == 12.5
    assert payload["sufficient"] is True
    assert payload["raw_summary"]["max_allowance_entry"] == 12.5


def test_allowance_zero_is_visible_but_not_sufficient() -> None:
    client = PolymarketClient(_live_config())
    client._sdk_client = _AllowanceSdkClient({"balance": "0", "allowances": {"a": "0"}})
    payload = asyncio.run(client.get_allowance())
    assert payload["query_visible"] is True
    assert payload["available"] == 0.0
    assert payload["sufficient"] is False


def test_allowance_tries_proxy_signature_type_and_keeps_exchange_side_result() -> None:
    client = PolymarketClient(_live_config())
    client._sdk_client = _SignatureAwareAllowanceSdkClient()
    with patch("src.polymarket_client.BalanceAllowanceParams", _FakeBalanceAllowanceParams), patch(
        "src.polymarket_client.AssetType", _FakeAssetType
    ):
        payload = asyncio.run(client.get_allowance())
    assert payload["query_visible"] is True
    assert payload["available"] == 4.25
    assert payload["sufficient"] is True
    assert payload["query_params"]["signature_type"] == 1
    assert payload["sdk_signature_type"] == 0
