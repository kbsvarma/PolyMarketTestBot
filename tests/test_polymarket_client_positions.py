from __future__ import annotations

import asyncio
from pathlib import Path

from src.config import load_config
from src.polymarket_client import PolymarketClient


class _PositionsSdkClient:
    def get_positions(self):
        return []


def test_live_get_positions_allows_empty_list() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    client = PolymarketClient(config)
    client._sdk_client = _PositionsSdkClient()
    positions = asyncio.run(client.get_positions())
    assert positions == []
