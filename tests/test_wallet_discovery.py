from __future__ import annotations

import asyncio
from pathlib import Path

from src.config import load_config
from src.models import DiscoveryState, Mode, SourceQuality
from src.wallet_discovery import WalletDiscoveryService


def _activity_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "timestamp": f"2026-03-12T00:{idx:02d}:00Z",
            "side": "BUY" if idx % 2 == 0 else "SELL",
            "price": 0.45 + idx * 0.01,
            "size": 10 + idx,
            "market_id": f"market-{idx % 2}",
            "category": "crypto price",
        }
        for idx in range(count)
    ]


def test_wallet_discovery_success(monkeypatch) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example").model_copy(update={"mode": Mode.PAPER})
    service = WalletDiscoveryService(config, root / "data")

    async def fake_leaderboard(limit: int = 20):
        return [{"wallet_address": "0xabc"}]

    async def fake_markets(limit: int = 60):
        return []

    async def fake_recent(limit: int = 250):
        return []

    async def fake_activity(wallet_address: str, limit: int = 80):
        return _activity_rows(20)

    monkeypatch.setattr(service.client, "fetch_leaderboard", fake_leaderboard)
    monkeypatch.setattr(service.client, "fetch_markets", fake_markets)
    monkeypatch.setattr(service.client, "fetch_recent_public_activity", fake_recent)
    monkeypatch.setattr(service.client, "fetch_wallet_activity", fake_activity)

    result = asyncio.run(service.run_discovery_cycle())
    assert result.state == DiscoveryState.SUCCESS
    assert result.wallets
    assert result.wallets[0].wallet_address == "0xabc"


def test_wallet_discovery_empty_result(monkeypatch) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example").model_copy(update={"mode": Mode.PAPER})
    service = WalletDiscoveryService(config, root / "data")

    async def fake_empty(*args, **kwargs):
        return []

    monkeypatch.setattr(service.client, "fetch_leaderboard", fake_empty)
    monkeypatch.setattr(service.client, "fetch_markets", fake_empty)
    monkeypatch.setattr(service.client, "fetch_recent_public_activity", fake_empty)

    result = asyncio.run(service.run_discovery_cycle())
    assert result.state == DiscoveryState.NO_DATA
    assert result.wallets == []


def test_wallet_discovery_malformed_result(monkeypatch) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example").model_copy(update={"mode": Mode.PAPER})
    service = WalletDiscoveryService(config, root / "data")

    async def fake_bad(*args, **kwargs):
        return {"bad": "payload"}

    async def fake_empty(*args, **kwargs):
        return []

    monkeypatch.setattr(service.client, "fetch_leaderboard", fake_bad)
    monkeypatch.setattr(service.client, "fetch_markets", fake_empty)
    monkeypatch.setattr(service.client, "fetch_recent_public_activity", fake_empty)

    result = asyncio.run(service.run_discovery_cycle())
    assert result.state == DiscoveryState.MALFORMED_RESPONSE
    assert result.wallets == []


def test_wallet_discovery_research_fallback_is_labeled(monkeypatch) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example").model_copy(update={"mode": Mode.RESEARCH})
    service = WalletDiscoveryService(config, root / "data")

    async def fake_empty(*args, **kwargs):
        return []

    monkeypatch.setattr(service.client, "fetch_leaderboard", fake_empty)
    monkeypatch.setattr(service.client, "fetch_markets", fake_empty)
    monkeypatch.setattr(service.client, "fetch_recent_public_activity", fake_empty)

    result = asyncio.run(service.run_discovery_cycle())
    assert result.wallets
    assert result.source_quality == SourceQuality.SYNTHETIC_FALLBACK
    assert result.state == DiscoveryState.SYNTHETIC_FALLBACK_USED
