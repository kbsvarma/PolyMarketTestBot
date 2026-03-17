from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import load_config
from src.models import Mode
from src.state import AppStateStore
from src.trade_monitor import TradeMonitor


def _monitor(tmp_path: Path) -> TradeMonitor:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    return TradeMonitor(config, tmp_path, AppStateStore(tmp_path / "app_state.json"))


def test_trade_monitor_ignores_redeem_events(tmp_path: Path) -> None:
    monitor = _monitor(tmp_path)

    detection = monitor._row_to_detection(  # noqa: SLF001
        "0xabc",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": "m1",
            "token_id": "t1",
            "side": "REDEEM",
            "price": 0.5,
            "size": 10,
        },
    )

    assert detection is None


def test_trade_monitor_preserves_outcome_in_market_metadata(tmp_path: Path) -> None:
    monitor = _monitor(tmp_path)

    detection = monitor._row_to_detection(  # noqa: SLF001
        "0xabc",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": "m1",
            "token_id": "unknown-token",
            "side": "BUY",
            "outcome": "Yes",
            "price": 0.5,
            "size": 10,
        },
    )

    assert detection is not None
    assert detection.market_metadata["outcome"] == "Yes"


def test_trade_monitor_ignores_stale_live_detection(tmp_path: Path) -> None:
    monitor = _monitor(tmp_path)
    monitor.config.mode = Mode.LIVE

    detection = monitor._row_to_detection(  # noqa: SLF001
        "0xabc",
        {
            "timestamp": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "market_id": "m1",
            "token_id": "t1",
            "side": "BUY",
            "price": 0.5,
            "size": 10,
        },
    )

    assert detection is None


def test_trade_monitor_ignores_expired_market_row(tmp_path: Path) -> None:
    monitor = _monitor(tmp_path)

    detection = monitor._row_to_detection(  # noqa: SLF001
        "0xabc",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": "m1",
            "token_id": "t1",
            "side": "BUY",
            "price": 0.5,
            "size": 10,
            "endDate": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        },
    )

    assert detection is None


def test_trade_monitor_inferrs_oscars_as_entertainment(tmp_path: Path) -> None:
    monitor = _monitor(tmp_path)

    detection = monitor._row_to_detection(  # noqa: SLF001
        "0xabc",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": "m-oscars",
            "token_id": "t-oscars",
            "side": "BUY",
            "title": "Who will win Best Picture at the Oscars?",
            "price": 0.62,
            "size": 10,
        },
    )

    assert detection is not None
    assert detection.category == "entertainment / pop culture"


def test_trade_monitor_poll_wallets_times_out_safely(tmp_path: Path) -> None:
    monitor = _monitor(tmp_path)
    monitor.config.mode = Mode.LIVE
    monitor._wallet_poll_timeout_seconds = lambda: 0.1  # type: ignore[method-assign]

    async def _slow_fetch(_wallet: str, limit: int = 20):  # noqa: ARG001
        await asyncio.sleep(2)
        return [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_id": "m1",
                "token_id": "t1",
                "side": "BUY",
                "price": 0.5,
                "size": 10,
            }
        ]

    monitor.client.fetch_wallet_activity = _slow_fetch  # type: ignore[method-assign]

    detections = asyncio.run(monitor.poll_wallets(["0xabc"]))

    assert detections == []
