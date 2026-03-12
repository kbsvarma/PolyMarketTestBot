from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.config import AppConfig
from src.models import DetectionEvent
from src.polymarket_client import PolymarketClient
from src.utils import append_csv_row, stable_event_key


class TradeMonitor:
    def __init__(self, config: AppConfig, data_dir: Path, state_store: object) -> None:
        self.config = config
        self.data_dir = data_dir
        self.state_store = state_store
        self.client = PolymarketClient(config)
        self.seen_keys: set[str] = set()

    def make_event_key(self, wallet: str, market_id: str, token_id: str, tx_hash: str, side: str) -> str:
        return stable_event_key(wallet, market_id, token_id, tx_hash, side)

    async def poll_wallets(self, wallets: list[str]) -> list[DetectionEvent]:
        detections: list[DetectionEvent] = []
        for wallet in wallets:
            activities = await self.client.fetch_wallet_activity(wallet, limit=20)
            for row in activities:
                detection = self._row_to_detection(wallet, row)
                if detection.event_key in self.seen_keys:
                    continue
                self.seen_keys.add(detection.event_key)
                detections.append(detection)
                append_csv_row(self.data_dir / "detected_wallet_trades.csv", detection.model_dump(mode="json"))
        return detections

    def _row_to_detection(self, wallet: str, row: dict) -> DetectionEvent:
        timestamp = self._parse_timestamp(row.get("timestamp") or row.get("time"))
        tx_hash = str(row.get("tx_hash") or row.get("transactionHash") or row.get("hash") or f"local-{wallet[-4:]}-{int(timestamp.timestamp())}")
        market_id = str(row.get("market_id") or row.get("conditionId") or row.get("market") or row.get("slug") or "unknown-market")
        token_id = str(row.get("token_id") or row.get("asset_id") or row.get("tokenId") or row.get("clobTokenId") or "unknown-token")
        side = str(row.get("side") or row.get("type") or "BUY").upper()
        event_key = self.make_event_key(wallet, market_id, token_id, tx_hash, side)
        local_now = datetime.now(timezone.utc)
        latency = max((local_now - timestamp).total_seconds(), 0.0)
        price = float(row.get("price") or row.get("outcomePrice") or 0.5)
        size = float(row.get("size") or row.get("amount") or row.get("shares") or 10.0)
        return DetectionEvent(
            event_key=event_key,
            local_detection_timestamp=local_now,
            source_trade_timestamp=timestamp,
            wallet_address=wallet,
            market_title=str(row.get("title") or row.get("market_title") or row.get("question") or market_id),
            market_slug=str(row.get("market_slug") or row.get("slug") or market_id),
            market_id=market_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            notional=price * size,
            transaction_hash=tx_hash,
            detection_latency_seconds=latency,
            best_bid=float(row.get("best_bid")) if row.get("best_bid") is not None else None,
            best_ask=float(row.get("best_ask")) if row.get("best_ask") is not None else None,
            depth_available=float(row.get("depth_available")) if row.get("depth_available") is not None else None,
            category=str(row.get("category") or "unknown"),
            source_alias=str(row.get("alias") or ""),
            market_metadata={"raw": row},
        )

    def _parse_timestamp(self, raw: str | None) -> datetime:
        if not raw:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
