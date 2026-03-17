from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from src.config import AppConfig
from src.logger import logger
from src.models import DetectionEvent, SourceQuality
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

    def _wallet_poll_timeout_seconds(self) -> float:
        live_bound = float(self.config.live.bounded_execution_seconds or 20)
        if self.config.mode.value == "LIVE":
            return max(6.0, min(live_bound, 12.0))
        return max(10.0, live_bound)

    async def _fetch_wallet_activity_safe(self, wallet: str, limit: int) -> tuple[str, list[dict]]:
        timeout_seconds = self._wallet_poll_timeout_seconds()
        try:
            activities = await asyncio.wait_for(
                self.client.fetch_wallet_activity(wallet, limit=limit),
                timeout=timeout_seconds,
            )
            return wallet, activities
        except asyncio.TimeoutError:
            logger.warning(
                "Wallet activity fetch timed out wallet={} timeout_seconds={}",
                wallet,
                timeout_seconds,
            )
        except RuntimeError as exc:
            logger.warning("Wallet activity fetch failed wallet={} error={}", wallet, exc)
        except Exception as exc:  # pragma: no cover - defensive safety net
            logger.warning("Wallet activity fetch errored wallet={} error={}", wallet, exc)
        return wallet, []

    async def poll_wallets(self, wallets: list[str]) -> list[DetectionEvent]:
        detections: list[DetectionEvent] = []
        results = await asyncio.gather(
            *(self._fetch_wallet_activity_safe(wallet, limit=20) for wallet in wallets)
        )
        for wallet, activities in results:
            for row in activities:
                detection = self._row_to_detection(wallet, row)
                if detection is None:
                    continue
                if detection.event_key in self.seen_keys:
                    continue
                self.seen_keys.add(detection.event_key)
                detections.append(detection)
                append_csv_row(self.data_dir / "detected_wallet_trades.csv", detection.model_dump(mode="json"))
        return detections

    def _row_to_detection(self, wallet: str, row: dict) -> DetectionEvent | None:
        timestamp = self._parse_timestamp(row.get("timestamp") or row.get("time"))
        if self._is_expired_or_stale(row, timestamp):
            return None
        tx_hash = str(row.get("tx_hash") or row.get("transactionHash") or row.get("hash") or f"local-{wallet[-4:]}-{int(timestamp.timestamp())}")
        market_id = str(row.get("market_id") or row.get("conditionId") or row.get("market") or row.get("slug") or "unknown-market")
        token_id = str(
            row.get("token_id")
            or row.get("asset_id")
            or row.get("tokenId")
            or row.get("clobTokenId")
            or row.get("asset")
            or "unknown-token"
        )
        side = str(row.get("side") or row.get("type") or "BUY").upper()
        if side in {"REDEEM", "REWARD", "MERGE", "SPLIT"}:
            return None
        event_key = self.make_event_key(wallet, market_id, token_id, tx_hash, side)
        local_now = datetime.now(timezone.utc)
        latency = max((local_now - timestamp).total_seconds(), 0.0)
        price = float(row.get("price") or row.get("outcomePrice") or 0.5)
        size = float(row.get("size") or row.get("amount") or row.get("shares") or 10.0)
        category = self._infer_category(row)
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
            category=category,
            source_alias=str(row.get("alias") or row.get("pseudonym") or row.get("name") or ""),
            market_metadata={"raw": row, "outcome": row.get("outcome") or row.get("outcome_name") or row.get("outcomeName") or ""},
            source_quality=SourceQuality(str(row.get("source_quality") or SourceQuality.REAL_PUBLIC_DATA.value)),
        )

    def _is_expired_or_stale(self, row: dict, timestamp: datetime) -> bool:
        now = datetime.now(timezone.utc)
        if bool(row.get("closed")):
            return True
        for key in ("endDate", "end_date_iso", "end_date", "expiration", "expiresAt"):
            raw = row.get(key)
            if not raw:
                continue
            end_at = self._parse_timestamp(raw)
            if end_at <= now:
                return True
        latency_seconds = max((now - timestamp).total_seconds(), 0.0)
        if self.config.mode.value == "LIVE":
            max_age = max(self.config.risk.stale_signal_seconds * 2, 600)
        else:
            max_age = max(self.config.risk.stale_signal_seconds * 20, 86_400)
        return latency_seconds > max_age

    def _infer_category(self, row: dict) -> str:
        explicit = str(row.get("category") or "").strip()
        if explicit:
            return explicit
        text = " ".join(
            str(row.get(key) or "")
            for key in ("slug", "eventSlug", "title", "outcome", "icon")
        ).lower()
        if any(keyword in text for keyword in ("btc", "bitcoin", "eth", "ethereum", "sol", "solana", "xrp", "crypto")):
            return "crypto price"
        if any(
            keyword in text
            for keyword in (
                "oscar",
                "oscars",
                "academy award",
                "academy awards",
                "best picture",
                "best actor",
                "best actress",
                "best director",
                "movie",
                "film",
                "box office",
                "golden globe",
                "golden globes",
                "grammy",
                "grammys",
                "emmy",
                "emmys",
                "album of the year",
                "song of the year",
            )
        ):
            return "entertainment / pop culture"
        if any(keyword in text for keyword in ("election", "candidate", "president", "senate", "supreme court", "biden", "trump")):
            return "politics"
        if any(keyword in text for keyword in ("temperature", "weather", "rain", "snow")):
            return "macro / economics"
        if any(keyword in text for keyword in ("championship", "match", "game", "nba", "nfl", "mlb", "soccer")):
            return "sports"
        return "unknown"

    def _parse_timestamp(self, raw: str | int | float | None) -> datetime:
        if not raw:
            return datetime.now(timezone.utc)
        if isinstance(raw, (int, float)):
            try:
                value = float(raw)
                if value > 1_000_000_000_000:
                    value = value / 1000.0
                return datetime.fromtimestamp(value, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
