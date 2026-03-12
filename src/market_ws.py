from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None


@dataclass
class MarketWSClient:
    url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    connected: bool = False
    latest_messages: dict[str, dict] = field(default_factory=dict)
    last_event_at: datetime | None = None
    last_error: str = ""
    reconnect_attempts: int = 0
    watched_token_ids: list[str] = field(default_factory=list)

    async def connect(self, token_ids: list[str] | None = None, timeout_seconds: int = 8, max_messages: int = 3) -> None:
        self.watched_token_ids = token_ids or []
        if websockets is None or not token_ids:
            self.connected = False
            self.last_error = "websockets unavailable or no token ids"
            return
        websocket = None
        received_any = False
        try:
            websocket = await asyncio.wait_for(websockets.connect(self.url), timeout=timeout_seconds)
            self.connected = True
            self.last_error = ""
            await websocket.send(json.dumps({"assets_ids": token_ids, "type": "market"}))
            for _ in range(max_messages):
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
                payload = json.loads(raw)
                if isinstance(payload, list):
                    rows = [item for item in payload if isinstance(item, dict)]
                elif isinstance(payload, dict):
                    rows = [payload]
                else:
                    rows = []
                for row in rows:
                    asset_id = str(row.get("asset_id") or row.get("token_id") or token_ids[0])
                    row["received_at"] = datetime.now(timezone.utc).isoformat()
                    self.latest_messages[asset_id] = row
                self.last_event_at = datetime.now(timezone.utc)
                received_any = True
        except Exception as exc:
            self.connected = received_any
            self.last_error = str(exc) or exc.__class__.__name__
            self.reconnect_attempts += 1
        finally:
            if websocket is not None:
                await websocket.close()

    async def refresh(self, timeout_seconds: int = 8) -> None:
        await self.connect(self.watched_token_ids, timeout_seconds=timeout_seconds, max_messages=1)

    def event_age_seconds(self) -> float:
        if self.last_event_at is None:
            return 9999.0
        return (datetime.now(timezone.utc) - self.last_event_at).total_seconds()

    async def close(self) -> None:
        self.connected = False
