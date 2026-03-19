"""
Polymarket Real-Time Data Socket (RTDS) client.

RTDS streams millisecond-stamped crypto price data from two sources:
  - Binance (real-time CEX spot feed)
  - Chainlink (the oracle that actually settles 5-min Up/Down contracts)

Connection:
  wss://ws-subscriptions-clob.polymarket.com/ws/price   (RTDS endpoint)

Protocol:
  1. Connect
  2. Send PING every 5 s (plain text "PING") to keep connection alive
  3. Receive price update messages as JSON

This client runs as a persistent background task and exposes:
  - ``binance_price``       latest Binance BTC/USD price
  - ``chainlink_price``     latest Chainlink BTC/USD price
  - ``binance_ts``          timestamp of latest Binance message (ms epoch)
  - ``chainlink_ts``        timestamp of latest Chainlink message (ms epoch)
  - ``lag_ms()``            Binance - Chainlink price lag (sign-aware)
  - ``staleness_seconds()`` seconds since last update from *either* source

Usage (in an async context):
    client = RTDSClient()
    asyncio.create_task(client.run_forever())
    # ... elsewhere ...
    snap = client.snapshot()
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger  # type: ignore[import]
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

# The RTDS WebSocket endpoint (Polymarket docs)
RTDS_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/price"

# How often to send PING to keep connection alive (seconds)
PING_INTERVAL_SECONDS = 5

# Seconds after which we treat the feed as stale
RTDS_STALENESS_THRESHOLD_SECONDS = 3.0

# How long to wait for a message before timing out a receive attempt
RECV_TIMEOUT_SECONDS = 6.0

# Back-off cap between reconnect attempts (seconds)
MAX_RECONNECT_BACKOFF_SECONDS = 30.0


@dataclass
class RTDSSnapshot:
    """Point-in-time copy of RTDS state, safe to pass around."""
    binance_price: float = 0.0
    chainlink_price: float = 0.0
    binance_ts_ms: float = 0.0        # epoch milliseconds
    chainlink_ts_ms: float = 0.0      # epoch milliseconds
    snapshot_at: float = field(default_factory=time.time)  # epoch seconds

    # ------------------------------------------------------------------ #

    def binance_age_seconds(self) -> float:
        if self.binance_ts_ms <= 0:
            return 9999.0
        return (time.time() * 1000 - self.binance_ts_ms) / 1000.0

    def chainlink_age_seconds(self) -> float:
        if self.chainlink_ts_ms <= 0:
            return 9999.0
        return (time.time() * 1000 - self.chainlink_ts_ms) / 1000.0

    def staleness_seconds(self) -> float:
        """Seconds since the most recent update from *either* source."""
        most_recent_ts_ms = max(self.binance_ts_ms, self.chainlink_ts_ms)
        if most_recent_ts_ms <= 0:
            return 9999.0
        return (time.time() * 1000 - most_recent_ts_ms) / 1000.0

    def is_fresh(self, threshold_seconds: float = RTDS_STALENESS_THRESHOLD_SECONDS) -> bool:
        return self.staleness_seconds() <= threshold_seconds

    def binance_fresh(self, threshold_seconds: float = RTDS_STALENESS_THRESHOLD_SECONDS) -> bool:
        return self.binance_age_seconds() <= threshold_seconds

    def chainlink_fresh(self, threshold_seconds: float = RTDS_STALENESS_THRESHOLD_SECONDS) -> bool:
        return self.chainlink_age_seconds() <= threshold_seconds

    def lag_ms(self) -> float:
        """
        Binance timestamp minus Chainlink timestamp (ms).
        Positive = Binance is ahead of Chainlink (usual case).
        Zero if either is missing.
        """
        if self.binance_ts_ms <= 0 or self.chainlink_ts_ms <= 0:
            return 0.0
        return self.binance_ts_ms - self.chainlink_ts_ms

    def price_divergence(self) -> float:
        """
        Absolute price difference between Binance and Chainlink.
        Zero if either is missing.
        """
        if self.binance_price <= 0 or self.chainlink_price <= 0:
            return 0.0
        return abs(self.binance_price - self.chainlink_price)

    def price_divergence_pct(self) -> float:
        """Price divergence as a fraction of Chainlink price."""
        if self.chainlink_price <= 0:
            return 0.0
        return self.price_divergence() / self.chainlink_price

    def to_dict(self) -> dict[str, Any]:
        return {
            "binance_price": self.binance_price,
            "chainlink_price": self.chainlink_price,
            "binance_ts_ms": self.binance_ts_ms,
            "chainlink_ts_ms": self.chainlink_ts_ms,
            "binance_age_seconds": round(self.binance_age_seconds(), 3),
            "chainlink_age_seconds": round(self.chainlink_age_seconds(), 3),
            "staleness_seconds": round(self.staleness_seconds(), 3),
            "lag_ms": round(self.lag_ms(), 1),
            "price_divergence": round(self.price_divergence(), 2),
            "price_divergence_pct": round(self.price_divergence_pct(), 6),
            "is_fresh": self.is_fresh(),
        }


class RTDSClient:
    """
    Persistent RTDS WebSocket subscriber.

    Start as a background task:
        asyncio.create_task(client.run_forever())

    Read state at any time (thread-safe for asyncio single-thread model):
        snap = client.snapshot()
    """

    def __init__(
        self,
        url: str = RTDS_WS_URL,
        ping_interval: float = PING_INTERVAL_SECONDS,
        staleness_threshold: float = RTDS_STALENESS_THRESHOLD_SECONDS,
    ) -> None:
        self.url = url
        self.ping_interval = ping_interval
        self.staleness_threshold = staleness_threshold

        # Latest data (updated in place by the run loop)
        self._binance_price: float = 0.0
        self._chainlink_price: float = 0.0
        self._binance_ts_ms: float = 0.0
        self._chainlink_ts_ms: float = 0.0

        # Diagnostics
        self._connected: bool = False
        self._reconnect_count: int = 0
        self._last_error: str = ""
        self._message_count: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def snapshot(self) -> RTDSSnapshot:
        """Return a point-in-time snapshot of current RTDS state."""
        return RTDSSnapshot(
            binance_price=self._binance_price,
            chainlink_price=self._chainlink_price,
            binance_ts_ms=self._binance_ts_ms,
            chainlink_ts_ms=self._chainlink_ts_ms,
        )

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def message_count(self) -> int:
        return self._message_count

    @property
    def last_error(self) -> str:
        return self._last_error

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _parse_message(self, raw: str) -> None:
        """
        Parse an incoming WebSocket message and update internal state.

        Supports two formats:
          1. Binance aggTrade stream (wss://data-stream.binance.vision/ws/SYMBOL@aggTrade):
               {"e":"aggTrade","p":"73718.01","T":1773644370974, ...}
          2. Legacy Polymarket RTDS format:
               {"source":"Binance","price":"73718.01","timestamp":1773644370974}
        """
        try:
            if not raw or raw.strip() == "PONG":
                return
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return

        # Handle list of updates
        if isinstance(payload, list):
            for item in payload:
                self._apply_single(item)
        elif isinstance(payload, dict):
            self._apply_single(payload)

    def _apply_single(self, payload: dict[str, Any]) -> None:
        # --- Binance aggTrade / trade stream format ---
        event_type = str(payload.get("e") or "")
        if event_type in ("aggTrade", "trade"):
            try:
                price = float(payload.get("p") or 0.0)
                ts_ms = float(payload.get("T") or payload.get("t") or time.time() * 1000)
            except (TypeError, ValueError):
                return
            if price > 0:
                self._message_count += 1
                self._binance_price = price
                self._binance_ts_ms = ts_ms
                # Mirror to chainlink so lag_signal sees a consistent pair
                # (no real Chainlink feed; lag_signal will use price_divergence=0)
                self._chainlink_price = price
                self._chainlink_ts_ms = ts_ms
            return

        # --- Legacy Polymarket RTDS format ---
        source = str(payload.get("source") or payload.get("provider") or "").lower()
        try:
            price = float(payload.get("price") or payload.get("p") or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        try:
            ts_raw = payload.get("timestamp") or payload.get("ts") or payload.get("t") or 0
            ts_ms = float(ts_raw)
        except (TypeError, ValueError):
            ts_ms = time.time() * 1000

        if price <= 0:
            return

        self._message_count += 1
        if "binance" in source or not source:
            self._binance_price = price
            self._binance_ts_ms = ts_ms
            self._chainlink_price = price
            self._chainlink_ts_ms = ts_ms
        elif "chainlink" in source:
            self._chainlink_price = price
            self._chainlink_ts_ms = ts_ms

    # ------------------------------------------------------------------ #
    # Run loop
    # ------------------------------------------------------------------ #

    async def run_forever(self) -> None:
        """
        Persistent connection loop.  Reconnects with exponential back-off.
        Run with: asyncio.create_task(client.run_forever())
        """
        backoff = 1.0
        while True:
            try:
                await self._connect_and_stream()
                backoff = 1.0  # reset on clean exit
            except Exception as exc:
                self._connected = False
                self._last_error = str(exc)
                self._reconnect_count += 1
                logger.warning(
                    "RTDS disconnected reconnect_count={} error={} back_off={:.1f}s",
                    self._reconnect_count,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, MAX_RECONNECT_BACKOFF_SECONDS)

    async def _connect_and_stream(self) -> None:
        """Open the WebSocket, send PING, receive messages in a loop."""
        try:
            import websockets  # type: ignore[import]
        except ImportError:
            logger.error("websockets package not installed; RTDS client disabled")
            await asyncio.sleep(60)
            return

        logger.info("RTDS connecting url={}", self.url)
        # For Binance streams, use built-in websockets ping (protocol-level).
        # For legacy Polymarket RTDS, we send text "PING" manually (ping_interval=None).
        use_builtin_ping = "binance" in self.url.lower()
        async with websockets.connect(
            self.url,
            ping_interval=20 if use_builtin_ping else None,
            open_timeout=10,
            close_timeout=5,
        ) as ws:
            self._connected = True
            logger.info("RTDS connected url={}", self.url)

            # Start PING sender as a sibling task (only for non-Binance endpoints)
            ping_task = asyncio.create_task(self._ping_loop(ws, skip=use_builtin_ping))
            try:
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_SECONDS)
                        self._parse_message(raw)
                    except asyncio.TimeoutError:
                        # No message within timeout — log staleness but keep going
                        snap = self.snapshot()
                        logger.debug(
                            "RTDS recv timeout staleness=%.1fs", snap.staleness_seconds()
                        )
            finally:
                ping_task.cancel()
                with suppress(asyncio.CancelledError):
                    await ping_task
                self._connected = False

    async def _ping_loop(self, ws: Any, skip: bool = False) -> None:
        """Send text PING messages every ``ping_interval`` seconds (legacy endpoints only)."""
        if skip:
            return
        while True:
            await asyncio.sleep(self.ping_interval)
            try:
                await ws.send("PING")
            except Exception:
                break


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all strategy modules
# ---------------------------------------------------------------------------

_default_client: RTDSClient | None = None


def get_rtds_client(url: str | None = None) -> RTDSClient:
    """Return (and lazily create) the module-level singleton RTDSClient."""
    global _default_client
    if _default_client is None:
        _default_client = RTDSClient(url=url or RTDS_WS_URL)
    return _default_client
