"""
Polymarket CLOB WebSocket price feed.

Streams live YES/NO ask prices from Polymarket's WebSocket API as a
drop-in alternative to REST orderbook polling.  The public interface
(yes_ask, no_ask, yes_ask_size, no_ask_size, last_update_ts,
is_connected) mirrors exactly what AssetWatcher reads from its own
_yes_ask / _no_ask fields after a REST fetch.

Usage
-----
feed = PolymarketWSFeed()
await feed.start(market_id, yes_token_id, no_token_id)
# ... read feed.yes_ask etc. from the polling loop ...
await feed.stop()

Handles
-------
- `price_change` events  (single-sided tick update)
- `book` snapshot events (full best-ask/bid refresh)
- Automatic reconnect with exponential backoff (cap 5 s)
- Logs connect / disconnect / reconnect / stale-fallback events
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

try:
    from loguru import logger  # type: ignore[import]
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_RECONNECT_BASE_SECONDS: float = 0.5
_RECONNECT_MAX_SECONDS: float = 5.0
_RECV_TIMEOUT_SECONDS: float = 10.0   # server sends heartbeat-ish traffic


class PolymarketWSFeed:
    """
    Streams YES/NO ask prices from the Polymarket CLOB WebSocket.

    Exposes the same price fields that AssetWatcher currently updates
    via REST orderbook fetches, so the watcher can read from either
    source without touching signal logic.

    Properties
    ----------
    yes_ask, no_ask         : float  best ask price (what we pay to BUY)
    yes_ask_size, no_ask_size: float  visible size at best ask
    last_update_ts          : float  time.time() when last price was received
    is_connected            : bool   True while the WS is connected and subscribed
    """

    def __init__(self) -> None:
        self._market_id: str = ""
        self._yes_token: str = ""
        self._no_token: str = ""

        # Prices — same names as AssetWatcher internal fields
        self._yes_ask: float = 0.0
        self._no_ask: float = 0.0
        self._yes_ask_size: float = 0.0
        self._no_ask_size: float = 0.0
        self._last_update_ts: float = 0.0

        self._connected: bool = False
        self._stop_requested: bool = False
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def yes_ask(self) -> float:
        return self._yes_ask

    @property
    def no_ask(self) -> float:
        return self._no_ask

    @property
    def yes_ask_size(self) -> float:
        return self._yes_ask_size

    @property
    def no_ask_size(self) -> float:
        return self._no_ask_size

    @property
    def last_update_ts(self) -> float:
        return self._last_update_ts

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        market_id: str,
        yes_token: str,
        no_token: str,
    ) -> None:
        """
        Connect to the Polymarket CLOB WebSocket and subscribe to YES/NO
        price updates for the given market.

        Starts a background asyncio task that reconnects automatically on
        disconnect.  Call stop() to shut it down cleanly.
        """
        self._market_id = market_id
        self._yes_token = yes_token
        self._no_token = no_token
        self._stop_requested = False

        # Reset prices so stale data from a previous window is not reused
        self._yes_ask = 0.0
        self._no_ask = 0.0
        self._yes_ask_size = 0.0
        self._no_ask_size = 0.0
        self._last_update_ts = 0.0
        self._connected = False

        self._task = asyncio.create_task(
            self._run_loop(),
            name=f"ws_feed_{market_id[:8]}",
        )
        logger.info(
            "PolymarketWSFeed starting market_id={} yes_token={} no_token={}",
            market_id,
            yes_token[:8] + "…" if len(yes_token) > 8 else yes_token,
            no_token[:8] + "…" if len(no_token) > 8 else no_token,
        )

    async def stop(self) -> None:
        """Disconnect the WebSocket feed and cancel the background task."""
        self._stop_requested = True
        self._connected = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("PolymarketWSFeed stopped market_id={}", self._market_id)

    # ------------------------------------------------------------------
    # Internal reconnect loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """
        Outer reconnect loop.  Retries with exponential backoff capped at
        _RECONNECT_MAX_SECONDS.  Exits cleanly when stop() has been called.
        """
        backoff = _RECONNECT_BASE_SECONDS
        attempt = 0

        while not self._stop_requested:
            attempt += 1
            try:
                await self._connect_and_stream()
                # _connect_and_stream returned normally (server closed)
                if self._stop_requested:
                    break
                logger.warning(
                    "PolymarketWSFeed connection closed (attempt {}), reconnecting in {:.1f}s",
                    attempt, backoff,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stop_requested:
                    break
                logger.warning(
                    "PolymarketWSFeed error attempt={} error={!r}, reconnecting in {:.1f}s",
                    attempt, exc, backoff,
                )

            self._connected = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_SECONDS)

    async def _connect_and_stream(self) -> None:
        """
        Open one WebSocket connection, subscribe, and stream messages until
        the connection closes or stop() is called.
        """
        # Import here to keep the top-level import section clean and to
        # avoid making the module un-importable on environments that lack
        # the websockets library at parse time.
        import websockets  # type: ignore[import]

        logger.info("PolymarketWSFeed connecting to {}", _WS_URL)
        async with websockets.connect(
            _WS_URL,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            # Subscribe to market price events.
            # Correct Polymarket CLOB WS format: lowercase type, assets_ids only.
            subscribe_msg = json.dumps({
                "type": "market",
                "assets_ids": [self._yes_token, self._no_token],
            })
            await ws.send(subscribe_msg)
            self._connected = True
            logger.info(
                "PolymarketWSFeed connected and subscribed market_id={}",
                self._market_id,
            )

            while not self._stop_requested:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(),
                        timeout=_RECV_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    # No message in _RECV_TIMEOUT_SECONDS — connection may be
                    # dead; exit this function to trigger reconnect.
                    logger.warning(
                        "PolymarketWSFeed recv timeout after {}s, reconnecting",
                        _RECV_TIMEOUT_SECONDS,
                    )
                    return

                self._handle_raw(raw)

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    def _handle_raw(self, raw: str | bytes) -> None:
        """
        Parse a raw WebSocket message and update internal price state.

        Polymarket CLOB WS sends two message shapes:
          1. Initial snapshot: list of book objects, one per token.
             Each has asset_id, bids[], asks[].
          2. Ongoing ticks: dict with price_changes[] array.
             Each entry has asset_id, best_ask, best_bid directly.
        """
        try:
            payload: Any = json.loads(raw)
        except Exception:
            return

        if isinstance(payload, list):
            # Initial book snapshot — one entry per subscribed token
            for item in payload:
                if isinstance(item, dict):
                    self._handle_book(item)
        elif isinstance(payload, dict):
            # Ongoing price_changes ticks
            changes = payload.get("price_changes")
            if isinstance(changes, list):
                for change in changes:
                    if isinstance(change, dict):
                        self._handle_price_change(change)

    def _handle_price_change(self, change: dict[str, Any]) -> None:
        """
        Handle one entry from a price_changes tick.

        Polymarket sends best_ask and best_bid directly in each entry —
        no need to parse book levels. We only care about best_ask (what
        we pay to BUY a YES or NO token).
        """
        asset_id = change.get("asset_id", "")
        best_ask_str = change.get("best_ask")
        if not best_ask_str:
            return
        try:
            best_ask = float(best_ask_str)
            # size not directly available at best_ask in tick; use 0 as sentinel
            # (AssetWatcher only gates on size for depth checks, not signal logic)
            best_ask_size = 0.0
        except (TypeError, ValueError):
            return

        if best_ask <= 0:
            return

        if asset_id == self._yes_token:
            self._yes_ask = best_ask
            self._yes_ask_size = best_ask_size
            self._last_update_ts = time.time()
        elif asset_id == self._no_token:
            self._no_ask = best_ask
            self._no_ask_size = best_ask_size
            self._last_update_ts = time.time()

    def _handle_book(self, event: dict[str, Any]) -> None:
        """
        Handle one token's initial book snapshot (from the list on connect).

        Extracts the best ask (lowest price in asks[]) and seeds the
        internal price state before the first price_change tick arrives.
        """
        asset_id = event.get("asset_id", "")
        asks: list[dict[str, Any]] = event.get("asks", [])

        best_ask_price: float = 0.0
        best_ask_size: float = 0.0

        if asks:
            try:
                best = min(asks, key=lambda x: float(x.get("price", 9999)))
                best_ask_price = float(best.get("price", 0))
                best_ask_size = float(best.get("size", 0))
            except (TypeError, ValueError):
                pass

        if best_ask_price <= 0:
            return

        if asset_id == self._yes_token:
            self._yes_ask = best_ask_price
            self._yes_ask_size = best_ask_size
            self._last_update_ts = time.time()
        elif asset_id == self._no_token:
            self._no_ask = best_ask_price
            self._no_ask_size = best_ask_size
            self._last_update_ts = time.time()
