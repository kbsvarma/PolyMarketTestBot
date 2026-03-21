"""
Crypto market watcher for the bracket strategy direction signal observer.

Responsibilities:
  - Maintain a live BTC/USD or ETH/USD price feed via Binance WebSocket
    (one RTDSClient instance per asset — completely separate from the main
    Polymarket RTDS feed used by the existing lag_signal strategy)
  - Find the currently active 15-minute Polymarket market for each asset
  - Poll YES and NO best-ask prices every few seconds
  - Detect 15-minute window transitions and signal the evaluator to reset
  - Compare BTC vs ETH market liquidity to pick the better asset each window

Design notes:
  - This module is OBSERVATION ONLY — no orders are placed here
  - Two classes: AssetWatcher (per-asset) and CryptoMarketWatcher (orchestrator)
  - The slug format for 15m Polymarket markets is VERIFIED on first run by
    logging the actual market title. Update config.yaml if the prefix differs.
  - Window timestamp formula: int(time.time() // 900) * 900
    Polymarket 15m windows align to :00/:15/:30/:45 of each hour.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp

try:
    from loguru import logger  # type: ignore[import]
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from src.config import AssetVolConfig, CryptoDirectionConfig
from src.models import MarketInfo
from src.polymarket_ws_feed import PolymarketWSFeed
from src.rtds_client import RTDSClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 15-minute window duration in seconds
WINDOW_DURATION_SECONDS: int = 900


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def current_window_ts(duration: int = WINDOW_DURATION_SECONDS) -> int:
    """
    Return the Unix timestamp of the START of the current window.

    Works for any window size — pass duration=300 for 5-min, 900 for 15-min.
    Polymarket windows align to multiples of the duration in Unix time.

    Examples (15-min / duration=900):
      14:02:30 → window started at 14:00:00
      14:16:00 → window started at 14:15:00
      14:45:59 → window started at 14:45:00
    """
    return int(time.time() // duration) * duration


def minutes_remaining_in_window(duration: int = WINDOW_DURATION_SECONDS) -> float:
    """Return how many minutes are left in the current window."""
    now = time.time()
    window_end = current_window_ts(duration) + duration
    return max(0.0, (window_end - now) / 60.0)


def _identify_token_side(info: MarketInfo) -> str:
    """
    Determine whether a MarketInfo object represents the YES or NO token.

    Polymarket 15m markets have two outcome tokens. We identify them by
    looking for directional keywords in the title and slug. Returns "YES",
    "NO", or "UNKNOWN" if we can't determine it.

    This heuristic is logged on first run — update if wrong for your markets.
    """
    text = (info.title + " " + info.slug).lower()

    # UP / YES indicators
    up_keywords = ("up", "higher", "above", "bull", "yes", "rise", "over")
    # DOWN / NO indicators
    down_keywords = ("down", "lower", "below", "bear", "no", "fall", "under")

    up_hits = sum(1 for kw in up_keywords if kw in text)
    down_hits = sum(1 for kw in down_keywords if kw in text)

    if up_hits > down_hits:
        return "YES"
    elif down_hits > up_hits:
        return "NO"
    else:
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# AssetWatcher — watches a single asset (BTC or ETH)
# ---------------------------------------------------------------------------

class AssetWatcher:
    """
    Watches a single asset's 15-minute Polymarket binary market.

    Owns:
      - An RTDSClient connected to the Binance aggTrade stream for live price
      - Polling logic to refresh YES/NO orderbook prices
      - Market resolution logic (finding the current window's market)

    Thread model: single asyncio event loop, all state updates on the loop.
    """

    def __init__(
        self,
        asset: str,
        asset_cfg: AssetVolConfig,
        polymarket_client: Any,  # PolymarketClient — typed as Any to avoid circular import
        cfg: CryptoDirectionConfig,
    ) -> None:
        self.asset = asset                      # "BTC" or "ETH"
        self._asset_cfg = asset_cfg
        self._client = polymarket_client
        self._cfg = cfg

        # --- Price feed (Binance WebSocket via RTDSClient) ---
        # Each asset gets its OWN RTDSClient so we get independent price feeds.
        # BTC → wss://stream.binance.com/ws/btcusdt@aggTrade
        # ETH → wss://stream.binance.com/ws/ethusdt@aggTrade
        self._rtds = RTDSClient(url=asset_cfg.binance_ws_url)

        # --- Market state (resolved each window) ---
        self._current_window_ts: int = 0
        self._market_id: str = ""
        self._yes_token_id: str = ""
        self._no_token_id: str = ""
        self._market_liquidity: float = 0.0
        self._market_volume: float = 0.0
        self._market_title: str = ""            # for verification logging
        self._market_resolved: bool = False
        self._resolve_retry_due_at: float = 0.0

        # --- WebSocket feed (optional) ---
        self._cfg_use_ws: bool = cfg.use_websocket_feed
        self._ws_feed: PolymarketWSFeed | None = None

        # --- Live YES/NO prices ---
        self._yes_bid: float = 0.0
        self._no_bid: float = 0.0
        self._yes_ask: float = 0.0
        self._no_ask: float = 0.0
        self._yes_bid_size: float = 0.0
        self._no_bid_size: float = 0.0
        self._yes_ask_size: float = 0.0
        self._no_ask_size: float = 0.0
        self._last_price_ts: float = 0.0        # time() when prices were last refreshed

        # --- Binance REST fallback (used when RTDS WebSocket is stale) ---
        self._rest_fallback_price: float = 0.0
        self._last_rest_poll_ts: float = 0.0

    def _reset_market_state(self, window_ts: int) -> None:
        """Clear market/orderbook state before resolving a new window."""
        self._current_window_ts = window_ts
        self._market_id = ""
        self._yes_token_id = ""
        self._no_token_id = ""
        self._market_liquidity = 0.0
        self._market_volume = 0.0
        self._market_title = ""
        self._market_resolved = False
        self._yes_bid = 0.0
        self._no_bid = 0.0
        self._yes_ask = 0.0
        self._no_ask = 0.0
        self._yes_bid_size = 0.0
        self._no_bid_size = 0.0
        self._yes_ask_size = 0.0
        self._no_ask_size = 0.0
        self._last_price_ts = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Start the Binance price feed as a background asyncio task.

        The RTDSClient reconnects automatically on disconnect, so this task
        runs for the lifetime of the observer.
        """
        asyncio.create_task(
            self._rtds.run_forever(),
            name=f"rtds_{self.asset.lower()}",
        )
        logger.info(
            "AssetWatcher started asset={} binance_url={}",
            self.asset, self._asset_cfg.binance_ws_url,
        )

    # ------------------------------------------------------------------ #
    # Market resolution
    # ------------------------------------------------------------------ #

    async def resolve_market(self, window_ts: int) -> bool:
        """
        Find the Polymarket 15m market for this asset and window timestamp.

        Tries the current window slug first, then the next window (handles
        the case where we're right at a window boundary and the market just
        opened). Logs the resolved market title for slug-format verification.

        Returns True if the market was successfully resolved.
        """
        self._reset_market_state(window_ts)
        slug = f"{self._asset_cfg.slug_prefix}-{window_ts}"

        # Fetch YES (Up) and NO (Down) tokens separately so we always get both
        # token IDs from the SAME market. Do not fall forward to the next
        # window slug; that can bind the current window to a future market and
        # poison the entire window with one-sided books.
        yes_info: MarketInfo | None = None
        no_info: MarketInfo | None = None
        fetch_error: str = ""

        for outcome_label in ("Up", "Down"):
            try:
                result = await self._client.fetch_market_lookup(
                    market_id="", market_slug=slug, outcome=outcome_label
                )
                if outcome_label == "Up":
                    yes_info = result
                else:
                    no_info = result
            except Exception as exc:
                fetch_error = str(exc)
                logger.warning(
                    "Market lookup failed asset={} slug={} outcome={} error={}",
                    self.asset, slug, outcome_label, exc,
                )

        if not yes_info and not no_info:
            if not fetch_error:
                logger.warning("Could not find any tokens asset={} slug={}", self.asset, slug)
            self._resolve_retry_due_at = time.time() + self._cfg.market_resolve_retry_seconds
            logger.warning(
                "Failed to resolve market asset={} window_ts={} slug={} — will retry in {:.1f}s",
                self.asset,
                window_ts,
                slug,
                self._cfg.market_resolve_retry_seconds,
            )
            return False

        if yes_info is None or no_info is None:
            missing = "YES" if yes_info is None else "NO"
            logger.warning(
                "Resolved partial market asset={} slug={} — missing {} token. "
                "Will retry instead of using one-sided books.",
                self.asset,
                slug,
                missing,
            )
            self._resolve_retry_due_at = time.time() + self._cfg.market_resolve_retry_seconds
            return False

        # --- Commit resolved market state ---
        self._market_id = yes_info.market_id
        self._yes_token_id = yes_info.token_id
        self._no_token_id = no_info.token_id
        self._market_title = yes_info.title
        self._market_liquidity = max(yes_info.liquidity, no_info.liquidity)
        self._market_volume = max(yes_info.volume, no_info.volume)
        self._market_resolved = True
        self._resolve_retry_due_at = 0.0

        logger.info(
            "Market resolved asset={} window_ts={} slug={} title='{}' "
            "market_id={} yes_token={} no_token={} liquidity={:.0f}",
            self.asset, window_ts, slug, self._market_title,
            self._market_id,
            self._yes_token_id or "MISSING",
            self._no_token_id or "MISSING",
            self._market_liquidity,
        )

        # --- Start WebSocket feed for the new window (if enabled) ---
        if self._cfg_use_ws:
            await self._start_ws_feed()

        return True

    async def _start_ws_feed(self) -> None:
        """
        Stop any previous WS feed and start a fresh one for the current
        market window.  Called after market resolution succeeds.
        """
        # Stop the previous feed cleanly before starting a new one
        if self._ws_feed is not None:
            try:
                await self._ws_feed.stop()
            except Exception as exc:
                logger.debug("WS feed stop error asset={} error={}", self.asset, exc)
            self._ws_feed = None

        if not self._market_id or not self._yes_token_id or not self._no_token_id:
            logger.warning(
                "Cannot start WS feed asset={} — market IDs not fully resolved",
                self.asset,
            )
            return

        self._ws_feed = PolymarketWSFeed()
        await self._ws_feed.start(
            self._market_id,
            self._yes_token_id,
            self._no_token_id,
        )
        logger.info("WS feed started asset={} market_id={}", self.asset, self._market_id)

    async def _stop_ws_feed(self) -> None:
        """Stop the WS feed cleanly (called on window close / watcher shutdown)."""
        if self._ws_feed is not None:
            try:
                await self._ws_feed.stop()
            except Exception as exc:
                logger.debug("WS feed stop error asset={} error={}", self.asset, exc)
            self._ws_feed = None

    # ------------------------------------------------------------------ #
    # Price refreshing
    # ------------------------------------------------------------------ #

    async def refresh_prices(self) -> None:
        """
        Fetch the latest YES and NO best-ask prices from the Polymarket orderbook.

        When use_websocket_feed is True and the WS feed has delivered a price
        update within the last 2 seconds, the WS values are copied directly
        into the internal price fields and the REST orderbook fetch is skipped
        (saving ~150 ms per poll cycle).

        If the WS feed is stale (no update for > 2 s) or not connected, the
        method falls through to the existing REST fetch path automatically.
        The ask price is what we'd pay to BUY — correct for entry cost calc.
        """
        if not self._market_resolved or (not self._yes_token_id and not self._no_token_id):
            now = time.time()
            if now >= self._resolve_retry_due_at:
                await self.resolve_market(current_window_ts(self._cfg.window_duration_seconds))
            if not self._market_resolved:
                return

        # Binance REST fallback: keep price warm when RTDS WebSocket is stale
        if not self._rtds.snapshot().is_fresh():
            await self._fetch_rest_price_fallback()

        # ---- WebSocket fast path (skips REST fetch when feed is fresh) ----
        if (
            self._cfg_use_ws
            and self._ws_feed is not None
            and self._ws_feed.is_connected
        ):
            ws_age = time.time() - self._ws_feed.last_update_ts
            if ws_age < 2.0 and self._ws_feed.last_update_ts > 0:
                # Feed is live and fresh — copy prices and skip REST
                self._yes_ask = self._ws_feed.yes_ask
                self._yes_ask_size = self._ws_feed.yes_ask_size
                self._no_ask = self._ws_feed.no_ask
                self._no_ask_size = self._ws_feed.no_ask_size
                self._last_price_ts = time.time()
                return
            else:
                logger.warning(
                    "WS feed stale asset={} age={:.2f}s, falling back to REST",
                    self.asset,
                    ws_age,
                )

        # Fetch YES orderbook
        if self._yes_token_id:
            try:
                yes_ob = await self._client.get_orderbook(self._yes_token_id)
                if yes_ob and yes_ob.bids:
                    self._yes_bid = float(yes_ob.bids[0].price)
                    self._yes_bid_size = float(yes_ob.bids[0].size)
                else:
                    self._yes_bid = 0.0
                    self._yes_bid_size = 0.0
                if yes_ob and yes_ob.asks:
                    self._yes_ask = float(yes_ob.asks[0].price)
                    self._yes_ask_size = float(yes_ob.asks[0].size)
                else:
                    self._yes_ask = 0.0
                    self._yes_ask_size = 0.0
            except Exception as exc:
                logger.debug("YES orderbook fetch failed asset={} error={}", self.asset, exc)
                self._yes_bid = 0.0
                self._yes_bid_size = 0.0
                self._yes_ask = 0.0
                self._yes_ask_size = 0.0

        # Fetch NO orderbook
        if self._no_token_id:
            try:
                no_ob = await self._client.get_orderbook(self._no_token_id)
                if no_ob and no_ob.bids:
                    self._no_bid = float(no_ob.bids[0].price)
                    self._no_bid_size = float(no_ob.bids[0].size)
                else:
                    self._no_bid = 0.0
                    self._no_bid_size = 0.0
                if no_ob and no_ob.asks:
                    self._no_ask = float(no_ob.asks[0].price)
                    self._no_ask_size = float(no_ob.asks[0].size)
                else:
                    self._no_ask = 0.0
                    self._no_ask_size = 0.0
            except Exception as exc:
                logger.debug("NO orderbook fetch failed asset={} error={}", self.asset, exc)
                self._no_bid = 0.0
                self._no_bid_size = 0.0
                self._no_ask = 0.0
                self._no_ask_size = 0.0

        self._last_price_ts = time.time()

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #

    async def _fetch_rest_price_fallback(self) -> None:
        """
        One-shot Binance REST poll to get the current spot price when the RTDS
        WebSocket feed is stale.  Updates _rest_fallback_price.

        Cooldown: at most once every 5 seconds to avoid hammering the API.
        Errors are caught silently (logged at DEBUG only).
        """
        now = time.time()
        if now - self._last_rest_poll_ts < 5.0:
            return
        self._last_rest_poll_ts = now

        # Derive symbol from the binance_ws_url, e.g.
        # wss://stream.binance.com/ws/btcusdt@aggTrade → BTCUSDT
        ws_url = self._asset_cfg.binance_ws_url
        symbol = "BTCUSDT"
        try:
            stream_part = ws_url.rstrip("/").split("/ws/")[-1]  # e.g. "btcusdt@aggTrade"
            pair = stream_part.split("@")[0].upper()            # e.g. "BTCUSDT"
            if pair:
                symbol = pair
        except Exception:
            pass

        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    data = await resp.json()
                    price = float(data["price"])
                    if price > 0:
                        self._rest_fallback_price = price
                        logger.info(
                            "RTDS stale — using Binance REST fallback price={}",
                            price,
                        )
        except Exception as exc:
            logger.debug(
                "Binance REST fallback fetch failed asset={} symbol={} error={}",
                self.asset, symbol, exc,
            )

    def asset_price(self) -> float:
        """Current BTC/USD or ETH/USD price from the Binance feed.

        Returns the RTDS WebSocket price when live; falls back to the last
        successful REST poll price when the WebSocket feed is stale.
        """
        snap = self._rtds.snapshot()
        if snap.is_fresh() and snap.binance_price > 0:
            return snap.binance_price
        if self._rest_fallback_price > 0:
            return self._rest_fallback_price
        return snap.binance_price

    def rtds_is_fresh(self) -> bool:
        """True if the Binance price feed is live and not stale."""
        return self._rtds.snapshot().is_fresh()

    def prices_are_fresh(self) -> bool:
        """True if YES/NO orderbook prices were refreshed recently."""
        age = time.time() - self._last_price_ts
        return age <= self._cfg.price_staleness_max_seconds

    def yes_ask(self) -> float:
        return self._yes_ask

    def no_ask(self) -> float:
        return self._no_ask

    def yes_bid(self) -> float:
        return self._yes_bid

    def no_bid(self) -> float:
        return self._no_bid

    def yes_ask_size(self) -> float:
        return self._yes_ask_size

    def no_ask_size(self) -> float:
        return self._no_ask_size

    def yes_bid_size(self) -> float:
        return self._yes_bid_size

    def no_bid_size(self) -> float:
        return self._no_bid_size

    def is_ready(self) -> bool:
        """True if all required data is available and fresh."""
        return (
            self._market_resolved
            and self.rtds_is_fresh()
            and self.prices_are_fresh()
            and self._yes_ask > 0
            and self._no_ask > 0
            and self.asset_price() > 0
        )

    def readiness_reason(self) -> str:
        """
        Explain why the watcher is not currently ready for signal evaluation.

        This keeps the observer/report honest when a generic PRICES_STALE gate
        is really being caused by one specific missing dependency.
        """
        if not self._market_resolved:
            return "market_unresolved"
        if not self.rtds_is_fresh():
            return "asset_feed_stale"
        if not self.prices_are_fresh():
            return "orderbook_refresh_stale"
        if self._yes_ask <= 0:
            return "yes_ask_missing"
        if self._no_ask <= 0:
            return "no_ask_missing"
        if self.asset_price() <= 0:
            return "asset_price_missing"
        return "ready"

    def market_meta(self) -> dict[str, Any]:
        """Return market identifiers and metadata for the signal event log."""
        return {
            "market_id": self._market_id,
            "yes_token_id": self._yes_token_id,
            "no_token_id": self._no_token_id,
            "liquidity": self._market_liquidity,
            "volume": self._market_volume,
            "yes_ask_size": self._yes_ask_size,
            "no_ask_size": self._no_ask_size,
        }


# ---------------------------------------------------------------------------
# CryptoMarketWatcher — orchestrates BTC and ETH asset watchers
# ---------------------------------------------------------------------------

class CryptoMarketWatcher:
    """
    Orchestrates AssetWatcher instances for BTC and/or ETH.

    On each poll cycle the run loop should call:
      1. check_window_transition() → returns list of assets that just transitioned
      2. refresh_all_prices()      → update YES/NO prices for all assets
      3. active_assets()           → get list of assets to evaluate this cycle
      4. get_watcher(asset)        → access per-asset state

    Window transitions trigger market re-resolution for affected assets and
    signal the direction signal evaluator to reset its per-window state.
    """

    def __init__(
        self,
        cfg: CryptoDirectionConfig,
        polymarket_client: Any,
    ) -> None:
        self._cfg = cfg
        self._watchers: dict[str, AssetWatcher] = {}

        if cfg.track_btc:
            self._watchers["BTC"] = AssetWatcher("BTC", cfg.btc, polymarket_client, cfg)
        if cfg.track_eth:
            self._watchers["ETH"] = AssetWatcher("ETH", cfg.eth, polymarket_client, cfg)

        self._last_window_ts: int = 0   # last known window timestamp
        self._price_refresh_due_at: float = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Start all asset Binance price feeds as background tasks."""
        for watcher in self._watchers.values():
            await watcher.start()

    # ------------------------------------------------------------------ #
    # Window management
    # ------------------------------------------------------------------ #

    async def check_window_transition(self) -> list[str]:
        """
        Detect 15-minute window rollovers.

        Returns a list of asset names that experienced a window transition
        this call. Market resolution may still be catching up after the
        rollover; evaluators should still roll their window state on time.
        Returns empty list if no transition occurred.

        The caller (direction signal run loop) should:
          - Call evaluator.on_window_open() for each transitioned asset
          - Log the transition prominently for monitoring
        """
        new_ts = current_window_ts(self._cfg.window_duration_seconds)
        if new_ts == self._last_window_ts:
            return []   # still in the same window

        logger.info(
            "Window transition detected old_ts={} new_ts={} assets={}",
            self._last_window_ts, new_ts, list(self._watchers.keys()),
        )
        self._last_window_ts = new_ts
        transitioned: list[str] = []

        for asset, watcher in self._watchers.items():
            success = await watcher.resolve_market(new_ts)
            transitioned.append(asset)
            if not success:
                logger.warning(
                    "Window transition: market NOT resolved asset={} window_ts={}",
                    asset, new_ts,
                )

        return transitioned

    # ------------------------------------------------------------------ #
    # Price refreshing
    # ------------------------------------------------------------------ #

    async def refresh_all_prices(self) -> None:
        """
        Refresh YES/NO orderbook prices for all watchers.

        Throttled by price_refresh_interval_seconds to avoid hammering
        the Polymarket API unnecessarily.
        """
        now = time.time()
        if now < self._price_refresh_due_at:
            return
        self._price_refresh_due_at = now + self._cfg.price_refresh_interval_seconds

        for watcher in self._watchers.values():
            await watcher.refresh_prices()

    # ------------------------------------------------------------------ #
    # Asset selection
    # ------------------------------------------------------------------ #

    def active_assets(self) -> list[str]:
        """
        Return the list of assets to evaluate on this poll cycle.

        If prefer_higher_liquidity is True: return only the single asset
        with higher Polymarket market liquidity (better fills, tighter spread).

        If prefer_higher_liquidity is False: return all tracked assets
        so signals can fire independently for BTC and ETH.

        Falls back to BTC if ETH is not resolved or has no liquidity data.
        """
        if not self._cfg.prefer_higher_liquidity:
            # Evaluate all assets independently
            return list(self._watchers.keys())

        # Pick the one with higher liquidity
        best_asset: str | None = None
        best_liquidity: float = -1.0

        for asset, watcher in self._watchers.items():
            liq = watcher._market_liquidity
            if liq > best_liquidity:
                best_liquidity = liq
                best_asset = asset

        if best_asset:
            return [best_asset]

        # Fallback: return BTC if available, else whatever we have
        if "BTC" in self._watchers:
            return ["BTC"]
        return list(self._watchers.keys())[:1]

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_watcher(self, asset: str) -> AssetWatcher | None:
        """Return the AssetWatcher for the given asset name, or None."""
        return self._watchers.get(asset)

    def current_window_ts(self) -> int:
        """Return the last known window timestamp (set on first transition)."""
        return self._last_window_ts
