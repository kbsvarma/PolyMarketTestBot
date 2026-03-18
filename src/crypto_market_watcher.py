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

try:
    from loguru import logger  # type: ignore[import]
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from src.config import AssetVolConfig, CryptoDirectionConfig
from src.models import MarketInfo
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

        # --- Live YES/NO prices ---
        self._yes_ask: float = 0.0
        self._no_ask: float = 0.0
        self._last_price_ts: float = 0.0        # time() when prices were last refreshed

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
        # Build candidate slugs: try current window, then next
        candidates = [
            f"{self._asset_cfg.slug_prefix}-{window_ts}",
            f"{self._asset_cfg.slug_prefix}-{window_ts + self._cfg.window_duration_seconds}",
        ]

        for slug in candidates:
            # Fetch YES (Up) and NO (Down) tokens separately so we always
            # get both token IDs from the same market.  The Gamma API returns
            # one market record with two clobTokenIds; fetch_market_lookup
            # matches the right one when outcome= is supplied.
            yes_info: MarketInfo | None = None
            no_info: MarketInfo | None = None
            fetch_error: str = ""

            for outcome_label, attr in (("Up", "yes_info"), ("Down", "no_info")):
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
                if fetch_error:
                    continue  # already logged
                logger.warning(
                    "Could not find any tokens asset={} slug={}",
                    self.asset, slug,
                )
                continue

            if yes_info and not no_info:
                logger.warning(
                    "Only found YES token for asset={} slug={} — "
                    "NO token missing. Prices for NO will be zero.",
                    self.asset, slug,
                )
            if no_info and not yes_info:
                logger.warning(
                    "Only found NO token for asset={} slug={} — "
                    "YES token missing. Prices for YES will be zero.",
                    self.asset, slug,
                )

            # --- Commit resolved market state ---
            self._current_window_ts = window_ts
            self._market_id = (yes_info or no_info).market_id           # type: ignore[union-attr]
            self._yes_token_id = yes_info.token_id if yes_info else ""
            self._no_token_id = no_info.token_id if no_info else ""
            self._market_title = (yes_info or no_info).title             # type: ignore[union-attr]
            self._market_liquidity = max(
                (yes_info.liquidity if yes_info else 0.0),
                (no_info.liquidity if no_info else 0.0),
            )
            self._market_volume = max(
                (yes_info.volume if yes_info else 0.0),
                (no_info.volume if no_info else 0.0),
            )
            self._market_resolved = True

            # CRITICAL: Log the actual market title every window so you can verify
            # that the slug prefix in config is producing the right market.
            logger.info(
                "Market resolved asset={} window_ts={} slug={} title='{}' "
                "market_id={} yes_token={} no_token={} liquidity={:.0f}",
                self.asset, window_ts, slug, self._market_title,
                self._market_id,
                self._yes_token_id or "MISSING",
                self._no_token_id or "MISSING",
                self._market_liquidity,
            )
            return True

        # Failed to resolve
        self._market_resolved = False
        logger.warning(
            "Failed to resolve market asset={} window_ts={} tried={}",
            self.asset, window_ts, candidates,
        )
        return False

    # ------------------------------------------------------------------ #
    # Price refreshing
    # ------------------------------------------------------------------ #

    async def refresh_prices(self) -> None:
        """
        Fetch the latest YES and NO best-ask prices from the Polymarket orderbook.

        Uses the CLOB orderbook endpoint directly (not WebSocket) since we
        only need snapshots every 2 seconds, not tick-by-tick streaming.
        The ask price is what we'd pay to BUY — correct for entry cost calc.
        """
        if not self._market_resolved or (not self._yes_token_id and not self._no_token_id):
            return

        # Fetch YES orderbook
        if self._yes_token_id:
            try:
                yes_ob = await self._client.get_orderbook(self._yes_token_id)
                if yes_ob and yes_ob.asks:
                    self._yes_ask = float(yes_ob.asks[0].price)
                else:
                    self._yes_ask = 0.0
            except Exception as exc:
                logger.debug("YES orderbook fetch failed asset={} error={}", self.asset, exc)
                self._yes_ask = 0.0

        # Fetch NO orderbook
        if self._no_token_id:
            try:
                no_ob = await self._client.get_orderbook(self._no_token_id)
                if no_ob and no_ob.asks:
                    self._no_ask = float(no_ob.asks[0].price)
                else:
                    self._no_ask = 0.0
            except Exception as exc:
                logger.debug("NO orderbook fetch failed asset={} error={}", self.asset, exc)
                self._no_ask = 0.0

        self._last_price_ts = time.time()

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #

    def asset_price(self) -> float:
        """Current BTC/USD or ETH/USD price from the Binance feed."""
        return self._rtds.snapshot().binance_price

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

    def market_meta(self) -> dict[str, Any]:
        """Return market identifiers and metadata for the signal event log."""
        return {
            "market_id": self._market_id,
            "yes_token_id": self._yes_token_id,
            "no_token_id": self._no_token_id,
            "liquidity": self._market_liquidity,
            "volume": self._market_volume,
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
        this call (i.e., successfully resolved their market for the new window).
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
            if success:
                transitioned.append(asset)
            else:
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
