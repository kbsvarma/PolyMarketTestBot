from __future__ import annotations

import base64
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
except ImportError:  # pragma: no cover
    hashes = None
    serialization = None
    padding = None

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover
    def retry(*_: object, **__: object):
        def decorator(func):
            return func
        return decorator

    def stop_after_attempt(*_: object):
        return None

    def wait_exponential(*_: object, **__: object):
        return None

    def retry_if_exception_type(*_: object):
        return None

from src.config import AppConfig
from src.models import MarketInfo, OrderbookLevel, OrderbookSnapshot, SourceQuality


class KalshiClient:
    """Kalshi venue adapter for the crypto bracket runtime."""

    def __init__(self, config: AppConfig, http_client: Any | None = None) -> None:
        self.config = config
        self.base_url = config.env.kalshi_api_base_url.rstrip("/")
        self.headers = {"User-Agent": config.endpoints.user_agent}
        self.client = http_client or (httpx.AsyncClient(timeout=12.0, headers=self.headers) if httpx else None)
        self._private_key = self._load_private_key()
        self._sdk_init_error = ""
        self._window_market_cache: dict[tuple[str, int], dict[str, Any]] = {}

    async def close(self) -> None:
        if self.client and hasattr(self.client, "aclose"):
            await self.client.aclose()

    def live_order_capable(self) -> tuple[bool, str]:
        if not self.config.env.live_trading_enabled:
            return False, "LIVE_TRADING_ENABLED is false."
        if self.client is None:
            return False, "httpx is not installed."
        if not self.config.env.kalshi_api_key_id:
            return False, "KALSHI_API_KEY_ID is missing."
        if self._private_key is None:
            return False, self._sdk_init_error or "Kalshi private key is not configured."
        return True, "Kalshi signed REST order methods are available."

    def _load_private_key(self) -> Any:
        if serialization is None:
            self._sdk_init_error = "cryptography import failed."
            return None

        pem_text = (self.config.env.kalshi_private_key or "").strip()
        pem_path = (self.config.env.kalshi_private_key_path or "").strip()

        if not pem_text and pem_path:
            path = Path(pem_path).expanduser()
            try:
                pem_text = path.read_text(encoding="utf-8")
            except Exception as exc:
                self._sdk_init_error = f"Unable to read KALSHI_PRIVATE_KEY_PATH: {exc}"
                return None

        if not pem_text:
            self._sdk_init_error = "Kalshi private key is missing."
            return None

        pem_text = pem_text.replace("\\n", "\n")
        try:
            self._sdk_init_error = ""
            return serialization.load_pem_private_key(pem_text.encode("utf-8"), password=None)
        except Exception as exc:
            self._sdk_init_error = f"Unable to parse Kalshi private key: {exc}"
            return None

    def _live_mode(self) -> bool:
        return self._mode_value() == "LIVE"

    def _mode_value(self) -> str:
        return self.config.mode.value if hasattr(self.config.mode, "value") else str(self.config.mode)

    def _ensure_live_data(self, payload: Any, context: str) -> Any:
        if self._live_mode() and (payload is None or payload == [] or payload == {}):
            raise RuntimeError(f"Live-critical payload missing for {context}")
        return payload

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        auth: bool = False,
    ) -> Any:
        if self.client is None:
            raise RuntimeError("httpx is not installed.")

        headers = dict(self.headers)
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        if auth:
            headers.update(self._auth_headers(method, path))

        response = await self.client.request(
            method.upper(),
            f"{self.base_url}{path}",
            params=params,
            json=json_body,
            headers=headers,
        )
        if response.status_code >= 400:
            detail = response.text[:500].strip()
            raise RuntimeError(f"Kalshi {method.upper()} {path} failed status={response.status_code} detail={detail}")
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if self._private_key is None or hashes is None or padding is None:
            raise RuntimeError(self._sdk_init_error or "Kalshi signing key unavailable.")
        if not self.config.env.kalshi_api_key_id:
            raise RuntimeError("KALSHI_API_KEY_ID is missing.")

        timestamp = str(int(time.time() * 1000))
        sign_path = path.split("?", 1)[0]
        message = f"{timestamp}{method.upper()}{sign_path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.config.env.kalshi_api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    @staticmethod
    def _parse_iso8601(value: Any) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None

    @staticmethod
    def _to_float(value: Any) -> float:
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _format_dollars(value: float) -> str:
        return f"{max(0.0, min(float(value), 0.9999)):.4f}"

    @staticmethod
    def _format_count_fp(value: float) -> str:
        return f"{float(value):.2f}"

    @staticmethod
    def _token_id_for(ticker: str, side: str) -> str:
        return f"kalshi:{ticker}:{side.lower()}"

    @staticmethod
    def _parse_token_id(token_id: str) -> tuple[str, str]:
        prefix, sep, side = str(token_id or "").rpartition(":")
        if sep and prefix.startswith("kalshi:") and side.lower() in {"yes", "no"}:
            return prefix.split("kalshi:", 1)[1], side.lower()
        raise ValueError(f"Malformed Kalshi token id: {token_id}")

    @staticmethod
    def _parse_synthetic_window_slug(market_slug: str) -> tuple[str, int] | None:
        prefix, sep, suffix = str(market_slug or "").rpartition("-")
        if not sep or not suffix.isdigit() or len(suffix) < 9:
            return None
        return prefix, int(suffix)

    @staticmethod
    def _unwrap_market(payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            market = payload.get("market")
            if isinstance(market, dict):
                return market
            markets = payload.get("markets")
            if isinstance(markets, list):
                for item in markets:
                    if isinstance(item, dict):
                        return item
            return payload if payload else None
        return None

    @staticmethod
    def _unwrap_markets(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            rows = payload.get("markets")
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, dict)]
            if payload.get("market") and isinstance(payload["market"], dict):
                return [payload["market"]]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _market_category(self, market: dict[str, Any]) -> str:
        category = str(market.get("category") or "").strip().lower()
        if category:
            return category
        ticker = str(market.get("ticker") or "")
        if "BTC" in ticker or "ETH" in ticker or "SOL" in ticker:
            return "crypto price"
        return "unknown"

    def _build_market_info(self, market: dict[str, Any], side: str) -> MarketInfo:
        ticker = str(market.get("ticker") or "")
        title = str(market.get("title") or ticker or "Kalshi market")
        outcome_name = "YES" if side == "yes" else "NO"
        yes_ask = self._to_float(market.get("yes_ask_dollars"))
        no_ask = self._to_float(market.get("no_ask_dollars"))
        return MarketInfo(
            market_id=ticker,
            token_id=self._token_id_for(ticker, side),
            title=f"{title} [{outcome_name}]",
            slug=ticker,
            category=self._market_category(market),
            outcome_name=outcome_name,
            outcome_index=0 if side == "yes" else 1,
            outcome_prices=[yes_ask, no_ask],
            active=str(market.get("status") or "").lower() == "active",
            closed=str(market.get("status") or "").lower() in {"finalized", "settled", "closed"},
            accepting_orders=str(market.get("status") or "").lower() == "active",
            liquidity=self._to_float(market.get("liquidity_dollars")),
            volume=self._to_float(market.get("volume_fp") or market.get("volume_24h_fp")),
            end_date_iso=str(market.get("close_time") or "") or None,
            resolution_source=str(market.get("rules_primary") or "") or None,
            source_quality=SourceQuality.REAL_PUBLIC_DATA,
        )

    async def _fetch_market_by_ticker(self, ticker: str) -> dict[str, Any] | None:
        payload = await self._request_json("GET", f"/markets/{quote(ticker, safe='')}")
        return self._unwrap_market(payload)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type(RuntimeError),
    )
    async def _fetch_series_market_for_window(self, series_ticker: str, window_ts: int) -> dict[str, Any] | None:
        cache_key = (series_ticker, int(window_ts))
        if cache_key in self._window_market_cache:
            return dict(self._window_market_cache[cache_key])

        target_open = datetime.fromtimestamp(window_ts, tz=timezone.utc)
        target_close = datetime.fromtimestamp(
            window_ts + int(self.config.crypto_direction.window_duration_seconds),
            tz=timezone.utc,
        )
        payload = await self._request_json(
            "GET",
            "/markets",
            params={"series_ticker": series_ticker, "status": "all", "limit": 200},
        )
        markets = self._unwrap_markets(payload)
        exact: dict[str, Any] | None = None
        containing: dict[str, Any] | None = None
        for market in markets:
            open_time = self._parse_iso8601(market.get("open_time"))
            close_time = self._parse_iso8601(market.get("close_time"))
            if open_time is None or close_time is None:
                continue
            if open_time == target_open and close_time == target_close:
                exact = market
                break
            if open_time <= target_open < close_time:
                containing = market
        chosen = exact or containing
        if chosen is not None:
            self._window_market_cache[cache_key] = dict(chosen)
        return chosen

    async def _resolve_market_item(
        self,
        *,
        market_id: str = "",
        token_id: str = "",
        market_slug: str = "",
        refresh: bool = False,
    ) -> dict[str, Any] | None:
        if token_id:
            ticker, _ = self._parse_token_id(token_id)
            return await self._fetch_market_by_ticker(ticker)
        if market_id:
            return await self._fetch_market_by_ticker(market_id)
        if market_slug:
            synthetic = self._parse_synthetic_window_slug(market_slug)
            if synthetic is not None:
                series_ticker, window_ts = synthetic
                if refresh:
                    self._window_market_cache.pop((series_ticker, int(window_ts)), None)
                return await self._fetch_series_market_for_window(series_ticker, window_ts)
            return await self._fetch_market_by_ticker(market_slug)
        return None

    async def fetch_market_lookup(
        self,
        market_id: str,
        token_id: str = "",
        market_slug: str = "",
        outcome: str = "",
    ) -> MarketInfo | None:
        market = await self._resolve_market_item(
            market_id=market_id,
            token_id=token_id,
            market_slug=market_slug,
        )
        if not market:
            return None

        if token_id:
            _, side = self._parse_token_id(token_id)
        else:
            label = str(outcome or "").strip().lower()
            side = "yes" if label in {"up", "yes", "higher", "above"} else "no"
        return self._build_market_info(market, side)

    async def fetch_market_resolution(self, market_id: str = "", market_slug: str = "") -> dict[str, Any] | None:
        market = await self._resolve_market_item(
            market_id=market_id,
            market_slug=market_slug,
            refresh=True,
        )
        if not market:
            return None
        result = str(market.get("result") or "").strip().lower()
        resolved_yes: bool | None = None
        if result == "yes":
            resolved_yes = True
        elif result == "no":
            resolved_yes = False
        return {
            "resolved_yes": resolved_yes,
            "source": "kalshi_market_result" if resolved_yes is not None else "kalshi_market_metadata",
            "market_slug": str(market.get("ticker") or ""),
            "market_id": str(market.get("ticker") or ""),
            "status": str(market.get("status") or ""),
            "result": result,
            "outcome_prices": [
                self._to_float(market.get("yes_ask_dollars")),
                self._to_float(market.get("no_ask_dollars")),
            ],
        }

    @staticmethod
    def _parse_orderbook_levels(raw_levels: Any) -> list[OrderbookLevel]:
        rows: list[OrderbookLevel] = []
        for item in raw_levels or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                price, size = item[0], item[1]
            elif isinstance(item, dict):
                price = item.get("price_dollars") or item.get("price")
                size = item.get("count_fp") or item.get("size")
            else:
                continue
            try:
                rows.append(OrderbookLevel(price=float(price), size=float(size)))
            except (TypeError, ValueError):
                continue
        return rows

    @staticmethod
    def _implied_asks_from_bids(levels: list[OrderbookLevel]) -> list[OrderbookLevel]:
        asks: list[OrderbookLevel] = []
        for level in levels:
            ask_price = max(0.0, min(1.0, round(1.0 - float(level.price), 4)))
            asks.append(OrderbookLevel(price=ask_price, size=float(level.size)))
        return asks

    async def get_orderbook(self, token_id: str) -> OrderbookSnapshot:
        ticker, side = self._parse_token_id(token_id)
        try:
            payload = await self._request_json("GET", f"/markets/{quote(ticker, safe='')}/orderbook")
            market = await self._fetch_market_by_ticker(ticker)
            orderbook = payload.get("orderbook_fp") or payload.get("orderbook") or {}
            yes_bid_levels = sorted(
                self._parse_orderbook_levels(orderbook.get("yes_dollars") or orderbook.get("yes")),
                key=lambda level: level.price,
                reverse=True,
            )
            no_bid_levels = sorted(
                self._parse_orderbook_levels(orderbook.get("no_dollars") or orderbook.get("no")),
                key=lambda level: level.price,
                reverse=True,
            )

            if market:
                if not yes_bid_levels and self._to_float(market.get("yes_bid_dollars")) > 0:
                    yes_bid_levels = [
                        OrderbookLevel(
                            price=self._to_float(market.get("yes_bid_dollars")),
                            size=self._to_float(market.get("yes_bid_size_fp")),
                        )
                    ]
                if not no_bid_levels and self._to_float(market.get("no_bid_dollars")) > 0:
                    no_bid_levels = [
                        OrderbookLevel(
                            price=self._to_float(market.get("no_bid_dollars")),
                            size=self._to_float(market.get("no_bid_size_fp")),
                        )
                    ]

            yes_ask_levels = sorted(self._implied_asks_from_bids(no_bid_levels), key=lambda level: level.price)
            no_ask_levels = sorted(self._implied_asks_from_bids(yes_bid_levels), key=lambda level: level.price)

            if market:
                if not yes_ask_levels and self._to_float(market.get("yes_ask_dollars")) > 0:
                    yes_ask_levels = [
                        OrderbookLevel(
                            price=self._to_float(market.get("yes_ask_dollars")),
                            size=self._to_float(market.get("yes_ask_size_fp")),
                        )
                    ]
                if not no_ask_levels and self._to_float(market.get("no_ask_dollars")) > 0:
                    no_ask_levels = [
                        OrderbookLevel(
                            price=self._to_float(market.get("no_ask_dollars")),
                            size=self._to_float(market.get("no_ask_size_fp")),
                        )
                    ]

            bids = yes_bid_levels if side == "yes" else no_bid_levels
            asks = yes_ask_levels if side == "yes" else no_ask_levels
            if bids or asks:
                return OrderbookSnapshot(token_id=token_id, bids=bids, asks=asks)
            raise RuntimeError("Empty Kalshi orderbook")
        except Exception as exc:
            if self._live_mode():
                raise RuntimeError(f"Unable to fetch real Kalshi orderbook in LIVE mode: {exc}") from exc
            return OrderbookSnapshot(token_id=token_id, bids=[], asks=[])

    @staticmethod
    def _normalize_order_status(status: Any) -> str:
        normalized = str(status or "").strip().lower()
        if normalized == "executed":
            return "FILLED"
        if normalized == "canceled":
            return "CANCELLED"
        if normalized == "resting":
            return "RESTING"
        return normalized.upper()

    def _normalize_order_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_order = payload.get("order") if isinstance(payload.get("order"), dict) else payload
        side = str(raw_order.get("side") or "").strip().lower()
        price_value = raw_order.get("yes_price_dollars") if side == "yes" else raw_order.get("no_price_dollars")
        average_fill_price = self._to_float(price_value)
        filled_size = self._to_float(raw_order.get("fill_count_fp") or raw_order.get("fill_count"))
        remaining_size = self._to_float(raw_order.get("remaining_count_fp") or raw_order.get("remaining_count"))
        return {
            "exchange_order_id": str(raw_order.get("order_id") or raw_order.get("id") or ""),
            "status": self._normalize_order_status(raw_order.get("status")),
            "filled_size": filled_size,
            "average_fill_price": average_fill_price if filled_size > 0 else 0.0,
            "remaining_size": remaining_size,
            "client_order_id": str(raw_order.get("client_order_id") or ""),
            "raw": raw_order,
        }

    async def ensure_token_sell_allowance(self, token_id: str) -> dict[str, Any]:
        return {"ok": True, "venue": "kalshi", "token_id": token_id, "skipped": True}

    async def place_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        entry_style: str,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        live_capable, detail = self.live_order_capable()
        if not live_capable:
            raise RuntimeError(detail)

        ticker, contract_side = self._parse_token_id(token_id)
        rounded_size = max(1.0, round(float(size), 2))
        market = await self._fetch_market_by_ticker(ticker)
        if market and not bool(market.get("fractional_trading_enabled", False)):
            rounded_size = float(max(1, math.floor(rounded_size)))
        whole_contracts = float(rounded_size).is_integer()
        action = "buy" if side.upper() == "BUY" else "sell"
        tif = "good_till_canceled"
        if entry_style == "FOLLOW_TAKER":
            tif = "fill_or_kill"
        elif entry_style == "FOLLOW_TAKER_PARTIAL":
            tif = "immediate_or_cancel"

        payload: dict[str, Any] = {
            "ticker": ticker,
            "side": contract_side,
            "action": action,
            "type": "limit",
            "count_fp": self._format_count_fp(rounded_size),
            "time_in_force": tif,
        }
        if whole_contracts:
            payload["count"] = int(round(rounded_size))
        price_field = "yes_price_dollars" if contract_side == "yes" else "no_price_dollars"
        payload[price_field] = self._format_dollars(price)
        if client_order_id:
            payload["client_order_id"] = client_order_id
        if action == "buy":
            payload["buy_max_cost"] = max(1, int(math.ceil(float(price) * rounded_size * 100)))
        else:
            payload["reduce_only"] = True

        response = await self._request_json("POST", "/portfolio/orders", json_body=payload, auth=True)
        normalized = self._normalize_order_payload(response)
        if not normalized.get("exchange_order_id"):
            raise RuntimeError("Kalshi order placement response did not include order_id.")
        return normalized

    async def place_buy_order(
        self,
        token_id: str,
        price: float,
        size: float,
        entry_style: str,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.place_order(
            token_id=token_id,
            price=price,
            size=size,
            side="BUY",
            entry_style=entry_style,
            client_order_id=client_order_id,
        )

    async def place_sell_order(
        self,
        token_id: str,
        price: float,
        size: float,
        entry_style: str,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.place_order(
            token_id=token_id,
            price=price,
            size=size,
            side="SELL",
            entry_style=entry_style,
            client_order_id=client_order_id,
        )

    async def get_order_status(self, exchange_order_id: str) -> dict[str, Any]:
        response = await self._request_json(
            "GET",
            f"/portfolio/orders/{quote(exchange_order_id, safe='')}",
            auth=True,
        )
        return self._normalize_order_payload(response)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        response = await self._request_json(
            "DELETE",
            f"/portfolio/orders/{quote(order_id, safe='')}",
            auth=True,
        )
        normalized = self._normalize_order_payload(response)
        if not normalized.get("status"):
            normalized["status"] = "CANCELLED"
        return normalized

    async def get_balance(self) -> dict[str, Any]:
        response = await self._request_json("GET", "/portfolio/balance", auth=True)
        balance_cents = float(response.get("balance") or 0.0)
        portfolio_value_cents = float(response.get("portfolio_value") or 0.0)
        return {
            "cash_usd": balance_cents / 100.0,
            "portfolio_value_usd": portfolio_value_cents / 100.0,
            "updated_ts": response.get("updated_ts"),
            "source": "kalshi_balance",
            "source_quality": SourceQuality.REAL_PUBLIC_DATA.value,
        }

    async def get_allowance(self) -> dict[str, Any]:
        balance = await self.get_balance()
        available = float(balance.get("cash_usd") or 0.0)
        return {
            "available": available,
            "sufficient": available > 0,
            "source": "kalshi_balance",
            "query_visible": True,
            "source_quality": SourceQuality.REAL_PUBLIC_DATA.value,
        }
