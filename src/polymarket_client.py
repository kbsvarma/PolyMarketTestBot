from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, AssetType, BalanceAllowanceParams, OrderArgs, OrderType
except ImportError:  # pragma: no cover
    ClobClient = None
    ApiCreds = None
    AssetType = None
    BalanceAllowanceParams = None
    OrderArgs = None
    OrderType = None

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
from src.models import HealthStatus, MarketInfo, OrderbookLevel, OrderbookSnapshot, SourceQuality


class PolymarketClient:
    """Polymarket-specific client with fail-closed live behavior."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.base_url = config.endpoints.clob_base_url.rstrip("/")
        self.gamma_url = config.endpoints.gamma_base_url.rstrip("/")
        self.data_url = config.endpoints.data_base_url.rstrip("/")
        self.headers = {"User-Agent": config.endpoints.user_agent}
        self.client = httpx.AsyncClient(timeout=12.0, headers=self.headers) if httpx else None
        self._sdk_init_error = ""
        self._sdk_client = self._build_sdk_client()

    def _build_sdk_client(self) -> Any:
        if ClobClient is None:
            self._sdk_init_error = "py-clob-client import failed."
            return None
        if not self.config.env.polymarket_private_key:
            self._sdk_init_error = "POLYMARKET_PRIVATE_KEY is missing."
            return None
        try:
            creds = None
            if (
                ApiCreds is not None
                and self.config.env.polymarket_api_key
                and self.config.env.polymarket_api_secret
                and self.config.env.polymarket_api_passphrase
            ):
                creds = ApiCreds(
                    api_key=self.config.env.polymarket_api_key,
                    api_secret=self.config.env.polymarket_api_secret,
                    api_passphrase=self.config.env.polymarket_api_passphrase,
                )
            client = ClobClient(
                host=self.base_url,
                chain_id=self.config.env.polymarket_chain_id,
                key=self.config.env.polymarket_private_key,
                creds=creds,
                funder=self.config.env.polymarket_funder or None,
            )
            if creds is not None and hasattr(client, "set_api_creds"):
                client.set_api_creds(creds)
            elif hasattr(client, "create_or_derive_api_creds"):
                creds = client.create_or_derive_api_creds()
                client.set_api_creds(creds)
            self._sdk_init_error = ""
            return client
        except Exception as exc:
            self._sdk_init_error = str(exc)
            return None

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()

    async def _rpc_call(self, method: str, params: list[Any]) -> Any:
        if not self.client:
            raise RuntimeError("httpx unavailable")
        candidates = [
            self.config.env.polygon_rpc_url,
            "https://polygon-bor-rpc.publicnode.com",
            "https://polygon.drpc.org",
        ]
        last_error = "Polygon RPC request failed."
        for url in dict.fromkeys(candidates):
            try:
                response = await self.client.post(
                    url,
                    json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                )
                response.raise_for_status()
                payload = response.json()
                if "error" in payload:
                    last_error = str(payload["error"])
                    continue
                return payload.get("result")
            except Exception as exc:
                last_error = str(exc)
        raise RuntimeError(last_error)

    async def _erc20_balance(self, wallet_address: str, token_address: str, decimals: int = 6) -> float:
        if not wallet_address or not token_address:
            return 0.0
        wallet = wallet_address.lower().replace("0x", "").rjust(64, "0")
        data = f"0x70a08231{wallet}"
        result = await self._rpc_call(
            "eth_call",
            [{"to": token_address, "data": data}, "latest"],
        )
        if not result:
            return 0.0
        raw_value = int(result, 16)
        return float(Decimal(raw_value) / Decimal(10**decimals))

    async def get_wallet_stablecoin_balances(self) -> dict[str, Any]:
        if not self.config.env.polymarket_funder:
            return {"wallet_address": "", "usdc": 0.0, "usdce": 0.0, "total_stablecoins": 0.0}
        usdc = await self._erc20_balance(self.config.env.polymarket_funder, self.config.env.polygon_usdc_address)
        usdce = await self._erc20_balance(self.config.env.polymarket_funder, self.config.env.polygon_usdce_address)
        return {
            "wallet_address": self.config.env.polymarket_funder,
            "usdc": usdc,
            "usdce": usdce,
            "total_stablecoins": usdc + usdce,
        }

    async def health_check(self) -> HealthStatus:
        if self._mode_value() == "LIVE":
            if self._sdk_client is None:
                detail = self._sdk_init_error or "py-clob-client not available or live credentials are incomplete."
                return HealthStatus(ok=False, detail=detail)
            missing = [
                name
                for name, value in {
                    "POLYMARKET_PRIVATE_KEY": self.config.env.polymarket_private_key,
                    "POLYMARKET_FUNDER": self.config.env.polymarket_funder,
                }.items()
                if not value
            ]
            if missing:
                return HealthStatus(ok=False, detail=f"Missing live auth inputs: {', '.join(missing)}")
        return HealthStatus(ok=True, detail="Client initialized.")

    async def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        if not self.client:
            raise RuntimeError("httpx unavailable")
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _live_mode(self) -> bool:
        return self._mode_value() == "LIVE"

    def _mode_value(self) -> str:
        return self.config.mode.value if hasattr(self.config.mode, "value") else str(self.config.mode)

    def _ensure_live_data(self, payload: Any, context: str) -> Any:
        if self._live_mode() and (payload is None or payload == [] or payload == {}):
            raise RuntimeError(f"Live-critical payload missing for {context}")
        return payload

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type(RuntimeError),
    )
    async def fetch_markets(self, limit: int = 100) -> list[MarketInfo]:
        try:
            payload = await self._get_json(f"{self.gamma_url}/markets", params={"limit": limit})
            markets = payload if isinstance(payload, list) else payload.get("markets", [])
            rows: list[MarketInfo] = []
            for item in markets:
                token_id = ""
                clob_token_ids = item.get("clobTokenIds")
                if isinstance(clob_token_ids, list) and clob_token_ids:
                    token_id = str(clob_token_ids[0])
                elif isinstance(clob_token_ids, str):
                    token_id = clob_token_ids.split(",")[0].strip("[]\" ")
                row = MarketInfo(
                    market_id=str(item.get("conditionId") or item.get("id") or item.get("slug", "")),
                    token_id=token_id or str(item.get("tokenId") or item.get("clobTokenId") or ""),
                    title=str(item.get("question") or item.get("title") or item.get("description") or "Unknown market"),
                    slug=str(item.get("slug") or item.get("marketSlug") or ""),
                    category=str(item.get("category") or "unknown"),
                    active=bool(item.get("active", True)),
                    closed=bool(item.get("closed", False)),
                    liquidity=float(item.get("liquidity") or 0.0),
                    volume=float(item.get("volume") or item.get("volumeNum") or 0.0),
                    end_date_iso=item.get("endDate") or item.get("end_date_iso"),
                    source_quality=SourceQuality.REAL_PUBLIC_DATA,
                )
                if row.market_id and row.token_id:
                    rows.append(row)
            rows = self._ensure_live_data(rows, "markets")
            return rows
        except Exception:
            if self._live_mode():
                raise RuntimeError("Unable to fetch real markets in LIVE mode.")
            return self._fallback_markets()

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type(RuntimeError),
    )
    async def fetch_wallet_activity(self, wallet_address: str, limit: int = 50) -> list[dict[str, Any]]:
        candidates = [
            (f"{self.data_url}/activity", {"user": wallet_address, "limit": limit}),
            (f"{self.data_url}/trades", {"user": wallet_address, "limit": limit}),
            (f"{self.gamma_url}/activity", {"address": wallet_address, "limit": limit}),
            (f"{self.gamma_url}/trades", {"address": wallet_address, "limit": limit}),
        ]
        for url, params in candidates:
            try:
                payload = await self._get_json(url, params=params)
                rows = payload if isinstance(payload, list) else payload.get("history", payload.get("data", []))
                if rows:
                    return rows
            except Exception:
                continue
        if self._live_mode():
            raise RuntimeError(f"Unable to fetch real wallet activity for {wallet_address} in LIVE mode.")
        return self._fallback_wallet_activity(wallet_address) if self._mode_value() == "RESEARCH" else []

    async def fetch_leaderboard(self, limit: int = 50) -> list[dict[str, Any]]:
        candidates = [
            (f"{self.data_url}/leaderboard", {"limit": limit}),
            (f"{self.gamma_url}/leaderboard", {"limit": limit}),
        ]
        for url, params in candidates:
            try:
                payload = await self._get_json(url, params=params)
                rows = payload if isinstance(payload, list) else payload.get("leaders", payload.get("data", []))
                if rows:
                    return rows
            except Exception:
                continue
        if self._live_mode():
            raise RuntimeError("Unable to fetch real leaderboard in LIVE mode.")
        return []

    async def fetch_top_holders(self, market_ids: list[str], per_market_limit: int = 10) -> list[dict[str, Any]]:
        if not market_ids:
            return []
        candidates = [
            (f"{self.data_url}/holders", {"market": ",".join(market_ids), "limit": per_market_limit, "minBalance": 1}),
        ]
        for url, params in candidates:
            try:
                payload = await self._get_json(url, params=params)
                rows = payload if isinstance(payload, list) else payload.get("data", payload.get("holders", []))
                if rows:
                    return rows
            except Exception:
                continue
        if self._live_mode():
            raise RuntimeError("Unable to fetch top holders in LIVE mode.")
        return []

    async def fetch_recent_public_activity(self, limit: int = 200) -> list[dict[str, Any]]:
        candidates = [
            (f"{self.data_url}/activity", {"limit": limit}),
            (f"{self.data_url}/trades", {"limit": limit}),
            (f"{self.gamma_url}/activity", {"limit": limit}),
            (f"{self.gamma_url}/trades", {"limit": limit}),
        ]
        for url, params in candidates:
            try:
                payload = await self._get_json(url, params=params)
                rows = payload if isinstance(payload, list) else payload.get("history", payload.get("data", []))
                if rows:
                    return rows
            except Exception:
                continue
        if self._live_mode():
            raise RuntimeError("Unable to fetch public recent activity in LIVE mode.")
        return []

    async def get_orderbook(self, token_id: str) -> OrderbookSnapshot:
        try:
            payload = await self._get_json(f"{self.base_url}/book", params={"token_id": token_id})
            bids = [OrderbookLevel(price=float(level["price"]), size=float(level["size"])) for level in payload.get("bids", [])]
            asks = [OrderbookLevel(price=float(level["price"]), size=float(level["size"])) for level in payload.get("asks", [])]
            if bids or asks:
                return OrderbookSnapshot(token_id=token_id, bids=bids, asks=asks)
            raise RuntimeError("Empty orderbook")
        except Exception as exc:
            if self._live_mode():
                raise RuntimeError(f"Unable to fetch real orderbook in LIVE mode: {exc}") from exc
            return OrderbookSnapshot(
                token_id=token_id,
                bids=[OrderbookLevel(price=0.47, size=120), OrderbookLevel(price=0.46, size=180)],
                asks=[OrderbookLevel(price=0.49, size=110), OrderbookLevel(price=0.50, size=150)],
            )

    async def get_balance(self) -> dict[str, Any]:
        if self._sdk_client is None:
            if self._live_mode():
                raise RuntimeError("Live balance unavailable because py-clob-client is not initialized.")
            return {"cash_usd": self.config.bankroll.live_bankroll_reference, "source": "reference", "source_quality": SourceQuality.SYNTHETIC_FALLBACK.value}
        try:
            if hasattr(self._sdk_client, "get_balance_allowance"):
                params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL) if BalanceAllowanceParams and AssetType else None
                payload = self._sdk_client.get_balance_allowance(params)
            elif hasattr(self._sdk_client, "get_balance"):
                payload = self._sdk_client.get_balance()
            else:
                raise RuntimeError("SDK balance method unavailable")
            return self._ensure_live_data(payload, "balance")
        except Exception as exc:
            if self._live_mode():
                raise RuntimeError(f"Unable to fetch real balance in LIVE mode: {exc}") from exc
            return {"cash_usd": self.config.bankroll.live_bankroll_reference, "source": "fallback", "source_quality": SourceQuality.SYNTHETIC_FALLBACK.value}

    async def get_allowance(self) -> dict[str, Any]:
        if self._sdk_client is None:
            if self._live_mode():
                raise RuntimeError("Allowance/spendability unavailable because py-clob-client is not initialized.")
            return {"available": self.config.bankroll.live_bankroll_reference, "sufficient": True, "source": "reference", "source_quality": SourceQuality.SYNTHETIC_FALLBACK.value}
        try:
            if hasattr(self._sdk_client, "get_balance_allowance"):
                params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL) if BalanceAllowanceParams and AssetType else None
                payload = self._sdk_client.get_balance_allowance(params)
                available = float(payload.get("available") or payload.get("balance") or payload.get("allowance") or 0.0)
                return {"available": available, "sufficient": available > 0, "raw": payload, "source_quality": SourceQuality.REAL_PUBLIC_DATA.value}
            if hasattr(self._sdk_client, "get_balance"):
                payload = self._sdk_client.get_balance()
                available = float(payload.get("available") or payload.get("balance") or 0.0)
                return {"available": available, "sufficient": available > 0, "raw": payload, "source_quality": SourceQuality.REAL_PUBLIC_DATA.value}
            raise RuntimeError("SDK allowance method unavailable")
        except Exception as exc:
            if self._live_mode():
                raise RuntimeError(f"Unable to fetch allowance/spendability in LIVE mode: {exc}") from exc
            return {"available": self.config.bankroll.live_bankroll_reference, "sufficient": True, "source": "fallback", "source_quality": SourceQuality.SYNTHETIC_FALLBACK.value}

    async def get_positions(self) -> list[dict[str, Any]]:
        if self._sdk_client and hasattr(self._sdk_client, "get_positions"):
            try:
                payload = self._sdk_client.get_positions()
                return self._ensure_live_data(payload, "positions")
            except Exception as exc:
                if self._live_mode():
                    raise RuntimeError(f"Unable to fetch positions in LIVE mode: {exc}") from exc
        if not self.config.env.polymarket_funder:
            if self._live_mode():
                raise RuntimeError("POLYMARKET_FUNDER is required for live position visibility.")
            return []
        candidates = [
            (f"{self.data_url}/positions", {"user": self.config.env.polymarket_funder}),
            (f"{self.gamma_url}/positions", {"user": self.config.env.polymarket_funder}),
        ]
        for url, params in candidates:
            try:
                payload = await self._get_json(url, params=params)
                rows = payload if isinstance(payload, list) else payload.get("data", payload.get("positions", []))
                if rows is not None and rows != []:
                    return rows
            except Exception:
                continue
        if self._live_mode():
            raise RuntimeError("Unable to fetch live positions.")
        return []

    async def get_open_orders(self) -> list[dict[str, Any]]:
        if self._sdk_client is None or not hasattr(self._sdk_client, "get_orders"):
            if self._live_mode():
                raise RuntimeError("Open order query unavailable in LIVE mode.")
            return []
        try:
            payload = self._sdk_client.get_orders()
            if payload is None:
                raise RuntimeError("Empty open orders response")
            normalized: list[dict[str, Any]] = []
            for item in payload:
                normalized.append(
                    {
                        "exchange_order_id": str(item.get("id") or item.get("orderID") or item.get("order_id") or ""),
                        "client_order_id": str(item.get("client_order_id") or item.get("clientOrderId") or ""),
                        "status": str(item.get("status") or item.get("state") or ""),
                        "market_id": str(item.get("market_id") or item.get("conditionId") or item.get("market") or ""),
                        "token_id": str(item.get("token_id") or item.get("asset_id") or item.get("tokenId") or ""),
                        "side": str(item.get("side") or "BUY"),
                        "size": float(item.get("size") or item.get("original_size") or item.get("shares") or 0.0),
                        "filled_size": float(item.get("filled_size") or item.get("filled") or 0.0),
                        "remaining_size": float(item.get("remaining_size") or item.get("remaining") or 0.0),
                        "price": float(item.get("price") or 0.0),
                        "raw": item,
                    }
                )
            return normalized
        except Exception as exc:
            if self._live_mode():
                raise RuntimeError(f"Unable to fetch open orders in LIVE mode: {exc}") from exc
            return []

    async def place_order(self, token_id: str, price: float, size: float, side: str, entry_style: str, client_order_id: str | None = None) -> dict[str, Any]:
        if not self.config.env.live_trading_enabled:
            raise RuntimeError("LIVE_TRADING_ENABLED is false.")
        if self._sdk_client is None or OrderArgs is None or OrderType is None:
            raise RuntimeError("py-clob-client is required for real live order placement.")
        order_type = OrderType.GTC
        try:
            if hasattr(OrderType, "FOK") and entry_style == "FOLLOW_TAKER":
                order_type = OrderType.FOK
            order_args = OrderArgs(
                price=price,
                size=size,
                side=side,
                token_id=token_id,
            )
            signed_order = self._sdk_client.create_order(order_args)
            if client_order_id and isinstance(signed_order, dict):
                signed_order["client_order_id"] = client_order_id
            result = self._sdk_client.post_order(signed_order, order_type)
            if not result:
                raise RuntimeError("Empty order placement response")
            return {
                "exchange_order_id": str(result.get("orderID") or result.get("id") or result.get("order_id") or ""),
                "status": str(result.get("status") or result.get("state") or "SUBMITTED"),
                "client_order_id": client_order_id or str(result.get("client_order_id") or ""),
                "raw": result,
            }
        except Exception as exc:
            raise RuntimeError(f"Real live order placement failed: {exc}") from exc

    async def place_buy_order(self, token_id: str, price: float, size: float, entry_style: str, client_order_id: str | None = None) -> dict[str, Any]:
        return await self.place_order(token_id=token_id, price=price, size=size, side="BUY", entry_style=entry_style, client_order_id=client_order_id)

    async def place_sell_order(self, token_id: str, price: float, size: float, entry_style: str, client_order_id: str | None = None) -> dict[str, Any]:
        return await self.place_order(token_id=token_id, price=price, size=size, side="SELL", entry_style=entry_style, client_order_id=client_order_id)

    async def get_order_status(self, exchange_order_id: str) -> dict[str, Any]:
        if self._sdk_client is None or not hasattr(self._sdk_client, "get_order"):
            raise RuntimeError("Order status query unavailable.")
        try:
            result = self._sdk_client.get_order(exchange_order_id)
            if not result:
                raise RuntimeError("Empty order status response")
            return {
                "exchange_order_id": str(result.get("id") or result.get("orderID") or exchange_order_id),
                "status": str(result.get("status") or result.get("state") or ""),
                "filled_size": float(result.get("filled_size") or result.get("filled") or 0.0),
                "average_fill_price": float(result.get("avg_price") or result.get("average_fill_price") or 0.0),
                "remaining_size": float(result.get("remaining_size") or result.get("remaining") or 0.0),
                "client_order_id": str(result.get("client_order_id") or result.get("clientOrderId") or ""),
                "raw": result,
            }
        except Exception as exc:
            raise RuntimeError(f"Order status query failed: {exc}") from exc

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        if self._sdk_client is None or not hasattr(self._sdk_client, "cancel"):
            raise RuntimeError("Order cancel unavailable.")
        try:
            result = self._sdk_client.cancel(order_id)
            if not result:
                raise RuntimeError("Empty cancel response")
            return {
                "exchange_order_id": str(result.get("id") or result.get("orderID") or order_id),
                "status": str(result.get("status") or result.get("state") or "CANCELLED"),
                "raw": result,
            }
        except Exception as exc:
            raise RuntimeError(f"Cancel order failed: {exc}") from exc

    async def cancel_all_orders(self) -> dict[str, Any]:
        if self._sdk_client is None or not hasattr(self._sdk_client, "cancel_all"):
            raise RuntimeError("Cancel all unavailable.")
        try:
            result = self._sdk_client.cancel_all()
            if not result:
                raise RuntimeError("Empty cancel all response")
            return result
        except Exception as exc:
            raise RuntimeError(f"Cancel all failed: {exc}") from exc

    async def send_heartbeat(self) -> dict[str, Any]:
        if self._sdk_client and hasattr(self._sdk_client, "get_server_time"):
            try:
                return {"ok": True, "server_time": self._sdk_client.get_server_time(), "timestamp": datetime.now(timezone.utc).isoformat()}
            except Exception as exc:
                raise RuntimeError(f"Heartbeat failed: {exc}") from exc
        if self._live_mode():
            raise RuntimeError("Heartbeat unavailable in LIVE mode.")
        return {"ok": True, "timestamp": datetime.now(timezone.utc).isoformat()}

    async def get_server_time(self) -> str:
        if self._sdk_client and hasattr(self._sdk_client, "get_server_time"):
            return str(self._sdk_client.get_server_time())
        return datetime.now(timezone.utc).isoformat()

    async def get_market_tradability(self, market_id: str, token_id: str) -> dict[str, Any]:
        markets = await self.fetch_markets(limit=200)
        for market in markets:
            if market.market_id == market_id and market.token_id == token_id:
                tradable = bool(market.active and not market.closed)
                return {
                    "market_id": market.market_id,
                    "token_id": market.token_id,
                    "tradable": tradable,
                    "orderbook_enabled": tradable,
                    "category": market.category,
                    "title": market.title,
                    "liquidity": market.liquidity,
                }
        if self._live_mode():
            raise RuntimeError(f"Market tradability metadata missing for {market_id}/{token_id}")
        return {"market_id": market_id, "token_id": token_id, "tradable": False, "orderbook_enabled": False}

    def _fallback_markets(self) -> list[MarketInfo]:
        return [
            MarketInfo(
                market_id=f"market-{idx}",
                token_id=f"token-{idx}",
                title=title,
                slug=title.lower().replace(" ", "-"),
                category=category,
                active=True,
                closed=False,
                liquidity=500 + idx * 120,
                volume=2500 + idx * 400,
                source_quality=SourceQuality.SYNTHETIC_FALLBACK,
            )
            for idx, (title, category) in enumerate(
                [
                    ("Will BTC settle above 100k this month", "crypto price"),
                    ("Will candidate X win state Y", "politics"),
                    ("Will CPI print above consensus", "macro / economics"),
                    ("Will team A win the championship", "sports"),
                ]
            )
        ]

    def _fallback_wallet_activity(self, wallet_address: str) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_slug": "will-btc-settle-above-100k-this-month",
                "market_id": "market-0",
                "token_id": "token-0",
                "side": "BUY",
                "price": 0.49,
                "size": 18.0,
                "tx_hash": f"fallback-{wallet_address[-4:]}-0",
                "title": "Will BTC settle above 100k this month",
                "category": "crypto price",
                "source_quality": SourceQuality.SYNTHETIC_FALLBACK.value,
            }
        ]
