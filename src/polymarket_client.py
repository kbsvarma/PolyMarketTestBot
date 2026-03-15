from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
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
        self._preferred_signature_type: int | None = config.env.polymarket_signature_type
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
                signature_type=self.config.env.polymarket_signature_type,
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

    def live_order_capable(self) -> tuple[bool, str]:
        if not self.config.env.live_trading_enabled:
            return False, "LIVE_TRADING_ENABLED is false."
        if self._sdk_client is None:
            return False, self._sdk_init_error or "py-clob-client is not initialized."
        required_methods = ("create_order", "post_order", "get_order", "cancel", "get_orders")
        missing = [name for name in required_methods if not hasattr(self._sdk_client, name)]
        if missing:
            return False, f"SDK client missing required live order methods: {', '.join(missing)}"
        return True, "SDK live order methods are available."

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()

    def _to_float(self, value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _to_collateral_amount(self, value: Any) -> float:
        if value in (None, ""):
            return 0.0
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("0x"):
                return 0.0
            if "." in stripped:
                return self._to_float(stripped)
            if stripped.lstrip("-").isdigit():
                return self._to_float(stripped) / 1_000_000.0
        if isinstance(value, int):
            return float(value) / 1_000_000.0
        return self._to_float(value)

    def _normalize_positions_payload(self, payload: Any) -> list[dict[str, Any]]:
        if payload is None:
            raise RuntimeError("Empty positions response")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("data", "positions", "items"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    return [item for item in rows if isinstance(item, dict)]
            if payload:
                return [payload]
        raise RuntimeError(f"Malformed positions response type: {type(payload).__name__}")

    def _summarize_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            keys = sorted(payload.keys())
            return {"type": "dict", "keys": keys[:10]}
        if isinstance(payload, list):
            return {"type": "list", "length": len(payload)}
        return {"type": type(payload).__name__}

    def _parse_listish(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [part.strip("[]\" '") for part in text.split(",") if part.strip("[]\" '")]
        return [str(value).strip()]

    def _market_infos_from_item(self, item: dict[str, Any]) -> list[MarketInfo]:
        market_id = str(item.get("conditionId") or item.get("id") or item.get("slug") or "")
        title = str(item.get("question") or item.get("title") or item.get("description") or "Unknown market")
        slug = str(item.get("slug") or item.get("marketSlug") or "")
        category = str(item.get("category") or "unknown")
        token_ids = self._parse_listish(item.get("clobTokenIds"))
        if not token_ids:
            token_ids = self._parse_listish(item.get("tokenId") or item.get("clobTokenId") or item.get("asset_id"))
        outcomes = self._parse_listish(item.get("outcomes") or item.get("outcomeNames") or item.get("outcome"))
        rows: list[MarketInfo] = []
        for idx, token_id in enumerate(token_ids):
            outcome_suffix = f" [{outcomes[idx]}]" if idx < len(outcomes) and outcomes[idx] else ""
            row = MarketInfo(
                market_id=market_id,
                token_id=str(token_id),
                title=f"{title}{outcome_suffix}",
                slug=slug,
                category=category,
                active=bool(item.get("active", True)),
                closed=bool(item.get("closed", False)),
                liquidity=float(item.get("liquidity") or 0.0),
                volume=float(item.get("volume") or item.get("volumeNum") or 0.0),
                end_date_iso=item.get("endDate") or item.get("end_date_iso"),
                source_quality=SourceQuality.REAL_PUBLIC_DATA,
            )
            if row.market_id and row.token_id:
                rows.append(row)
        return rows

    def _extract_market_rows(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("markets", "data", "items"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    return [item for item in rows if isinstance(item, dict)]
            return [payload]
        return []

    def _normalize_allowance_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError(f"Malformed allowance response type: {type(payload).__name__}")
        allowances = payload.get("allowances", {})
        allowance_values: list[float] = []
        if isinstance(allowances, dict):
            for value in allowances.values():
                allowance_values.append(self._to_collateral_amount(value))
        top_level_available = self._to_collateral_amount(payload.get("available"))
        top_level_balance = self._to_collateral_amount(payload.get("balance"))
        top_level_allowance = self._to_collateral_amount(payload.get("allowance"))
        approval_capacity = max([top_level_allowance, *allowance_values], default=0.0)
        spendable_balance = max(top_level_available, top_level_balance, 0.0)
        available = spendable_balance if spendable_balance > 0 else approval_capacity
        sufficient = available > 0 and (approval_capacity > 0 or not isinstance(allowances, dict) or len(allowances) == 0)
        return {
            "available": available,
            "sufficient": sufficient,
            "approval_capacity": approval_capacity,
            "raw": payload,
            "raw_summary": {
                "top_level_balance": top_level_balance,
                "top_level_available": top_level_available,
                "top_level_allowance": top_level_allowance,
                "allowance_entry_count": len(allowances) if isinstance(allowances, dict) else 0,
                "max_allowance_entry": max(allowance_values, default=0.0),
                "approval_capacity": approval_capacity,
            },
            "query_visible": True,
            "source_quality": SourceQuality.REAL_PUBLIC_DATA.value,
        }

    def _allowance_params_variants(self) -> list[Any]:
        if BalanceAllowanceParams is None or AssetType is None:
            return []
        signature_type = getattr(getattr(self, "_sdk_client", None), "builder", None)
        signature_type = getattr(signature_type, "sig_type", None)
        configured_signature_type = self.config.env.polymarket_signature_type
        signature_types = [configured_signature_type, signature_type, 0, 1, 2, None]
        variants = []
        for value in signature_types:
            if value is None:
                variants.append(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
            else:
                variants.append(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=value))
        seen: set[tuple[str, str, object]] = set()
        deduped: list[Any] = []
        for params in variants:
            key = (str(getattr(params, "asset_type", "")), str(getattr(params, "token_id", "")), getattr(params, "signature_type", None))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(params)
        return deduped

    def _extract_sdk_signature_type(self) -> int | None:
        builder = getattr(self._sdk_client, "builder", None)
        signature_type = getattr(builder, "sig_type", None)
        if signature_type is None:
            return self.config.env.polymarket_signature_type
        return int(signature_type)

    def _set_order_signature_type(self, signature_type: int | None) -> None:
        builder = getattr(self._sdk_client, "builder", None)
        if builder is not None and signature_type is not None:
            builder.sig_type = int(signature_type)
            self._preferred_signature_type = int(signature_type)

    def _order_signature_type_candidates(self) -> list[int | None]:
        current = self._extract_sdk_signature_type()
        configured = self.config.env.polymarket_signature_type
        candidates = [self._preferred_signature_type, configured, current, 1, 0]
        deduped: list[int | None] = []
        for item in candidates:
            if item in deduped:
                continue
            deduped.append(item)
        return deduped

    def _best_allowance_attempt(self, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        return max(
            attempts,
            key=lambda item: (
                float(item.get("available", 0.0)),
                int(bool(item.get("query_visible", False))),
                1 if item.get("query_method") == "get_balance_allowance" else 0,
            ),
        )

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
    async def fetch_markets(self, limit: int = 100, offset: int = 0, active_only: bool = False) -> list[MarketInfo]:
        try:
            params: dict[str, Any] = {"limit": limit}
            if offset > 0:
                params["offset"] = offset
            if active_only:
                params["active"] = "true"
                params["closed"] = "false"
            payload = await self._get_json(f"{self.gamma_url}/markets", params=params)
            rows: list[MarketInfo] = []
            for item in self._extract_market_rows(payload):
                rows.extend(self._market_infos_from_item(item))
            rows = self._ensure_live_data(rows, "markets")
            return rows
        except Exception:
            if self._live_mode():
                raise RuntimeError("Unable to fetch real markets in LIVE mode.")
            return self._fallback_markets()

    async def fetch_market_lookup(self, market_id: str, token_id: str = "", market_slug: str = "", outcome: str = "") -> MarketInfo | None:
        candidates: list[tuple[str, dict[str, Any] | None]] = []
        if market_id:
            candidates.extend(
                [
                    (f"{self.gamma_url}/markets/{market_id}", None),
                    (f"{self.gamma_url}/markets", {"conditionId": market_id}),
                    (f"{self.gamma_url}/markets", {"id": market_id}),
                    (f"{self.gamma_url}/markets", {"market": market_id}),
                ]
            )
        if market_slug:
            candidates.extend(
                [
                    (f"{self.gamma_url}/markets", {"slug": market_slug}),
                    (f"{self.gamma_url}/markets", {"marketSlug": market_slug}),
                ]
            )
        if token_id and token_id not in {"unknown-token", "unknown", "None"}:
            candidates.extend(
                [
                    (f"{self.gamma_url}/markets", {"clobTokenIds": token_id}),
                    (f"{self.gamma_url}/markets", {"clobTokenId": token_id}),
                    (f"{self.gamma_url}/markets", {"tokenId": token_id}),
                    (f"{self.gamma_url}/events", {"tokenId": token_id}),
                ]
            )
        for url, params in candidates:
            try:
                payload = await self._get_json(url, params=params)
            except Exception:
                continue
            rows = self._extract_market_rows(payload)
            exact_match: MarketInfo | None = None
            fallback_match: MarketInfo | None = None
            for item in rows:
                outcome_names = [name.lower() for name in self._parse_listish(item.get("outcomes") or item.get("outcomeNames") or item.get("outcome"))]
                for idx, market in enumerate(self._market_infos_from_item(item)):
                    if market.market_id == market_id and token_id and market.token_id == token_id:
                        exact_match = market
                        break
                    if market.market_id == market_id and not token_id and outcome and idx < len(outcome_names) and outcome.lower() == outcome_names[idx]:
                        exact_match = market
                        break
                    if market_slug and market.slug == market_slug and token_id and market.token_id == token_id:
                        exact_match = market
                        break
                    if market_slug and market.slug == market_slug and not token_id and outcome and idx < len(outcome_names) and outcome.lower() == outcome_names[idx]:
                        exact_match = market
                        break
                    if token_id and market.token_id == token_id:
                        fallback_match = market
                    elif market.market_id == market_id and fallback_match is None:
                        fallback_match = market
                    elif market_slug and market.slug == market_slug and fallback_match is None:
                        fallback_match = market
                if exact_match is not None:
                    break
            if exact_match is not None:
                return exact_match
            if fallback_match is not None:
                return fallback_match
        return None

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
                attempts = []
                errors: list[str] = []
                for params in self._allowance_params_variants():
                    try:
                        payload = self._sdk_client.get_balance_allowance(params)
                        normalized = self._normalize_allowance_payload(payload)
                        normalized["query_method"] = "get_balance_allowance"
                        normalized["query_params"] = {
                            "asset_type": str(getattr(params, "asset_type", "")),
                            "token_id": str(getattr(params, "token_id", "")),
                            "signature_type": getattr(params, "signature_type", None),
                        }
                        attempts.append(normalized)
                    except Exception as exc:
                        errors.append(str(exc))
                if not attempts:
                    raise RuntimeError("; ".join(errors) if errors else "SDK balance method unavailable")
                best = self._best_allowance_attempt(attempts)
                payload = best.get("raw", {})
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
            attempts: list[dict[str, Any]] = []
            errors: list[str] = []
            if hasattr(self._sdk_client, "get_balance_allowance"):
                variants = self._allowance_params_variants() or [None]
                for params in variants:
                    try:
                        payload = self._sdk_client.get_balance_allowance(params) if params is not None else self._sdk_client.get_balance_allowance()
                        normalized = self._normalize_allowance_payload(payload)
                        normalized["query_method"] = "get_balance_allowance"
                        normalized["query_params"] = {
                            "asset_type": str(getattr(params, "asset_type", "")),
                            "token_id": str(getattr(params, "token_id", "")),
                            "signature_type": getattr(params, "signature_type", None),
                        } if params is not None else {"params": None}
                        attempts.append(normalized)
                    except Exception as exc:
                        errors.append(f"get_balance_allowance:{exc}")
                if hasattr(self._sdk_client, "update_balance_allowance"):
                    for params in self._allowance_params_variants():
                        try:
                            payload = self._sdk_client.update_balance_allowance(params)
                            normalized = self._normalize_allowance_payload(payload)
                            normalized["query_method"] = "update_balance_allowance"
                            normalized["query_params"] = {
                                "asset_type": str(getattr(params, "asset_type", "")),
                                "token_id": str(getattr(params, "token_id", "")),
                                "signature_type": getattr(params, "signature_type", None),
                            }
                            attempts.append(normalized)
                        except Exception as exc:
                            errors.append(f"update_balance_allowance:{exc}")
            if hasattr(self._sdk_client, "get_balance"):
                try:
                    normalized = self._normalize_allowance_payload(self._sdk_client.get_balance())
                    normalized["query_method"] = "get_balance"
                    normalized["query_params"] = {}
                    attempts.append(normalized)
                except Exception as exc:
                    errors.append(f"get_balance:{exc}")
            if not attempts:
                raise RuntimeError("; ".join(errors) if errors else "SDK allowance method unavailable")
            best = self._best_allowance_attempt(attempts)
            best["query_attempts"] = [
                {
                    "method": item.get("query_method", ""),
                    "params": item.get("query_params", {}),
                    "summary": item.get("raw_summary", {}),
                }
                for item in attempts
            ]
            best["query_errors"] = errors
            best["sdk_signature_type"] = self._extract_sdk_signature_type()
            best["configured_signature_type"] = self.config.env.polymarket_signature_type
            preferred_signature_type = best.get("query_params", {}).get("signature_type")
            if preferred_signature_type is not None:
                self._set_order_signature_type(int(preferred_signature_type))
            return best
        except Exception as exc:
            if self._live_mode():
                raise RuntimeError(f"Unable to fetch allowance/spendability in LIVE mode: {exc}") from exc
            return {
                "available": self.config.bankroll.live_bankroll_reference,
                "sufficient": True,
                "source": "fallback",
                "query_visible": True,
                "source_quality": SourceQuality.SYNTHETIC_FALLBACK.value,
            }

    async def get_positions(self) -> list[dict[str, Any]]:
        errors: list[str] = []
        if self._sdk_client and hasattr(self._sdk_client, "get_positions"):
            try:
                payload = self._sdk_client.get_positions()
                return self._normalize_positions_payload(payload)
            except Exception as exc:
                errors.append(f"sdk:{exc}")
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
                return self._normalize_positions_payload(payload)
            except Exception as exc:
                errors.append(f"{url}:{exc}")
                continue
        if self._live_mode():
            detail = "; ".join(errors) if errors else "positions query unavailable"
            raise RuntimeError(f"Unable to fetch live positions. {detail}")
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
                remaining_size = self._extract_order_remaining_size(item)
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
                        "remaining_size": remaining_size,
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
        if hasattr(OrderType, "FOK") and entry_style == "FOLLOW_TAKER":
            order_type = OrderType.FOK
        order_args = OrderArgs(
            price=price,
            size=size,
            side=side,
            token_id=token_id,
        )
        errors: list[str] = []
        for signature_type in self._order_signature_type_candidates():
            try:
                self._set_order_signature_type(signature_type)
                signed_order = self._sdk_client.create_order(order_args)
                if client_order_id and isinstance(signed_order, dict):
                    signed_order["client_order_id"] = client_order_id
                result = self._sdk_client.post_order(signed_order, order_type)
                if not result:
                    raise RuntimeError("Empty order placement response")
                exchange_order_id = str(result.get("orderID") or result.get("id") or result.get("order_id") or "")
                status = str(result.get("status") or result.get("state") or "SUBMITTED")
                echoed_client_order_id = client_order_id or str(result.get("client_order_id") or result.get("clientOrderId") or "")
                if not exchange_order_id:
                    raise RuntimeError("Order placement response did not include an exchange_order_id.")
                if not status:
                    raise RuntimeError("Order placement response did not include an order status.")
                return {
                    "exchange_order_id": exchange_order_id,
                    "status": status,
                    "client_order_id": echoed_client_order_id,
                    "raw": result,
                    "signature_type_used": self._extract_sdk_signature_type(),
                }
            except Exception as exc:
                errors.append(f"signature_type={signature_type}:{exc}")
                if "invalid signature" not in str(exc).lower():
                    break
        raise RuntimeError(f"Real live order placement failed: {'; '.join(errors)}")

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
                "remaining_size": self._extract_order_remaining_size(result),
                "client_order_id": str(result.get("client_order_id") or result.get("clientOrderId") or ""),
                "raw": result,
            }
        except Exception as exc:
            raise RuntimeError(f"Order status query failed: {exc}") from exc

    def _extract_order_remaining_size(self, payload: dict[str, Any]) -> float:
        try:
            direct = float(payload.get("remaining_size") or payload.get("remaining") or 0.0)
        except (TypeError, ValueError):
            direct = 0.0
        if direct > 0:
            return direct
        try:
            original_size = float(payload.get("original_size") or payload.get("size") or payload.get("shares") or 0.0)
            size_matched = float(payload.get("size_matched") or payload.get("filled_size") or payload.get("filled") or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if original_size > 0:
            return max(original_size - size_matched, 0.0)
        return 0.0

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
        direct_market = await self.fetch_market_lookup(market_id, token_id)
        if direct_market is not None:
            tradable = bool(direct_market.active and not direct_market.closed)
            return {
                "market_id": direct_market.market_id,
                "token_id": direct_market.token_id,
                "tradable": tradable,
                "orderbook_enabled": tradable,
                "category": direct_market.category,
                "title": direct_market.title,
                "liquidity": direct_market.liquidity,
            }
        if token_id and token_id not in {"unknown-token", "unknown", "None"}:
            try:
                orderbook = await self.get_orderbook(token_id)
                has_depth = bool(orderbook.bids or orderbook.asks)
                return {
                    "market_id": market_id,
                    "token_id": token_id,
                    "tradable": has_depth,
                    "orderbook_enabled": has_depth,
                    "category": "unknown",
                    "title": market_id,
                    "liquidity": 0.0,
                    "derived_from_orderbook": True,
                }
            except Exception:
                pass
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
