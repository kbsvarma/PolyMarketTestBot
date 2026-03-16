from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from src.config import AppConfig
from src.logger import logger
from src.models import DiscoveryResult, DiscoveryState, SourceQuality, ValidationMode, WalletMetrics
from src.polymarket_client import PolymarketClient
from src.source_quality import quality_from_discovery_state, quality_rank
from src.utils import clamp, write_json


class WalletDiscoveryService:
    def __init__(self, config: AppConfig, data_dir: Path) -> None:
        self.config = config
        self.data_dir = data_dir
        self.client = PolymarketClient(config)

    def _source_timeout_seconds(self, name: str) -> float:
        base = float(self.config.live.bounded_execution_seconds or 20)
        timeout = min(max(base * 0.4, 3.0), 10.0)
        if name.startswith("wallet_activity:"):
            return min(timeout, 5.0)
        if name in {"market_holders", "recent_activity"}:
            return min(max(base * 0.5, 4.0), 8.0)
        return timeout

    def _candidate_activity_limit(self) -> int:
        configured = max(
            self.config.wallet_selection.top_research_wallets,
            self.config.wallet_selection.approved_paper_wallets,
            self.config.wallet_selection.approved_live_wallets,
            3,
        )
        return min(max(configured * 4, 12), 24)

    async def run_discovery_cycle(self) -> DiscoveryResult:
        categories = self.config.categories.tracked or [
            "politics",
            "crypto price",
            "entertainment / pop culture",
            "macro / economics",
        ]
        diagnostics: dict[str, object] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "sources": [],
            "counts": {},
            "errors": [],
            "fallback_used": False,
        }
        source_rows: list[tuple[str, list[dict[str, object]], SourceQuality]] = []

        leaderboard_rows, leaderboard_error = await self._safe_source("leaderboard", self.client.fetch_leaderboard(limit=20))
        diagnostics["sources"].append({"name": "leaderboard", "row_count": len(leaderboard_rows), "error": leaderboard_error})
        if leaderboard_rows:
            source_rows.append(("leaderboard", leaderboard_rows, SourceQuality.REAL_PUBLIC_DATA))

        markets, markets_error = await self._safe_source("markets", self.client.fetch_markets(limit=60))
        diagnostics["sources"].append({"name": "markets", "row_count": len(markets), "error": markets_error})
        market_by_id = {market.market_id: market for market in markets}

        holder_rows: list[dict[str, object]] = []
        if markets:
            holder_rows, holders_error = await self._safe_source("market_holders", self._discover_from_market_holders(markets))
            diagnostics["sources"].append({"name": "market_holders", "row_count": len(holder_rows), "error": holders_error})
            if holder_rows:
                source_rows.append(("market_holders", holder_rows, SourceQuality.REAL_PUBLIC_DATA))

        activity_rows, activity_error = await self._safe_source("recent_activity", self._discover_from_recent_activity())
        diagnostics["sources"].append({"name": "recent_activity", "row_count": len(activity_rows), "error": activity_error})
        if activity_rows:
            source_rows.append(("recent_activity", activity_rows, SourceQuality.REAL_PUBLIC_DATA))

        merged_candidates: dict[str, dict[str, object]] = {}
        for source_name, rows, quality in source_rows:
            for row in rows:
                wallet_address = self._extract_wallet_address(row)
                if not wallet_address or self._is_placeholder_address(wallet_address):
                    continue
                existing = merged_candidates.get(wallet_address)
                if existing is None:
                    merged_candidates[wallet_address] = {
                        "wallet_address": wallet_address,
                        "volume": float(row.get("volume") or row.get("amount") or row.get("shares") or row.get("size") or 0.0),
                        "activity_count": int(row.get("activity_count") or 1),
                        "sources": [source_name],
                        "source_quality": quality.value,
                    }
                else:
                    existing["volume"] = float(existing.get("volume") or 0.0) + float(
                        row.get("volume") or row.get("amount") or row.get("shares") or row.get("size") or 0.0
                    )
                    existing["activity_count"] = int(existing.get("activity_count") or 0) + int(row.get("activity_count") or 1)
                    existing_sources = list(existing.get("sources") or [])
                    if source_name not in existing_sources:
                        existing_sources.append(source_name)
                    existing["sources"] = existing_sources
                    existing["source_quality"] = max(
                        [existing.get("source_quality", SourceQuality.DEGRADED_PUBLIC_DATA.value), quality.value],
                        key=lambda value: quality_rank(SourceQuality(value)),
                    )

        candidate_rows = sorted(
            merged_candidates.values(),
            key=lambda item: (
                int(item.get("activity_count") or 0),
                float(item.get("volume") or 0.0),
                str(item.get("wallet_address") or ""),
            ),
            reverse=True,
        )
        diagnostics["counts"] = {
            "leaderboard_rows": len(leaderboard_rows),
            "market_rows": len(markets),
            "holder_rows": len(holder_rows),
            "activity_rows": len(activity_rows),
            "candidate_wallets": len(candidate_rows),
        }
        candidate_limit = self._candidate_activity_limit()
        activity_candidates = candidate_rows[:candidate_limit]
        diagnostics["counts"]["candidate_wallets_evaluated"] = len(activity_candidates)
        diagnostics["counts"]["candidate_wallets_skipped_due_to_limit"] = max(len(candidate_rows) - len(activity_candidates), 0)

        filtered_wallets: list[dict[str, object]] = []
        rejected_wallets: list[dict[str, object]] = []
        wallets: list[WalletMetrics] = []
        activity_results = await asyncio.gather(
            *[
                self._safe_source(
                    f"wallet_activity:{str(candidate.get('wallet_address') or '')}",
                    self.client.fetch_wallet_activity(str(candidate.get("wallet_address") or ""), limit=80),
                )
                for candidate in activity_candidates
            ]
        )
        for candidate, (activities, activity_fetch_error) in zip(activity_candidates, activity_results):
            wallet_address = str(candidate.get("wallet_address") or "")
            if activity_fetch_error:
                rejected_wallets.append({"wallet_address": wallet_address, "reason_code": "FETCH_FAILED", "reason": activity_fetch_error})
                continue
            if not isinstance(activities, list):
                rejected_wallets.append({"wallet_address": wallet_address, "reason_code": "MALFORMED_RESPONSE", "reason": "Wallet activity is not a list."})
                continue
            if len(activities) < self.config.wallet_selection.min_trade_count:
                filtered_wallets.append(
                    {
                        "wallet_address": wallet_address,
                        "reason_code": "INSUFFICIENT_HISTORY",
                        "reason": f"Only {len(activities)} activities found.",
                    }
                )
                continue
            metrics = self._build_metrics(
                wallet_address=wallet_address,
                activities=activities,
                market_by_id=market_by_id,
                categories=categories,
                source_quality=SourceQuality(str(candidate.get("source_quality") or SourceQuality.DEGRADED_PUBLIC_DATA.value)),
            )
            wallets.append(metrics)

        fallback_used = False
        state = DiscoveryState.SUCCESS
        reason = "Real public wallet discovery succeeded."
        if candidate_rows and not wallets:
            state = DiscoveryState.FILTERED_TO_ZERO
            reason = "Candidates were found, but all were rejected by filters."
        elif not candidate_rows:
            errors = [entry["error"] for entry in diagnostics["sources"] if entry.get("error")]
            malformed = [error for error in errors if "malformed payload" in str(error).lower() or "null payload" in str(error).lower()]
            if malformed:
                state = DiscoveryState.MALFORMED_RESPONSE
                reason = "Public discovery returned malformed responses."
            elif errors:
                state = DiscoveryState.FETCH_FAILED
                reason = "Public discovery sources failed or returned unusable payloads."
            else:
                state = DiscoveryState.NO_DATA
                reason = "No usable public wallets were discovered."

        if not wallets and self.config.mode.value == "RESEARCH":
            wallets = self._fallback_wallets(categories)
            fallback_used = True
            state = DiscoveryState.SYNTHETIC_FALLBACK_USED
            reason = "Research mode is using synthetic fallback wallets because public discovery was insufficient."

        source_quality = quality_from_discovery_state(state, fallback_used=fallback_used)
        diagnostics["fallback_used"] = fallback_used
        diagnostics["finished_at"] = datetime.now(timezone.utc).isoformat()
        diagnostics["discovery_state"] = state.value
        diagnostics["source_quality"] = source_quality.value
        diagnostics["validation_mode"] = (
            ValidationMode.DEV_ONLY.value if fallback_used else ValidationMode.VALIDATION_GRADE.value
        )
        diagnostics["raw_candidate_count"] = len(candidate_rows)
        diagnostics["filtered_wallet_count"] = len(filtered_wallets)
        diagnostics["wallet_count"] = len(wallets)
        diagnostics["rejected_wallet_count"] = len(rejected_wallets)
        diagnostics["approved_candidate_count"] = len(wallets)

        result = DiscoveryResult(
            wallets=wallets,
            state=state,
            source_quality=source_quality,
            reason=reason,
            diagnostics=diagnostics,
            candidate_wallets=candidate_rows,
            filtered_wallets=filtered_wallets,
            rejected_wallets=rejected_wallets,
        )
        write_json(self.data_dir / "wallet_discovery_diagnostics.json", result.model_dump(mode="json"))
        write_json(self.data_dir / "top_wallets.json", [wallet.model_dump(mode="json") for wallet in wallets])
        return result

    async def _safe_source(self, name: str, awaitable) -> tuple[list, str]:
        try:
            rows = await asyncio.wait_for(awaitable, timeout=self._source_timeout_seconds(name))
            if rows is None:
                return [], f"{name} returned null payload."
            if not isinstance(rows, list):
                return [], f"{name} returned malformed payload type {type(rows).__name__}."
            return rows, ""
        except asyncio.TimeoutError:
            logger.warning("Wallet discovery source timed out source={} timeout_seconds={}", name, self._source_timeout_seconds(name))
            return [], f"{name} timed out."
        except Exception as exc:
            logger.warning("Wallet discovery source failed source={} error={}", name, exc)
            return [], str(exc)

    async def _discover_from_market_holders(self, markets: list[object]) -> list[dict[str, object]]:
        active_markets = [
            market
            for market in markets
            if getattr(market, "active", False) and not getattr(market, "closed", False)
        ]
        # Prefer markets in live-selected categories to find wallets that trade what the bot can actually follow.
        live_selected: list[str] = list(getattr(getattr(self.config, "live", None), "selected_categories", None) or [])
        if live_selected:
            preferred = [m for m in active_markets if getattr(m, "category", "") in live_selected]
            if preferred:
                active_markets = preferred
        active_markets.sort(key=lambda market: float(getattr(market, "liquidity", 0.0) or 0.0), reverse=True)
        market_ids = [str(getattr(market, "market_id", "")) for market in active_markets[:10] if getattr(market, "market_id", "")]
        holders = await self.client.fetch_top_holders(market_ids, per_market_limit=12)
        candidates: dict[str, dict[str, object]] = {}
        for row in holders:
            wallet_address = self._extract_wallet_address(row)
            if self._is_placeholder_address(wallet_address) or not wallet_address:
                continue
            candidates.setdefault(
                wallet_address,
                {"wallet_address": wallet_address, "volume": 0.0, "activity_count": 0},
            )
            candidates[wallet_address]["volume"] = float(candidates[wallet_address]["volume"]) + float(
                row.get("shares") or row.get("amount") or row.get("balance") or 0.0
            )
            candidates[wallet_address]["activity_count"] = int(candidates[wallet_address]["activity_count"]) + 1
        return list(candidates.values())

    async def _discover_from_recent_activity(self) -> list[dict[str, object]]:
        activity = await self.client.fetch_recent_public_activity(limit=250)
        candidates: dict[str, dict[str, object]] = {}
        for row in activity:
            wallet_address = self._extract_wallet_address(row)
            if self._is_placeholder_address(wallet_address) or not wallet_address:
                continue
            score = float(row.get("size") or row.get("amount") or row.get("shares") or row.get("volume") or 0.0)
            existing = candidates.get(wallet_address)
            if existing is None:
                candidates[wallet_address] = {"wallet_address": wallet_address, "volume": score, "activity_count": 1}
            else:
                existing["volume"] = float(existing.get("volume") or 0.0) + score
                existing["activity_count"] = int(existing.get("activity_count") or 0) + 1
        return list(candidates.values())

    def _extract_wallet_address(self, row: dict[str, object]) -> str:
        return str(
            row.get("wallet_address")
            or row.get("wallet")
            or row.get("address")
            or row.get("proxyWallet")
            or row.get("user")
            or row.get("makerAddress")
            or row.get("takerAddress")
            or ""
        )

    def _is_placeholder_address(self, wallet_address: str) -> bool:
        return wallet_address.upper().startswith("0XWALLET")

    def _build_metrics(
        self,
        wallet_address: str,
        activities: list[dict],
        market_by_id: dict[str, object],
        categories: list[str],
        source_quality: SourceQuality,
    ) -> WalletMetrics:
        parsed = []
        category_counter: Counter[str] = Counter()
        market_counter: Counter[str] = Counter()
        notional_values: list[float] = []
        buy_count = 0
        sell_count = 0

        _short_dur_kws = ("updown-5m", "updown-1m", "updown-15m", "updown-30m",
                          "up-or-down", "up or down", "5-minute", "1-minute",
                          "15-minute", "30-minute", "5min", "1min",
                          "intraday-price", "price-up", "price-down")
        short_duration_count = 0
        for row in activities:
            price = float(row.get("price") or row.get("outcomePrice") or 0.5)
            size = float(row.get("size") or row.get("amount") or row.get("shares") or 10.0)
            timestamp = self._parse_timestamp(row.get("timestamp") or row.get("time"))
            side = str(row.get("side") or row.get("type") or "BUY").upper()
            market_id = str(row.get("market_id") or row.get("conditionId") or row.get("market") or "")
            market_obj = market_by_id.get(market_id)
            category = str(row.get("category") or getattr(market_obj, "category", None) or "unknown")
            if category == "unknown":
                category = categories[len(parsed) % len(categories)]
            category_counter[category] += 1
            market_counter[market_id or f"unknown-{len(parsed)%3}"] += 1
            notional_values.append(price * size)
            if side == "BUY":
                buy_count += 1
            else:
                sell_count += 1
            # Track short-duration market trades for scoring penalty
            _title = str(row.get("title") or getattr(market_obj, "title", None) or "").lower()
            _slug = str(row.get("slug") or getattr(market_obj, "slug", None) or "").lower()
            if any(kw in _title or kw in _slug for kw in _short_dur_kws):
                short_duration_count += 1
            parsed.append({"timestamp": timestamp, "price": price, "size": size, "side": side, "category": category})

        parsed.sort(key=lambda item: item["timestamp"])
        trade_count = len(parsed)
        eval_days = max(self.config.runtime.wallet_evaluation_window_days, 1)
        trades_per_day = trade_count / eval_days
        pnl_proxy = self._estimate_pnl(parsed)
        win_rate = self._estimate_win_rate(parsed)
        avg_trade_size = mean(notional_values) if notional_values else 0.0
        dominant_category, dominant_count = category_counter.most_common(1)[0]
        conviction_score = clamp(avg_trade_size / max(max(notional_values), 1.0), 0.0, 1.0)
        market_concentration = max(market_counter.values()) / trade_count if trade_count else 1.0
        category_concentration = dominant_count / trade_count if trade_count else 1.0
        holding_hours = self._holding_time_estimate(parsed)
        drawdown_proxy = clamp(1.0 - max(pnl_proxy + 0.5, 0.0), 0.0, 1.0)
        low_velocity_score = clamp(1.0 - trades_per_day / 5.0, 0.0, 1.0)
        short_duration_rate = short_duration_count / max(trade_count, 1)
        copyability_score = clamp(
            0.35 * low_velocity_score
            + 0.20 * win_rate
            + 0.20 * min(1.0, holding_hours / 24.0)
            + 0.15 * min(1.0, 80.0 / max(avg_trade_size, 1.0))
            + 0.10 * (1.0 - market_concentration * 0.5),
            0.0,
            1.0,
        )
        if short_duration_rate > 0.3:
            copyability_score = clamp(copyability_score * (1.0 - short_duration_rate * 0.8), 0.0, 1.0)
        delay_base = clamp(copyability_score * 0.9 + low_velocity_score * 0.1, 0.0, 1.0)

        return WalletMetrics(
            wallet_address=wallet_address,
            evaluation_window_days=eval_days,
            trade_count=trade_count,
            trades_per_day=round(trades_per_day, 4),
            buy_count=buy_count,
            sell_count=sell_count,
            estimated_pnl_percent=round(pnl_proxy, 4),
            win_rate=round(win_rate, 4),
            average_trade_size=round(avg_trade_size, 4),
            conviction_score=round(conviction_score, 4),
            market_concentration=round(market_concentration, 4),
            category_concentration=round(category_concentration, 4),
            holding_time_estimate_hours=round(holding_hours, 4),
            drawdown_proxy=round(drawdown_proxy, 4),
            copyability_score=round(copyability_score, 4),
            low_velocity_score=round(low_velocity_score, 4),
            delay_5s=round(clamp(delay_base - 0.02, 0.0, 1.0), 4),
            delay_15s=round(clamp(delay_base - 0.05, 0.0, 1.0), 4),
            delay_30s=round(clamp(delay_base - 0.09, 0.0, 1.0), 4),
            delay_60s=round(clamp(delay_base - 0.16, 0.0, 1.0), 4),
            dominant_category=dominant_category,
            source_quality=source_quality,
        )

    def _estimate_pnl(self, parsed: list[dict]) -> float:
        if not parsed:
            return 0.0
        buys = [row["price"] for row in parsed if row["side"] == "BUY"]
        sells = [row["price"] for row in parsed if row["side"] != "BUY"]
        if not buys:
            return 0.0
        avg_buy = mean(buys)
        avg_sell = mean(sells) if sells else avg_buy + 0.03
        return clamp((avg_sell - avg_buy) / max(avg_buy, 1e-6), -1.0, 1.0)

    def _estimate_win_rate(self, parsed: list[dict]) -> float:
        if not parsed:
            return 0.0
        rolling = []
        last_buy = None
        for row in parsed:
            if row["side"] == "BUY":
                last_buy = row["price"]
            elif last_buy is not None:
                rolling.append(1.0 if row["price"] >= last_buy else 0.0)
        return round(sum(rolling) / len(rolling), 4) if rolling else 0.5

    def _holding_time_estimate(self, parsed: list[dict]) -> float:
        if len(parsed) < 2:
            return 24.0
        deltas = []
        for left, right in zip(parsed, parsed[1:]):
            deltas.append((right["timestamp"] - left["timestamp"]).total_seconds() / 3600.0)
        return max(mean(deltas), 1.0)

    def _fallback_wallets(self, categories: list[str]) -> list[WalletMetrics]:
        rows = []
        for idx in range(3):
            rows.append(
                WalletMetrics(
                    wallet_address=f"0xWALLET{idx:02d}",
                    evaluation_window_days=self.config.runtime.wallet_evaluation_window_days,
                    trade_count=25 + idx * 8,
                    trades_per_day=1.2 + idx * 0.2,
                    buy_count=18 + idx,
                    sell_count=6 + idx,
                    estimated_pnl_percent=0.10 + idx * 0.02,
                    win_rate=0.58 + idx * 0.03,
                    average_trade_size=45.0 + idx * 5,
                    conviction_score=0.62,
                    market_concentration=0.35,
                    category_concentration=0.60,
                    holding_time_estimate_hours=22.0,
                    drawdown_proxy=0.20,
                    copyability_score=0.66,
                    low_velocity_score=0.72,
                    delay_5s=0.72,
                    delay_15s=0.68,
                    delay_30s=0.63,
                    delay_60s=0.57,
                    dominant_category=categories[idx % len(categories)],
                    source_quality=SourceQuality.SYNTHETIC_FALLBACK,
                )
            )
        return rows

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
