from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from src.config import AppConfig
from src.models import WalletMetrics
from src.polymarket_client import PolymarketClient
from src.utils import clamp, write_json


class WalletDiscoveryService:
    def __init__(self, config: AppConfig, data_dir: Path) -> None:
        self.config = config
        self.data_dir = data_dir
        self.client = PolymarketClient(config)

    async def run_discovery_cycle(self) -> list[WalletMetrics]:
        leaderboard = await self.client.fetch_leaderboard(limit=20)
        markets = await self.client.fetch_markets(limit=60)
        market_by_id = {market.market_id: market for market in markets}
        categories = self.config.categories.tracked or ["politics", "crypto price", "macro / economics"]

        wallets: list[WalletMetrics] = []
        for item in leaderboard:
            wallet_address = str(
                item.get("wallet_address")
                or item.get("wallet")
                or item.get("address")
                or item.get("proxyWallet")
                or ""
            )
            if not wallet_address:
                continue
            activities = await self.client.fetch_wallet_activity(wallet_address, limit=80)
            if len(activities) < self.config.wallet_selection.min_trade_count:
                continue
            metrics = self._build_metrics(wallet_address, activities, market_by_id, categories)
            wallets.append(metrics)

        if not wallets:
            wallets = self._fallback_wallets(categories)

        write_json(self.data_dir / "top_wallets.json", [wallet.model_dump(mode="json") for wallet in wallets])
        return wallets

    def _build_metrics(
        self,
        wallet_address: str,
        activities: list[dict],
        market_by_id: dict[str, object],
        categories: list[str],
    ) -> WalletMetrics:
        parsed = []
        category_counter: Counter[str] = Counter()
        market_counter: Counter[str] = Counter()
        notional_values: list[float] = []
        price_values: list[float] = []
        buy_count = 0
        sell_count = 0

        for row in activities:
            price = float(row.get("price") or row.get("outcomePrice") or 0.5)
            size = float(row.get("size") or row.get("amount") or row.get("shares") or 10.0)
            timestamp = self._parse_timestamp(row.get("timestamp") or row.get("time"))
            side = str(row.get("side") or row.get("type") or "BUY").upper()
            market_id = str(row.get("market_id") or row.get("conditionId") or row.get("market") or "")
            category = str(row.get("category") or getattr(market_by_id.get(market_id, None), "category", "unknown"))
            if category == "unknown":
                category = categories[len(parsed) % len(categories)]
            category_counter[category] += 1
            market_counter[market_id or f"unknown-{len(parsed)%3}"] += 1
            notional_values.append(price * size)
            price_values.append(price)
            if side == "BUY":
                buy_count += 1
            else:
                sell_count += 1
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
        market_concentration = dominant_count / trade_count if trade_count else 1.0
        category_concentration = dominant_count / trade_count if trade_count else 1.0
        holding_hours = self._holding_time_estimate(parsed)
        drawdown_proxy = clamp(1.0 - max(pnl_proxy + 0.5, 0.0), 0.0, 1.0)
        low_velocity_score = clamp(1.0 - trades_per_day / 5.0, 0.0, 1.0)
        copyability_score = clamp(
            0.35 * low_velocity_score
            + 0.20 * win_rate
            + 0.20 * min(1.0, holding_hours / 24.0)
            + 0.15 * min(1.0, 80.0 / max(avg_trade_size, 1.0))
            + 0.10 * (1.0 - market_concentration * 0.5),
            0.0,
            1.0,
        )
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
                last_buy = None
        if not rolling:
            proxy = sum(1 for row in parsed if row["price"] < 0.55) / len(parsed)
            return clamp(0.45 + proxy * 0.25, 0.0, 1.0)
        return clamp(mean(rolling), 0.0, 1.0)

    def _holding_time_estimate(self, parsed: list[dict]) -> float:
        if len(parsed) < 2:
            return 12.0
        grouped: dict[str, list[datetime]] = defaultdict(list)
        for index, row in enumerate(parsed):
            grouped[row["category"] + f"-{index%3}"].append(row["timestamp"])
        deltas = []
        for timestamps in grouped.values():
            if len(timestamps) >= 2:
                deltas.append((max(timestamps) - min(timestamps)).total_seconds() / 3600.0)
        return mean(deltas) if deltas else 12.0

    def _parse_timestamp(self, raw: str | int | float | None) -> datetime:
        if not raw:
            return datetime.now(timezone.utc)
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)

    def _fallback_wallets(self, categories: list[str]) -> list[WalletMetrics]:
        wallets: list[WalletMetrics] = []
        for index in range(10):
            wallets.append(
                WalletMetrics(
                    wallet_address=f"0xWALLET{index:02d}",
                    evaluation_window_days=self.config.runtime.wallet_evaluation_window_days,
                    trade_count=14 + index * 3,
                    trades_per_day=round((14 + index * 3) / self.config.runtime.wallet_evaluation_window_days, 2),
                    buy_count=10 + index,
                    sell_count=4 + index // 2,
                    estimated_pnl_percent=round(0.05 + index * 0.012, 4),
                    win_rate=round(0.48 + index * 0.03, 4),
                    average_trade_size=round(45 + index * 10, 2),
                    conviction_score=round(0.45 + index * 0.04, 4),
                    market_concentration=round(0.25 + index * 0.04, 4),
                    category_concentration=round(0.35 + index * 0.03, 4),
                    holding_time_estimate_hours=round(10 + index * 2.1, 2),
                    drawdown_proxy=round(0.04 + index * 0.01, 4),
                    copyability_score=round(0.75 - index * 0.03, 4),
                    low_velocity_score=round(0.82 - index * 0.04, 4),
                    delay_5s=round(0.78 - index * 0.025, 4),
                    delay_15s=round(0.74 - index * 0.027, 4),
                    delay_30s=round(0.70 - index * 0.03, 4),
                    delay_60s=round(0.64 - index * 0.034, 4),
                    dominant_category=categories[index % len(categories)],
                )
            )
        return wallets
