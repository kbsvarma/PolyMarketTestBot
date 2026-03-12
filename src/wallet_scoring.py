from __future__ import annotations

from pathlib import Path

from src.config import AppConfig
from src.hedge_filter import hedge_suspicion_score
from src.models import ApprovedWallets, WalletMetrics
from src.utils import clamp, write_csv, write_json


class WalletScoringService:
    def __init__(self, config: AppConfig, data_dir: Path) -> None:
        self.config = config
        self.data_dir = data_dir

    def score_wallets(self, wallets: list[WalletMetrics]) -> list[WalletMetrics]:
        for wallet in wallets:
            wallet.performance_score = clamp((wallet.estimated_pnl_percent + wallet.win_rate) / 1.5, 0.0, 1.0)
            wallet.consistency_score = clamp((1 - wallet.drawdown_proxy) * wallet.win_rate, 0.0, 1.0)
            wallet.sample_size_score = clamp(wallet.trade_count / 50.0, 0.0, 1.0)
            wallet.market_quality_score = clamp(1.0 - wallet.market_concentration * 0.5, 0.0, 1.0)
            wallet.delayed_viability_score = round((wallet.delay_5s + wallet.delay_15s + wallet.delay_30s + wallet.delay_60s) / 4.0, 4)
            wallet.category_strength_score = clamp(wallet.category_concentration * wallet.win_rate, 0.0, 1.0)
            wallet.hedge_suspicion_score = hedge_suspicion_score(wallet)
            wallet.complexity_penalty = clamp(wallet.hedge_suspicion_score * 0.8 + wallet.market_concentration * 0.2, 0.0, 1.0)
            wallet.copied_performance_score = clamp(wallet.delayed_viability_score * wallet.copyability_score, 0.0, 1.0)
            wallet.global_score = round(
                0.18 * wallet.performance_score
                + 0.12 * wallet.consistency_score
                + 0.08 * wallet.sample_size_score
                + 0.08 * wallet.market_quality_score
                + 0.18 * wallet.copyability_score
                + 0.14 * wallet.delayed_viability_score
                + 0.10 * wallet.category_strength_score
                + 0.10 * wallet.low_velocity_score
                - 0.10 * wallet.hedge_suspicion_score
                - 0.05 * wallet.complexity_penalty
                + 0.07 * wallet.copied_performance_score,
                4,
            )
        wallets.sort(key=lambda wallet: wallet.global_score, reverse=True)
        write_csv(self.data_dir / "wallet_scorecard.csv", [wallet.model_dump() for wallet in wallets])
        write_json(self.data_dir / "top_wallets.json", [wallet.model_dump(mode="json") for wallet in wallets[: self.config.wallet_selection.top_research_wallets]])
        return wallets

    def select_wallets(self, wallets: list[WalletMetrics], replay_rows: list[dict[str, object]]) -> ApprovedWallets:
        replay_expectancy: dict[str, float] = {}
        for row in replay_rows:
            wallet = str(row["wallet_address"])
            replay_expectancy[wallet] = max(float(row["expectancy"]), replay_expectancy.get(wallet, float("-inf")))

        eligible = [
            wallet
            for wallet in wallets
            if wallet.trade_count >= self.config.wallet_selection.min_trade_count
            and wallet.copyability_score >= self.config.wallet_selection.min_copyability_score
            and wallet.delayed_viability_score >= self.config.wallet_selection.min_delay_viability_score
            and replay_expectancy.get(wallet.wallet_address, 0.0) >= self.config.backtest.min_wallet_replay_expectancy
        ]
        research = [wallet.wallet_address for wallet in wallets[: self.config.wallet_selection.top_research_wallets]]
        paper = [wallet.wallet_address for wallet in eligible[: self.config.wallet_selection.approved_paper_wallets]]
        return ApprovedWallets(research_wallets=research, paper_wallets=paper, live_wallets=paper[:1])
