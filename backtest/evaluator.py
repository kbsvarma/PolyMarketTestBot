from __future__ import annotations

from pathlib import Path

from backtest.replay import replay_wallet
from src.config import AppConfig
from src.models import WalletMetrics
from src.utils import write_csv


class BacktestEvaluator:
    def __init__(self, config: AppConfig, data_dir: Path) -> None:
        self.config = config
        self.data_dir = data_dir

    def evaluate_wallets(self, wallets: list[WalletMetrics]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for wallet in wallets:
            for delay_bucket in self.config.backtest.delay_buckets_seconds:
                for style in self.config.entry_styles.compare:
                    rows.append(replay_wallet(wallet, delay_bucket, style))
        write_csv(self.data_dir / "wallet_backtest_summary.csv", rows)
        return rows
