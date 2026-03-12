from __future__ import annotations

from pathlib import Path

from src.config import AppConfig
from src.models import CategoryScore, WalletMetrics
from src.utils import write_csv


class CategoryScorer:
    def __init__(self, config: AppConfig, data_dir: Path) -> None:
        self.config = config
        self.data_dir = data_dir

    def build_scorecards(self, wallets: list[WalletMetrics]) -> list[CategoryScore]:
        rows: list[CategoryScore] = []
        for wallet in wallets:
            score = round(
                0.35 * wallet.copyability_score
                + 0.25 * wallet.delayed_viability_score
                + 0.20 * wallet.win_rate
                + 0.20 * wallet.low_velocity_score,
                4,
            )
            rows.append(
                CategoryScore(
                    wallet_address=wallet.wallet_address,
                    category=wallet.dominant_category,
                    score=score,
                    trade_count=wallet.trade_count,
                    win_rate=wallet.win_rate,
                    copyability_score=wallet.copyability_score,
                    delay_viability_score=wallet.delayed_viability_score,
                )
            )
        write_csv(self.data_dir / "category_wallet_scorecard.csv", [row.model_dump() for row in rows])
        return rows
