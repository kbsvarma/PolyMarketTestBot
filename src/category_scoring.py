from __future__ import annotations

from pathlib import Path

from src.models import CategoryScore, CategoryScoringResult, WalletMetrics
from src.utils import write_csv, write_json


class CategoryScorer:
    def __init__(self, config, data_dir: Path) -> None:
        self.config = config
        self.data_dir = data_dir

    def build_scorecards(self, wallets: list[WalletMetrics]) -> CategoryScoringResult:
        rows: list[CategoryScore] = []
        for wallet in sorted(wallets, key=lambda item: (item.dominant_category, item.wallet_address)):
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
        rows.sort(key=lambda row: (row.category, -row.score, row.wallet_address))
        result = CategoryScoringResult(
            rows=rows,
            diagnostics={
                "row_count": len(rows),
                "categories": sorted({row.category for row in rows}),
                "wallet_count": len(wallets),
            },
        )
        if rows:
            write_csv(self.data_dir / "category_wallet_scorecard.csv", [row.model_dump(mode="json") for row in rows])
        else:
            (self.data_dir / "category_wallet_scorecard.csv").write_text(
                "wallet_address,category,score,trade_count,win_rate,copyability_score,delay_viability_score\n",
                encoding="utf-8",
            )
        write_json(self.data_dir / "category_wallet_scorecard_diagnostics.json", result.model_dump(mode="json"))
        return result
