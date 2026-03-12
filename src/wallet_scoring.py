from __future__ import annotations

from pathlib import Path

from src.hedge_filter import hedge_suspicion_score
from src.models import ApprovedWallets, SourceQuality, WalletMetrics, WalletScoringResult
from src.source_quality import quality_rank
from src.utils import clamp, write_csv, write_json


class WalletScoringService:
    def __init__(self, config, data_dir: Path) -> None:
        self.config = config
        self.data_dir = data_dir

    def score_wallets(self, wallets: list[WalletMetrics]) -> WalletScoringResult:
        if not wallets:
            result = WalletScoringResult(
                scored_wallets=[],
                skipped_wallets=[],
                rejected_wallets=[],
                state="EMPTY",
                source_quality=SourceQuality.DEGRADED_PUBLIC_DATA,
                diagnostics={"wallet_count": 0, "message": "No wallets available for scoring."},
            )
            self._persist(result)
            return result

        scored_wallets: list[WalletMetrics] = []
        skipped_wallets: list[dict[str, object]] = []
        rejected_wallets: list[dict[str, object]] = []
        for wallet in wallets:
            if wallet.trade_count < self.config.wallet_selection.min_trade_count:
                skipped_wallets.append(
                    {
                        "wallet_address": wallet.wallet_address,
                        "reason_code": "INSUFFICIENT_TRADES",
                        "reason": f"Trade count {wallet.trade_count} below minimum {self.config.wallet_selection.min_trade_count}.",
                    }
                )
                continue
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
            scored_wallets.append(wallet)

        scored_wallets.sort(
            key=lambda wallet: (
                wallet.global_score,
                wallet.copyability_score,
                wallet.wallet_address,
            ),
            reverse=True,
        )
        source_quality = max((wallet.source_quality for wallet in scored_wallets), key=quality_rank, default=SourceQuality.DEGRADED_PUBLIC_DATA)
        result = WalletScoringResult(
            scored_wallets=scored_wallets,
            skipped_wallets=skipped_wallets,
            rejected_wallets=rejected_wallets,
            state="PARTIAL_SUCCESS" if scored_wallets and (rejected_wallets or skipped_wallets) else ("SUCCESS" if scored_wallets else "EMPTY"),
            source_quality=source_quality,
            diagnostics={
                "wallet_count": len(wallets),
                "scored_count": len(scored_wallets),
                "skipped_count": len(skipped_wallets),
                "rejected_count": len(rejected_wallets),
                "top_wallets": [wallet.wallet_address for wallet in scored_wallets[:3]],
                "synthetic_wallet_count": len([wallet for wallet in wallets if wallet.source_quality == SourceQuality.SYNTHETIC_FALLBACK]),
            },
        )
        self._persist(result)
        return result

    def select_wallets(self, scoring: WalletScoringResult, replay_rows: list[dict[str, object]]) -> ApprovedWallets:
        replay_expectancy: dict[str, float] = {}
        for row in replay_rows:
            wallet = str(row["wallet_address"])
            replay_expectancy[wallet] = max(float(row["expectancy"]), replay_expectancy.get(wallet, float("-inf")))

        eligible = [
            wallet
            for wallet in scoring.scored_wallets
            if wallet.copyability_score >= self.config.wallet_selection.min_copyability_score
            and wallet.delayed_viability_score >= self.config.wallet_selection.min_delay_viability_score
            and replay_expectancy.get(wallet.wallet_address, 0.0) >= self.config.backtest.min_wallet_replay_expectancy
            and wallet.source_quality == SourceQuality.REAL_PUBLIC_DATA
        ]
        research = [wallet.wallet_address for wallet in scoring.scored_wallets[: self.config.wallet_selection.top_research_wallets]]
        paper = [wallet.wallet_address for wallet in eligible[: self.config.wallet_selection.approved_paper_wallets]]
        return ApprovedWallets(research_wallets=research, paper_wallets=paper, live_wallets=paper[:1])

    def _persist(self, result: WalletScoringResult) -> None:
        if result.scored_wallets:
            write_csv(self.data_dir / "wallet_scorecard.csv", [wallet.model_dump(mode="json") for wallet in result.scored_wallets])
        else:
            (self.data_dir / "wallet_scorecard.csv").write_text(
                "wallet_address,global_score,copyability_score,delayed_viability_score,dominant_category\n",
                encoding="utf-8",
            )
        write_json(
            self.data_dir / "wallet_scoring_diagnostics.json",
            {
                **result.model_dump(mode="json"),
                "scored_wallets": [wallet.model_dump(mode="json") for wallet in result.scored_wallets],
                "approved_wallets": [wallet.wallet_address for wallet in result.scored_wallets if wallet.source_quality == SourceQuality.REAL_PUBLIC_DATA],
            },
        )
        write_json(self.data_dir / "top_wallets.json", [wallet.model_dump(mode="json") for wallet in result.scored_wallets[: self.config.wallet_selection.top_research_wallets]])
