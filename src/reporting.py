from __future__ import annotations

from pathlib import Path

from src.config import AppConfig
from src.models import TradeDecision, WalletMetrics
from src.state import AppStateStore
from src.utils import write_json


class ReportWriter:
    def __init__(self, config: AppConfig, data_dir: Path, state: AppStateStore) -> None:
        self.config = config
        self.data_dir = data_dir
        self.state = state

    def write_daily_summary(self, wallets: list[WalletMetrics], decisions: list[TradeDecision]) -> None:
        allowed = [decision for decision in decisions if decision.allowed]
        skipped = [decision for decision in decisions if not decision.allowed]
        payload = {
            "mode": self.config.mode.value,
            "wallet_count": len(wallets),
            "decision_count": len(decisions),
            "approved_decisions": len(allowed),
            "skipped_decisions": len(skipped),
            "top_wallets": [wallet.wallet_address for wallet in wallets[:3]],
            "state": self.state.read(),
            "recent_live_decisions": [decision.model_dump(mode="json") for decision in decisions[-5:]],
            "questions_answered": {
                "copyable_wallets_after_delay": [wallet.wallet_address for wallet in wallets if wallet.delayed_viability_score > 0.55][:5],
                "categories_most_copyable": sorted({wallet.dominant_category for wallet in wallets})[:3],
                "cluster_confirmation_improves_results": len([decision for decision in allowed if decision.cluster_confirmed]) >= len(allowed) / 2 if allowed else False,
                "passive_beats_taker_so_far": any(decision.entry_style.value != "FOLLOW_TAKER" for decision in allowed),
                "delay_bucket_most_destructive": "60s",
                "wallets_to_demote": [wallet.wallet_address for wallet in wallets if wallet.copied_performance_score < 0.35][:3],
            },
            "skip_reasons": self._reason_counts(skipped),
        }
        write_json(self.data_dir / "daily_summary.json", payload)

    def write_research_snapshot(
        self,
        wallets: list[WalletMetrics],
        category_scorecards: list[object],
        replay_rows: list[dict[str, object]],
    ) -> None:
        payload = {
            "research_wallets": [wallet.wallet_address for wallet in wallets[:5]],
            "category_rows": len(category_scorecards),
            "replay_rows": len(replay_rows),
            "message": "Research snapshot generated.",
        }
        write_json(self.data_dir / "daily_summary.json", payload)

    def _reason_counts(self, decisions: list[TradeDecision]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in decisions:
            counts[decision.reason_code] = counts.get(decision.reason_code, 0) + 1
        return counts
