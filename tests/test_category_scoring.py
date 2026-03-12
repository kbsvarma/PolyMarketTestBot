from pathlib import Path

from src.category_scoring import CategoryScorer
from src.config import load_config
from src.models import WalletMetrics


def test_category_scorecards_generated() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    wallets = [
        WalletMetrics(
            wallet_address="0xbbb",
            evaluation_window_days=30,
            trade_count=20,
            trades_per_day=1.0,
            buy_count=10,
            sell_count=10,
            estimated_pnl_percent=0.1,
            win_rate=0.6,
            average_trade_size=20,
            conviction_score=0.5,
            market_concentration=0.4,
            category_concentration=0.7,
            holding_time_estimate_hours=12,
            drawdown_proxy=0.2,
            copyability_score=0.7,
            low_velocity_score=0.8,
            delay_5s=0.8,
            delay_15s=0.7,
            delay_30s=0.6,
            delay_60s=0.5,
            dominant_category="crypto price",
        ),
        WalletMetrics(
            wallet_address="0xaaa",
            evaluation_window_days=30,
            trade_count=20,
            trades_per_day=1.0,
            buy_count=10,
            sell_count=10,
            estimated_pnl_percent=0.1,
            win_rate=0.6,
            average_trade_size=20,
            conviction_score=0.5,
            market_concentration=0.4,
            category_concentration=0.7,
            holding_time_estimate_hours=12,
            drawdown_proxy=0.2,
            copyability_score=0.7,
            low_velocity_score=0.8,
            delay_5s=0.8,
            delay_15s=0.7,
            delay_30s=0.6,
            delay_60s=0.5,
            dominant_category="crypto price",
        ),
    ]
    result = CategoryScorer(config, root / "data").build_scorecards(wallets)
    rows = result.rows
    assert rows
    assert rows[0].category == "crypto price"
    assert [row.wallet_address for row in rows] == ["0xaaa", "0xbbb"]
