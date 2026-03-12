from pathlib import Path

from src.config import load_config
from src.models import WalletMetrics
from src.wallet_scoring import WalletScoringService


def test_wallet_scoring_orders_by_global_score() -> None:
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
    result = WalletScoringService(config, root / "data").score_wallets(wallets)
    scored = result.scored_wallets
    assert scored[0].global_score >= scored[-1].global_score
    assert scored[0].copyability_score >= 0.4
    assert [wallet.wallet_address for wallet in scored] == ["0xbbb", "0xaaa"]


def test_wallet_scoring_empty_input() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    result = WalletScoringService(config, root / "data").score_wallets([])
    assert result.state == "EMPTY"
    assert result.scored_wallets == []
