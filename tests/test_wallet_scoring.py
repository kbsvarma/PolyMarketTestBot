from pathlib import Path

from src.config import load_config
from src.models import SourceQuality, WalletMetrics, WalletScoringResult
from src.wallet_scoring import WalletScoringService


def _wallet(address: str, copyability: float = 0.8, delay: float = 0.8) -> WalletMetrics:
    return WalletMetrics(
        wallet_address=address,
        evaluation_window_days=30,
        trade_count=20,
        trades_per_day=1.0,
        buy_count=10,
        sell_count=10,
        estimated_pnl_percent=0.2,
        win_rate=0.7,
        average_trade_size=100.0,
        conviction_score=0.8,
        market_concentration=0.2,
        category_concentration=0.6,
        holding_time_estimate_hours=4.0,
        drawdown_proxy=0.1,
        copyability_score=copyability,
        low_velocity_score=0.7,
        delay_5s=delay,
        delay_15s=delay,
        delay_30s=delay,
        delay_60s=delay,
        delayed_viability_score=delay,
        source_quality=SourceQuality.REAL_PUBLIC_DATA,
    )


def _config():
    root = Path(__file__).resolve().parent.parent
    return load_config(root / "config.live_smoke.yaml", root / ".env.example")


def test_select_wallets_uses_operator_live_wallet_count_override() -> None:
    config = _config()
    config.env = config.env.model_copy(update={"operator_live_wallet_count": 5})
    service = WalletScoringService(config, Path("/tmp"))
    wallets = [_wallet(f"0x{i}") for i in range(6)]
    scoring = WalletScoringResult(scored_wallets=wallets, state="SUCCESS", source_quality=SourceQuality.REAL_PUBLIC_DATA)

    approved = service.select_wallets(scoring, [{"wallet_address": f"0x{i}", "expectancy": 0.02} for i in range(6)])

    assert len(approved.live_wallets) == 5
    assert approved.live_wallets == approved.paper_wallets


def test_select_wallets_uses_configured_live_wallet_count_when_no_override() -> None:
    config = _config()
    config = config.model_copy(
        update={
            "wallet_selection": config.wallet_selection.model_copy(update={"approved_live_wallets": 2}),
        }
    )
    service = WalletScoringService(config, Path("/tmp"))
    wallets = [_wallet(f"0x{i}") for i in range(4)]
    scoring = WalletScoringResult(scored_wallets=wallets, state="SUCCESS", source_quality=SourceQuality.REAL_PUBLIC_DATA)

    approved = service.select_wallets(scoring, [{"wallet_address": f"0x{i}", "expectancy": 0.02} for i in range(4)])

    assert len(approved.live_wallets) == 2
