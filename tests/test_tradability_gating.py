from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config
from src.models import DetectionEvent, FillEstimate, WalletMetrics
from src.risk_manager import RiskManager


def build_wallet() -> WalletMetrics:
    return WalletMetrics(
        wallet_address="0xabc",
        evaluation_window_days=30,
        trade_count=20,
        trades_per_day=0.5,
        buy_count=14,
        sell_count=6,
        estimated_pnl_percent=0.12,
        win_rate=0.61,
        average_trade_size=55,
        conviction_score=0.7,
        market_concentration=0.3,
        category_concentration=0.5,
        holding_time_estimate_hours=12,
        drawdown_proxy=0.08,
        copyability_score=0.7,
        low_velocity_score=0.75,
        delay_5s=0.72,
        delay_15s=0.7,
        delay_30s=0.68,
        delay_60s=0.6,
        dominant_category="politics",
        delayed_viability_score=0.675,
        hedge_suspicion_score=0.4,
    )


def build_detection() -> DetectionEvent:
    return DetectionEvent(
        event_key="k",
        wallet_address="0xabc",
        market_title="M",
        market_slug="m",
        market_id="m1",
        token_id="t1",
        side="BUY",
        price=0.5,
        size=10,
        notional=5,
        transaction_hash="tx",
        detection_latency_seconds=5,
        source_trade_timestamp=datetime.now(timezone.utc),
        category="politics",
    )


def test_live_risk_blocks_untradable_market() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    fill = FillEstimate(fillable=True, executable_price=0.51, spread_pct=0.01, slippage_pct=0.01, filled_notional=5, reason="OK")
    result = RiskManager(config).evaluate(
        build_detection(), build_wallet(), fill, "LIVE", 0, 0, 0, 0, True, True, True,
        category="politics", market_id="m1", live_ready=True, balance_visible=True, allowance_sufficient=True, tradable=False
    )
    assert not result.allowed
    assert result.reason_code == "MARKET_NOT_TRADABLE"
