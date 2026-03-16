from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config
from src.models import DetectionEvent, FillEstimate, WalletMetrics
from src.risk_manager import RiskManager


def build_wallet() -> WalletMetrics:
    wallet = WalletMetrics(
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
    return wallet


def test_risk_manager_blocks_stale_signal() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    detection = DetectionEvent(
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
        detection_latency_seconds=999,
        source_trade_timestamp=datetime.now(timezone.utc),
    )
    fill = FillEstimate(fillable=True, executable_price=0.51, spread_pct=0.01, slippage_pct=0.01, filled_notional=5, reason="OK")
    result = RiskManager(config).evaluate(detection, build_wallet(), fill, "PAPER", 0, 0, 0, 0, True, True, True)
    assert not result.allowed
    assert result.reason_code == "STALE_SIGNAL"


def test_risk_manager_allows_stale_override() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    detection = DetectionEvent(
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
        detection_latency_seconds=400,
        source_trade_timestamp=datetime.now(timezone.utc),
        category="politics",
    )
    fill = FillEstimate(fillable=True, executable_price=0.5, spread_pct=0.0, slippage_pct=0.0, filled_notional=5, reason="OK")
    result = RiskManager(config).evaluate(
        detection,
        build_wallet(),
        fill,
        "LIVE",
        0,
        0,
        0,
        0,
        True,
        True,
        True,
        category="politics",
        tradable=True,
        live_ready=True,
        kill_switch=False,
        manual_live_enable=True,
        manual_resume_required=False,
        health_state="HEALTHY",
        reconciliation_clean=True,
        heartbeat_ok=True,
        balance_visible=True,
        allowance_sufficient=True,
        entries_last_hour_override=0,
        stale_signal_seconds_override=900,
    )
    assert result.allowed


def test_risk_manager_evaluate_does_not_consume_entry_slot() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    detection = DetectionEvent(
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
    )
    fill = FillEstimate(fillable=True, executable_price=0.51, spread_pct=0.01, slippage_pct=0.01, filled_notional=5, reason="OK")
    risk = RiskManager(config)

    result = risk.evaluate(detection, build_wallet(), fill, "PAPER", 0, 0, 0, 0, True, True, True)

    assert result.allowed
    assert risk.entries_last_hour() == 0


def test_risk_manager_register_entry_enforces_hourly_limit() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    config.risk.max_new_entries_per_hour = 1
    detection = DetectionEvent(
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
    )
    fill = FillEstimate(fillable=True, executable_price=0.51, spread_pct=0.01, slippage_pct=0.01, filled_notional=5, reason="OK")
    risk = RiskManager(config)
    risk.register_entry()

    result = risk.evaluate(detection, build_wallet(), fill, "PAPER", 0, 0, 0, 0, True, True, True)

    assert not result.allowed
    assert result.reason_code == "ENTRY_RATE_LIMIT"


def test_risk_manager_respects_entries_last_hour_override() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    config.risk.max_new_entries_per_hour = 1
    detection = DetectionEvent(
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
    )
    fill = FillEstimate(fillable=True, executable_price=0.51, spread_pct=0.01, slippage_pct=0.01, filled_notional=5, reason="OK")
    risk = RiskManager(config)

    result = risk.evaluate(
        detection,
        build_wallet(),
        fill,
        "PAPER",
        0,
        0,
        0,
        0,
        True,
        True,
        True,
        entries_last_hour_override=1,
    )

    assert not result.allowed
    assert result.reason_code == "ENTRY_RATE_LIMIT"
