from pathlib import Path

from src.config import load_config
from src.geoblock import GeoblockChecker
from src.live_engine import LiveTradingEngine
from src.models import EntryStyle, LiveOrder, Mode, Position, TradeDecision, DecisionAction
from src.state import AppStateStore


def _engine(tmp_path: Path) -> LiveTradingEngine:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config = config.model_copy(update={"mode": "LIVE"})
    config.env = config.env.model_copy(
        update={
            "live_trading_enabled": True,
            "polymarket_private_key": "0xabc",
            "polymarket_funder": "0xdef",
        }
    )
    return LiveTradingEngine(config, tmp_path, AppStateStore(tmp_path / "app_state.json"), GeoblockChecker(config))


def _decision(notional: float) -> TradeDecision:
    return TradeDecision(
        allowed=True,
        action=DecisionAction.LIVE_COPY,
        reason_code="OK",
        human_readable_reason="ok",
        local_decision_id="d1",
        wallet_address="0xabc",
        market_id="m1",
        token_id="t1",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        category="politics",
        scaled_notional=notional,
        source_price=0.5,
        executable_price=0.5,
        cluster_confirmed=True,
        hedge_suspicion_score=0.0,
    )


def test_operator_cap_blocks_trade_over_max_trade(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.config.env = engine.config.env.model_copy(update={"operator_live_max_trade_usd": 5.0})

    reason = engine._operator_entry_block_reason(_decision(6.0), [], [])

    assert reason is not None
    assert "OPERATOR_MAX_TRADE_USD" in reason


def test_operator_cap_blocks_when_max_positions_reached(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.config.env = engine.config.env.model_copy(update={"operator_live_max_positions": 1})
    position = Position(
        position_id="p1",
        mode=Mode.LIVE,
        wallet_address="0xabc",
        market_id="m1",
        token_id="t1",
        category="politics",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        entry_price=0.5,
        current_mark_price=0.5,
        quantity=10,
        notional=5,
        fees_paid=0.0,
        source_trade_timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        remaining_size=10,
    )

    reason = engine._operator_entry_block_reason(_decision(1.0), [position], [])

    assert reason is not None
    assert "OPERATOR_MAX_POSITIONS" in reason


def test_operator_cap_blocks_when_session_exposure_exceeded(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.config.env = engine.config.env.model_copy(update={"operator_live_session_max_usd": 30.0})
    order = LiveOrder(
        local_decision_id="d0",
        local_order_id="o0",
        client_order_id="c0",
        market_id="m0",
        token_id="t0",
        side="BUY",
        intended_price=0.5,
        intended_size=50,
        remaining_size=50,
        entry_style=EntryStyle.PASSIVE_LIMIT,
    )

    reason = engine._operator_entry_block_reason(_decision(6.0), [], [order])

    assert reason is not None
    assert "OPERATOR_SESSION_MAX_USD" in reason
