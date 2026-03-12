from datetime import datetime, timedelta, timezone

from src.models import EntryStyle, Mode, Position
from src.exits import evaluate_exit


def test_exit_stop_loss() -> None:
    position = Position(
        position_id="1",
        mode=Mode.PAPER,
        wallet_address="0x1",
        market_id="m1",
        token_id="t1",
        category="politics",
        entry_style=EntryStyle.FOLLOW_TAKER,
        entry_price=0.6,
        current_mark_price=0.6,
        quantity=10,
        notional=6,
        fees_paid=0,
        source_trade_timestamp=datetime.now(timezone.utc),
    )
    should_exit, reason = evaluate_exit(position, 0.5)
    assert should_exit
    assert reason == "STOP_LOSS"


def test_exit_time_stop() -> None:
    position = Position(
        position_id="1",
        mode=Mode.PAPER,
        wallet_address="0x1",
        market_id="m1",
        token_id="t1",
        category="politics",
        entry_style=EntryStyle.FOLLOW_TAKER,
        entry_price=0.6,
        current_mark_price=0.6,
        quantity=10,
        notional=6,
        fees_paid=0,
        source_trade_timestamp=datetime.now(timezone.utc),
        opened_at=datetime.now(timezone.utc) - timedelta(hours=60),
    )
    should_exit, reason = evaluate_exit(position, 0.61)
    assert should_exit
    assert reason == "TIME_STOP"
