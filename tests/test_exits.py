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


def test_exit_arms_trailing_profit_after_peak_move() -> None:
    position = Position(
        position_id="1",
        mode=Mode.PAPER,
        wallet_address="0x1",
        market_id="m1",
        token_id="t1",
        category="politics",
        entry_style=EntryStyle.FOLLOW_TAKER,
        entry_price=0.5,
        current_mark_price=0.5,
        quantity=10,
        notional=5,
        fees_paid=0,
        source_trade_timestamp=datetime.now(timezone.utc),
    )

    should_exit, reason = evaluate_exit(position, 0.7)

    assert should_exit is False
    assert reason == ""
    assert position.profit_lock_armed is True
    assert position.peak_mark_price == 0.7

    should_exit, reason = evaluate_exit(position, 0.59)

    assert should_exit is True
    assert reason == "TRAILING_PROFIT"


def test_paired_arb_holds_until_extended_time_stop() -> None:
    position = Position(
        position_id="1",
        mode=Mode.PAPER,
        wallet_address="0x1",
        market_id="m1",
        token_id="t1",
        category="entertainment / pop culture",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        entry_price=0.47,
        current_mark_price=0.47,
        quantity=10,
        notional=4.7,
        fees_paid=0,
        source_trade_timestamp=datetime.now(timezone.utc),
        opened_at=datetime.now(timezone.utc) - timedelta(hours=200),
        thesis_type="paired_arb",
        bundle_id="bundle-1",
    )

    should_exit, reason = evaluate_exit(position, 0.6)

    assert should_exit is True
    assert reason == "PAIRED_TIME_STOP"
