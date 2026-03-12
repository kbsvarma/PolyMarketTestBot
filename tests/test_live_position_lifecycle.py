from datetime import datetime, timezone

from src.models import EntryStyle, Mode, Position


def test_position_not_closed_until_exit_fill_confirmed() -> None:
    position = Position(
        position_id="p1",
        mode=Mode.LIVE,
        wallet_address="0xabc",
        source_wallet="0xabc",
        market_id="m1",
        token_id="t1",
        category="politics",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        entry_price=0.5,
        current_mark_price=0.5,
        quantity=4.0,
        notional=2.0,
        fees_paid=0.0,
        source_trade_timestamp=datetime.now(timezone.utc),
        entry_size=4.0,
        remaining_size=4.0,
        exit_state="EXIT_ORDER_OPEN",
    )
    assert not position.closed
    assert position.remaining_size == 4.0
