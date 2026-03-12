from src.models import OrderbookLevel, OrderbookSnapshot
from src.orderbook import estimate_fill


def test_fillability_gate_rejects_thin_book() -> None:
    orderbook = OrderbookSnapshot(
        token_id="t1",
        bids=[OrderbookLevel(price=0.45, size=10)],
        asks=[OrderbookLevel(price=0.6, size=1)],
    )
    fill = estimate_fill(orderbook, target_notional=10, max_slippage_pct=0.03)
    assert not fill.fillable
