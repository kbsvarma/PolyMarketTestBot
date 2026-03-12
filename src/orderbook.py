from __future__ import annotations

from src.models import FillEstimate, OrderbookSnapshot


def estimate_fill(orderbook: OrderbookSnapshot, target_notional: float, max_slippage_pct: float) -> FillEstimate:
    if not orderbook.asks:
        return FillEstimate(
            fillable=False,
            executable_price=0.0,
            spread_pct=1.0,
            slippage_pct=1.0,
            filled_notional=0.0,
            reason="No asks available.",
        )

    best_bid = orderbook.bids[0].price if orderbook.bids else orderbook.asks[0].price
    best_ask = orderbook.asks[0].price
    spread_pct = (best_ask - best_bid) / max(best_ask, 1e-6)

    total_depth = sum(level.price * level.size for level in orderbook.asks)
    remaining = target_notional
    cost = 0.0
    filled_size = 0.0
    filled_notional = 0.0
    consumed = 0.0
    for level in orderbook.asks:
        level_notional = level.price * level.size
        take = min(level_notional, remaining)
        if take <= 0:
            break
        take_size = take / max(level.price, 1e-9)
        cost += take
        filled_size += take_size
        filled_notional += take
        consumed += level_notional
        remaining -= take
        if remaining <= 1e-9:
            break
    fillable = remaining <= 1e-9
    executable_price = cost / filled_size if filled_size else best_ask
    slippage_pct = (executable_price - best_ask) / max(best_ask, 1e-6)
    reason = "OK" if fillable and slippage_pct <= max_slippage_pct else "Insufficient depth or slippage too high."
    return FillEstimate(
        fillable=fillable and slippage_pct <= max_slippage_pct,
        executable_price=round(executable_price, 6),
        spread_pct=round(spread_pct, 6),
        slippage_pct=round(slippage_pct, 6),
        filled_notional=round(filled_notional, 6),
        reason=reason,
        depth_consumed_pct=round((target_notional / max(total_depth, 1e-9)), 6),
        max_size_within_slippage=round(sum(level.size for level in orderbook.asks if (level.price - best_ask) / max(best_ask, 1e-6) <= max_slippage_pct), 6),
    )
