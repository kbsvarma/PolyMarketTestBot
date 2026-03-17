from __future__ import annotations

from src.fees import taker_fee, net_edge_after_fees, meets_min_edge
from src.models import FillEstimate, OrderbookSnapshot


def compute_executable_edge(
    *,
    fair_value: float,
    fill: FillEstimate,
    category: str = "crypto price",
    is_taker: bool = True,
    min_edge_taker: float = 0.020,
    min_edge_maker: float = 0.010,
) -> dict:
    """
    Given a fair-value estimate and a FillEstimate, compute:
      - gross_edge    : fair_value - executable_price
      - fee           : taker/maker fee at executable_price
      - net_edge      : gross_edge - fee
      - passes_gate   : True if net_edge >= threshold

    Returns a dict safe for JSON serialisation and decision context.
    """
    exec_price = float(fill.executable_price or 0.0)
    gross = round(fair_value - exec_price, 8)
    fee = taker_fee(exec_price, category=category) if is_taker else 0.0
    net = round(gross - fee, 8)
    passes = meets_min_edge(net, is_taker=is_taker, min_taker=min_edge_taker, min_maker=min_edge_maker)
    return {
        "fair_value": round(fair_value, 6),
        "executable_price": round(exec_price, 6),
        "gross_edge": round(gross, 6),
        "taker_fee": round(fee, 6),
        "net_edge": round(net, 6),
        "passes_min_edge_gate": passes,
        "is_taker": is_taker,
        "category": category,
        "min_edge_threshold": min_edge_taker if is_taker else min_edge_maker,
    }


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
