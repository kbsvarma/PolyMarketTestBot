"""
Polymarket fee model.

Taker fee formula (crypto markets):
    fee = C * p * feeRate * (p * (1 - p)) ^ exponent
    where:
        p        = probability (token price, 0..1)
        feeRate  = 0.25  (crypto)
        exponent = 2     (crypto)
        C        = 1.0   (scaling constant)

Fee peaks at ~1.56% at p=0.50, falls to near-zero at extremes.

For non-crypto markets the fee structure differs; this module defaults to
zero for non-crypto categories to avoid over-penalising non-fee markets
while still capturing the full crypto drag.

Maker rebates use the same fee-equivalent curve but credit back to the
market-maker; takers always pay the fee.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CRYPTO_FEE_RATE: float = 0.25
CRYPTO_FEE_EXPONENT: int = 2

# Minimum edge we require AFTER fees for taker fills.
# Near mid-probability the fee peaks at ~1.56%, so 2 ¢ minimum gives buffer.
MIN_EDGE_AFTER_FEES_TAKER: float = 0.020   # 2.0 ¢
MIN_EDGE_AFTER_FEES_MAKER: float = 0.010   # 1.0 ¢

_CRYPTO_CATEGORY_TOKENS = frozenset({"crypto", "crypto price"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_crypto(category: str) -> bool:
    """Return True if the market category carries Polymarket crypto taker fees."""
    return str(category or "").strip().lower() in _CRYPTO_CATEGORY_TOKENS


def taker_fee(price: float, *, category: str = "crypto price") -> float:
    """
    Compute the taker fee fraction for a single leg at *price*.

    Returns the fee as a fraction of the trade notional (e.g. 0.0156 = 1.56 %).
    For non-crypto categories the fee is zero (conservative; adjust when
    Polymarket publishes non-crypto fee schedules).

    Parameters
    ----------
    price    : token price in [0, 1]
    category : market category string
    """
    if not _is_crypto(category):
        return 0.0
    p = max(min(float(price), 0.99), 0.01)
    return round(p * CRYPTO_FEE_RATE * math.pow(p * (1.0 - p), CRYPTO_FEE_EXPONENT), 8)


def taker_fee_pct(price: float, *, category: str = "crypto price") -> float:
    """Same as ``taker_fee`` but expressed as a percentage (0..100)."""
    return round(taker_fee(price, category=category) * 100.0, 6)


def max_profitable_opposite_price(
    entry_price: float,
    *,
    min_net_margin: float = 0.0,
    category: str = "crypto price",
) -> float:
    """
    Return the highest opposite-side price that still leaves a profitable bracket.

    Solves for ``y`` in:

        1 - x*(1+fee_x) - y*(1+fee_y) >= min_net_margin

    where ``x`` is the already-filled Phase 1 price and ``y`` is the opposite
    token price we could still buy while meeting the configured minimum locked
    profit after fees.
    """
    x = max(min(float(entry_price), 0.99), 0.01)
    target_margin = max(float(min_net_margin), 0.0)
    x_cost = x * (1.0 + taker_fee(x, category=category))
    threshold = 1.0 - target_margin

    lo = 0.01
    hi = 0.99

    def _lhs(y: float) -> float:
        return x_cost + y * (1.0 + taker_fee(y, category=category))

    if _lhs(lo) > threshold:
        return 0.0

    for _ in range(64):
        mid = (lo + hi) / 2.0
        if _lhs(mid) > threshold:
            hi = mid
        else:
            lo = mid
    return round(lo, 8)


def net_edge_after_fees(
    *,
    gross_edge: float,
    entry_price: float,
    category: str = "crypto price",
    is_taker: bool = True,
) -> float:
    """
    Compute net edge after subtracting the taker (or maker) fee for a single leg.

    Parameters
    ----------
    gross_edge   : raw price edge (e.g. fair_value - ask_price), positive = good
    entry_price  : the price at which the trade would be executed
    category     : market category
    is_taker     : True → subtract full taker fee; False → assume maker rebate
                   offsets the fee (net cost = 0 for maker)
    """
    fee = taker_fee(entry_price, category=category) if is_taker else 0.0
    return round(gross_edge - fee, 8)


def net_edge_after_fees_both_legs(
    *,
    gross_edge: float,
    yes_price: float,
    no_price: float,
    category: str = "crypto price",
    is_taker: bool = True,
) -> tuple[float, float, float]:
    """
    Compute net edge after fees for a full YES+NO paired trade.

    Returns (net_edge, fee_yes, fee_no).
    """
    fee_yes = taker_fee(yes_price, category=category) if is_taker else 0.0
    fee_no = taker_fee(no_price, category=category) if is_taker else 0.0
    net = round(gross_edge - fee_yes - fee_no, 8)
    return net, round(fee_yes, 8), round(fee_no, 8)


def meets_min_edge(
    net_edge: float,
    *,
    is_taker: bool = True,
    min_taker: float = MIN_EDGE_AFTER_FEES_TAKER,
    min_maker: float = MIN_EDGE_AFTER_FEES_MAKER,
) -> bool:
    """Return True if *net_edge* clears the minimum threshold for the fill type."""
    threshold = min_taker if is_taker else min_maker
    return net_edge >= threshold


def fee_summary(price: float, *, category: str = "crypto price") -> dict[str, float]:
    """
    Return a dict with fee diagnostics at *price* for logging / tracing.
    """
    fee = taker_fee(price, category=category)
    return {
        "price": round(price, 6),
        "taker_fee_fraction": round(fee, 8),
        "taker_fee_pct": round(fee * 100.0, 6),
        "fee_rate": CRYPTO_FEE_RATE if _is_crypto(category) else 0.0,
        "fee_exponent": CRYPTO_FEE_EXPONENT if _is_crypto(category) else 0,
        "is_crypto": _is_crypto(category),
    }
