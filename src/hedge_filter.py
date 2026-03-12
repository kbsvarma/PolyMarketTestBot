from __future__ import annotations

from src.models import WalletMetrics
from src.utils import clamp


def hedge_suspicion_score(wallet: WalletMetrics) -> float:
    rapid_multi_market = clamp(wallet.trades_per_day / 8.0, 0.0, 1.0)
    basket_bias = clamp(wallet.market_concentration * 0.6 + (1 - wallet.holding_time_estimate_hours / 48.0), 0.0, 1.0)
    low_conviction_penalty = clamp(1.0 - wallet.conviction_score, 0.0, 1.0)
    return round(clamp(0.45 * rapid_multi_market + 0.35 * basket_bias + 0.20 * low_conviction_penalty, 0.0, 1.0), 4)
