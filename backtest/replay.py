from __future__ import annotations

from src.models import EntryStyle, WalletMetrics
from src.utils import clamp


def replay_wallet(wallet: WalletMetrics, delay_bucket: int, entry_style: EntryStyle) -> dict[str, object]:
    delay_penalty = {5: 0.01, 15: 0.02, 30: 0.035, 60: 0.06}.get(delay_bucket, 0.08)
    style_bonus = {
        EntryStyle.FOLLOW_TAKER: -0.005,
        EntryStyle.PASSIVE_LIMIT: 0.012,
        EntryStyle.POST_ONLY_MAKER: 0.009,
    }[entry_style]
    expectancy = clamp(wallet.copyability_score * wallet.delayed_viability_score * 0.08 + style_bonus - delay_penalty, -1.0, 1.0)
    fill_rate = clamp(0.82 - delay_penalty * 5 + (0.05 if entry_style != EntryStyle.POST_ONLY_MAKER else -0.05), 0.0, 1.0)
    return {
        "wallet_address": wallet.wallet_address,
        "delay_bucket": delay_bucket,
        "entry_style": entry_style.value,
        "expectancy": round(expectancy, 4),
        "fill_rate": round(fill_rate, 4),
        "net_pnl": round(expectancy * 25, 4),
        "copyable": expectancy >= 0.01,
    }
