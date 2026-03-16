from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.models import Position


def evaluate_exit(
    position: Position,
    mark_price: float,
    stop_loss_pct: float = 0.10,
    take_profit_pct: float = 0.15,
    time_stop_hours: int = 48,
    hold_to_resolution: bool = False,
    profit_arm_pct: float = 0.08,
    min_profit_lock_pct: float = 0.03,
    trailing_profit_retrace_pct: float = 0.35,
    strong_winner_profit_pct: float = 0.35,
    strong_winner_retrace_pct: float = 0.20,
    paired_arb_time_stop_hours: int = 168,
) -> tuple[bool, str]:
    entry_price = position.entry_price_actual or position.entry_price
    pnl_pct = (mark_price - entry_price) / max(entry_price, 1e-9)
    age = datetime.now(timezone.utc) - position.opened_at
    peak_price = max(position.peak_mark_price or entry_price, mark_price, entry_price)
    peak_pnl_pct = max(position.peak_pnl_pct or 0.0, (peak_price - entry_price) / max(entry_price, 1e-9))
    position.current_mark_price = mark_price
    position.peak_mark_price = round(peak_price, 6)
    position.peak_pnl_pct = round(peak_pnl_pct, 6)
    position.peak_mark_seen_at = datetime.now(timezone.utc)

    if position.thesis_type == "paired_arb":
        if not hold_to_resolution and age >= timedelta(hours=max(time_stop_hours, paired_arb_time_stop_hours)):
            return True, "PAIRED_TIME_STOP"
        return False, ""

    if pnl_pct <= -abs(stop_loss_pct):
        return True, "STOP_LOSS"

    if peak_pnl_pct >= abs(profit_arm_pct):
        position.profit_lock_armed = True
        retrace_pct = strong_winner_retrace_pct if peak_pnl_pct >= abs(strong_winner_profit_pct) else trailing_profit_retrace_pct
        locked_profit_pct = max(abs(min_profit_lock_pct), peak_pnl_pct * (1.0 - abs(retrace_pct)))
        trailing_stop_price = entry_price * (1.0 + locked_profit_pct)
        position.trailing_stop_price = round(trailing_stop_price, 6)
        if pnl_pct <= locked_profit_pct:
            return True, "TRAILING_PROFIT"
    elif pnl_pct >= abs(take_profit_pct) and age >= timedelta(hours=max(time_stop_hours / 3, 6)):
        return True, "TAKE_PROFIT"

    if not hold_to_resolution and age >= timedelta(hours=time_stop_hours):
        if pnl_pct > 0 and position.profit_lock_armed:
            return True, "TIME_STOP_PROFITABLE"
        return True, "TIME_STOP"
    return False, ""
