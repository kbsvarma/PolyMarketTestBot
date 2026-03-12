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
) -> tuple[bool, str]:
    pnl_pct = (mark_price - position.entry_price) / max(position.entry_price, 1e-9)
    age = datetime.now(timezone.utc) - position.opened_at
    if pnl_pct <= -abs(stop_loss_pct):
        return True, "STOP_LOSS"
    if pnl_pct >= abs(take_profit_pct):
        return True, "TAKE_PROFIT"
    if not hold_to_resolution and age >= timedelta(hours=time_stop_hours):
        return True, "TIME_STOP"
    return False, ""
