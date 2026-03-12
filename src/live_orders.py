from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.models import EntryStyle, LiveOrder, OrderLifecycleStatus, TradeDecision
from src.utils import read_json, stable_event_key, write_json


class LiveOrderStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[LiveOrder]:
        payload = read_json(self.path, [])
        return [LiveOrder.model_validate(item) for item in payload]  # type: ignore[arg-type]

    def save(self, orders: list[LiveOrder]) -> None:
        write_json(self.path, [order.model_dump(mode="json") for order in orders])

    def find_by_local_id(self, orders: list[LiveOrder], local_order_id: str) -> LiveOrder | None:
        for order in orders:
            if order.local_order_id == local_order_id:
                return order
        return None

    def find_by_client_order_id(self, orders: list[LiveOrder], client_order_id: str) -> LiveOrder | None:
        for order in orders:
            if order.client_order_id == client_order_id:
                return order
        return None

    def create_from_decision(self, decision: TradeDecision) -> LiveOrder:
        local_order_id = stable_event_key(decision.local_decision_id, decision.market_id, decision.token_id, "order")
        return LiveOrder(
            local_decision_id=decision.local_decision_id,
            local_order_id=local_order_id,
            client_order_id=stable_event_key(decision.local_decision_id, decision.market_id, decision.token_id, decision.entry_style.value),
            market_id=decision.market_id,
            token_id=decision.token_id,
            side="BUY",
            intended_price=decision.executable_price,
            intended_size=round(decision.scaled_notional / max(decision.executable_price, 1e-6), 6),
            entry_style=EntryStyle(decision.entry_style),
            remaining_size=round(decision.scaled_notional / max(decision.executable_price, 1e-6), 6),
            lifecycle_status=OrderLifecycleStatus.CREATED,
            timeout_at=datetime.now(timezone.utc),
        )
