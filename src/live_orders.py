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
        orders = [LiveOrder.model_validate(item) for item in payload]  # type: ignore[arg-type]
        for order in orders:
            self._normalize_loaded_order(order)
        return self._dedupe_orders(orders)

    def save(self, orders: list[LiveOrder]) -> None:
        write_json(self.path, [order.model_dump(mode="json") for order in self._dedupe_orders(orders)])

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
        intended_size = round(decision.scaled_notional / max(decision.executable_price, 1e-6), 6)
        if decision.thesis_type == "paired_arb":
            try:
                bundle_shares = float(decision.context.get("bundle_shares", 0.0) or 0.0)
            except (TypeError, ValueError):
                bundle_shares = 0.0
            if bundle_shares > 0:
                intended_size = round(bundle_shares, 6)
        return LiveOrder(
            local_decision_id=decision.local_decision_id,
            local_order_id=local_order_id,
            client_order_id=stable_event_key(decision.local_decision_id, decision.market_id, decision.token_id, decision.entry_style.value),
            strategy_name=decision.strategy_name,
            wallet_address=decision.wallet_address,
            category=decision.category,
            source_price=decision.source_price,
            market_id=decision.market_id,
            token_id=decision.token_id,
            side="BUY",
            intended_price=decision.executable_price,
            intended_size=intended_size,
            entry_style=EntryStyle(decision.entry_style),
            thesis_type=decision.thesis_type,
            bundle_id=decision.bundle_id,
            bundle_role=decision.bundle_role,
            paired_token_id=decision.paired_token_id,
            remaining_size=intended_size,
            lifecycle_status=OrderLifecycleStatus.CREATED,
            timeout_at=datetime.now(timezone.utc),
        )

    def _normalize_loaded_order(self, order: LiveOrder) -> None:
        if order.lifecycle_status in {
            OrderLifecycleStatus.CANCELLED,
            OrderLifecycleStatus.REJECTED,
            OrderLifecycleStatus.EXPIRED,
            OrderLifecycleStatus.FILLED,
        }:
            order.terminal_state = True
            return
        if order.lifecycle_status != OrderLifecycleStatus.UNKNOWN:
            return
        normalized = str(order.last_exchange_status or "").upper()
        mapping = {
            "LIVE": OrderLifecycleStatus.RESTING,
            "RESTING": OrderLifecycleStatus.RESTING,
            "OPEN": OrderLifecycleStatus.RESTING,
            "SUBMITTED": OrderLifecycleStatus.SUBMITTED,
            "ACKNOWLEDGED": OrderLifecycleStatus.ACKNOWLEDGED,
            "PARTIALLY_FILLED": OrderLifecycleStatus.PARTIALLY_FILLED,
            "FILLED": OrderLifecycleStatus.FILLED,
            "MATCHED": OrderLifecycleStatus.FILLED,
            "CANCELLED": OrderLifecycleStatus.CANCELLED,
            "CANCELED": OrderLifecycleStatus.CANCELLED,
            "CANCELED_MARKET_RESOLVED": OrderLifecycleStatus.CANCELLED,
            "REJECTED": OrderLifecycleStatus.REJECTED,
            "EXPIRED": OrderLifecycleStatus.EXPIRED,
        }
        mapped = mapping.get(normalized)
        if mapped is None:
            return
        order.lifecycle_status = mapped
        if mapped in {
            OrderLifecycleStatus.CANCELLED,
            OrderLifecycleStatus.REJECTED,
            OrderLifecycleStatus.EXPIRED,
            OrderLifecycleStatus.FILLED,
        }:
            order.terminal_state = True

    def _dedupe_orders(self, orders: list[LiveOrder]) -> list[LiveOrder]:
        deduped: dict[str, LiveOrder] = {}
        for order in orders:
            key = order.local_order_id or order.exchange_order_id or order.client_order_id
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = order.model_copy(deep=True)
                continue
            deduped[key] = self._merge_order_versions(existing, order)
        return sorted(
            deduped.values(),
            key=lambda order: (
                order.submitted_at or order.created_at,
                order.created_at,
                order.local_order_id,
            ),
        )

    def _merge_order_versions(self, current: LiveOrder, candidate: LiveOrder) -> LiveOrder:
        current_key = self._order_sort_key(current)
        candidate_key = self._order_sort_key(candidate)
        winner = candidate if candidate_key >= current_key else current
        loser = current if winner is candidate else candidate

        merged = winner.model_copy(deep=True)
        merged.created_at = min(current.created_at, candidate.created_at)
        merged.last_update_at = max(current.last_update_at, candidate.last_update_at)
        if current.submitted_at and candidate.submitted_at:
            merged.submitted_at = max(current.submitted_at, candidate.submitted_at)
        else:
            merged.submitted_at = current.submitted_at or candidate.submitted_at
        merged.raw_exchange_response_refs = list(
            dict.fromkeys(current.raw_exchange_response_refs + candidate.raw_exchange_response_refs)
        )
        if not merged.exchange_order_id:
            merged.exchange_order_id = loser.exchange_order_id
        if not merged.linked_position_id:
            merged.linked_position_id = loser.linked_position_id
        merged.filled_size = max(current.filled_size, candidate.filled_size)
        if merged.average_fill_price <= 0:
            merged.average_fill_price = max(current.average_fill_price, candidate.average_fill_price)
        if merged.remaining_size <= 0 and max(current.remaining_size, candidate.remaining_size) > 0 and not merged.terminal_state:
            merged.remaining_size = max(current.remaining_size, candidate.remaining_size)
        return merged

    def _order_sort_key(self, order: LiveOrder) -> tuple[datetime, int, float, float]:
        observed_at = order.last_update_at or order.submitted_at or order.created_at
        active_rank = 1 if not order.terminal_state else 0
        return (
            observed_at,
            active_rank,
            order.filled_size,
            max(order.remaining_size, order.intended_size),
        )
