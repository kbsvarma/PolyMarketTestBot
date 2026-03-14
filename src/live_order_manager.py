from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.audit import AuditLogger
from src.config import AppConfig
from src.live_orders import LiveOrderStore
from src.models import LiveOrder, Mode, OrderLifecycleStatus, Position, TradeDecision
from src.polymarket_client import PolymarketClient
from src.utils import stable_event_key


class LiveOrderManager:
    def __init__(self, config: AppConfig, data_dir: Path, client: PolymarketClient, audit: AuditLogger) -> None:
        self.config = config
        self.client = client
        self.audit = audit
        self.store = LiveOrderStore(data_dir / "live_orders.json")

    def create_entry_order(self, decision: TradeDecision) -> LiveOrder:
        order = self.store.create_from_decision(decision)
        order.timeout_at = datetime.now(timezone.utc) + timedelta(seconds=self.config.risk.order_timeout_seconds)
        return order

    async def submit_order(self, order: LiveOrder, side: str) -> dict:
        order.lifecycle_status = OrderLifecycleStatus.SUBMITTING
        order.submitted_at = datetime.now(timezone.utc)
        order.last_update_at = datetime.now(timezone.utc)
        try:
            if side == "BUY":
                response = await self.client.place_buy_order(
                    token_id=order.token_id,
                    price=order.intended_price,
                    size=order.intended_size,
                    entry_style=order.entry_style.value,
                    client_order_id=order.client_order_id,
                )
            else:
                response = await self.client.place_sell_order(
                    token_id=order.token_id,
                    price=order.intended_price,
                    size=order.intended_size,
                    entry_style=order.entry_style.value,
                    client_order_id=order.client_order_id,
                )
        except Exception:
            order.last_exchange_status = "REJECTED"
            order.lifecycle_status = OrderLifecycleStatus.REJECTED
            order.terminal_state = True
            order.last_update_at = datetime.now(timezone.utc)
            raise
        order.exchange_order_id = str(response.get("exchange_order_id") or "")
        order.last_exchange_status = str(response.get("status") or "SUBMITTED")
        order.lifecycle_status = self._map_status(order.last_exchange_status)
        order.last_update_at = datetime.now(timezone.utc)
        order.raw_exchange_response_refs.append(self.audit.record("submit_response", {"ts": datetime.now(timezone.utc).isoformat(), "client_order_id": order.client_order_id, "response": response}))
        return response

    async def refresh_order(self, order: LiveOrder) -> dict:
        if not order.exchange_order_id:
            return {"status": order.last_exchange_status or "UNKNOWN"}
        status = await self.client.get_order_status(order.exchange_order_id)
        order.last_exchange_status = str(status.get("status") or order.last_exchange_status)
        order.lifecycle_status = self._map_status(order.last_exchange_status)
        order.filled_size = float(status.get("filled_size") or order.filled_size)
        order.average_fill_price = float(status.get("average_fill_price") or order.average_fill_price)
        order.remaining_size = self._resolve_remaining_size(order, status)
        order.last_update_at = datetime.now(timezone.utc)
        order.raw_exchange_response_refs.append(self.audit.record("status_response", {"ts": datetime.now(timezone.utc).isoformat(), "exchange_order_id": order.exchange_order_id, "status": status}))
        if order.lifecycle_status in {OrderLifecycleStatus.CANCELLED, OrderLifecycleStatus.REJECTED, OrderLifecycleStatus.EXPIRED, OrderLifecycleStatus.FILLED}:
            order.terminal_state = True
        return status

    async def handle_timeout(self, order: LiveOrder) -> str:
        if order.timeout_at is None or datetime.now(timezone.utc) < order.timeout_at:
            return "NO_TIMEOUT"
        if order.filled_size > 0:
            return "PARTIAL_TIMEOUT"
        if not order.cancel_requested:
            order.cancel_requested = True
            order.lifecycle_status = OrderLifecycleStatus.CANCELLING
            try:
                await self.client.cancel_order(order.exchange_order_id or order.client_order_id)
                order.cancel_confirmed = True
                order.lifecycle_status = OrderLifecycleStatus.CANCELLED
                order.terminal_state = True
                order.last_update_at = datetime.now(timezone.utc)
                return "CANCELLED"
            except Exception:
                order.lifecycle_status = OrderLifecycleStatus.RECONCILING
                order.last_update_at = datetime.now(timezone.utc)
                return "TIMED_OUT"
        return "TIMED_OUT"

    async def maybe_reprice(self, order: LiveOrder, tradable: bool, drift_ok: bool) -> bool:
        if order.repriced_once or not tradable or not drift_ok:
            return False
        order.repriced_once = True
        order.timeout_at = datetime.now(timezone.utc) + timedelta(seconds=self.config.risk.order_timeout_seconds)
        return True

    def apply_fill_to_position(self, order: LiveOrder, decision: TradeDecision, positions: list[Position]) -> Position | None:
        if order.filled_size <= 0:
            return None
        position = next((item for item in positions if item.position_id == order.linked_position_id), None)
        if position is None:
            position = Position(
                position_id=stable_event_key(order.local_order_id, "position"),
                mode=Mode.LIVE,
                wallet_address=decision.wallet_address,
                source_wallet=decision.wallet_address,
                market_id=decision.market_id,
                token_id=decision.token_id,
                category=decision.category,
                entry_style=decision.entry_style,
                entry_price=decision.executable_price,
                current_mark_price=decision.executable_price,
                quantity=0.0,
                notional=0.0,
                fees_paid=0.0,
                source_trade_timestamp=datetime.now(timezone.utc),
                side="BUY",
                entry_order_ids=[order.local_order_id],
                entry_price_estimated=decision.executable_price,
                stop_loss_rule="10pct",
                take_profit_rule="15pct",
                time_stop_rule="48h",
                source_exit_following_enabled=True,
                exit_state="OPEN",
                entry_time=datetime.now(timezone.utc),
            )
            order.linked_position_id = position.position_id
            positions.append(position)

        confirmed_qty = order.filled_size
        existing_qty = position.entry_size
        if confirmed_qty > existing_qty:
            delta = confirmed_qty - existing_qty
            position.quantity += delta
            position.entry_size = confirmed_qty
            position.remaining_size = confirmed_qty - position.exited_size
            position.notional = round(position.notional + delta * (order.average_fill_price or decision.executable_price), 4)
            position.entry_price_actual = order.average_fill_price or decision.executable_price
            if order.exchange_order_id and order.exchange_order_id not in position.entry_order_ids:
                position.entry_order_ids.append(order.exchange_order_id)
            position.last_reconciled_at = datetime.now(timezone.utc)
        return position

    def _map_status(self, status: str) -> OrderLifecycleStatus:
        normalized = status.upper()
        mapping = {
            "CREATED": OrderLifecycleStatus.CREATED,
            "SUBMITTED": OrderLifecycleStatus.SUBMITTED,
            "ACKNOWLEDGED": OrderLifecycleStatus.ACKNOWLEDGED,
            "LIVE": OrderLifecycleStatus.RESTING,
            "RESTING": OrderLifecycleStatus.RESTING,
            "OPEN": OrderLifecycleStatus.RESTING,
            "PARTIALLY_FILLED": OrderLifecycleStatus.PARTIALLY_FILLED,
            "FILLED": OrderLifecycleStatus.FILLED,
            "CANCELLED": OrderLifecycleStatus.CANCELLED,
            "REJECTED": OrderLifecycleStatus.REJECTED,
            "EXPIRED": OrderLifecycleStatus.EXPIRED,
        }
        return mapping.get(normalized, OrderLifecycleStatus.UNKNOWN)

    def _resolve_remaining_size(self, order: LiveOrder, status: dict) -> float:
        raw_remaining = status.get("remaining_size")
        if raw_remaining not in (None, ""):
            try:
                remaining = float(raw_remaining)
                if remaining > 0 or order.lifecycle_status in {OrderLifecycleStatus.FILLED, OrderLifecycleStatus.CANCELLED, OrderLifecycleStatus.REJECTED, OrderLifecycleStatus.EXPIRED}:
                    return remaining
            except (TypeError, ValueError):
                pass

        raw = status.get("raw")
        if isinstance(raw, dict):
            try:
                original_size = float(raw.get("original_size") or 0.0)
                size_matched = float(raw.get("size_matched") or 0.0)
                if original_size > 0:
                    return max(original_size - size_matched, 0.0)
            except (TypeError, ValueError):
                pass

        return max(order.intended_size - order.filled_size, 0.0)
