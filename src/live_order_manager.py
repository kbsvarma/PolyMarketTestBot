from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.audit import AuditLogger
from src.config import AppConfig
from src.live_orders import LiveOrderStore
from src.models import EntryStyle, LiveOrder, Mode, OrderLifecycleStatus, OrderbookSnapshot, Position, TradeDecision
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
        order.timeout_at = datetime.now(timezone.utc) + timedelta(seconds=self._timeout_seconds_for_order(order))
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
        can_reprice = self._can_reprice(order)
        if not order.cancel_requested:
            order.cancel_requested = True
            order.lifecycle_status = OrderLifecycleStatus.CANCELLING
            try:
                await self.client.cancel_order(order.exchange_order_id or order.client_order_id)
                order.cancel_confirmed = True
                order.lifecycle_status = OrderLifecycleStatus.CANCELLED
                order.terminal_state = True
                order.last_update_at = datetime.now(timezone.utc)
                self.audit.record(
                    "timeout_cancelled",
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "local_order_id": order.local_order_id,
                        "exchange_order_id": order.exchange_order_id,
                        "client_order_id": order.client_order_id,
                        "reprice_ready": can_reprice,
                    },
                )
                if can_reprice:
                    return "REPRICE_READY"
                return "CANCELLED"
            except Exception:
                order.lifecycle_status = OrderLifecycleStatus.RECONCILING
                order.last_update_at = datetime.now(timezone.utc)
                return "TIMED_OUT"
        return "TIMED_OUT"

    async def cancel_open_order(self, order: LiveOrder) -> dict:
        response = await self.client.cancel_order(order.exchange_order_id or order.client_order_id)
        self._mark_cancelled(
            order,
            status=str(response.get("status") or "CANCELLED"),
            audit_event="cancel_response",
            response=response,
        )
        return response

    def mark_cancelled_by_reconciliation(self, order: LiveOrder, detail: dict | None = None) -> None:
        self._mark_cancelled(
            order,
            status="CANCELLED",
            audit_event="cancel_inferred",
            response=detail or {"reason": "order missing from exchange open orders"},
        )

    def prepare_reprice(self, order: LiveOrder, orderbook: OrderbookSnapshot, tradable: bool, drift_ok: bool) -> bool:
        if not tradable or not drift_ok or not self._can_reprice(order):
            return False
        new_price = self._repriced_price(order, orderbook)
        if new_price <= 0 or abs(new_price - order.intended_price) < 1e-9:
            return False
        previous_price = order.intended_price
        previous_client_order_id = order.client_order_id
        order.reprice_attempts += 1
        order.repriced_once = True
        order.intended_price = new_price
        order.remaining_size = max(order.intended_size - order.filled_size, 0.0)
        order.client_order_id = stable_event_key(order.local_order_id, "reprice", str(order.reprice_attempts))
        order.exchange_order_id = ""
        order.cancel_requested = False
        order.cancel_confirmed = False
        order.terminal_state = False
        order.last_exchange_status = "REPRICED"
        order.lifecycle_status = OrderLifecycleStatus.CREATED
        order.timeout_at = datetime.now(timezone.utc) + timedelta(seconds=self._timeout_seconds_for_order(order))
        order.last_update_at = datetime.now(timezone.utc)
        self.audit.record(
            "reprice_prepared",
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "local_order_id": order.local_order_id,
                "previous_price": previous_price,
                "new_price": new_price,
                "previous_client_order_id": previous_client_order_id,
                "client_order_id": order.client_order_id,
                "reprice_attempts": order.reprice_attempts,
            },
        )
        return True

    def apply_fill_to_position(self, order: LiveOrder, decision: TradeDecision | None, positions: list[Position]) -> Position | None:
        if order.is_exit or order.side != "BUY" or order.filled_size <= 0:
            return None
        position = next((item for item in positions if item.position_id == order.linked_position_id), None)
        if position is None:
            position = next(
                (
                    item
                    for item in positions
                    if not item.closed
                    and item.market_id == order.market_id
                    and item.token_id == order.token_id
                    and item.wallet_address == (decision.wallet_address if decision is not None else order.wallet_address)
                    and item.side == order.side
                ),
                None,
            )
        if position is None:
            entry_price = (
                decision.executable_price
                if decision is not None
                else order.average_fill_price or order.intended_price
            )
            wallet_address = decision.wallet_address if decision is not None else order.wallet_address
            category = decision.category if decision is not None else order.category
            strategy_name = decision.strategy_name if decision is not None else order.strategy_name
            position = Position(
                position_id=stable_event_key(order.local_order_id, "position"),
                mode=Mode.LIVE,
                strategy_name=strategy_name,
                wallet_address=wallet_address,
                source_wallet=wallet_address,
                market_id=order.market_id,
                token_id=order.token_id,
                category=category or "unknown",
                entry_style=decision.entry_style if decision is not None else order.entry_style,
                entry_price=entry_price,
                current_mark_price=entry_price,
                quantity=0.0,
                notional=0.0,
                fees_paid=0.0,
                source_trade_timestamp=order.created_at,
                entry_reason=decision.reason_code if decision is not None else "LIVE_FILL_RECONCILED",
                cluster_confirmed=decision.cluster_confirmed if decision is not None else False,
                hedge_suspicion_score=decision.hedge_suspicion_score if decision is not None else 0.0,
                side=order.side,
                thesis_type=decision.thesis_type if decision is not None else order.thesis_type,
                bundle_id=decision.bundle_id if decision is not None else order.bundle_id,
                bundle_role=decision.bundle_role if decision is not None else order.bundle_role,
                paired_token_id=decision.paired_token_id if decision is not None else order.paired_token_id,
                entry_order_ids=[order.local_order_id],
                entry_price_estimated=entry_price,
                entry_price_actual=entry_price,
                stop_loss_rule="10pct",
                take_profit_rule="15pct",
                time_stop_rule="48h",
                source_exit_following_enabled=True,
                exit_state="OPEN",
                entry_time=order.submitted_at or datetime.now(timezone.utc),
                peak_mark_price=entry_price,
                peak_pnl_pct=0.0,
                peak_mark_seen_at=order.submitted_at or datetime.now(timezone.utc),
            )
            order.linked_position_id = position.position_id
            positions.append(position)
        elif not order.linked_position_id:
            order.linked_position_id = position.position_id
        if not position.thesis_type:
            position.thesis_type = order.thesis_type or "directional"
        if not position.bundle_id:
            position.bundle_id = order.bundle_id
        if not position.bundle_role:
            position.bundle_role = order.bundle_role
        if not position.paired_token_id:
            position.paired_token_id = order.paired_token_id

        confirmed_qty = order.filled_size
        existing_qty = position.entry_size
        if confirmed_qty > existing_qty:
            delta = confirmed_qty - existing_qty
            position.quantity += delta
            position.entry_size = confirmed_qty
            position.remaining_size = confirmed_qty - position.exited_size
            fill_price = order.average_fill_price or (decision.executable_price if decision is not None else order.intended_price)
            position.notional = round(position.notional + delta * fill_price, 4)
            position.entry_price_actual = fill_price
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
            "MATCHED": OrderLifecycleStatus.FILLED,
            "CANCELED": OrderLifecycleStatus.CANCELLED,
            "CANCELLED": OrderLifecycleStatus.CANCELLED,
            "CANCELED_MARKET_RESOLVED": OrderLifecycleStatus.CANCELLED,
            "REJECTED": OrderLifecycleStatus.REJECTED,
            "EXPIRED": OrderLifecycleStatus.EXPIRED,
        }
        return mapping.get(normalized, OrderLifecycleStatus.UNKNOWN)

    def _resolve_remaining_size(self, order: LiveOrder, status: dict) -> float:
        if order.lifecycle_status in {OrderLifecycleStatus.CANCELLED, OrderLifecycleStatus.REJECTED, OrderLifecycleStatus.EXPIRED, OrderLifecycleStatus.FILLED}:
            return 0.0
        raw_remaining = status.get("remaining_size")
        if raw_remaining not in (None, ""):
            try:
                remaining = float(raw_remaining)
                if remaining > 0:
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

    def _mark_cancelled(self, order: LiveOrder, status: str, audit_event: str, response: dict) -> None:
        order.cancel_requested = True
        order.cancel_confirmed = True
        order.last_exchange_status = status
        order.lifecycle_status = self._map_status(order.last_exchange_status)
        if order.lifecycle_status == OrderLifecycleStatus.UNKNOWN:
            order.lifecycle_status = OrderLifecycleStatus.CANCELLED
        order.terminal_state = True
        order.remaining_size = 0.0
        order.last_update_at = datetime.now(timezone.utc)
        order.raw_exchange_response_refs.append(
            self.audit.record(
                audit_event,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "exchange_order_id": order.exchange_order_id,
                    "client_order_id": order.client_order_id,
                    "response": response,
                },
            )
        )

    def _timeout_seconds_for_order(self, order: LiveOrder) -> int:
        base_timeout = int(self.config.risk.order_timeout_seconds)
        if order.is_exit:
            return base_timeout
        if order.entry_style in {EntryStyle.PASSIVE_LIMIT, EntryStyle.POST_ONLY_MAKER}:
            return max(base_timeout, 60)
        return base_timeout

    def _can_reprice(self, order: LiveOrder) -> bool:
        return (
            not order.is_exit
            and order.side == "BUY"
            and order.entry_style in {EntryStyle.PASSIVE_LIMIT, EntryStyle.POST_ONLY_MAKER}
            and order.reprice_attempts < self.config.risk.max_reprice_attempts
        )

    def _repriced_price(self, order: LiveOrder, orderbook: OrderbookSnapshot) -> float:
        tick_size = 0.01
        best_bid = orderbook.bids[0].price if orderbook.bids else 0.0
        best_ask = orderbook.asks[0].price if orderbook.asks else 0.0
        current_price = min(max(order.intended_price, tick_size), 0.99)
        if order.entry_style == EntryStyle.POST_ONLY_MAKER:
            maker_ceiling = max(best_ask - tick_size, tick_size) if best_ask > 0 else current_price + tick_size
            improved_price = max(current_price + tick_size, best_bid + tick_size if best_bid > 0 else current_price + tick_size)
            return round(min(maker_ceiling, improved_price, 0.99), 6)
        if best_ask > 0:
            improved_price = max(current_price + tick_size, best_bid + tick_size if best_bid > 0 else current_price + tick_size)
            return round(min(best_ask, improved_price, 0.99), 6)
        if best_bid > 0:
            return round(min(max(current_price + tick_size, best_bid + tick_size), 0.99), 6)
        return round(min(current_price + tick_size, 0.99), 6)
