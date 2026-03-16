from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.audit import AuditLogger
from src.config import AppConfig
from src.exits import evaluate_exit
from src.geoblock import GeoblockChecker
from src.health import HealthMonitor
from src.live_order_manager import LiveOrderManager
from src.live_orders import LiveOrderStore
from src.live_readiness import build_readiness_result
from src.market_data import MarketDataService
from src.models import (
    DecisionAction,
    HealthComponent,
    HealthState,
    LiveOrder,
    Mode,
    OrderLifecycleStatus,
    Position,
    ReconciliationIssue,
    SystemStatus,
    TradeDecision,
)
from src.polymarket_client import PolymarketClient
from src.positions import PositionStore
from src.reconciliation import RECENT_EXCHANGE_VISIBILITY_GRACE_SECONDS, reconcile_live_state
from src.state import AppStateStore
from src.state_machine import SystemStateMachine
from src.utils import append_csv_row, stable_event_key


class LiveTradingEngine:
    def __init__(self, config: AppConfig, data_dir: Path, state: AppStateStore, geoblock: GeoblockChecker) -> None:
        self.config = config
        self.data_dir = data_dir
        self.state = state
        self.geoblock = geoblock
        self.client = PolymarketClient(config)
        self.market_data = MarketDataService(config, data_dir)
        self.positions = PositionStore(data_dir / "positions.json")
        self.orders = LiveOrderStore(data_dir / "live_orders.json")
        self.audit = AuditLogger(data_dir / "live_audit.jsonl")
        self.decisions_audit = AuditLogger(data_dir / "live_decisions.jsonl")
        self.health_monitor = HealthMonitor(data_dir / "health_status.json")
        self.state_machine = SystemStateMachine(state)
        self.order_manager = LiveOrderManager(config, data_dir, self.client, self.audit)

    def _is_soft_pause_reason(self, reason: str) -> bool:
        normalized = str(reason or "").strip()
        if not normalized:
            return False
        return normalized.startswith("Live exit ambiguity for ")

    def _entries_last_hour(self, live_orders: list[LiveOrder]) -> int:
        cutoff = datetime.now(timezone.utc).timestamp() - 3600.0
        seen_order_ids: set[str] = set()
        count = 0
        for order in live_orders:
            order_key = order.local_order_id or order.exchange_order_id or order.client_order_id
            if order_key in seen_order_ids:
                continue
            seen_order_ids.add(order_key)
            if order.is_exit:
                continue
            if order.lifecycle_status == OrderLifecycleStatus.REJECTED:
                continue
            if (order.submitted_at or order.created_at).timestamp() < cutoff:
                continue
            count += 1
        return count

    def _exchange_position_market_id(self, payload: dict[str, object]) -> str:
        return str(payload.get("market_id") or payload.get("conditionId") or payload.get("market") or "")

    def _exchange_position_token_id(self, payload: dict[str, object]) -> str:
        return str(payload.get("token_id") or payload.get("asset") or payload.get("asset_id") or payload.get("tokenId") or "")

    def _exchange_position_side(self, payload: dict[str, object]) -> str:
        return str(payload.get("side") or "BUY")

    def _exchange_position_quantity(self, payload: dict[str, object]) -> float:
        try:
            return round(float(payload.get("quantity") or payload.get("size") or payload.get("shares") or 0.0), 6)
        except (TypeError, ValueError):
            return 0.0

    def _exchange_position_price(self, payload: dict[str, object], value_keys: tuple[str, ...], fallback: float = 0.0) -> float:
        quantity = self._exchange_position_quantity(payload)
        if quantity <= 0:
            return fallback
        for key in value_keys:
            try:
                numeric = float(payload.get(key) or 0.0)
            except (TypeError, ValueError):
                numeric = 0.0
            if numeric > 0:
                return round(numeric / quantity, 6)
        return fallback

    def _position_open_size(self, position: Position) -> float:
        if position.closed:
            return 0.0
        if position.remaining_size > 0:
            return round(position.remaining_size, 6)
        return round(max(position.quantity, 0.0), 6)

    def _position_recently_updated(self, position: Position, now: datetime) -> bool:
        for value in (position.last_reconciled_at, position.entry_time, position.opened_at):
            if value is None:
                continue
            if (now - value).total_seconds() <= RECENT_EXCHANGE_VISIBILITY_GRACE_SECONDS:
                return True
        return False

    def _is_unmanaged_exit_hold(self, position: Position) -> bool:
        return (
            not position.closed
            and self._position_open_size(position) > 0
            and position.exit_state == "EXIT_DATA_UNAVAILABLE"
            and not position.source_exit_following_enabled
        )

    def _quarantine_unmanaged_exit_positions(self, live_positions: list[Position]) -> bool:
        updated = False
        now = datetime.now(timezone.utc)
        for position in live_positions:
            if position.closed or self._position_open_size(position) <= 0:
                continue
            if position.exit_state != "EXIT_DATA_UNAVAILABLE":
                continue
            if not position.source_exit_following_enabled:
                continue
            position.source_exit_following_enabled = False
            position.last_reconciled_at = now
            self.audit.record(
                "live_exit_management_quarantined",
                {
                    "ts": now.isoformat(),
                    "position_id": position.position_id,
                    "market_id": position.market_id,
                    "token_id": position.token_id,
                    "reason": "NO_EXIT_ORDERBOOK",
                },
            )
            updated = True
        return updated

    def _find_latest_filled_exit_order(
        self,
        live_orders: list[LiveOrder],
        market_id: str,
        token_id: str,
        position_side: str,
    ) -> LiveOrder | None:
        expected_exit_side = "SELL" if position_side.upper() == "BUY" else "BUY"
        return next(
            (
                order
                for order in reversed(live_orders)
                if order.is_exit
                and order.market_id == market_id
                and order.token_id == token_id
                and order.side.upper() == expected_exit_side
                and (order.filled_size > 0 or order.lifecycle_status == OrderLifecycleStatus.FILLED)
            ),
            None,
        )

    def _sync_position_open_size(
        self,
        position: Position,
        target_open_size: float,
        current_mark_price: float,
        now: datetime,
    ) -> bool:
        target_open_size = round(max(target_open_size, 0.0), 6)
        current_open_size = self._position_open_size(position)
        if target_open_size <= 0 and current_open_size > 0:
            entry_price = position.entry_price_actual or position.entry_price
            exit_price = current_mark_price or position.current_mark_price or entry_price
            total_size = max(position.entry_size or position.quantity, position.quantity, position.exited_size + current_open_size)
            position.exited_size = round(position.exited_size + current_open_size, 6)
            position.remaining_size = 0.0
            position.closed = True
            position.closed_at = now
            position.exit_state = "CLOSED"
            position.exit_reason = "EXCHANGE_NET_POSITION_REDUCED"
            position.current_mark_price = exit_price
            position.unrealized_pnl = 0.0
            position.realized_pnl = round((exit_price - entry_price) * total_size - position.fees_paid, 4)
            position.last_reconciled_at = now
            self.audit.record(
                "live_position_closed_from_exchange_net_sync",
                {
                    "ts": now.isoformat(),
                    "position_id": position.position_id,
                    "market_id": position.market_id,
                    "token_id": position.token_id,
                    "previous_open_size": current_open_size,
                    "exit_price": exit_price,
                },
            )
            return True
        changed = False
        if abs(current_open_size - target_open_size) > 1e-6:
            total_size = round(position.exited_size + target_open_size, 6)
            position.quantity = target_open_size
            position.entry_size = total_size
            position.remaining_size = target_open_size
            position.notional = round((position.entry_price_actual or position.entry_price) * total_size, 4)
            changed = True
            self.audit.record(
                "live_position_quantity_synced_from_exchange",
                {
                    "ts": now.isoformat(),
                    "position_id": position.position_id,
                    "market_id": position.market_id,
                    "token_id": position.token_id,
                    "previous_open_size": current_open_size,
                    "synced_open_size": target_open_size,
                },
            )
        if current_mark_price > 0:
            position.current_mark_price = current_mark_price
            changed = True
        if changed:
            position.last_reconciled_at = now
        return changed

    def _close_position_missing_from_exchange(
        self,
        position: Position,
        live_orders: list[LiveOrder],
        now: datetime,
    ) -> bool:
        if position.closed or self._position_open_size(position) <= 0:
            return False
        if self._position_recently_updated(position, now):
            return False

        latest_exit = self._find_latest_filled_exit_order(live_orders, position.market_id, position.token_id, position.side)
        if (
            latest_exit is not None
            and latest_exit.linked_position_id == position.position_id
            and self._apply_exit_fill_to_position(position, latest_exit, "EXCHANGE_POSITION_MISSING")
        ):
            self.audit.record(
                "live_position_closed_missing_from_exchange",
                {
                    "ts": now.isoformat(),
                    "position_id": position.position_id,
                    "market_id": position.market_id,
                    "token_id": position.token_id,
                    "exit_order_id": latest_exit.local_order_id,
                    "exchange_order_id": latest_exit.exchange_order_id,
                },
            )
            return True

        open_size = self._position_open_size(position)
        exit_price = position.current_mark_price or position.entry_price_actual or position.entry_price
        entry_price = position.entry_price_actual or position.entry_price
        total_size = max(position.entry_size or position.quantity, position.quantity, position.exited_size + open_size)
        position.exited_size = round(position.exited_size + open_size, 6)
        position.remaining_size = 0.0
        position.closed = True
        position.closed_at = now
        position.exit_state = "CLOSED"
        position.exit_reason = "EXCHANGE_POSITION_MISSING"
        position.current_mark_price = exit_price
        position.unrealized_pnl = 0.0
        position.realized_pnl = round((exit_price - entry_price) * total_size - position.fees_paid, 4)
        position.last_reconciled_at = now
        self.audit.record(
            "live_position_closed_missing_from_exchange",
            {
                "ts": now.isoformat(),
                "position_id": position.position_id,
                "market_id": position.market_id,
                "token_id": position.token_id,
                "exit_price": exit_price,
                "reason": "no_exchange_position_after_grace_window",
            },
        )
        return True

    async def _repair_missing_live_positions(
        self,
        live_positions: list[Position] | None = None,
        live_orders: list[LiveOrder] | None = None,
        exchange_positions: list[dict[str, object]] | None = None,
        exchange_open_orders: list[dict[str, object]] | None = None,
    ) -> bool:
        if self.config.mode != Mode.LIVE:
            return False

        live_positions = live_positions if live_positions is not None else self.positions.positions_for_mode(Mode.LIVE)
        live_orders = live_orders if live_orders is not None else self.orders.load()
        if exchange_positions is None:
            try:
                exchange_positions = await self.client.get_positions()
            except Exception:
                return False
        if not exchange_positions:
            return False
        if exchange_open_orders is None:
            try:
                exchange_open_orders = await self.client.get_open_orders()
            except Exception:
                exchange_open_orders = []

        open_exchange_order_ids = {
            str(item.get("exchange_order_id") or item.get("id") or item.get("orderID") or "")
            for item in (exchange_open_orders or [])
            if str(item.get("exchange_order_id") or item.get("id") or item.get("orderID") or "")
        }

        repaired = False
        now = datetime.now(timezone.utc)
        exchange_position_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
        for exchange_position in exchange_positions:
            market_id = self._exchange_position_market_id(exchange_position)
            token_id = self._exchange_position_token_id(exchange_position)
            side = self._exchange_position_side(exchange_position)
            quantity = self._exchange_position_quantity(exchange_position)
            if not market_id or not token_id or quantity <= 0:
                continue
            exchange_position_by_key[(market_id, token_id, side)] = exchange_position

        grouped_local_positions: dict[tuple[str, str, str], list[Position]] = {}
        for position in live_positions:
            if position.closed or self._position_open_size(position) <= 0:
                continue
            grouped_local_positions.setdefault((position.market_id, position.token_id, position.side), []).append(position)

        for key, positions_for_key in grouped_local_positions.items():
            exchange_position = exchange_position_by_key.get(key)
            if exchange_position is None:
                for position in positions_for_key:
                    if self._close_position_missing_from_exchange(position, live_orders, now):
                        repaired = True
                continue

            exchange_quantity = self._exchange_position_quantity(exchange_position)
            current_mark_price = self._exchange_position_price(
                exchange_position,
                ("currentValue", "current_value", "markValue", "mark_value"),
                fallback=0.0,
            )
            remaining_exchange_quantity = exchange_quantity
            for position in sorted(positions_for_key, key=lambda item: item.opened_at):
                current_open_size = self._position_open_size(position)
                target_open_size = min(current_open_size, remaining_exchange_quantity)
                if self._sync_position_open_size(position, target_open_size, current_mark_price, now):
                    repaired = True
                remaining_exchange_quantity = max(remaining_exchange_quantity - target_open_size, 0.0)
            exchange_position["_uncovered_quantity"] = remaining_exchange_quantity

        for exchange_position in exchange_positions:
            market_id = self._exchange_position_market_id(exchange_position)
            token_id = self._exchange_position_token_id(exchange_position)
            side = self._exchange_position_side(exchange_position)
            if "_uncovered_quantity" in exchange_position:
                quantity = round(float(exchange_position.get("_uncovered_quantity") or 0.0), 6)
            else:
                quantity = self._exchange_position_quantity(exchange_position)
            if not market_id or not token_id or quantity <= 0:
                continue

            matching_order = next(
                (
                    order
                    for order in reversed(live_orders)
                    if not order.is_exit
                    and order.market_id == market_id
                    and order.token_id == token_id
                    and order.side == side
                    and order.exchange_order_id not in open_exchange_order_ids
                ),
                None,
            )

            metadata: dict[str, object] = {}
            try:
                metadata = await self.market_data.fetch_market_metadata(market_id, token_id)
            except Exception:
                metadata = {}

            fallback_entry_price = matching_order.average_fill_price if matching_order and matching_order.average_fill_price > 0 else (
                matching_order.intended_price if matching_order is not None else 0.0
            )
            entry_price = self._exchange_position_price(
                exchange_position,
                ("initialValue", "initial_value", "costBasis", "cost_basis"),
                fallback=fallback_entry_price,
            )
            current_mark_price = self._exchange_position_price(
                exchange_position,
                ("currentValue", "current_value", "markValue", "mark_value"),
                fallback=entry_price,
            )
            if entry_price <= 0:
                entry_price = current_mark_price
            if current_mark_price <= 0:
                current_mark_price = entry_price
            if entry_price <= 0:
                continue

            wallet_address = (matching_order.wallet_address or "") if matching_order is not None else ""
            category = (matching_order.category or "") if matching_order is not None else ""
            if not category or category == "unknown":
                category = str(metadata.get("category") or "unknown")
            strategy_name = matching_order.strategy_name if matching_order is not None and matching_order.strategy_name else "wallet_follow"
            entry_style = matching_order.entry_style if matching_order is not None else self.config.entry_styles.preferred_live_entry_style
            source_wallet = wallet_address or "exchange-repaired"
            legacy_terminal_hint = False
            if matching_order is not None:
                legacy_terminal_hint = bool(
                    matching_order.terminal_state
                    or matching_order.cancel_confirmed
                    or matching_order.lifecycle_status
                    in {
                        OrderLifecycleStatus.CANCELLED,
                        OrderLifecycleStatus.REJECTED,
                        OrderLifecycleStatus.EXPIRED,
                    }
                )

            position = Position(
                position_id=stable_event_key(market_id, token_id, side, "exchange-repaired-position"),
                mode=Mode.LIVE,
                strategy_name=strategy_name,
                wallet_address=source_wallet,
                source_wallet=source_wallet,
                market_id=market_id,
                token_id=token_id,
                category=category,
                entry_style=entry_style,
                entry_price=entry_price,
                current_mark_price=current_mark_price,
                quantity=quantity,
                notional=round(quantity * entry_price, 4),
                fees_paid=0.0,
                source_trade_timestamp=(matching_order.created_at if matching_order is not None else now),
                entry_reason="LIVE_POSITION_REPAIRED",
                cluster_confirmed=False,
                hedge_suspicion_score=0.0,
                side=side,
                thesis_type=matching_order.thesis_type if matching_order is not None else "directional",
                bundle_id=matching_order.bundle_id if matching_order is not None else "",
                bundle_role=matching_order.bundle_role if matching_order is not None else "",
                paired_token_id=matching_order.paired_token_id if matching_order is not None else "",
                entry_order_ids=[
                    ref
                    for ref in [
                        matching_order.local_order_id if matching_order is not None else "",
                        matching_order.exchange_order_id if matching_order is not None else "",
                    ]
                    if ref
                ],
                entry_price_estimated=entry_price,
                entry_price_actual=entry_price,
                stop_loss_rule="10pct",
                take_profit_rule="15pct",
                time_stop_rule="48h",
                source_exit_following_enabled=True,
                exit_state="OPEN",
                last_reconciled_at=now,
                entry_time=(matching_order.submitted_at if matching_order is not None else now),
                entry_size=quantity,
                exited_size=0.0,
                remaining_size=quantity,
                peak_mark_price=current_mark_price,
                peak_pnl_pct=max((current_mark_price - entry_price) / max(entry_price, 1e-9), 0.0),
                peak_mark_seen_at=now,
            )
            live_positions.append(position)

            if matching_order is not None:
                matching_order.linked_position_id = position.position_id
                matching_order.filled_size = max(matching_order.filled_size, quantity)
                if matching_order.average_fill_price <= 0:
                    matching_order.average_fill_price = entry_price
                matching_order.remaining_size = max(matching_order.intended_size - matching_order.filled_size, 0.0)
                if matching_order.remaining_size <= 0:
                    matching_order.lifecycle_status = OrderLifecycleStatus.FILLED
                    matching_order.last_exchange_status = "MATCHED"
                    matching_order.terminal_state = True
                elif legacy_terminal_hint:
                    matching_order.lifecycle_status = OrderLifecycleStatus.PARTIALLY_FILLED
                    matching_order.last_exchange_status = "PARTIALLY_FILLED"
                    matching_order.remaining_size = 0.0
                    matching_order.terminal_state = True
                else:
                    matching_order.lifecycle_status = OrderLifecycleStatus.PARTIALLY_FILLED
                    matching_order.last_exchange_status = "PARTIALLY_FILLED"
                    matching_order.terminal_state = False

            self.audit.record(
                "live_position_repaired_from_exchange",
                {
                    "ts": now.isoformat(),
                    "market_id": market_id,
                    "token_id": token_id,
                    "side": side,
                    "quantity": quantity,
                    "entry_price_actual": entry_price,
                    "current_mark_price": current_mark_price,
                    "local_order_id": matching_order.local_order_id if matching_order is not None else "",
                    "exchange_order_id": matching_order.exchange_order_id if matching_order is not None else "",
                },
            )
            repaired = True

        if repaired:
            paper_positions = self.positions.positions_for_mode(Mode.PAPER)
            self.positions.save_positions(paper_positions, live_positions)
            self.orders.save(live_orders)
        return repaired

    def _revive_or_recover_exchange_open_orders(
        self,
        live_orders: list[LiveOrder],
        exchange_open_orders: list[dict[str, object]],
    ) -> bool:
        updated = False
        now = datetime.now(timezone.utc)
        for item in exchange_open_orders:
            exchange_order_id = str(item.get("exchange_order_id") or item.get("id") or item.get("orderID") or "")
            client_order_id = str(item.get("client_order_id") or item.get("clientOrderId") or "")
            if not exchange_order_id and not client_order_id:
                continue
            status = str(item.get("status") or item.get("state") or "")
            lifecycle_status = self.order_manager._map_status(status)
            filled_size = float(item.get("filled_size") or 0.0)
            remaining_size = float(item.get("remaining_size") or 0.0)
            size = float(item.get("size") or 0.0)
            price = float(item.get("price") or 0.0)
            side = str(item.get("side") or "BUY")
            market_id = str(item.get("market_id") or item.get("conditionId") or item.get("market") or "")
            token_id = str(item.get("token_id") or item.get("asset_id") or item.get("tokenId") or "")
            local_order = next(
                (
                    order
                    for order in reversed(live_orders)
                    if (exchange_order_id and order.exchange_order_id == exchange_order_id)
                    or (client_order_id and order.client_order_id == client_order_id)
                ),
                None,
            )
            if local_order is None:
                local_order = LiveOrder(
                    local_decision_id=stable_event_key(exchange_order_id or client_order_id, market_id, token_id, "exchange-recovered-decision"),
                    local_order_id=stable_event_key(exchange_order_id or client_order_id, market_id, token_id, "exchange-recovered-order"),
                    client_order_id=client_order_id or stable_event_key(exchange_order_id, "exchange-recovered-client"),
                    strategy_name="wallet_follow",
                    market_id=market_id,
                    token_id=token_id,
                    side=side,
                    intended_price=price,
                    intended_size=size or remaining_size,
                    entry_style=self.config.entry_styles.preferred_live_entry_style,
                    submitted_at=now,
                    last_exchange_status=status,
                    lifecycle_status=lifecycle_status,
                    filled_size=filled_size,
                    average_fill_price=price if filled_size > 0 and price > 0 else 0.0,
                    remaining_size=remaining_size or max((size or remaining_size) - filled_size, 0.0),
                    terminal_state=lifecycle_status in {
                        OrderLifecycleStatus.CANCELLED,
                        OrderLifecycleStatus.REJECTED,
                        OrderLifecycleStatus.EXPIRED,
                        OrderLifecycleStatus.FILLED,
                    },
                    exchange_order_id=exchange_order_id,
                )
                live_orders.append(local_order)
                self.audit.record(
                    "live_order_recovered_from_exchange",
                    {
                        "ts": now.isoformat(),
                        "exchange_order_id": exchange_order_id,
                        "client_order_id": client_order_id,
                        "market_id": market_id,
                        "token_id": token_id,
                        "status": status,
                    },
                )
                updated = True
                continue

            was_terminal = local_order.terminal_state
            if exchange_order_id:
                local_order.exchange_order_id = exchange_order_id
            if client_order_id:
                local_order.client_order_id = client_order_id
            if market_id:
                local_order.market_id = market_id
            if token_id:
                local_order.token_id = token_id
            if side:
                local_order.side = side
            if price > 0:
                local_order.intended_price = price
            if size > 0:
                local_order.intended_size = size
            local_order.last_exchange_status = status
            local_order.lifecycle_status = lifecycle_status
            local_order.filled_size = filled_size
            if filled_size > 0 and price > 0:
                local_order.average_fill_price = price
            local_order.remaining_size = remaining_size or max(local_order.intended_size - filled_size, 0.0)
            local_order.terminal_state = lifecycle_status in {
                OrderLifecycleStatus.CANCELLED,
                OrderLifecycleStatus.REJECTED,
                OrderLifecycleStatus.EXPIRED,
                OrderLifecycleStatus.FILLED,
            }
            if local_order.lifecycle_status in {
                OrderLifecycleStatus.SUBMITTED,
                OrderLifecycleStatus.ACKNOWLEDGED,
                OrderLifecycleStatus.RESTING,
                OrderLifecycleStatus.PARTIALLY_FILLED,
            }:
                local_order.terminal_state = False
                local_order.cancel_requested = False
                local_order.cancel_confirmed = False
            local_order.last_update_at = now
            if was_terminal and not local_order.terminal_state:
                self.audit.record(
                    "live_order_reactivated_from_exchange",
                    {
                        "ts": now.isoformat(),
                        "local_order_id": local_order.local_order_id,
                        "exchange_order_id": local_order.exchange_order_id,
                        "status": status,
                    },
                )
            updated = True
        return updated

    def _apply_exit_fill_to_position(self, position: Position, exit_order: LiveOrder, reason: str) -> bool:
        if exit_order.filled_size <= 0:
            return False
        total_size = max(position.entry_size or position.quantity, position.quantity, position.remaining_size + position.exited_size)
        exited_size = min(max(position.exited_size, exit_order.filled_size), total_size)
        remaining_size = max(total_size - exited_size, 0.0)
        position.exited_size = exited_size
        position.remaining_size = 0.0 if remaining_size <= 0.01 else remaining_size
        exit_price = exit_order.average_fill_price or position.current_mark_price or position.entry_price_actual or position.entry_price
        position.current_mark_price = exit_price
        position.last_reconciled_at = datetime.now(timezone.utc)
        if position.remaining_size > 0:
            position.exit_state = "PARTIAL_EXIT"
            return True
        position.closed = True
        position.closed_at = position.last_reconciled_at
        position.exit_state = "CLOSED"
        position.exit_reason = reason
        entry_price = position.entry_price_actual or position.entry_price
        position.unrealized_pnl = 0.0
        position.realized_pnl = round((exit_price - entry_price) * total_size - position.fees_paid, 4)
        return True

    async def _repair_closed_live_positions_from_exit_orders(
        self,
        live_positions: list[Position] | None = None,
        live_orders: list[LiveOrder] | None = None,
    ) -> bool:
        if self.config.mode != Mode.LIVE:
            return False
        live_positions = live_positions if live_positions is not None else self.positions.positions_for_mode(Mode.LIVE)
        live_orders = live_orders if live_orders is not None else self.orders.load()
        repaired = False
        for position in live_positions:
            if position.closed:
                continue
            filled_exit = next(
                (
                    order
                    for order in reversed(live_orders)
                    if order.is_exit
                    and order.linked_position_id == position.position_id
                    and (order.filled_size > 0 or order.lifecycle_status == OrderLifecycleStatus.FILLED)
                ),
                None,
            )
            if filled_exit is None:
                continue
            if not self._apply_exit_fill_to_position(position, filled_exit, "EXIT_FILLED"):
                continue
            self.audit.record(
                "live_position_closed_from_exit_fill",
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "position_id": position.position_id,
                    "market_id": position.market_id,
                    "token_id": position.token_id,
                    "exit_order_id": filled_exit.local_order_id,
                    "exchange_order_id": filled_exit.exchange_order_id,
                    "filled_size": filled_exit.filled_size,
                    "average_fill_price": filled_exit.average_fill_price,
                },
            )
            repaired = True
        if repaired:
            paper_positions = self.positions.positions_for_mode(Mode.PAPER)
            self.positions.save_positions(paper_positions, live_positions)
        return repaired

    async def startup_validation(self) -> dict[str, object]:
        checks: dict[str, object] = {}
        client_health = await self.client.health_check()
        checks["auth_valid"] = client_health.ok
        checks["auth_detail"] = client_health.detail
        live_order_capable, live_order_capable_detail = self.client.live_order_capable()
        checks["live_order_capable"] = live_order_capable
        checks["live_order_capable_detail"] = live_order_capable_detail

        try:
            balance = await self.client.get_balance()
            checks["balance_visible"] = True
            checks["balance_detail"] = str(balance)
        except Exception as exc:
            checks["balance_visible"] = False
            checks["balance_detail"] = str(exc)

        try:
            allowance = await self.client.get_allowance()
            checks["allowance_visible"] = bool(allowance.get("query_visible", True))
            checks["allowance_sufficient"] = bool(allowance.get("sufficient", False))
            checks["allowance_available"] = float(allowance.get("available", 0.0) or 0.0)
            checks["allowance_raw_summary"] = allowance.get("raw_summary", {})
            checks["allowance_sdk_signature_type"] = allowance.get("sdk_signature_type")
            checks["allowance_configured_signature_type"] = allowance.get("configured_signature_type")
            checks["allowance_detail"] = str(allowance)
        except Exception as exc:
            checks["allowance_visible"] = False
            checks["allowance_sufficient"] = False
            checks["allowance_available"] = 0.0
            checks["allowance_raw_summary"] = {"error": str(exc)}
            checks["allowance_sdk_signature_type"] = None
            checks["allowance_configured_signature_type"] = self.config.env.polymarket_signature_type
            checks["allowance_detail"] = str(exc)

        try:
            wallet_balances = await self.client.get_wallet_stablecoin_balances()
            checks["wallet_balance_visible"] = True
            checks["wallet_balance_detail"] = str(wallet_balances)
            checks["wallet_total_stablecoins"] = float(wallet_balances.get("total_stablecoins", 0.0) or 0.0)
        except Exception as exc:
            checks["wallet_balance_visible"] = False
            checks["wallet_balance_detail"] = str(exc)
            checks["wallet_total_stablecoins"] = 0.0

        try:
            open_orders = await self.client.get_open_orders()
            checks["open_orders_visible"] = True
            checks["open_orders_detail"] = f"{len(open_orders)} orders visible"
        except Exception as exc:
            checks["open_orders_visible"] = False
            checks["open_orders_detail"] = str(exc)

        try:
            positions = await self.client.get_positions()
            checks["positions_visible"] = True
            checks["positions_count"] = len(positions)
            checks["positions_query_error"] = ""
            checks["positions_detail"] = f"{len(positions)} positions visible"
        except Exception as exc:
            checks["positions_visible"] = False
            checks["positions_count"] = 0
            checks["positions_query_error"] = str(exc)
            checks["positions_detail"] = str(exc)

        try:
            markets = await self.market_data.refresh_markets()
            token_ids = [market.token_id for market in list(markets.values())[:3]]
            await self.market_data.stream_watchlist(token_ids)
            tradability = []
            for market in list(markets.values())[:3]:
                tradability.append(await self.market_data.get_tradability(market.market_id, market.token_id))
            checks["tradability_ok"] = all(
                isinstance(item, dict) and bool(item.get("market_id")) and bool(item.get("token_id")) and "tradable" in item
                for item in tradability
            ) if tradability else False
            checks["tradability_detail"] = str(tradability)
            checks["rest_ok"] = True
            checks["rest_detail"] = "REST and websocket checks passed."
        except Exception as exc:
            checks["tradability_ok"] = False
            checks["tradability_detail"] = str(exc)
            checks["rest_ok"] = False
            checks["rest_detail"] = str(exc)

        reconciliation = await self.reconcile()
        checks["reconciliation_clean"] = reconciliation.clean
        checks["reconciliation_detail"] = reconciliation.severity
        checks["reconciliation_summary"] = reconciliation.model_dump(mode="json")
        return checks

    async def refresh_live_status(self) -> None:
        live_positions_for_state: list[Position] = []
        live_orders_for_state: list[LiveOrder] = []
        if self.config.mode.value == "LIVE":
            live_positions = self.positions.positions_for_mode(Mode.LIVE)
            persist_quarantine_state = any(
                position.exit_state == "EXIT_DATA_UNAVAILABLE" and not position.source_exit_following_enabled
                for position in live_positions
            )
            if self._quarantine_unmanaged_exit_positions(live_positions) or persist_quarantine_state:
                paper_positions = self.positions.positions_for_mode(Mode.PAPER)
                self.positions.save_positions(paper_positions, live_positions)
            await self._sync_existing_open_orders()
            await self._repair_closed_live_positions_from_exit_orders()
            await self._repair_missing_live_positions()
            live_positions_for_state = self.positions.positions_for_mode(Mode.LIVE)
            live_orders_for_state = self.orders.load()
        checks = await self.startup_validation()
        reconciliation_summary = checks.get("reconciliation_summary", {})
        unresolved = [
            order.local_order_id
            for order in self.orders.load()
            if order.lifecycle_status == OrderLifecycleStatus.UNKNOWN and not order.terminal_state
        ]
        if bool(checks.get("reconciliation_clean", False)):
            current = self.state.read()
            current_pause_reason = str(current.get("pause_reason", ""))
            if (
                current_pause_reason == "Live readiness gate failed."
                or self._is_soft_pause_reason(current_pause_reason)
            ) and not unresolved:
                self.state.clear_pause()
        health = await self.collect_health(
            reconciliation_override={
                "clean": bool(checks.get("reconciliation_clean", False)),
                "severity": str(checks.get("reconciliation_detail", "")),
                "summary": reconciliation_summary,
            }
        )
        readiness = build_readiness_result(self.config, self.state, health, checks)
        positions_payload: list[dict] = []
        try:
            positions_payload = await self.client.get_positions()
        except Exception:
            positions_payload = []
        portfolio_current_value = sum(float(item.get("currentValue") or 0.0) for item in positions_payload)
        portfolio_initial_value = sum(float(item.get("initialValue") or 0.0) for item in positions_payload)
        connected_proxy_wallet = ""
        if positions_payload:
            connected_proxy_wallet = str(positions_payload[0].get("proxyWallet") or "")
        wallet_stablecoin_balances: dict[str, object] = {}
        try:
            wallet_stablecoin_balances = await self.client.get_wallet_stablecoin_balances()
        except Exception:
            wallet_stablecoin_balances = {}
        self.state.update_system_status(
            **self._operator_cap_state(live_positions_for_state, live_orders_for_state),
            live_health_state=health.overall.value,
            live_readiness_last_result=readiness.model_dump(mode="json"),
            heartbeat_ok=not any(component.name == "heartbeat" and component.state != HealthState.HEALTHY for component in health.components),
            balance_visible=bool(checks.get("balance_visible", False)),
            balance_detail=str(checks.get("balance_detail", "")),
            allowance_visible=bool(checks.get("allowance_visible", False)),
            manual_live_enable=bool(self.config.live.manual_live_enable),
            allowance_sufficient=bool(checks.get("allowance_sufficient", False)),
            allowance_available=float(checks.get("allowance_available", 0.0) or 0.0),
            allowance_raw_summary=checks.get("allowance_raw_summary", {}),
            allowance_detail=str(checks.get("allowance_detail", "")),
            allowance_sdk_signature_type=checks.get("allowance_sdk_signature_type"),
            allowance_configured_signature_type=checks.get("allowance_configured_signature_type"),
            auth_detail=str(checks.get("auth_detail", "")),
            open_orders_visible=bool(checks.get("open_orders_visible", False)),
            open_orders_detail=str(checks.get("open_orders_detail", "")),
            positions_visible=bool(checks.get("positions_visible", False)),
            positions_count=int(checks.get("positions_count", 0) or 0),
            positions_query_error=str(checks.get("positions_query_error", "")),
            positions_detail=str(checks.get("positions_detail", "")),
            connected_funder_wallet=self.config.env.polymarket_funder,
            connected_proxy_wallet=connected_proxy_wallet,
            wallet_balance_visible=bool(checks.get("wallet_balance_visible", False)),
            wallet_balance_detail=str(checks.get("wallet_balance_detail", "")),
            wallet_usdc_balance=float(wallet_stablecoin_balances.get("usdc", 0.0) or 0.0),
            wallet_usdce_balance=float(wallet_stablecoin_balances.get("usdce", 0.0) or 0.0),
            wallet_total_stablecoins=float(wallet_stablecoin_balances.get("total_stablecoins", 0.0) or 0.0),
            portfolio_position_value=portfolio_current_value,
            portfolio_cost_basis=portfolio_initial_value,
            tradability_ok=bool(checks.get("tradability_ok", False)),
            tradability_detail=str(checks.get("tradability_detail", "")),
            reconciliation_clean=bool(checks.get("reconciliation_clean", False)),
            reconciliation_summary=reconciliation_summary,
            unresolved_live_order_ids=unresolved,
            live_last_reconciled_at=datetime.now(timezone.utc).isoformat(),
        )
        if self.config.mode.value == "LIVE":
            if readiness.ready:
                current_status = self.state.read().get("system_status")
                target = SystemStatus.LIVE_READY if current_status != SystemStatus.LIVE_ACTIVE.value else SystemStatus.LIVE_ACTIVE
                if current_status != target.value:
                    self.state_machine.transition(target, "Live readiness gate passed.")
            else:
                self.state.pause("Live readiness gate failed.")

    async def collect_health(self, reconciliation_override: dict[str, object] | None = None):
        now = datetime.now(timezone.utc)
        components: list[HealthComponent] = []

        client_health = await self.client.health_check()
        components.append(HealthComponent(name="auth", state=HealthState.HEALTHY if client_health.ok else HealthState.UNHEALTHY, detail=client_health.detail))

        try:
            heartbeat = await self.client.send_heartbeat()
            heartbeat_ts = heartbeat.get("timestamp")
            age = 0.0
            if heartbeat_ts:
                age = (now - datetime.fromisoformat(str(heartbeat_ts).replace("Z", "+00:00"))).total_seconds()
            hb_state = HealthState.HEALTHY if age <= self.config.risk.heartbeat_timeout_seconds else HealthState.DEGRADED
            components.append(HealthComponent(name="heartbeat", state=hb_state, detail="Heartbeat ok", age_seconds=age, metadata=heartbeat))
        except Exception as exc:
            components.append(HealthComponent(name="heartbeat", state=HealthState.UNHEALTHY, detail=str(exc)))

        ws_age = self.market_data.ws.event_age_seconds()
        ws_state = HealthState.HEALTHY if self.market_data.ws.connected and ws_age <= self.config.risk.stale_market_data_seconds else HealthState.DEGRADED
        components.append(
            HealthComponent(
                name="market_ws",
                state=ws_state,
                detail=self.market_data.ws.last_error or "websocket status",
                age_seconds=ws_age,
                metadata={"reconnect_attempts": self.market_data.ws.reconnect_attempts, "watched_token_ids": self.market_data.ws.watched_token_ids},
            )
        )

        try:
            balance = await self.client.get_balance()
            components.append(HealthComponent(name="balance", state=HealthState.HEALTHY, detail="Balance visible", metadata=balance))
        except Exception as exc:
            components.append(HealthComponent(name="balance", state=HealthState.UNHEALTHY, detail=str(exc)))

        try:
            allowance = await self.client.get_allowance()
            allow_state = HealthState.HEALTHY if allowance.get("sufficient") else HealthState.UNHEALTHY
            components.append(HealthComponent(name="allowance", state=allow_state, detail="Allowance/spendability check", metadata=allowance))
        except Exception as exc:
            components.append(HealthComponent(name="allowance", state=HealthState.UNHEALTHY, detail=str(exc)))

        if reconciliation_override is None:
            reconciliation = await self.reconcile()
            reconciliation_clean = reconciliation.clean
            reconciliation_severity = reconciliation.severity
            reconciliation_metadata = reconciliation.model_dump(mode="json")
        else:
            reconciliation_clean = bool(reconciliation_override.get("clean", False))
            reconciliation_severity = str(reconciliation_override.get("severity", "UNKNOWN"))
            reconciliation_metadata = reconciliation_override.get("summary", {})
        components.append(
            HealthComponent(
                name="reconciliation",
                state=HealthState.HEALTHY if reconciliation_clean else HealthState.UNHEALTHY,
                detail=reconciliation_severity,
                metadata=reconciliation_metadata,
            )
        )
        return self.health_monitor.aggregate(components)

    async def _sync_existing_open_orders(self) -> None:
        live_orders = self.orders.load()
        try:
            open_orders = await self.client.get_open_orders()
        except Exception:
            return
        updated = self._revive_or_recover_exchange_open_orders(live_orders, open_orders)
        active_orders = [order for order in live_orders if not order.terminal_state and order.exchange_order_id]
        if not active_orders:
            if updated:
                self.orders.save(live_orders)
            return
        open_ids = {
            str(item.get("exchange_order_id") or item.get("id") or item.get("orderID") or "")
            for item in open_orders
        }
        for order in active_orders:
            if order.exchange_order_id in open_ids:
                continue
            try:
                await self.order_manager.refresh_order(order)
            except Exception:
                self.order_manager.mark_cancelled_by_reconciliation(
                    order,
                    detail={"reason": "order missing from exchange open orders"},
                )
                updated = True
            else:
                updated = True
        if updated:
            self.orders.save(live_orders)

    async def reconcile(
        self,
        live_positions: list[Position] | None = None,
        live_orders: list[LiveOrder] | None = None,
        exchange_positions: list[dict[str, object]] | None = None,
        exchange_orders: list[dict[str, object]] | None = None,
    ):
        local_position_payload = [
            position.model_dump(mode="json")
            for position in (live_positions if live_positions is not None else self.positions.positions_for_mode(Mode.LIVE))
        ]
        local_order_payload = [
            order.model_dump(mode="json")
            for order in (live_orders if live_orders is not None else self.orders.load())
        ]
        try:
            if exchange_positions is None:
                exchange_positions = await self.client.get_positions()
            if exchange_orders is None:
                exchange_orders = await self.client.get_open_orders()
        except Exception as exc:
            summary = reconcile_live_state(local_position_payload, [], local_order_payload, [])
            summary.issues.append(ReconciliationIssue(severity="SEVERE", issue_type="EXCHANGE_VISIBILITY", detail=str(exc)))
            summary.clean = False
            summary.severity = "SEVERE"
            return summary
        return reconcile_live_state(local_position_payload, exchange_positions, local_order_payload, exchange_orders)

    async def handle_decisions(self, decisions: list[TradeDecision]) -> None:
        await self.refresh_live_status()
        if self.config.mode.value != "LIVE" or not self.config.live.enabled or not self.config.env.live_trading_enabled:
            return

        state_snapshot = self.state.read()
        if state_snapshot.get("paused") or state_snapshot.get("kill_switch") or state_snapshot.get("manual_resume_required"):
            return
        if not state_snapshot.get("live_readiness_last_result", {}).get("ready", False):
            return

        geostatus = self.geoblock.live_trading_allowed()
        if not geostatus.eligible:
            self.state.pause(geostatus.detail)
            return

        if self.config.live.emergency_flatten_flag:
            await self._emergency_flatten()
            return

        health = await self.collect_health()
        if health.overall == HealthState.UNHEALTHY:
            self.state.update_system_status(live_health_state=health.overall.value)
            return

        try:
            self.state_machine.transition(SystemStatus.LIVE_ACTIVE, "Managing live orders.")
        except ValueError:
            pass

        paper_positions = self.positions.positions_for_mode(Mode.PAPER)
        live_positions = self.positions.positions_for_mode(Mode.LIVE)
        live_orders = self.orders.load()
        entries_last_hour = self._entries_last_hour(live_orders)

        self.state.update_system_status(**self._operator_cap_state(live_positions, live_orders))

        await self._manage_existing_orders(live_orders, live_positions, decisions)
        await self._apply_live_exits(live_positions, live_orders)
        await self._repair_closed_live_positions_from_exit_orders(live_positions=live_positions, live_orders=live_orders)
        await self._repair_missing_live_positions(live_positions=live_positions, live_orders=live_orders)

        for decision in decisions:
            effective_decision = decision
            if decision.action == DecisionAction.LIVE_COPY and decision.allowed:
                effective_decision = self._effective_live_decision(decision)
                if effective_decision.scaled_notional != decision.scaled_notional:
                    self.audit.record(
                        "live_order_size_floor_applied",
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "decision_id": decision.local_decision_id,
                            "original_notional": decision.scaled_notional,
                            "effective_notional": effective_decision.scaled_notional,
                            "minimum_order_size_shares": effective_decision.context.get("minimum_live_order_size_shares"),
                        },
                    )
            self.decisions_audit.record("live_decision", {"ts": datetime.now(timezone.utc).isoformat(), "decision": effective_decision.model_dump(mode="json")})
            if effective_decision.action != DecisionAction.LIVE_COPY or not effective_decision.allowed:
                continue
            if effective_decision.category not in self.config.live.selected_categories:
                continue
            if any(order.local_decision_id == effective_decision.local_decision_id and not order.terminal_state for order in live_orders):
                continue
            # Prevent duplicate entries: skip if an open order or live position already exists for this token.
            if any(
                str(order.token_id or "") == str(effective_decision.token_id or "")
                and str(order.side or "").upper() == str(effective_decision.side or "BUY").upper()
                and not order.terminal_state
                for order in live_orders
            ):
                self.audit.record("duplicate_token_skip", {"ts": datetime.now(timezone.utc).isoformat(), "token_id": effective_decision.token_id, "market_id": effective_decision.market_id})
                continue
            if any(
                str(p.get("token_id") or "") == str(effective_decision.token_id or "")
                and str(p.get("side") or "").upper() == str(effective_decision.side or "BUY").upper()
                and p.get("status") not in ("CLOSED", "RESOLVED")
                for p in live_positions
            ):
                self.audit.record("duplicate_position_skip", {"ts": datetime.now(timezone.utc).isoformat(), "token_id": effective_decision.token_id, "market_id": effective_decision.market_id})
                continue
            if entries_last_hour >= self.config.risk.max_new_entries_per_hour:
                self.audit.record(
                    "entry_rate_limit_block",
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "decision_id": effective_decision.local_decision_id,
                        "entries_last_hour": entries_last_hour,
                    },
                )
                continue
            cap_block = self._operator_entry_block_reason(effective_decision, live_positions, live_orders)
            if cap_block is not None:
                self.audit.record(
                    "operator_cap_block",
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "decision_id": effective_decision.local_decision_id,
                        "reason": cap_block,
                    },
                )
                continue

            cached_tradability = effective_decision.context.get("tradability")
            cached_tradability_valid = (
                isinstance(cached_tradability, dict)
                and str(cached_tradability.get("market_id") or effective_decision.market_id) == effective_decision.market_id
                and str(cached_tradability.get("token_id") or effective_decision.token_id) == effective_decision.token_id
                and "tradable" in cached_tradability
                and "orderbook_enabled" in cached_tradability
            )
            if cached_tradability_valid:
                tradability = dict(cached_tradability)
            else:
                try:
                    tradability = await self.market_data.get_tradability(effective_decision.market_id, effective_decision.token_id)
                except Exception as exc:
                    self.audit.record(
                        "tradability_lookup_error",
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "decision_id": effective_decision.local_decision_id,
                            "market_id": effective_decision.market_id,
                            "token_id": effective_decision.token_id,
                            "error": str(exc),
                            "used_cached_tradability": False,
                        },
                    )
                    continue
            self.state.update_system_status(watched_market_tradability_cache={effective_decision.market_id: tradability})
            if not (tradability.get("tradable") and tradability.get("orderbook_enabled")):
                self.audit.record(
                    "tradability_block",
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "decision_id": effective_decision.local_decision_id,
                        "tradability": tradability,
                    },
                )
                continue

            live_order = self.order_manager.create_entry_order(effective_decision)
            live_orders.append(live_order)
            self.orders.save(live_orders)
            self.audit.record("order_created", {"ts": datetime.now(timezone.utc).isoformat(), "local_order_id": live_order.local_order_id, "client_order_id": live_order.client_order_id})

            try:
                await self.order_manager.submit_order(live_order, side="BUY")
                self.orders.save(live_orders)
                entries_last_hour += 1
            except Exception as exc:
                self.audit.record("order_error", {"ts": datetime.now(timezone.utc).isoformat(), "local_order_id": live_order.local_order_id, "error": str(exc)})
                self.orders.save(live_orders)
                if live_order.lifecycle_status == OrderLifecycleStatus.REJECTED:
                    continue
                self.state.pause(f"Live order error: {exc}")
                break

        await self._manage_existing_orders(live_orders, live_positions, decisions)
        await self._repair_closed_live_positions_from_exit_orders(live_positions=live_positions, live_orders=live_orders)
        await self._repair_missing_live_positions(live_positions=live_positions, live_orders=live_orders)
        self.positions.save_positions(paper_positions, live_positions)
        self.orders.save(live_orders)
        try:
            reconciliation = await self.reconcile(live_positions=live_positions, live_orders=live_orders)
        except TypeError:
            reconciliation = await self.reconcile()
        unresolved = [order.local_order_id for order in live_orders if order.lifecycle_status == OrderLifecycleStatus.UNKNOWN and not order.terminal_state]
        self.state.update_system_status(
            reconciliation_clean=reconciliation.clean,
            reconciliation_summary=reconciliation.model_dump(mode="json"),
            unresolved_live_order_ids=unresolved,
            **self._operator_cap_state(live_positions, live_orders),
        )
        if not reconciliation.clean:
            self.state.pause(f"Live reconciliation mismatch: {reconciliation.severity}")

    def _operator_cap_state(self, live_positions: list[Position], live_orders: list[LiveOrder]) -> dict[str, object]:
        active_positions = [position for position in live_positions if not position.closed and position.remaining_size > 0]
        active_entry_orders = [order for order in live_orders if not order.terminal_state and not order.is_exit]
        session_exposure = round(
            sum(position.remaining_size * (position.current_mark_price or position.entry_price_actual or position.entry_price) for position in active_positions)
            + sum(max(order.remaining_size, order.intended_size) * order.intended_price for order in active_entry_orders),
            4,
        )
        actively_managed_positions = [
            position for position in active_positions if not self._is_unmanaged_exit_hold(position)
        ]
        occupied_slots = len(actively_managed_positions) + len(active_entry_orders)
        return {
            "operator_live_session_max_usd": self.config.env.operator_live_session_max_usd,
            "operator_live_max_trade_usd": self.config.env.operator_live_max_trade_usd,
            "operator_live_max_positions": self.config.env.operator_live_max_positions,
            "operator_live_current_exposure_usd": session_exposure,
            "operator_live_current_slots": occupied_slots,
        }

    def _effective_live_decision(self, decision: TradeDecision) -> TradeDecision:
        min_shares = float(getattr(self.config.live, "minimum_order_size_shares", 0.0) or 0.0)
        min_notional_usd = float(getattr(self.config.live, "minimum_order_notional_usd", 0.0) or 0.0)
        if (min_shares <= 0 and min_notional_usd <= 0) or decision.executable_price <= 0:
            return decision
        minimum_notional = max(min_shares * decision.executable_price, min_notional_usd)
        minimum_notional = round(minimum_notional, 4)
        if decision.scaled_notional >= minimum_notional:
            return decision
        updated_context = {
            **decision.context,
            "minimum_live_order_size_shares": min_shares,
            "minimum_live_order_notional": minimum_notional,
            "minimum_live_order_notional_usd": min_notional_usd,
            "minimum_live_order_size_applied": True,
        }
        return decision.model_copy(update={"scaled_notional": minimum_notional, "context": updated_context})

    def _operator_entry_block_reason(self, decision: TradeDecision, live_positions: list[Position], live_orders: list[LiveOrder]) -> str | None:
        max_trade = self.config.env.operator_live_max_trade_usd
        if max_trade is not None and max_trade > 0 and decision.scaled_notional > max_trade:
            return f"OPERATOR_MAX_TRADE_USD exceeded: {decision.scaled_notional:.4f} > {max_trade:.4f}"

        active_positions = [position for position in live_positions if not position.closed and position.remaining_size > 0]
        active_entry_orders = [order for order in live_orders if not order.terminal_state and not order.is_exit]

        max_positions = self.config.env.operator_live_max_positions
        if max_positions is not None and max_positions > 0:
            actively_managed_positions = [
                position for position in active_positions if not self._is_unmanaged_exit_hold(position)
            ]
            occupied_slots = len(actively_managed_positions) + len(active_entry_orders)
            if occupied_slots >= max_positions:
                return f"OPERATOR_MAX_POSITIONS reached: {occupied_slots} >= {max_positions}"

        session_cap = self.config.env.operator_live_session_max_usd
        if session_cap is not None and session_cap > 0:
            current_exposure = sum(
                position.remaining_size * (position.current_mark_price or position.entry_price_actual or position.entry_price)
                for position in active_positions
            ) + sum(max(order.remaining_size, order.intended_size) * order.intended_price for order in active_entry_orders)
            if current_exposure + decision.scaled_notional > session_cap:
                return (
                    f"OPERATOR_SESSION_MAX_USD exceeded: "
                    f"{current_exposure + decision.scaled_notional:.4f} > {session_cap:.4f}"
                )
        return None

    async def _manage_existing_orders(self, live_orders: list[LiveOrder], live_positions: list[Position], decisions: list[TradeDecision]) -> None:
        for order in live_orders:
            if order.lifecycle_status in {OrderLifecycleStatus.PARTIALLY_FILLED, OrderLifecycleStatus.FILLED}:
                decision = next((item for item in decisions if item.local_decision_id == order.local_decision_id), None)
                self.order_manager.apply_fill_to_position(order, decision, live_positions)
            if order.terminal_state:
                continue
            try:
                await self.order_manager.refresh_order(order)
            except Exception as exc:
                order.lifecycle_status = OrderLifecycleStatus.UNKNOWN
                self.state.pause(f"Order status ambiguity for {order.local_order_id}: {exc}")
                return

            if order.lifecycle_status in {OrderLifecycleStatus.PARTIALLY_FILLED, OrderLifecycleStatus.FILLED}:
                decision = next((item for item in decisions if item.local_decision_id == order.local_decision_id), None)
                self.order_manager.apply_fill_to_position(order, decision, live_positions)
            if order.lifecycle_status in {OrderLifecycleStatus.SUBMITTED, OrderLifecycleStatus.ACKNOWLEDGED, OrderLifecycleStatus.RESTING}:
                timeout_result = await self.order_manager.handle_timeout(order)
                if timeout_result == "REPRICE_READY":
                    try:
                        tradability = await self.market_data.get_tradability(order.market_id, order.token_id)
                    except Exception as exc:
                        self.audit.record(
                            "reprice_tradability_error",
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "local_order_id": order.local_order_id,
                                "market_id": order.market_id,
                                "token_id": order.token_id,
                                "error": str(exc),
                            },
                        )
                        continue
                    try:
                        orderbook = await self.client.get_orderbook(order.token_id)
                    except Exception as exc:
                        self.audit.record(
                            "reprice_orderbook_error",
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "local_order_id": order.local_order_id,
                                "token_id": order.token_id,
                                "error": str(exc),
                            },
                        )
                        continue
                    drift_ok = True
                    if self.order_manager.prepare_reprice(order, orderbook, bool(tradability.get("tradable")), drift_ok):
                        try:
                            await self.order_manager.submit_order(order, side=order.side)
                        except Exception as exc:
                            self.audit.record(
                                "reprice_submit_error",
                                {
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                    "local_order_id": order.local_order_id,
                                    "client_order_id": order.client_order_id,
                                    "error": str(exc),
                                },
                            )
                            continue
                elif timeout_result == "TIMED_OUT":
                    order.lifecycle_status = OrderLifecycleStatus.RECONCILING
                    self.state.pause(f"Order timed out unresolved: {order.local_order_id}")
                    return
            if order.lifecycle_status == OrderLifecycleStatus.UNKNOWN:
                self.state.pause(f"Unknown order state: {order.local_order_id}")
                return

    async def _apply_live_exits(self, live_positions: list[Position], live_orders: list[LiveOrder]) -> None:
        for position in live_positions:
            if position.closed or position.remaining_size <= 0:
                continue
            if self._is_unmanaged_exit_hold(position):
                continue
            try:
                orderbook = await self.client.get_orderbook(position.token_id)
                best_bid = orderbook.bids[0].price if orderbook.bids else position.entry_price_actual or position.entry_price
                position.current_mark_price = best_bid
                position.unrealized_pnl = round((best_bid - (position.entry_price_actual or position.entry_price)) * position.remaining_size - position.fees_paid, 4)
                should_exit, reason = evaluate_exit(
                    position,
                    best_bid,
                    hold_to_resolution=position.thesis_type == "paired_arb",
                    profit_arm_pct=self.config.live.adaptive_profit_arm_pct,
                    min_profit_lock_pct=self.config.live.adaptive_profit_min_lock_pct,
                    trailing_profit_retrace_pct=self.config.live.trailing_profit_retrace_pct,
                    strong_winner_profit_pct=self.config.live.strong_winner_profit_pct,
                    strong_winner_retrace_pct=self.config.live.strong_winner_retrace_pct,
                    paired_arb_time_stop_hours=self.config.live.paired_arb_time_stop_hours,
                )
                if not should_exit:
                    continue

                latest_exit = next((order for order in reversed(live_orders) if order.is_exit and order.linked_position_id == position.position_id), None)
                existing_exit = next((order for order in reversed(live_orders) if order.is_exit and order.linked_position_id == position.position_id and not order.terminal_state), None)
                if existing_exit is None:
                    if latest_exit is not None and latest_exit.filled_size > 0:
                        if self._apply_exit_fill_to_position(position, latest_exit, reason):
                            if position.closed:
                                append_csv_row(self.data_dir / "live_trade_history.csv", {**position.model_dump(mode="json"), "status": "CLOSED", "reason_code": reason})
                        continue
                    exit_order = LiveOrder(
                        local_decision_id=stable_event_key(position.position_id, "exit-decision"),
                        local_order_id=stable_event_key(position.position_id, "exit-order"),
                        client_order_id=stable_event_key(position.position_id, "exit-client"),
                        market_id=position.market_id,
                        token_id=position.token_id,
                        side="SELL",
                        intended_price=best_bid,
                        intended_size=position.remaining_size,
                        entry_style=self.config.entry_styles.preferred_live_entry_style,
                        remaining_size=position.remaining_size,
                        is_exit=True,
                        linked_position_id=position.position_id,
                        timeout_at=datetime.now(timezone.utc),
                    )
                    live_orders.append(exit_order)
                    await self.order_manager.submit_order(exit_order, side="SELL")
                    position.exit_order_ids.append(exit_order.local_order_id)
                    position.exit_state = "EXIT_ORDER_OPEN"
                else:
                    await self.order_manager.refresh_order(existing_exit)
                    if existing_exit.filled_size > 0:
                        if self._apply_exit_fill_to_position(position, existing_exit, reason) and position.closed:
                            append_csv_row(self.data_dir / "live_trade_history.csv", {**position.model_dump(mode="json"), "status": "CLOSED", "reason_code": reason})
                        if existing_exit.lifecycle_status == OrderLifecycleStatus.UNKNOWN:
                            self.state.pause(f"Exit ambiguity for {position.position_id}")
                            return
                position.last_reconciled_at = datetime.now(timezone.utc)
            except Exception as exc:
                self.audit.record(
                    "live_exit_orderbook_unavailable",
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "position_id": position.position_id,
                        "market_id": position.market_id,
                        "token_id": position.token_id,
                        "error": str(exc),
                    },
                )
                position.exit_state = "EXIT_DATA_UNAVAILABLE"
                if "404" in str(exc):
                    position.source_exit_following_enabled = False
                    self.audit.record(
                        "live_exit_management_quarantined",
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "position_id": position.position_id,
                            "market_id": position.market_id,
                            "token_id": position.token_id,
                            "reason": "NO_EXIT_ORDERBOOK",
                        },
                    )
                position.last_reconciled_at = datetime.now(timezone.utc)
                continue

    async def _emergency_flatten(self) -> None:
        try:
            await self.client.cancel_all_orders()
        except Exception as exc:
            self.state.pause(f"Emergency flatten cancel-all failed: {exc}")
            return
        self.state.pause("Emergency flatten flag triggered. Manual reconciliation required.")
