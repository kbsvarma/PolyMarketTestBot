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
from src.reconciliation import reconcile_live_state
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

    async def startup_validation(self) -> dict[str, object]:
        checks: dict[str, object] = {}
        client_health = await self.client.health_check()
        checks["auth_valid"] = client_health.ok
        checks["auth_detail"] = client_health.detail

        try:
            balance = await self.client.get_balance()
            checks["balance_visible"] = True
            checks["balance_detail"] = str(balance)
        except Exception as exc:
            checks["balance_visible"] = False
            checks["balance_detail"] = str(exc)

        try:
            allowance = await self.client.get_allowance()
            checks["allowance_sufficient"] = bool(allowance.get("sufficient", False))
            checks["allowance_detail"] = str(allowance)
        except Exception as exc:
            checks["allowance_sufficient"] = False
            checks["allowance_detail"] = str(exc)

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
            checks["positions_detail"] = f"{len(positions)} positions visible"
        except Exception as exc:
            checks["positions_visible"] = False
            checks["positions_detail"] = str(exc)

        try:
            markets = await self.market_data.refresh_markets()
            token_ids = [market.token_id for market in list(markets.values())[:3]]
            await self.market_data.stream_watchlist(token_ids)
            tradability = []
            for market in list(markets.values())[:3]:
                tradability.append(await self.market_data.get_tradability(market.market_id, market.token_id))
            checks["tradability_ok"] = all(bool(item.get("tradable")) for item in tradability) if tradability else False
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
        return checks

    async def refresh_live_status(self) -> None:
        health = await self.collect_health()
        checks = await self.startup_validation()
        readiness = build_readiness_result(self.config, self.state, health, checks)
        self.state.update_system_status(
            live_health_state=health.overall.value,
            live_readiness_last_result=readiness.model_dump(mode="json"),
            heartbeat_ok=not any(component.name == "heartbeat" and component.state != HealthState.HEALTHY for component in health.components),
            balance_visible=bool(checks.get("balance_visible", False)),
            allowance_visible=bool(checks.get("allowance_sufficient", False)),
            allowance_sufficient=bool(checks.get("allowance_sufficient", False)),
            reconciliation_clean=bool(checks.get("reconciliation_clean", False)),
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

    async def collect_health(self):
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

        reconciliation = await self.reconcile()
        components.append(
            HealthComponent(
                name="reconciliation",
                state=HealthState.HEALTHY if reconciliation.clean else HealthState.UNHEALTHY,
                detail=reconciliation.severity,
                metadata=reconciliation.model_dump(mode="json"),
            )
        )
        return self.health_monitor.aggregate(components)

    async def reconcile(self):
        live_positions = [position.model_dump(mode="json") for position in self.positions.positions_for_mode(Mode.LIVE)]
        live_orders = [order.model_dump(mode="json") for order in self.orders.load()]
        try:
            exchange_positions = await self.client.get_positions()
            exchange_orders = await self.client.get_open_orders()
        except Exception as exc:
            summary = reconcile_live_state(live_positions, [], live_orders, [])
            summary.issues.append(ReconciliationIssue(severity="SEVERE", issue_type="EXCHANGE_VISIBILITY", detail=str(exc)))
            summary.clean = False
            summary.severity = "SEVERE"
            return summary
        return reconcile_live_state(live_positions, exchange_positions, live_orders, exchange_orders)

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
        if health.overall != HealthState.HEALTHY:
            self.state.update_system_status(live_health_state=health.overall.value)
            return

        try:
            self.state_machine.transition(SystemStatus.LIVE_ACTIVE, "Managing live orders.")
        except ValueError:
            pass

        paper_positions = self.positions.positions_for_mode(Mode.PAPER)
        live_positions = self.positions.positions_for_mode(Mode.LIVE)
        live_orders = self.orders.load()

        await self._manage_existing_orders(live_orders, live_positions, decisions)
        await self._apply_live_exits(live_positions, live_orders)

        for decision in decisions:
            self.decisions_audit.record("live_decision", {"ts": datetime.now(timezone.utc).isoformat(), "decision": decision.model_dump(mode="json")})
            if decision.action != DecisionAction.LIVE_COPY or not decision.allowed:
                continue
            if decision.category not in self.config.live.selected_categories:
                continue
            if any(order.local_decision_id == decision.local_decision_id and not order.terminal_state for order in live_orders):
                continue

            tradability = await self.market_data.get_tradability(decision.market_id, decision.token_id)
            self.state.update_system_status(watched_market_tradability_cache={decision.market_id: tradability})
            if not (tradability.get("tradable") and tradability.get("orderbook_enabled")):
                self.audit.record("tradability_block", {"ts": datetime.now(timezone.utc).isoformat(), "decision_id": decision.local_decision_id, "tradability": tradability})
                continue

            live_order = self.order_manager.create_entry_order(decision)
            live_orders.append(live_order)
            self.orders.save(live_orders)
            self.audit.record("order_created", {"ts": datetime.now(timezone.utc).isoformat(), "local_order_id": live_order.local_order_id, "client_order_id": live_order.client_order_id})

            try:
                await self.order_manager.submit_order(live_order, side="BUY")
                self.orders.save(live_orders)
            except Exception as exc:
                self.audit.record("order_error", {"ts": datetime.now(timezone.utc).isoformat(), "local_order_id": live_order.local_order_id, "error": str(exc)})
                self.state.pause(f"Live order error: {exc}")
                break

        await self._manage_existing_orders(live_orders, live_positions, decisions)
        reconciliation = await self.reconcile()
        unresolved = [order.local_order_id for order in live_orders if order.lifecycle_status == OrderLifecycleStatus.UNKNOWN and not order.terminal_state]
        self.state.update_system_status(
            reconciliation_clean=reconciliation.clean,
            reconciliation_summary=reconciliation.model_dump(mode="json"),
            unresolved_live_order_ids=unresolved,
        )
        if not reconciliation.clean:
            self.state.pause(f"Live reconciliation mismatch: {reconciliation.severity}")
        self.positions.save_positions(paper_positions, live_positions)
        self.orders.save(live_orders)

    async def _manage_existing_orders(self, live_orders: list[LiveOrder], live_positions: list[Position], decisions: list[TradeDecision]) -> None:
        decision_map = {decision.local_decision_id: decision for decision in decisions}
        for order in live_orders:
            if order.terminal_state:
                continue
            try:
                await self.order_manager.refresh_order(order)
            except Exception as exc:
                order.lifecycle_status = OrderLifecycleStatus.UNKNOWN
                self.state.pause(f"Order status ambiguity for {order.local_order_id}: {exc}")
                return

            if order.lifecycle_status in {OrderLifecycleStatus.PARTIALLY_FILLED, OrderLifecycleStatus.FILLED}:
                decision = decision_map.get(order.local_decision_id)
                if decision is not None:
                    self.order_manager.apply_fill_to_position(order, decision, live_positions)
            if order.lifecycle_status in {OrderLifecycleStatus.SUBMITTED, OrderLifecycleStatus.ACKNOWLEDGED, OrderLifecycleStatus.RESTING}:
                timeout_result = await self.order_manager.handle_timeout(order)
                if timeout_result == "CANCELLED":
                    drift_ok = True
                    tradability = await self.market_data.get_tradability(order.market_id, order.token_id)
                    if await self.order_manager.maybe_reprice(order, bool(tradability.get("tradable")), drift_ok):
                        decision = decision_map.get(order.local_decision_id)
                        if decision is not None:
                            await self.order_manager.submit_order(order, side=order.side)
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
            try:
                orderbook = await self.client.get_orderbook(position.token_id)
                best_bid = orderbook.bids[0].price if orderbook.bids else position.entry_price_actual or position.entry_price
                position.current_mark_price = best_bid
                position.unrealized_pnl = round((best_bid - (position.entry_price_actual or position.entry_price)) * position.remaining_size - position.fees_paid, 4)
                should_exit, reason = evaluate_exit(position, best_bid)
                if not should_exit:
                    continue

                existing_exit = next((order for order in live_orders if order.is_exit and order.linked_position_id == position.position_id and not order.terminal_state), None)
                if existing_exit is None:
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
                        newly_exited = max(existing_exit.filled_size - position.exited_size, 0.0)
                        if newly_exited > 0:
                            position.exited_size += newly_exited
                            position.remaining_size = max(position.entry_size - position.exited_size, 0.0)
                            if position.remaining_size <= 0:
                                position.closed = True
                                position.exit_reason = reason
                                position.exit_state = "CLOSED"
                                position.realized_pnl = position.unrealized_pnl
                                append_csv_row(self.data_dir / "live_trade_history.csv", {**position.model_dump(mode="json"), "status": "CLOSED", "reason_code": reason})
                            else:
                                position.exit_state = "PARTIAL_EXIT"
                        if existing_exit.lifecycle_status == OrderLifecycleStatus.UNKNOWN:
                            self.state.pause(f"Exit ambiguity for {position.position_id}")
                            return
                position.last_reconciled_at = datetime.now(timezone.utc)
            except Exception as exc:
                self.state.pause(f"Live exit ambiguity for {position.position_id}: {exc}")
                return

    async def _emergency_flatten(self) -> None:
        try:
            await self.client.cancel_all_orders()
        except Exception as exc:
            self.state.pause(f"Emergency flatten cancel-all failed: {exc}")
            return
        self.state.pause("Emergency flatten flag triggered. Manual reconciliation required.")
