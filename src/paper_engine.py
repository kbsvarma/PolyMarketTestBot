from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import AppConfig
from src.exits import evaluate_exit
from src.market_data import MarketDataService
from src.logger import logger
from src.models import DecisionAction, Mode, Position, TradeDecision
from src.positions import PositionStore
from src.state import AppStateStore
from src.utils import append_csv_row, append_jsonl, stable_event_key


class PaperTradingEngine:
    def __init__(self, config: AppConfig, data_dir: Path, state_store: AppStateStore) -> None:
        self.config = config
        self.data_dir = data_dir
        self.state_store = state_store
        self.positions = PositionStore(data_dir / "positions.json")
        self.market_data = MarketDataService(config, data_dir)
        self.audit_path = data_dir / "paper_audit.jsonl"

    def _entries_last_hour(self, paper_positions: list[Position]) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        return len(
            [
                position
                for position in paper_positions
                if (position.entry_time or position.opened_at) >= cutoff
            ]
        )

    async def handle_decisions(self, decisions: list[TradeDecision]) -> None:
        paper_positions = self.positions.positions_for_mode(Mode.PAPER)
        live_positions = self.positions.positions_for_mode(Mode.LIVE)
        opened_count = 0
        skipped_count = 0
        entries_last_hour = self._entries_last_hour(paper_positions)

        await self._mark_positions_to_market(paper_positions)
        self._apply_exit_logic(paper_positions)

        for decision in decisions:
            effective_decision = decision
            if decision.action == DecisionAction.PAPER_COPY and decision.allowed and entries_last_hour >= self.config.risk.max_new_entries_per_hour:
                effective_decision = decision.model_copy(
                    update={
                        "allowed": False,
                        "action": DecisionAction.SKIP,
                        "reason_code": "ENTRY_RATE_LIMIT",
                        "human_readable_reason": "Hourly entry limit reached during paper execution.",
                    }
                )
            if effective_decision.action != DecisionAction.PAPER_COPY or not effective_decision.allowed:
                skipped_count += 1
                append_jsonl(
                    self.audit_path,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "event": "paper_skipped",
                        "market_id": effective_decision.market_id,
                        "token_id": effective_decision.token_id,
                        "strategy_name": effective_decision.strategy_name,
                        "wallet_address": effective_decision.wallet_address,
                        "reason_code": effective_decision.reason_code,
                        "reason": effective_decision.human_readable_reason,
                        "notional": effective_decision.scaled_notional,
                        "source_quality": effective_decision.context.get("source_quality", ""),
                        "trust_level": effective_decision.context.get("trust_level", ""),
                        "fallback_used": effective_decision.context.get("fallback_used", False),
                        "counts_as_trustworthy_approval": effective_decision.context.get("counts_as_trustworthy_approval", False),
                        "discovery_state": effective_decision.context.get("discovery_state", ""),
                        "scoring_state": effective_decision.context.get("scoring_state", ""),
                        "style_evaluations": effective_decision.context.get("style_evaluations", []),
                    },
                )
                append_csv_row(
                    self.data_dir / "paper_trade_history.csv",
                    {
                        "position_id": "",
                        "market_id": effective_decision.market_id,
                        "token_id": effective_decision.token_id,
                        "wallet_address": effective_decision.wallet_address,
                        "strategy_name": effective_decision.strategy_name,
                        "category": effective_decision.category,
                        "entry_style": effective_decision.entry_style.value,
                        "entry_price": effective_decision.executable_price,
                        "quantity": 0.0,
                        "notional": effective_decision.scaled_notional,
                        "fees_paid": 0.0,
                        "realized_pnl": 0.0,
                        "unrealized_pnl": 0.0,
                        "entry_reason": effective_decision.human_readable_reason,
                        "exit_reason": "",
                        "cluster_confirmed": effective_decision.cluster_confirmed,
                        "hedge_suspicion_score": effective_decision.hedge_suspicion_score,
                        "opened_at": datetime.now(timezone.utc).isoformat(),
                        "closed": True,
                        "status": "SKIPPED",
                        "reason_code": effective_decision.reason_code,
                    },
                )
                continue

            quantity = effective_decision.scaled_notional / max(effective_decision.executable_price, 1e-6)
            fee_rate = 0.01 if effective_decision.entry_style.value == "FOLLOW_TAKER" else 0.005
            position = Position(
                position_id=stable_event_key(effective_decision.wallet_address, effective_decision.market_id, effective_decision.token_id, effective_decision.executable_price),
                mode=Mode.PAPER,
                strategy_name=effective_decision.strategy_name,
                wallet_address=effective_decision.wallet_address,
                market_id=effective_decision.market_id,
                token_id=effective_decision.token_id,
                category=effective_decision.category,
                entry_style=effective_decision.entry_style,
                entry_price=effective_decision.executable_price,
                current_mark_price=effective_decision.executable_price,
                quantity=round(quantity, 6),
                notional=effective_decision.scaled_notional,
                fees_paid=round(effective_decision.scaled_notional * fee_rate, 4),
                source_trade_timestamp=datetime.now(timezone.utc),
                entry_reason=effective_decision.human_readable_reason,
                cluster_confirmed=effective_decision.cluster_confirmed,
                hedge_suspicion_score=effective_decision.hedge_suspicion_score,
                source_wallet=effective_decision.wallet_address,
                entry_time=datetime.now(timezone.utc),
                entry_size=round(quantity, 6),
                remaining_size=round(quantity, 6),
                thesis_type=effective_decision.thesis_type,
                bundle_id=effective_decision.bundle_id,
                bundle_role=effective_decision.bundle_role,
                paired_token_id=effective_decision.paired_token_id,
                entry_price_estimated=effective_decision.executable_price,
                entry_price_actual=effective_decision.executable_price,
                stop_loss_rule="hard_stop",
                take_profit_rule="take_profit",
                time_stop_rule="time_stop",
                peak_mark_price=effective_decision.executable_price,
                peak_pnl_pct=0.0,
                peak_mark_seen_at=datetime.now(timezone.utc),
            )
            paper_positions.append(position)
            opened_count += 1
            entries_last_hour += 1
            append_jsonl(
                self.audit_path,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "paper_opened",
                    "position_id": position.position_id,
                    "market_id": position.market_id,
                    "token_id": position.token_id,
                    "strategy_name": position.strategy_name,
                    "wallet_address": position.wallet_address,
                    "entry_price": position.entry_price,
                    "quantity": position.quantity,
                    "notional": position.notional,
                    "entry_style": position.entry_style.value,
                    "reason_code": effective_decision.reason_code,
                    "source_quality": effective_decision.context.get("source_quality", ""),
                    "trust_level": effective_decision.context.get("trust_level", ""),
                    "fallback_used": effective_decision.context.get("fallback_used", False),
                    "counts_as_trustworthy_approval": effective_decision.context.get("counts_as_trustworthy_approval", False),
                    "style_evaluations": effective_decision.context.get("style_evaluations", []),
                },
            )
            append_csv_row(
                self.data_dir / "paper_trade_history.csv",
                {
                    **position.model_dump(mode="json"),
                    "status": "OPENED",
                    "reason_code": effective_decision.reason_code,
                },
            )

        self.positions.save_positions(paper_positions, live_positions)
        self._update_paper_summary(paper_positions)
        logger.info(
            "Paper cycle decisions={} opened={} skipped={} open_positions={}",
            len(decisions),
            opened_count,
            skipped_count,
            len([position for position in paper_positions if not position.closed]),
        )

    async def _mark_positions_to_market(self, paper_positions: list[Position]) -> None:
        for position in paper_positions:
            if position.closed:
                continue
            try:
                orderbook = await self.market_data.fetch_orderbook(position.token_id)
                best_bid = orderbook.bids[0].price if orderbook.bids else position.current_mark_price
                best_ask = orderbook.asks[0].price if orderbook.asks else position.current_mark_price
                if orderbook.bids and orderbook.asks:
                    mark_price = round((best_bid + best_ask) / 2, 6)
                else:
                    mark_price = round(best_bid or best_ask or position.current_mark_price, 6)
                position.current_mark_price = mark_price
                position.unrealized_pnl = round((mark_price - position.entry_price) * position.quantity - position.fees_paid, 4)
            except Exception:
                continue

    def _apply_exit_logic(self, paper_positions: list[Position]) -> None:
        for position in paper_positions:
            if position.closed:
                continue
            should_exit, reason = evaluate_exit(
                position,
                position.current_mark_price,
                hold_to_resolution=position.thesis_type == "paired_arb",
                profit_arm_pct=self.config.live.adaptive_profit_arm_pct,
                min_profit_lock_pct=self.config.live.adaptive_profit_min_lock_pct,
                trailing_profit_retrace_pct=self.config.live.trailing_profit_retrace_pct,
                strong_winner_profit_pct=self.config.live.strong_winner_profit_pct,
                strong_winner_retrace_pct=self.config.live.strong_winner_retrace_pct,
                paired_arb_time_stop_hours=self.config.live.paired_arb_time_stop_hours,
            )
            if should_exit:
                position.closed = True
                position.exit_reason = reason
                position.closed_at = datetime.now(timezone.utc)
                position.realized_pnl = position.unrealized_pnl
                position.remaining_size = 0.0
                append_jsonl(
                    self.audit_path,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "event": "paper_closed",
                        "position_id": position.position_id,
                        "market_id": position.market_id,
                        "token_id": position.token_id,
                        "exit_reason": reason,
                        "realized_pnl": position.realized_pnl,
                    },
                )
                append_csv_row(
                    self.data_dir / "paper_trade_history.csv",
                    {
                        **position.model_dump(mode="json"),
                        "status": "CLOSED",
                        "reason_code": reason,
                    },
                )

    def _update_paper_summary(self, paper_positions: list[Position]) -> None:
        now = datetime.now(timezone.utc)
        bankroll_override = float(self.state_store.read().get("paper_bankroll_override", 0.0) or 0.0)
        bankroll = bankroll_override if bankroll_override > 0 else self.config.bankroll.paper_starting_bankroll
        trade_override = float(self.state_store.read().get("paper_trade_notional_override", 0.0) or 0.0)

        open_positions = [position for position in paper_positions if not position.closed]
        closed_positions = [position for position in paper_positions if position.closed]

        def window_net(hours: int) -> float:
            cutoff = now - timedelta(hours=hours)
            realized = sum(
                position.realized_pnl
                for position in closed_positions
                if position.closed_at and position.closed_at >= cutoff
            )
            unrealized = sum(
                position.unrealized_pnl
                for position in open_positions
                if position.opened_at >= cutoff
            )
            return round(realized + unrealized, 4)

        summary = {
            "paper_bankroll": bankroll,
            "paper_trade_notional_override": trade_override,
            "open_positions": len(open_positions),
            "open_notional": round(sum(position.notional for position in open_positions), 4),
            "realized_pnl_total": round(sum(position.realized_pnl for position in closed_positions), 4),
            "unrealized_pnl_total": round(sum(position.unrealized_pnl for position in open_positions), 4),
            "net_pnl_total": round(sum(position.realized_pnl for position in closed_positions) + sum(position.unrealized_pnl for position in open_positions), 4),
            "last_hour_net_pnl": window_net(1),
            "last_24h_net_pnl": window_net(24),
            "last_7d_net_pnl": window_net(24 * 7),
            "updated_at": now.isoformat(),
        }
        self.state_store.update_system_status(
            paper_summary=summary,
            daily_pnl=summary["last_24h_net_pnl"],
        )
