from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.config import AppConfig
from src.exits import evaluate_exit
from src.models import DecisionAction, Mode, Position, TradeDecision
from src.positions import PositionStore
from src.utils import append_csv_row, stable_event_key


class PaperTradingEngine:
    def __init__(self, config: AppConfig, data_dir: Path, state_store: object) -> None:
        self.config = config
        self.data_dir = data_dir
        self.state_store = state_store
        self.positions = PositionStore(data_dir / "positions.json")

    def handle_decisions(self, decisions: list[TradeDecision]) -> None:
        paper_positions = self.positions.positions_for_mode(Mode.PAPER)
        live_positions = self.positions.positions_for_mode(Mode.LIVE)

        self._apply_exit_logic(paper_positions)

        for decision in decisions:
            if decision.action != DecisionAction.PAPER_COPY or not decision.allowed:
                append_csv_row(
                    self.data_dir / "paper_trade_history.csv",
                    {
                        "position_id": "",
                        "market_id": decision.market_id,
                        "token_id": decision.token_id,
                        "wallet_address": decision.wallet_address,
                        "category": decision.category,
                        "entry_style": decision.entry_style.value,
                        "entry_price": decision.executable_price,
                        "quantity": 0.0,
                        "notional": decision.scaled_notional,
                        "fees_paid": 0.0,
                        "realized_pnl": 0.0,
                        "unrealized_pnl": 0.0,
                        "entry_reason": decision.human_readable_reason,
                        "exit_reason": "",
                        "cluster_confirmed": decision.cluster_confirmed,
                        "hedge_suspicion_score": decision.hedge_suspicion_score,
                        "opened_at": datetime.now(timezone.utc).isoformat(),
                        "closed": True,
                        "status": "SKIPPED",
                        "reason_code": decision.reason_code,
                    },
                )
                continue

            quantity = decision.scaled_notional / max(decision.executable_price, 1e-6)
            fee_rate = 0.01 if decision.entry_style.value == "FOLLOW_TAKER" else 0.005
            position = Position(
                position_id=stable_event_key(decision.wallet_address, decision.market_id, decision.token_id, decision.executable_price),
                mode=Mode.PAPER,
                wallet_address=decision.wallet_address,
                market_id=decision.market_id,
                token_id=decision.token_id,
                category=decision.category,
                entry_style=decision.entry_style,
                entry_price=decision.executable_price,
                current_mark_price=decision.executable_price,
                quantity=round(quantity, 6),
                notional=decision.scaled_notional,
                fees_paid=round(decision.scaled_notional * fee_rate, 4),
                source_trade_timestamp=datetime.now(timezone.utc),
                entry_reason=decision.human_readable_reason,
                cluster_confirmed=decision.cluster_confirmed,
                hedge_suspicion_score=decision.hedge_suspicion_score,
            )
            paper_positions.append(position)
            append_csv_row(
                self.data_dir / "paper_trade_history.csv",
                {
                    **position.model_dump(mode="json"),
                    "status": "OPENED",
                    "reason_code": decision.reason_code,
                },
            )
        self.positions.save_positions(paper_positions, live_positions)

    def _apply_exit_logic(self, paper_positions: list[Position]) -> None:
        for position in paper_positions:
            if position.closed:
                continue
            mark_price = round(min(position.entry_price * 1.08, position.entry_price + 0.03), 6)
            position.current_mark_price = mark_price
            position.unrealized_pnl = round((mark_price - position.entry_price) * position.quantity - position.fees_paid, 4)
            should_exit, reason = evaluate_exit(position, mark_price)
            if should_exit:
                position.closed = True
                position.exit_reason = reason
                position.realized_pnl = position.unrealized_pnl
                append_csv_row(
                    self.data_dir / "paper_trade_history.csv",
                    {
                        **position.model_dump(mode="json"),
                        "status": "CLOSED",
                        "reason_code": reason,
                    },
                )
