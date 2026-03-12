from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.clustering import cluster_detections
from src.config import AppConfig
from src.market_data import MarketDataService
from src.models import ApprovedWallets, DecisionAction, DetectionEvent, EntryStyle, TradeDecision, WalletMetrics
from src.orderbook import estimate_fill
from src.positions import PositionStore
from src.risk_manager import RiskManager
from src.state import AppStateStore
from src.utils import clamp, stable_event_key


class StrategyEngine:
    def __init__(self, config: AppConfig, data_dir: Path, state: AppStateStore) -> None:
        self.config = config
        self.data_dir = data_dir
        self.state = state
        self.market_data = MarketDataService(config, data_dir)
        self.risk = RiskManager(config)
        self.positions = PositionStore(data_dir / "positions.json")

    def _wallet_map(self, wallets: list[WalletMetrics]) -> dict[str, WalletMetrics]:
        return {wallet.wallet_address: wallet for wallet in wallets}

    async def process_detections(
        self,
        detections: list[DetectionEvent],
        approved_wallets: ApprovedWallets,
        wallets: list[WalletMetrics],
    ) -> list[TradeDecision]:
        wallet_map = self._wallet_map(wallets)
        clusters = cluster_detections(self.config, detections, self.data_dir)
        cluster_lookup = {(cluster.market_id, cluster.token_id): cluster for cluster in clusters}
        decisions: list[TradeDecision] = []

        stored_positions = self.positions.load()
        active_positions = stored_positions.get("paper", []) + stored_positions.get("live", [])
        active_market_exposure: dict[str, float] = {}
        active_wallet_exposure: dict[str, float] = {}
        total_exposure = 0.0
        for position in active_positions:
            if position.get("closed"):
                continue
            notional = float(position.get("notional", 0.0))
            market_id = str(position.get("market_id", ""))
            wallet_address = str(position.get("wallet_address", ""))
            active_market_exposure[market_id] = active_market_exposure.get(market_id, 0.0) + notional
            active_wallet_exposure[wallet_address] = active_wallet_exposure.get(wallet_address, 0.0) + notional
            total_exposure += notional

        await self.market_data.refresh_markets()
        ws_snapshots = await self.market_data.stream_watchlist(sorted({detection.token_id for detection in detections}))

        for detection in detections:
            wallet = wallet_map.get(detection.wallet_address)
            if not wallet:
                continue
            market_meta = await self.market_data.fetch_market_metadata(detection.market_id)
            tradability = await self.market_data.get_tradability(detection.market_id, detection.token_id)
            orderbook = await self.market_data.fetch_orderbook(detection.token_id)
            cluster = cluster_lookup.get((detection.market_id, detection.token_id))
            cluster_confirmed = cluster is not None
            copy_fraction = self._select_copy_fraction(wallet)
            scaled_notional = min(detection.notional * copy_fraction, self._max_notional_for_mode())

            best_decision = self._skip_decision(detection, wallet, scaled_notional, "NO_VALID_ENTRY_STYLE", "No entry style passed all checks.")
            for entry_style in self.config.entry_styles.compare:
                fill = estimate_fill(orderbook, scaled_notional, self.config.risk.max_slippage_pct)
                drift_pct = abs(fill.executable_price - detection.price) / max(detection.price, 1e-6)
                hybrid_modifier = self._hybrid_confirmation_modifier(detection, ws_snapshots.get(detection.token_id, {}))
                entry_style_allowed = self._entry_style_allowed(entry_style, cluster_confirmed)
                state_snapshot = self.state.read()
                risk = self.risk.evaluate(
                    detection=detection,
                    wallet=wallet,
                    fill=fill,
                    mode=self.config.mode.value,
                    total_exposure=total_exposure,
                    market_exposure=active_market_exposure.get(detection.market_id, 0.0),
                    wallet_exposure=active_wallet_exposure.get(detection.wallet_address, 0.0),
                    daily_pnl=self.state.read().get("daily_pnl", 0.0),
                    cluster_confirmed=cluster_confirmed,
                    infra_ok=bool(market_meta.get("active", True)) and not state_snapshot.get("paused", False),
                    entry_style_allowed=entry_style_allowed,
                    category=detection.category,
                    market_id=detection.market_id,
                    has_conflicting_position=self._has_conflicting_position(active_positions, detection),
                    data_age_seconds=(datetime.now(timezone.utc) - orderbook.timestamp).total_seconds(),
                    entry_drift_pct=drift_pct,
                    live_ready=bool(state_snapshot.get("live_readiness_last_result", {}).get("ready", True)),
                    kill_switch=bool(state_snapshot.get("kill_switch", False)),
                    manual_live_enable=bool(state_snapshot.get("manual_live_enable", True)),
                    manual_resume_required=bool(state_snapshot.get("manual_resume_required", False)),
                    health_state=str(state_snapshot.get("live_health_state", "HEALTHY")),
                    reconciliation_clean=bool(state_snapshot.get("reconciliation_clean", True)),
                    heartbeat_ok=bool(state_snapshot.get("heartbeat_ok", True)),
                    balance_visible=bool(state_snapshot.get("balance_visible", True)),
                    allowance_sufficient=bool(state_snapshot.get("allowance_sufficient", True)),
                    tradable=bool(tradability.get("tradable")) and bool(tradability.get("orderbook_enabled")),
                )
                if drift_pct > self.config.risk.max_entry_drift_pct:
                    risk = risk.model_copy(
                        update={
                            "allowed": False,
                            "reason_code": "ENTRY_DRIFT",
                            "human_readable_reason": "Executable entry drift exceeds threshold.",
                            "context": {**risk.context, "entry_drift_pct": round(drift_pct, 6)},
                        }
                    )
                if hybrid_modifier < -0.15:
                    risk = risk.model_copy(
                        update={
                            "allowed": False,
                            "reason_code": "HYBRID_CONFIRMATION",
                            "human_readable_reason": "Optional hybrid confirmation weakened the signal too much.",
                            "context": {**risk.context, "hybrid_modifier": hybrid_modifier},
                        }
                    )

                action = DecisionAction.SKIP
                if risk.allowed and detection.wallet_address in approved_wallets.paper_wallets:
                    action = DecisionAction.PAPER_COPY if self.config.mode.value != "LIVE" else DecisionAction.LIVE_COPY
                decision = TradeDecision(
                    allowed=risk.allowed,
                    action=action,
                    reason_code=risk.reason_code,
                    human_readable_reason=risk.human_readable_reason,
                    local_decision_id=stable_event_key(detection.event_key, entry_style.value, datetime.now(timezone.utc).isoformat()),
                    wallet_address=detection.wallet_address,
                    market_id=detection.market_id,
                    token_id=detection.token_id,
                    entry_style=entry_style,
                    category=detection.category,
                    scaled_notional=round(scaled_notional, 4),
                    source_price=detection.price,
                    executable_price=fill.executable_price,
                    cluster_confirmed=cluster_confirmed,
                    hedge_suspicion_score=wallet.hedge_suspicion_score,
                    context={
                        "fill": fill.model_dump(),
                        "risk_context": risk.context,
                        "market_meta": market_meta,
                        "tradability": tradability,
                        "cluster_strength": cluster.cluster_strength if cluster else 0.0,
                        "hybrid_modifier": hybrid_modifier,
                        "ws_snapshot": ws_snapshots.get(detection.token_id, {}),
                    },
                )
                if self._decision_rank(decision, wallet, hybrid_modifier) > self._decision_rank(best_decision, wallet, 0.0):
                    best_decision = decision

            decisions.append(best_decision)
        return decisions

    def _select_copy_fraction(self, wallet: WalletMetrics) -> float:
        base = self.config.risk.copy_fraction_min + (self.config.risk.copy_fraction_max - self.config.risk.copy_fraction_min) * wallet.copyability_score
        return clamp(base, self.config.risk.copy_fraction_min, self.config.risk.copy_fraction_max)

    def _max_notional_for_mode(self) -> float:
        if self.config.mode.value == "LIVE":
            return self.config.risk.max_single_live_trade_usd
        return self.config.bankroll.paper_starting_bankroll * self.config.risk.max_market_exposure_pct

    def _entry_style_allowed(self, entry_style: EntryStyle, cluster_confirmed: bool) -> bool:
        if self.config.mode.value == "LIVE" and self.config.live.only_cluster_confirmed and not cluster_confirmed:
            return False
        if self.config.mode.value == "LIVE" and entry_style != self.config.entry_styles.preferred_live_entry_style:
            return False
        return True

    def _decision_rank(self, decision: TradeDecision, wallet: WalletMetrics, hybrid_modifier: float) -> float:
        if not decision.allowed:
            return -100.0
        fill = decision.context.get("fill", {})
        slippage = float(fill.get("slippage_pct", 1.0))
        return (
            wallet.global_score
            + (0.15 if decision.cluster_confirmed else 0.0)
            + hybrid_modifier
            - slippage
            - wallet.hedge_suspicion_score * 0.4
        )

    def _skip_decision(self, detection: DetectionEvent, wallet: WalletMetrics, scaled_notional: float, reason_code: str, reason: str) -> TradeDecision:
        return TradeDecision(
            allowed=False,
            action=DecisionAction.SKIP,
            reason_code=reason_code,
            human_readable_reason=reason,
            local_decision_id=stable_event_key(detection.event_key, reason_code),
            wallet_address=detection.wallet_address,
            market_id=detection.market_id,
            token_id=detection.token_id,
            entry_style=EntryStyle.FOLLOW_TAKER,
            category=detection.category,
            scaled_notional=scaled_notional,
            source_price=detection.price,
            executable_price=detection.price,
            cluster_confirmed=False,
            hedge_suspicion_score=wallet.hedge_suspicion_score,
            context={},
        )

    def _has_conflicting_position(self, active_positions: list[dict], detection: DetectionEvent) -> bool:
        for position in active_positions:
            if position.get("closed"):
                continue
            if position.get("market_id") == detection.market_id and position.get("wallet_address") == detection.wallet_address:
                return True
        return False

    def _hybrid_confirmation_modifier(self, detection: DetectionEvent, ws_snapshot: dict) -> float:
        if detection.category == "crypto price":
            best_bid = detection.best_bid or detection.price
            best_ask = detection.best_ask or detection.price
            tightness = 1.0 - ((best_ask - best_bid) / max(best_ask, 1e-6))
            ws_liquidity_signal = 0.05 if ws_snapshot else 0.0
            return round(clamp((tightness - 0.95) * 2.0 + ws_liquidity_signal, -0.2, 0.15), 4)
        return 0.0
