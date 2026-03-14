from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.clustering import cluster_detections
from src.config import AppConfig
from src.market_data import MarketDataService
from src.models import ApprovedWallets, DecisionAction, DetectionEvent, EntryStyle, SourceQuality, TradeDecision, WalletMetrics
from src.orderbook import estimate_fill
from src.paper_quality import classify_trust_level, counts_as_trustworthy_approval
from src.positions import PositionStore
from src.risk_manager import RiskManager
from src.state import AppStateStore
from src.source_quality import quality_rank
from src.utils import append_jsonl, clamp, stable_event_key


class StrategyEngine:
    def __init__(self, config: AppConfig, data_dir: Path, state: AppStateStore) -> None:
        self.config = config
        self.data_dir = data_dir
        self.state = state
        self.market_data = MarketDataService(config, data_dir)
        self.risk = RiskManager(config)
        self.positions = PositionStore(data_dir / "positions.json")
        self.trace_path = data_dir / "paper_decision_trace.jsonl"

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
        valid_watchlist_token_ids = sorted(
            {
                token_id
                for token_id in (str(detection.token_id or "").strip() for detection in detections)
                if token_id
            }
        )
        ws_snapshots = await self.market_data.stream_watchlist(valid_watchlist_token_ids) if valid_watchlist_token_ids else {}
        effective_paper_wallets = approved_wallets.paper_wallets or approved_wallets.research_wallets[: self.config.wallet_selection.approved_paper_wallets]
        effective_live_wallets = approved_wallets.live_wallets or effective_paper_wallets[:1]

        for detection in detections:
            wallet = wallet_map.get(detection.wallet_address)
            if not wallet:
                continue
            state_snapshot = self.state.read()
            decision_category = detection.category if detection.category != "unknown" else wallet.dominant_category
            discovery_state = str(state_snapshot.get("wallet_discovery_state", "UNKNOWN"))
            scoring_state = str(state_snapshot.get("wallet_scoring_state", "UNKNOWN"))
            decision_source_quality = min([wallet.source_quality, detection.source_quality], key=quality_rank)
            if not str(detection.token_id or "").strip():
                skip = self._skip_decision(
                    detection,
                    wallet,
                    min(detection.notional * self._select_copy_fraction(wallet), self._max_notional_for_mode()),
                    "MISSING_TOKEN_ID",
                    "Signal skipped because the source trade did not include a usable token_id.",
                )
                skip.context.update(
                    {
                        "discovery_state": discovery_state,
                        "scoring_state": scoring_state,
                        "source_quality": decision_source_quality.value,
                        "trust_level": classify_trust_level(
                            source_quality=decision_source_quality.value,
                            discovery_state=discovery_state,
                            scoring_state=scoring_state,
                            fallback_in_use=decision_source_quality == SourceQuality.SYNTHETIC_FALLBACK,
                        ),
                        "fallback_used": decision_source_quality == SourceQuality.SYNTHETIC_FALLBACK,
                        "counts_as_trustworthy_approval": False,
                    }
                )
                self._write_decision_trace(
                    detection,
                    skip,
                    False,
                    [],
                    discovery_state,
                    scoring_state,
                    decision_source_quality,
                )
                decisions.append(skip)
                continue
            try:
                market_meta = await self.market_data.fetch_market_metadata(detection.market_id, detection.token_id)
            except RuntimeError as exc:
                if str(detection.token_id or "").strip() and detection.token_id != "unknown-token":
                    market_meta = {
                        "market_id": detection.market_id,
                        "token_id": detection.token_id,
                        "title": detection.market_title,
                        "slug": detection.market_slug,
                        "category": decision_category,
                        "active": True,
                        "closed": False,
                        "metadata_fallback": True,
                        "market_meta_error": str(exc),
                    }
                else:
                    skip = self._skip_decision(
                        detection,
                        wallet,
                        min(detection.notional * self._select_copy_fraction(wallet), self._max_notional_for_mode()),
                        "MISSING_MARKET_METADATA",
                        f"Live market metadata unavailable for {detection.market_id}; signal skipped conservatively.",
                    )
                    skip.context.update(
                        {
                            "market_meta_error": str(exc),
                            "market_meta": {"market_id": detection.market_id, "title": detection.market_title, "slug": detection.market_slug},
                            "tradability": {"market_id": detection.market_id, "token_id": detection.token_id, "tradable": False, "orderbook_enabled": False},
                            "discovery_state": discovery_state,
                            "scoring_state": scoring_state,
                            "source_quality": decision_source_quality.value,
                            "trust_level": classify_trust_level(
                                source_quality=decision_source_quality.value,
                                discovery_state=discovery_state,
                                scoring_state=scoring_state,
                                fallback_in_use=decision_source_quality == SourceQuality.SYNTHETIC_FALLBACK,
                            ),
                            "fallback_used": decision_source_quality == SourceQuality.SYNTHETIC_FALLBACK,
                            "counts_as_trustworthy_approval": False,
                        }
                    )
                    self._write_decision_trace(
                        detection,
                        skip,
                        False,
                        [],
                        discovery_state,
                        scoring_state,
                        decision_source_quality,
                    )
                    decisions.append(skip)
                    continue
            try:
                tradability = await self.market_data.get_tradability(detection.market_id, detection.token_id)
            except RuntimeError as exc:
                skip = self._skip_decision(
                    detection,
                    wallet,
                    min(detection.notional * self._select_copy_fraction(wallet), self._max_notional_for_mode()),
                    "MISSING_TRADABILITY",
                    f"Live tradability unavailable for {detection.market_id}/{detection.token_id}; signal skipped conservatively.",
                )
                skip.context.update(
                    {
                        "market_meta": market_meta,
                        "tradability_error": str(exc),
                        "tradability": {
                            "market_id": detection.market_id,
                            "token_id": detection.token_id,
                            "tradable": False,
                            "orderbook_enabled": False,
                        },
                        "discovery_state": discovery_state,
                        "scoring_state": scoring_state,
                        "source_quality": decision_source_quality.value,
                        "trust_level": classify_trust_level(
                            source_quality=decision_source_quality.value,
                            discovery_state=discovery_state,
                            scoring_state=scoring_state,
                            fallback_in_use=decision_source_quality == SourceQuality.SYNTHETIC_FALLBACK,
                        ),
                        "fallback_used": decision_source_quality == SourceQuality.SYNTHETIC_FALLBACK,
                        "counts_as_trustworthy_approval": False,
                    }
                )
                self._write_decision_trace(
                    detection,
                    skip,
                    False,
                    [],
                    discovery_state,
                    scoring_state,
                    decision_source_quality,
                )
                decisions.append(skip)
                continue
            try:
                orderbook = await self.market_data.fetch_orderbook(detection.token_id)
            except RuntimeError as exc:
                skip = self._skip_decision(
                    detection,
                    wallet,
                    min(detection.notional * self._select_copy_fraction(wallet), self._max_notional_for_mode()),
                    "ORDERBOOK_UNAVAILABLE",
                    f"Live orderbook unavailable for token {detection.token_id}; signal skipped conservatively.",
                )
                skip.context.update(
                    {
                        "market_meta": market_meta,
                        "tradability": tradability,
                        "orderbook_error": str(exc),
                        "discovery_state": discovery_state,
                        "scoring_state": scoring_state,
                        "source_quality": decision_source_quality.value,
                        "trust_level": classify_trust_level(
                            source_quality=decision_source_quality.value,
                            discovery_state=discovery_state,
                            scoring_state=scoring_state,
                            fallback_in_use=decision_source_quality == SourceQuality.SYNTHETIC_FALLBACK,
                        ),
                        "fallback_used": decision_source_quality == SourceQuality.SYNTHETIC_FALLBACK,
                        "counts_as_trustworthy_approval": False,
                    }
                )
                self._write_decision_trace(
                    detection,
                    skip,
                    False,
                    [],
                    discovery_state,
                    scoring_state,
                    decision_source_quality,
                )
                decisions.append(skip)
                continue
            cluster = cluster_lookup.get((detection.market_id, detection.token_id))
            cluster_confirmed = cluster is not None
            copy_fraction = self._select_copy_fraction(wallet)
            scaled_notional = min(detection.notional * copy_fraction, self._max_notional_for_mode())

            best_decision = self._skip_decision(detection, wallet, scaled_notional, "NO_VALID_ENTRY_STYLE", "No entry style passed all checks.")
            style_evaluations: list[dict[str, object]] = []
            for entry_style in self.config.entry_styles.compare:
                fill = estimate_fill(orderbook, scaled_notional, self.config.risk.max_slippage_pct)
                drift_pct = abs(fill.executable_price - detection.price) / max(detection.price, 1e-6)
                hybrid_modifier = self._hybrid_confirmation_modifier(detection, ws_snapshots.get(detection.token_id, {}))
                entry_style_allowed = self._entry_style_allowed(entry_style, cluster_confirmed)
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
                    bankroll_override=self._paper_bankroll() if self.config.mode.value != "LIVE" else None,
                )
                if self.config.mode.value != "LIVE" and decision_source_quality == SourceQuality.SYNTHETIC_FALLBACK:
                    risk = risk.model_copy(
                        update={
                            "allowed": False,
                            "reason_code": "SYNTHETIC_SOURCE_BLOCKED",
                            "human_readable_reason": "Synthetic fallback data is blocked from trustable paper approvals.",
                            "context": {**risk.context, "source_quality": decision_source_quality.value},
                        }
                    )
                elif self.config.mode.value != "LIVE" and decision_source_quality == SourceQuality.DEGRADED_PUBLIC_DATA and risk.allowed:
                    risk = risk.model_copy(
                        update={
                            "allowed": False,
                            "reason_code": "DEGRADED_SOURCE_SHADOW",
                            "human_readable_reason": "Degraded public data keeps paper in shadow mode for this signal.",
                            "context": {**risk.context, "source_quality": decision_source_quality.value},
                        }
                    )
                if self.config.mode.value != "LIVE" and scoring_state == "EMPTY":
                    risk = risk.model_copy(
                        update={
                            "allowed": False,
                            "reason_code": "NO_APPROVED_SCORED_WALLETS",
                            "human_readable_reason": "Scoring is empty, so paper entries are blocked from trusted approval.",
                            "context": {**risk.context, "scoring_state": scoring_state},
                        }
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
                style_evaluations.append(
                    {
                        "entry_style": entry_style.value,
                        "allowed": risk.allowed,
                        "reason_code": risk.reason_code,
                        "reason": risk.human_readable_reason,
                        "source_quality": decision_source_quality.value,
                        "executable_price": fill.executable_price,
                        "spread_pct": fill.spread_pct,
                        "slippage_pct": fill.slippage_pct,
                        "entry_drift_pct": round(drift_pct, 6),
                    }
                )

                action = DecisionAction.SKIP
                if risk.allowed:
                    if self.config.mode.value == "LIVE" and detection.wallet_address in effective_live_wallets:
                        action = DecisionAction.LIVE_COPY
                    elif self.config.mode.value != "LIVE" and detection.wallet_address in effective_paper_wallets:
                        action = DecisionAction.PAPER_COPY
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
                    category=decision_category,
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
                        "style_evaluations": style_evaluations,
                        "discovery_state": discovery_state,
                        "scoring_state": scoring_state,
                        "source_quality": decision_source_quality.value,
                    },
                )
                if self._decision_rank(decision, wallet, hybrid_modifier) > self._decision_rank(best_decision, wallet, 0.0):
                    best_decision = decision

            if (
                self.config.mode.value != "LIVE"
                and detection.wallet_address in effective_paper_wallets
                and not best_decision.allowed
            ):
                relaxed = self._paper_relaxed_fallback_decision(
                    detection=detection,
                    wallet=wallet,
                    category=decision_category,
                    scaled_notional=scaled_notional,
                    style_evaluations=style_evaluations,
                    source_quality=decision_source_quality,
                    discovery_state=discovery_state,
                    scoring_state=scoring_state,
                )
                if relaxed is not None:
                    best_decision = relaxed

            best_decision.context["style_evaluations"] = style_evaluations
            trust_level = classify_trust_level(
                source_quality=decision_source_quality.value,
                discovery_state=discovery_state,
                scoring_state=scoring_state,
                fallback_in_use=decision_source_quality == SourceQuality.SYNTHETIC_FALLBACK,
            )
            fallback_used = decision_source_quality == SourceQuality.SYNTHETIC_FALLBACK
            if best_decision.action == DecisionAction.PAPER_COPY and (
                fallback_used or scoring_state == "EMPTY"
            ):
                best_decision = best_decision.model_copy(
                    update={
                        "allowed": False,
                        "action": DecisionAction.SKIP,
                        "reason_code": "NON_VALIDATION_SOURCE",
                        "human_readable_reason": "Paper decision blocked because the current source/scoring state is not validation-grade.",
                    }
                )
            best_decision.context["trust_level"] = trust_level
            best_decision.context["fallback_used"] = fallback_used
            best_decision.context["counts_as_trustworthy_approval"] = counts_as_trustworthy_approval(
                final_action=best_decision.action.value,
                trust_level=trust_level,
                fallback_in_use=fallback_used,
                scoring_state=scoring_state,
            )
            self._write_decision_trace(detection, best_decision, cluster_confirmed, style_evaluations, discovery_state, scoring_state, decision_source_quality)
            decisions.append(best_decision)
        return decisions

    def _select_copy_fraction(self, wallet: WalletMetrics) -> float:
        base = self.config.risk.copy_fraction_min + (self.config.risk.copy_fraction_max - self.config.risk.copy_fraction_min) * wallet.copyability_score
        return clamp(base, self.config.risk.copy_fraction_min, self.config.risk.copy_fraction_max)

    def _paper_bankroll(self) -> float:
        state_snapshot = self.state.read()
        override = float(state_snapshot.get("paper_bankroll_override", 0.0) or 0.0)
        return override if override > 0 else self.config.bankroll.paper_starting_bankroll

    def _max_notional_for_mode(self) -> float:
        if self.config.mode.value == "LIVE":
            operator_cap = self.config.env.operator_live_max_trade_usd
            if operator_cap is not None and operator_cap > 0:
                return min(self.config.risk.max_single_live_trade_usd, operator_cap)
            return self.config.risk.max_single_live_trade_usd
        state_snapshot = self.state.read()
        override = float(state_snapshot.get("paper_trade_notional_override", 0.0) or 0.0)
        if override > 0:
            return override
        return self._paper_bankroll() * self.config.risk.max_market_exposure_pct

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

    def _paper_relaxed_fallback_decision(
        self,
        detection: DetectionEvent,
        wallet: WalletMetrics,
        category: str,
        scaled_notional: float,
        style_evaluations: list[dict[str, object]],
        source_quality: SourceQuality,
        discovery_state: str,
        scoring_state: str,
    ) -> TradeDecision | None:
        soft_failures = {
            "NO_VALID_ENTRY_STYLE",
            "WIDE_SPREAD",
            "FILLABILITY",
            "SLIPPAGE",
            "ENTRY_DRIFT",
            "CATEGORY_BLOCKED",
        }
        if detection.side != "BUY":
            return None
        if not style_evaluations:
            return None
        if source_quality != SourceQuality.REAL_PUBLIC_DATA:
            return None
        if discovery_state != "SUCCESS" or scoring_state not in {"SUCCESS", "PARTIAL_SUCCESS"}:
            return None
        if any(str(item.get("reason_code")) not in soft_failures for item in style_evaluations):
            return None
        return TradeDecision(
            allowed=True,
            action=DecisionAction.PAPER_COPY,
            reason_code="PAPER_RELAXED_FALLBACK",
            human_readable_reason="Paper-only fallback accepted the signal despite microstructure gate failures.",
            local_decision_id=stable_event_key(detection.event_key, "PAPER_RELAXED_FALLBACK"),
            wallet_address=detection.wallet_address,
            market_id=detection.market_id,
            token_id=detection.token_id,
            entry_style=EntryStyle.FOLLOW_TAKER,
            category=category,
            scaled_notional=round(min(scaled_notional, max(self._max_notional_for_mode(), 1.0)), 4),
            source_price=detection.price,
            executable_price=detection.price,
            cluster_confirmed=False,
            hedge_suspicion_score=wallet.hedge_suspicion_score,
            context={
                "paper_relaxed": True,
                "style_evaluations": style_evaluations,
                "source_quality": source_quality.value,
                "discovery_state": discovery_state,
                "scoring_state": scoring_state,
                "trust_level": classify_trust_level(
                    source_quality=source_quality.value,
                    discovery_state=discovery_state,
                    scoring_state=scoring_state,
                    fallback_in_use=False,
                ),
                "fallback_used": False,
                "counts_as_trustworthy_approval": False,
            },
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

    def _write_decision_trace(
        self,
        detection: DetectionEvent,
        decision: TradeDecision,
        cluster_confirmed: bool,
        style_evaluations: list[dict[str, object]],
        discovery_state: str,
        scoring_state: str,
        source_quality: SourceQuality,
    ) -> None:
        risk_context = decision.context.get("risk_context", {})
        fill = decision.context.get("fill", {})
        append_jsonl(
            self.trace_path,
            {
                "signal_id": decision.local_decision_id,
                "detected_at": detection.local_detection_timestamp.isoformat(),
                "source_trade_timestamp": detection.source_trade_timestamp.isoformat(),
                "wallet_address": detection.wallet_address,
                "source_alias": detection.source_alias,
                "category": decision.category,
                "market_id": detection.market_id,
                "token_id": detection.token_id,
                "source_wallet": detection.wallet_address,
                "source_quality": source_quality.value,
                "trust_level": decision.context.get(
                    "trust_level",
                    classify_trust_level(
                        source_quality=source_quality.value,
                        discovery_state=discovery_state,
                        scoring_state=scoring_state,
                        fallback_in_use=source_quality == SourceQuality.SYNTHETIC_FALLBACK,
                    ),
                ),
                "fallback_used": decision.context.get("fallback_used", source_quality == SourceQuality.SYNTHETIC_FALLBACK),
                "counts_as_trustworthy_approval": decision.context.get("counts_as_trustworthy_approval", False),
                "discovery_state": discovery_state,
                "scoring_state": scoring_state,
                "cluster_state": "CONFIRMED" if cluster_confirmed else "SINGLE_WALLET",
                "freshness_state": "FRESH" if detection.detection_latency_seconds <= self.config.risk.stale_signal_seconds else "STALE",
                "fillability_state": "FILLABLE" if fill.get("fillable") else "UNFILLABLE",
                "risk_allowed": decision.allowed,
                "risk_reason_code": decision.reason_code,
                "risk_reason": decision.human_readable_reason,
                "human_readable_reason": decision.human_readable_reason,
                "risk_context": risk_context,
                "final_action": decision.action.value,
                "reason_code": decision.reason_code,
                "entry_style": decision.entry_style.value,
                "style_evaluations": style_evaluations,
                "scaled_notional": decision.scaled_notional,
            },
        )
