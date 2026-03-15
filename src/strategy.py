from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.clustering import cluster_detections
from src.config import AppConfig
from src.external_signals import OfficialSignalStore, mapping_confidence, parse_signal_timestamp, resolve_official_signal_market
from src.market_data import MarketDataService
from src.market_relationships import build_relationship_groups, find_best_dislocation_pair
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
        self.signal_log_path = data_dir / "strategy_signal_log.jsonl"
        self.official_signals = OfficialSignalStore(data_dir / Path(config.strategies.official_signal_file).name)

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
                try:
                    market_meta = await self.market_data.fetch_market_metadata(
                        detection.market_id,
                        detection.token_id,
                        detection.market_slug,
                        str(detection.market_metadata.get("outcome") or ""),
                    )
                except TypeError:
                    market_meta = await self.market_data.fetch_market_metadata(
                        detection.market_id,
                        detection.token_id,
                    )
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
            resolved_market_id = str(market_meta.get("market_id") or detection.market_id)
            resolved_token_id = str(detection.token_id or "").strip()
            if not resolved_token_id or resolved_token_id == "unknown-token":
                resolved_token_id = str(market_meta.get("token_id") or "").strip()
            if not resolved_token_id or resolved_token_id == "unknown-token":
                skip = self._skip_decision(
                    detection,
                    wallet,
                    min(detection.notional * self._select_copy_fraction(wallet), self._max_notional_for_mode()),
                    "MISSING_TOKEN_ID",
                    "Signal skipped because no usable token_id could be resolved from market metadata.",
                )
                skip.context.update(
                    {
                        "market_meta": market_meta,
                        "discovery_state": discovery_state,
                        "scoring_state": scoring_state,
                        "source_quality": decision_source_quality.value,
                    }
                )
                self._write_decision_trace(detection, skip, False, [], discovery_state, scoring_state, decision_source_quality)
                decisions.append(skip)
                continue
            try:
                tradability = await self.market_data.get_tradability(resolved_market_id, resolved_token_id)
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
                            "token_id": resolved_token_id,
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
                orderbook = await self.market_data.fetch_orderbook(resolved_token_id)
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
                    market_id=resolved_market_id,
                    token_id=resolved_token_id,
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
        decisions.extend(
            await self._generate_event_driven_official_decisions(
                stored_positions=active_positions,
                total_exposure=total_exposure,
                active_market_exposure=active_market_exposure,
                active_wallet_exposure=active_wallet_exposure,
            )
        )
        decisions.extend(
            await self._generate_correlation_dislocation_decisions(
                stored_positions=active_positions,
                total_exposure=total_exposure,
                active_market_exposure=active_market_exposure,
                active_wallet_exposure=active_wallet_exposure,
            )
        )
        decisions.extend(
            await self._generate_resolution_window_decisions(
                stored_positions=active_positions,
                total_exposure=total_exposure,
                active_market_exposure=active_market_exposure,
                active_wallet_exposure=active_wallet_exposure,
            )
        )
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
        if (
            self.config.mode.value == "LIVE"
            and not self.config.live.enable_multi_entry_style_live
            and entry_style != self.config.entry_styles.preferred_live_entry_style
        ):
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
            strategy_name="wallet_follow",
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
            strategy_name="wallet_follow",
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
                "strategy_name": decision.strategy_name,
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

    def _strategy_wallet(self, strategy_name: str) -> WalletMetrics:
        return WalletMetrics(
            wallet_address=f"strategy:{strategy_name}",
            evaluation_window_days=30,
            trade_count=50,
            trades_per_day=1.0,
            buy_count=25,
            sell_count=25,
            estimated_pnl_percent=0.2,
            win_rate=0.6,
            average_trade_size=100.0,
            conviction_score=0.8,
            market_concentration=0.2,
            category_concentration=0.5,
            holding_time_estimate_hours=6.0,
            drawdown_proxy=0.1,
            copyability_score=0.8,
            low_velocity_score=0.8,
            delay_5s=0.8,
            delay_15s=0.8,
            delay_30s=0.8,
            delay_60s=0.8,
            delayed_viability_score=0.8,
            hedge_suspicion_score=0.0,
            global_score=0.75,
            dominant_category="unknown",
            source_quality=SourceQuality.REAL_PUBLIC_DATA,
        )

    def _has_market_position(self, active_positions: list[dict], market_id: str) -> bool:
        for position in active_positions:
            if position.get("closed"):
                continue
            if position.get("market_id") == market_id:
                return True
        return False

    def _strategy_live_enabled(self, strategy_name: str) -> bool:
        if strategy_name == "event_driven_official":
            return self.config.strategies.official_signal_live_enabled
        if strategy_name == "correlation_dislocation":
            return self.config.strategies.correlation_live_enabled
        if strategy_name == "resolution_window":
            return self.config.strategies.resolution_window_live_enabled
        return False

    def _official_signal_confidence(
        self,
        *,
        row: dict[str, object],
        mapping_reason: str,
        fair_price: float,
        source_price: float,
        age_seconds: float,
    ) -> float:
        try:
            explicit_confidence_raw = float(row.get("confidence_score", 0.75) or 0.75)
        except (TypeError, ValueError):
            explicit_confidence_raw = 0.75
        try:
            source_reliability_raw = float(row.get("source_reliability", 0.8) or 0.8)
        except (TypeError, ValueError):
            source_reliability_raw = 0.8
        explicit_confidence = clamp(explicit_confidence_raw, 0.0, 1.0)
        source_reliability = clamp(source_reliability_raw, 0.0, 1.0)
        edge_component = clamp((fair_price - source_price) / max(1.0 - source_price, 0.15), 0.0, 1.0)
        freshness_component = clamp(
            1.0 - (age_seconds / max(self.config.strategies.official_signal_max_age_minutes * 60, 1)),
            0.0,
            1.0,
        )
        mapping_component = mapping_confidence(mapping_reason)
        confidence = (
            explicit_confidence * 0.35
            + source_reliability * 0.25
            + edge_component * 0.2
            + freshness_component * 0.1
            + mapping_component * 0.1
        )
        return round(clamp(confidence, 0.0, 1.0), 4)

    def _log_strategy_signal(
        self,
        *,
        strategy_name: str,
        signal_id: str,
        market_id: str,
        token_id: str,
        final_action: str,
        reason_code: str,
        extra: dict[str, object] | None = None,
    ) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "strategy_name": strategy_name,
            "signal_id": signal_id,
            "market_id": market_id,
            "token_id": token_id,
            "final_action": final_action,
            "reason_code": reason_code,
        }
        if extra:
            payload.update(extra)
        append_jsonl(self.signal_log_path, payload)

    def _paper_relaxed_supplemental_decision(
        self,
        *,
        strategy_name: str,
        detection: DetectionEvent,
        wallet: WalletMetrics,
        market_id: str,
        token_id: str,
        category: str,
        scaled_notional: float,
        fill: object,
        confidence_score: float,
        extra_context: dict[str, object],
    ) -> TradeDecision | None:
        if self.config.mode.value == "LIVE":
            return None
        if not self.config.strategies.supplemental_paper_relaxed_enabled:
            return None
        if confidence_score < self.config.strategies.supplemental_paper_relaxed_min_confidence:
            return None
        return TradeDecision(
            strategy_name=strategy_name,
            allowed=True,
            action=DecisionAction.PAPER_COPY,
            reason_code="SUPPLEMENTAL_PAPER_RELAXED",
            human_readable_reason="Paper-only supplemental fallback accepted a near-miss candidate for evaluation.",
            local_decision_id=stable_event_key(detection.event_key, strategy_name, "SUPPLEMENTAL_PAPER_RELAXED"),
            wallet_address=wallet.wallet_address,
            market_id=market_id,
            token_id=token_id,
            entry_style=EntryStyle.PASSIVE_LIMIT,
            category=category,
            scaled_notional=round(scaled_notional, 4),
            source_price=detection.price,
            executable_price=detection.price,
            cluster_confirmed=False,
            hedge_suspicion_score=wallet.hedge_suspicion_score,
            context={
                "paper_relaxed": True,
                "fill": fill.model_dump() if hasattr(fill, "model_dump") else {},
                "confidence_score": confidence_score,
                **extra_context,
            },
        )

    async def _build_supplemental_decision(
        self,
        *,
        strategy_name: str,
        signal_id: str,
        market_id: str,
        token_id: str,
        category: str,
        market_title: str,
        source_price: float,
        fair_price: float,
        scaled_notional: float,
        reason: str,
        confidence_score: float,
        extra_context: dict[str, object],
        stored_positions: list[dict],
        total_exposure: float,
        active_market_exposure: dict[str, float],
        active_wallet_exposure: dict[str, float],
        age_seconds: float = 0.0,
    ) -> TradeDecision:
        wallet = self._strategy_wallet(strategy_name)
        wallet.dominant_category = category
        detection = DetectionEvent(
            event_key=signal_id,
            wallet_address=wallet.wallet_address,
            market_title=market_title,
            market_slug=market_title.lower().replace(" ", "-"),
            market_id=market_id,
            token_id=token_id,
            side="BUY",
            price=source_price,
            size=max(scaled_notional / max(source_price, 1e-6), 1.0),
            notional=scaled_notional,
            transaction_hash=signal_id,
            detection_latency_seconds=age_seconds,
            category=category,
            source_alias=strategy_name,
        )
        if self.config.mode.value == "LIVE" and not self._strategy_live_enabled(strategy_name):
            decision = self._skip_decision(
                detection,
                wallet,
                scaled_notional,
                "STRATEGY_LIVE_DISABLED",
                f"{strategy_name} is paper/research-only and is disabled for live trading.",
            )
            decision.strategy_name = strategy_name
            decision.context.update(extra_context)
            decision.context.update(
                {
                    "source_quality": SourceQuality.REAL_PUBLIC_DATA.value,
                    "trust_level": classify_trust_level(
                        source_quality=SourceQuality.REAL_PUBLIC_DATA.value,
                        discovery_state="SUCCESS",
                        scoring_state="SUCCESS",
                        fallback_in_use=False,
                    ),
                    "confidence_score": confidence_score,
                }
            )
            self._write_decision_trace(
                detection,
                decision,
                False,
                [],
                "SUCCESS",
                "SUCCESS",
                SourceQuality.REAL_PUBLIC_DATA,
            )
            self._log_strategy_signal(
                strategy_name=strategy_name,
                signal_id=signal_id,
                market_id=market_id,
                token_id=token_id,
                final_action=decision.action.value,
                reason_code=decision.reason_code,
                extra={"confidence_score": confidence_score, **extra_context},
            )
            return decision

        try:
            market_meta = await self.market_data.fetch_market_metadata(market_id, token_id)
            tradability = await self.market_data.get_tradability(market_id, token_id)
            orderbook = await self.market_data.fetch_orderbook(token_id)
        except RuntimeError as exc:
            decision = self._skip_decision(
                detection,
                wallet,
                scaled_notional,
                "STRATEGY_MARKET_UNAVAILABLE",
                f"{strategy_name} signal skipped because market data is unavailable: {exc}",
            )
            decision.strategy_name = strategy_name
            decision.context.update(extra_context)
            decision.context["market_error"] = str(exc)
            self._write_decision_trace(detection, decision, False, [], "SUCCESS", "SUCCESS", SourceQuality.REAL_PUBLIC_DATA)
            self._log_strategy_signal(
                strategy_name=strategy_name,
                signal_id=signal_id,
                market_id=market_id,
                token_id=token_id,
                final_action=decision.action.value,
                reason_code=decision.reason_code,
                extra={"confidence_score": confidence_score, "market_error": str(exc), **extra_context},
            )
            return decision

        fill = estimate_fill(orderbook, scaled_notional, self.config.risk.max_slippage_pct)
        best_ask = orderbook.asks[0].price if orderbook.asks else source_price
        drift_pct = abs(fill.executable_price - best_ask) / max(best_ask, 1e-6)
        risk = self.risk.evaluate(
            detection=detection,
            wallet=wallet,
            fill=fill,
            mode=self.config.mode.value,
            total_exposure=total_exposure,
            market_exposure=active_market_exposure.get(market_id, 0.0),
            wallet_exposure=active_wallet_exposure.get(wallet.wallet_address, 0.0),
            daily_pnl=self.state.read().get("daily_pnl", 0.0),
            cluster_confirmed=False,
            infra_ok=bool(market_meta.get("active", True)) and not self.state.read().get("paused", False),
            entry_style_allowed=True,
            category=category,
            market_id=market_id,
            has_conflicting_position=self._has_market_position(stored_positions, market_id),
            data_age_seconds=(datetime.now(timezone.utc) - orderbook.timestamp).total_seconds(),
            entry_drift_pct=drift_pct,
            live_ready=bool(self.state.read().get("live_readiness_last_result", {}).get("ready", True)),
            kill_switch=bool(self.state.read().get("kill_switch", False)),
            manual_live_enable=bool(self.state.read().get("manual_live_enable", True)),
            manual_resume_required=bool(self.state.read().get("manual_resume_required", False)),
            health_state=str(self.state.read().get("live_health_state", "HEALTHY")),
            reconciliation_clean=bool(self.state.read().get("reconciliation_clean", True)),
            heartbeat_ok=bool(self.state.read().get("heartbeat_ok", True)),
            balance_visible=bool(self.state.read().get("balance_visible", True)),
            allowance_sufficient=bool(self.state.read().get("allowance_sufficient", True)),
            tradable=bool(tradability.get("tradable")) and bool(tradability.get("orderbook_enabled")),
            bankroll_override=self._paper_bankroll() if self.config.mode.value != "LIVE" else None,
        )
        action = DecisionAction.PAPER_COPY if risk.allowed and self.config.mode.value != "LIVE" else DecisionAction.SKIP
        decision = TradeDecision(
            strategy_name=strategy_name,
            allowed=risk.allowed,
            action=action,
            reason_code=risk.reason_code,
            human_readable_reason=risk.human_readable_reason if risk.allowed else risk.human_readable_reason,
            local_decision_id=stable_event_key(signal_id, strategy_name, market_id, token_id),
            wallet_address=wallet.wallet_address,
            market_id=market_id,
            token_id=token_id,
            entry_style=EntryStyle.PASSIVE_LIMIT,
            category=category,
            scaled_notional=round(scaled_notional, 4),
            source_price=source_price,
            executable_price=fill.executable_price or best_ask,
            cluster_confirmed=False,
            hedge_suspicion_score=wallet.hedge_suspicion_score,
            context={
                "fill": fill.model_dump(),
                "risk_context": risk.context,
                "market_meta": market_meta,
                "tradability": tradability,
                "source_quality": SourceQuality.REAL_PUBLIC_DATA.value,
                "trust_level": classify_trust_level(
                    source_quality=SourceQuality.REAL_PUBLIC_DATA.value,
                    discovery_state="SUCCESS",
                    scoring_state="SUCCESS",
                    fallback_in_use=False,
                ),
                "fallback_used": False,
                "counts_as_trustworthy_approval": False,
                "confidence_score": confidence_score,
                "fair_price": fair_price,
                **extra_context,
            },
        )
        if (
            not risk.allowed
            and strategy_name in {"correlation_dislocation", "resolution_window"}
            and risk.reason_code in {"WIDE_SPREAD", "FILLABILITY", "SLIPPAGE", "ENTRY_DRIFT"}
        ):
            relaxed = self._paper_relaxed_supplemental_decision(
                strategy_name=strategy_name,
                detection=detection,
                wallet=wallet,
                market_id=market_id,
                token_id=token_id,
                category=category,
                scaled_notional=scaled_notional,
                fill=fill,
                confidence_score=confidence_score,
                extra_context={
                    "risk_context": risk.context,
                    "market_meta": market_meta,
                    "tradability": tradability,
                    "source_quality": SourceQuality.REAL_PUBLIC_DATA.value,
                    "trust_level": classify_trust_level(
                        source_quality=SourceQuality.REAL_PUBLIC_DATA.value,
                        discovery_state="SUCCESS",
                        scoring_state="SUCCESS",
                        fallback_in_use=False,
                    ),
                    "fallback_used": False,
                    "counts_as_trustworthy_approval": False,
                    "fair_price": fair_price,
                    "soft_failure_reason_code": risk.reason_code,
                    **extra_context,
                },
            )
            if relaxed is not None:
                decision = relaxed
        self._write_decision_trace(detection, decision, False, [], "SUCCESS", "SUCCESS", SourceQuality.REAL_PUBLIC_DATA)
        self._log_strategy_signal(
            strategy_name=strategy_name,
            signal_id=signal_id,
            market_id=market_id,
            token_id=token_id,
            final_action=decision.action.value,
            reason_code=decision.reason_code,
            extra={"confidence_score": confidence_score, **extra_context},
        )
        return decision

    async def _generate_event_driven_official_decisions(
        self,
        *,
        stored_positions: list[dict],
        total_exposure: float,
        active_market_exposure: dict[str, float],
        active_wallet_exposure: dict[str, float],
    ) -> list[TradeDecision]:
        if not self.config.strategies.enable_event_driven_official:
            return []
        now = datetime.now(timezone.utc)
        rows = self.official_signals.load()
        decisions: list[TradeDecision] = []
        market_rows = list(self.market_data.market_cache.values())
        if not rows:
            self._log_strategy_signal(
                strategy_name="event_driven_official",
                signal_id=stable_event_key("event_driven_official", "cycle", now.isoformat()),
                market_id="",
                token_id="",
                final_action=DecisionAction.SKIP.value,
                reason_code="NO_OFFICIAL_SIGNALS_CONFIGURED",
                extra={"cycle_ts": now.isoformat()},
            )
            return []
        emitted = 0
        for row in rows:
            published_at = parse_signal_timestamp(row.get("published_at") or row.get("updated_at"))
            if published_at is None:
                continue
            age_seconds = max((now - published_at).total_seconds(), 0.0)
            if age_seconds > self.config.strategies.official_signal_max_age_minutes * 60:
                continue
            try:
                fair_price = float(row.get("fair_price"))
            except (TypeError, ValueError):
                continue
            try:
                source_reliability_raw = float(row.get("source_reliability", 0.8) or 0.8)
            except (TypeError, ValueError):
                source_reliability_raw = 0.8
            source_reliability = clamp(source_reliability_raw, 0.0, 1.0)
            if source_reliability < self.config.strategies.official_signal_min_source_reliability:
                self._log_strategy_signal(
                    strategy_name="event_driven_official",
                    signal_id=str(row.get("event_id") or stable_event_key("official", row.get("title"), published_at.isoformat())),
                    market_id=str(row.get("market_id") or ""),
                    token_id=str(row.get("token_id") or ""),
                    final_action=DecisionAction.SKIP.value,
                    reason_code="LOW_SOURCE_RELIABILITY",
                    extra={"source_reliability": source_reliability, "published_at": published_at.isoformat()},
                )
                continue
            resolved_market, mapping_reason = resolve_official_signal_market(row, market_rows)
            signal_id = str(row.get("event_id") or stable_event_key("official", row.get("title"), row.get("market_slug"), published_at.isoformat()))
            if resolved_market is None:
                self._log_strategy_signal(
                    strategy_name="event_driven_official",
                    signal_id=signal_id,
                    market_id=str(row.get("market_id") or ""),
                    token_id=str(row.get("token_id") or ""),
                    final_action=DecisionAction.SKIP.value,
                    reason_code=mapping_reason.upper(),
                    extra={
                        "published_at": published_at.isoformat(),
                        "title": str(row.get("title") or ""),
                        "market_slug": str(row.get("market_slug") or row.get("slug") or ""),
                        "outcome": str(row.get("outcome") or ""),
                    },
                )
                continue
            market_id = resolved_market.market_id
            token_id = resolved_market.token_id
            scaled_notional = min(float(row.get("notional_usd", self._max_notional_for_mode())), self._max_notional_for_mode())
            source_price = float(row.get("source_price") or fair_price)
            if fair_price - source_price < self.config.strategies.official_signal_min_edge_pct:
                continue
            confidence_score = self._official_signal_confidence(
                row=row,
                mapping_reason=mapping_reason,
                fair_price=fair_price,
                source_price=source_price,
                age_seconds=age_seconds,
            )
            if confidence_score < self.config.strategies.official_signal_min_confidence:
                self._log_strategy_signal(
                    strategy_name="event_driven_official",
                    signal_id=signal_id,
                    market_id=market_id,
                    token_id=token_id,
                    final_action=DecisionAction.SKIP.value,
                    reason_code="LOW_SIGNAL_CONFIDENCE",
                    extra={
                        "confidence_score": confidence_score,
                        "source_reliability": source_reliability,
                        "mapping_reason": mapping_reason,
                        "published_at": published_at.isoformat(),
                    },
                )
                continue
            decisions.append(
                await self._build_supplemental_decision(
                    strategy_name="event_driven_official",
                    signal_id=signal_id,
                    market_id=market_id,
                    token_id=token_id,
                    category=str(row.get("category") or resolved_market.category or "unknown"),
                    market_title=str(row.get("title") or row.get("market_title") or resolved_market.title),
                    source_price=source_price,
                    fair_price=fair_price,
                    scaled_notional=scaled_notional,
                    reason=str(row.get("rationale") or "Official event signal mapped to market."),
                    confidence_score=confidence_score,
                    extra_context={
                        "strategy_rationale": str(row.get("rationale") or "Official event signal mapped to market."),
                        "source_name": str(row.get("source_name") or ""),
                        "source_url": str(row.get("source_url") or ""),
                        "published_at": published_at.isoformat(),
                        "source_reliability": source_reliability,
                        "edge_pct": round(fair_price - source_price, 6),
                        "mapping_reason": mapping_reason,
                        "mapped_market_title": resolved_market.title,
                        "mapped_market_slug": resolved_market.slug,
                        "required_conditions": row.get("required_conditions", []),
                        "disqualifiers": row.get("disqualifiers", []),
                    },
                    stored_positions=stored_positions,
                    total_exposure=total_exposure,
                    active_market_exposure=active_market_exposure,
                    active_wallet_exposure=active_wallet_exposure,
                    age_seconds=age_seconds,
                )
            )
            emitted += 1
        if emitted == 0:
            self._log_strategy_signal(
                strategy_name="event_driven_official",
                signal_id=stable_event_key("event_driven_official", "cycle-empty", now.isoformat()),
                market_id="",
                token_id="",
                final_action=DecisionAction.SKIP.value,
                reason_code="NO_ELIGIBLE_OFFICIAL_SIGNALS",
                extra={"cycle_ts": now.isoformat(), "loaded_signal_count": len(rows)},
            )
        return decisions

    async def _generate_correlation_dislocation_decisions(
        self,
        *,
        stored_positions: list[dict],
        total_exposure: float,
        active_market_exposure: dict[str, float],
        active_wallet_exposure: dict[str, float],
    ) -> list[TradeDecision]:
        if not self.config.strategies.enable_correlation_dislocation:
            return []
        groups = build_relationship_groups(
            list(self.market_data.market_cache.values()),
            self.config.strategies.correlation_min_group_size,
        )
        decisions: list[TradeDecision] = []
        cycle_ts = datetime.now(timezone.utc)
        if not groups:
            self._log_strategy_signal(
                strategy_name="correlation_dislocation",
                signal_id=stable_event_key("correlation_dislocation", "cycle", cycle_ts.isoformat()),
                market_id="",
                token_id="",
                final_action=DecisionAction.SKIP.value,
                reason_code="NO_RELATIONSHIP_GROUPS",
                extra={"cycle_ts": cycle_ts.isoformat(), "market_cache_size": len(self.market_data.market_cache)},
            )
            return []
        emitted = 0
        near_miss_groups = 0
        best_near_miss: tuple[float, object, object, float, float, list[str]] | None = None
        best_observed_pair: tuple[float, object, object, float, float, list[str]] | None = None
        for group in groups[: self.config.strategies.correlation_max_groups]:
            if len(group) < 2:
                continue
            try:
                orderbooks = {market.market_id: await self.market_data.fetch_orderbook(market.token_id) for market in group}
            except RuntimeError:
                continue
            mid_prices: dict[str, float] = {}
            for market in group:
                orderbook = orderbooks[market.market_id]
                if not orderbook.bids or not orderbook.asks:
                    continue
                mid_prices[market.market_id] = round((orderbook.bids[0].price + orderbook.asks[0].price) / 2.0, 6)
            if len(mid_prices) < 2:
                continue
            best_pair = find_best_dislocation_pair(group, mid_prices, self.config.strategies.correlation_min_gap_pct)
            if best_pair is None:
                near_miss_groups += 1
                raw_best_pair = find_best_dislocation_pair(group, mid_prices, 0.0)
                if raw_best_pair is not None:
                    earlier, later, confidence = raw_best_pair
                    raw_gap = mid_prices[earlier.market_id] - mid_prices[later.market_id]
                    if raw_gap <= 0.0:
                        continue
                    observed = (
                        raw_gap,
                        earlier,
                        later,
                        mid_prices[earlier.market_id],
                        mid_prices[later.market_id],
                        [market.market_id for market in group],
                    )
                    if best_observed_pair is None or observed[0] > best_observed_pair[0]:
                        best_observed_pair = observed
                    min_gap = self.config.strategies.correlation_min_gap_pct
                    if raw_gap >= min_gap * self.config.strategies.correlation_near_miss_gap_ratio:
                        candidate = observed
                        if best_near_miss is None or candidate[0] > best_near_miss[0]:
                            best_near_miss = candidate
                continue
            earlier, later, confidence = best_pair
            earlier_mid = mid_prices[earlier.market_id]
            later_mid = mid_prices[later.market_id]
            fair_price = earlier_mid
            source_price = later_mid
            decisions.append(
                await self._build_supplemental_decision(
                    strategy_name="correlation_dislocation",
                    signal_id=stable_event_key("correlation", earlier.market_id, later.market_id),
                    market_id=later.market_id,
                    token_id=later.token_id,
                    category=later.category,
                    market_title=later.title,
                    source_price=source_price,
                    fair_price=fair_price,
                    scaled_notional=self._max_notional_for_mode(),
                    reason="Later-dated related market is priced materially below the earlier market in the same relationship group.",
                    confidence_score=round(min(confidence, 1.0), 4),
                    extra_context={
                        "relationship_group": [market.market_id for market in group],
                        "reference_market_id": earlier.market_id,
                        "reference_market_title": earlier.title,
                        "reference_mid_price": earlier_mid,
                        "target_mid_price": later_mid,
                        "price_gap_pct": round(earlier_mid - later_mid, 6),
                        "relative_gap_pct": round(confidence, 6),
                        "strategy_rationale": "Later-dated related market is cheaper than the earlier-dated market beyond the configured dislocation threshold.",
                    },
                    stored_positions=stored_positions,
                    total_exposure=total_exposure,
                    active_market_exposure=active_market_exposure,
                    active_wallet_exposure=active_wallet_exposure,
                    age_seconds=0.0,
                )
            )
            emitted += 1
        if (
            emitted == 0
            and best_near_miss is not None
            and self.config.mode.value != "LIVE"
            and self.config.strategies.supplemental_paper_relaxed_enabled
            and not self._strategy_live_enabled("correlation_dislocation")
        ):
            raw_gap, earlier, later, earlier_mid, later_mid, relationship_group = best_near_miss
            gap_ratio = raw_gap / max(self.config.strategies.correlation_min_gap_pct, 1e-6)
            confidence = clamp(
                0.62
                + min(gap_ratio, 1.0) * 0.16
                + min(earlier_mid, 1.0) * 0.08,
                0.0,
                0.92,
            )
            decisions.append(
                await self._build_supplemental_decision(
                    strategy_name="correlation_dislocation",
                    signal_id=stable_event_key("correlation-near-miss", earlier.market_id, later.market_id),
                    market_id=later.market_id,
                    token_id=later.token_id,
                    category=later.category,
                    market_title=later.title,
                    source_price=later_mid,
                    fair_price=earlier_mid,
                    scaled_notional=self._max_notional_for_mode(),
                    reason="Later-dated related market is only slightly below the dislocation threshold but still merits paper evaluation.",
                    confidence_score=round(confidence, 4),
                    extra_context={
                        "relationship_group": relationship_group,
                        "reference_market_id": earlier.market_id,
                        "reference_market_title": earlier.title,
                        "reference_mid_price": earlier_mid,
                        "target_mid_price": later_mid,
                        "price_gap_pct": round(raw_gap, 6),
                        "near_miss_gap_ratio": round(gap_ratio, 6),
                        "strategy_rationale": "Paper-only near-miss dislocation candidate promoted for evaluation after grouping succeeded but the gap landed just below the strict threshold.",
                        "strategy_near_miss": True,
                    },
                    stored_positions=stored_positions,
                    total_exposure=total_exposure,
                    active_market_exposure=active_market_exposure,
                    active_wallet_exposure=active_wallet_exposure,
                    age_seconds=0.0,
                )
            )
            emitted += 1
        if emitted == 0:
            extra = {
                "cycle_ts": cycle_ts.isoformat(),
                "group_count": len(groups),
                "near_miss_groups": near_miss_groups,
            }
            top_candidate = best_near_miss or best_observed_pair
            if top_candidate is not None:
                raw_gap, earlier, later, earlier_mid, later_mid, relationship_group = top_candidate
                extra.update(
                    {
                        "top_near_miss_market_id": later.market_id,
                        "top_near_miss_token_id": later.token_id,
                        "top_near_miss_market_title": later.title,
                        "top_near_miss_reference_market_id": earlier.market_id,
                        "top_near_miss_reference_market_title": earlier.title,
                        "top_near_miss_reference_mid_price": round(earlier_mid, 6),
                        "top_near_miss_target_mid_price": round(later_mid, 6),
                        "top_near_miss_gap_pct": round(raw_gap, 6),
                        "top_near_miss_gap_ratio": round(
                            raw_gap / max(self.config.strategies.correlation_min_gap_pct, 1e-6),
                            6,
                        ),
                        "top_near_miss_qualified": bool(best_near_miss is not None and top_candidate == best_near_miss),
                        "top_near_miss_relationship_group": relationship_group,
                    }
                )
            self._log_strategy_signal(
                strategy_name="correlation_dislocation",
                signal_id=stable_event_key("correlation_dislocation", "cycle-empty", cycle_ts.isoformat()),
                market_id="",
                token_id="",
                final_action=DecisionAction.SKIP.value,
                reason_code="NO_PRICE_DISLOCATIONS",
                extra=extra,
            )
        return decisions

    async def _generate_resolution_window_decisions(
        self,
        *,
        stored_positions: list[dict],
        total_exposure: float,
        active_market_exposure: dict[str, float],
        active_wallet_exposure: dict[str, float],
    ) -> list[TradeDecision]:
        if not self.config.strategies.enable_resolution_window:
            return []
        now = datetime.now(timezone.utc)
        candidates: list[tuple[float, object]] = []
        for market in self.market_data.market_cache.values():
            if market.closed or not market.active or not market.market_id or not market.token_id:
                continue
            if "sports" in (market.category or "").lower():
                continue
            end_at = None
            if market.end_date_iso:
                try:
                    end_at = datetime.fromisoformat(market.end_date_iso.replace("Z", "+00:00"))
                except ValueError:
                    end_at = None
            if end_at is None or end_at <= now:
                continue
            hours_to_resolution = (end_at - now).total_seconds() / 3600.0
            if hours_to_resolution > self.config.strategies.resolution_window_max_hours:
                continue
            if market.liquidity < self.config.strategies.resolution_window_min_liquidity:
                continue
            candidates.append((hours_to_resolution, market))

        cycle_ts = now
        if not candidates:
            self._log_strategy_signal(
                strategy_name="resolution_window",
                signal_id=stable_event_key("resolution_window", "cycle", cycle_ts.isoformat()),
                market_id="",
                token_id="",
                final_action=DecisionAction.SKIP.value,
                reason_code="NO_NEAR_RESOLUTION_MARKETS",
                extra={"cycle_ts": cycle_ts.isoformat(), "market_cache_size": len(self.market_data.market_cache)},
            )
            return []

        decisions: list[TradeDecision] = []
        emitted = 0
        diagnostic_counts = {
            "missing_orderbook": 0,
            "no_asks": 0,
            "below_price_floor": 0,
            "edge_too_small": 0,
            "above_fair_anchor": 0,
        }
        best_near_miss: tuple[float, object, float, float, float, str] | None = None
        best_observed_failure: tuple[float, object, float, float, float, str] | None = None
        for hours_to_resolution, market in sorted(candidates, key=lambda item: (item[0], -item[1].liquidity))[:10]:
            try:
                orderbook = await self.market_data.fetch_orderbook(market.token_id)
            except RuntimeError:
                diagnostic_counts["missing_orderbook"] += 1
                continue
            if not orderbook.asks:
                diagnostic_counts["no_asks"] += 1
                continue
            ask_price = orderbook.asks[0].price
            if ask_price < self.config.strategies.resolution_window_min_price:
                diagnostic_counts["below_price_floor"] += 1
                fair_price = max(self.config.strategies.resolution_window_target_fair_price, ask_price)
                observed = (
                    ask_price - self.config.strategies.resolution_window_min_price,
                    market,
                    ask_price,
                    fair_price,
                    hours_to_resolution,
                    "PRICE_FLOOR",
                )
                if best_observed_failure is None or observed[0] > best_observed_failure[0]:
                    best_observed_failure = observed
                if ask_price >= self.config.strategies.resolution_window_min_price - self.config.strategies.resolution_window_near_miss_price_buffer:
                    confidence_score = clamp(
                        0.6
                        + min(market.liquidity / 10_000.0, 1.0) * 0.08
                        + (1.0 - min(hours_to_resolution / max(self.config.strategies.resolution_window_max_hours, 1), 1.0)) * 0.14,
                        0.0,
                        0.9,
                    )
                    candidate = (
                        ask_price - (self.config.strategies.resolution_window_min_price - self.config.strategies.resolution_window_near_miss_price_buffer),
                        market,
                        ask_price,
                        fair_price,
                        hours_to_resolution,
                        "PRICE_FLOOR_NEAR_MISS",
                    )
                    if best_near_miss is None or candidate[0] > best_near_miss[0]:
                        best_near_miss = candidate
                continue
            fair_price = max(self.config.strategies.resolution_window_target_fair_price, ask_price)
            if ask_price >= self.config.strategies.resolution_window_target_fair_price:
                diagnostic_counts["above_fair_anchor"] += 1
                observed = (
                    self.config.strategies.resolution_window_target_fair_price - ask_price,
                    market,
                    ask_price,
                    self.config.strategies.resolution_window_target_fair_price,
                    hours_to_resolution,
                    "ABOVE_FAIR_ANCHOR",
                )
                if best_observed_failure is None or observed[0] > best_observed_failure[0]:
                    best_observed_failure = observed
                continue
            if fair_price - ask_price < self.config.strategies.resolution_window_min_edge_pct:
                diagnostic_counts["edge_too_small"] += 1
                edge = fair_price - ask_price
                observed = (
                    edge / max(self.config.strategies.resolution_window_min_edge_pct, 1e-6),
                    market,
                    ask_price,
                    fair_price,
                    hours_to_resolution,
                    "EDGE",
                )
                if best_observed_failure is None or observed[0] > best_observed_failure[0]:
                    best_observed_failure = observed
                if edge >= self.config.strategies.resolution_window_min_edge_pct * self.config.strategies.resolution_window_near_miss_edge_ratio:
                    candidate = (
                        edge / max(self.config.strategies.resolution_window_min_edge_pct, 1e-6),
                        market,
                        ask_price,
                        fair_price,
                        hours_to_resolution,
                        "EDGE_NEAR_MISS",
                    )
                    if best_near_miss is None or candidate[0] > best_near_miss[0]:
                        best_near_miss = candidate
                continue
            confidence_score = clamp(
                0.55
                + min(market.liquidity / 10_000.0, 1.0) * 0.1
                + (1.0 - min(hours_to_resolution / max(self.config.strategies.resolution_window_max_hours, 1), 1.0)) * 0.15
                + min((ask_price - self.config.strategies.resolution_window_min_price) / 0.15, 0.15),
                0.0,
                0.95,
            )
            decisions.append(
                await self._build_supplemental_decision(
                    strategy_name="resolution_window",
                    signal_id=stable_event_key("resolution_window", market.market_id, market.token_id),
                    market_id=market.market_id,
                    token_id=market.token_id,
                    category=market.category,
                    market_title=market.title,
                    source_price=ask_price,
                    fair_price=fair_price,
                    scaled_notional=self._max_notional_for_mode(),
                    reason="Near-resolution market is still discounted relative to conservative settlement value.",
                    confidence_score=round(confidence_score, 4),
                    extra_context={
                        "hours_to_resolution": round(hours_to_resolution, 3),
                        "market_liquidity": market.liquidity,
                        "strategy_rationale": "Near-resolution market still trades below a conservative settlement anchor despite approaching expiry.",
                    },
                    stored_positions=stored_positions,
                    total_exposure=total_exposure,
                    active_market_exposure=active_market_exposure,
                    active_wallet_exposure=active_wallet_exposure,
                    age_seconds=0.0,
                )
            )
            emitted += 1
        if (
            emitted == 0
            and best_near_miss is not None
            and self.config.mode.value != "LIVE"
            and self.config.strategies.supplemental_paper_relaxed_enabled
            and not self._strategy_live_enabled("resolution_window")
        ):
            near_miss_score, market, ask_price, fair_price, hours_to_resolution, near_miss_reason = best_near_miss
            confidence_score = clamp(
                0.64
                + min(near_miss_score, 1.0) * 0.14
                + min(market.liquidity / 10_000.0, 1.0) * 0.08
                + (1.0 - min(hours_to_resolution / max(self.config.strategies.resolution_window_max_hours, 1), 1.0)) * 0.1,
                0.0,
                0.91,
            )
            decisions.append(
                await self._build_supplemental_decision(
                    strategy_name="resolution_window",
                    signal_id=stable_event_key("resolution-window-near-miss", market.market_id, market.token_id),
                    market_id=market.market_id,
                    token_id=market.token_id,
                    category=market.category,
                    market_title=market.title,
                    source_price=ask_price,
                    fair_price=fair_price,
                    scaled_notional=self._max_notional_for_mode(),
                    reason="Near-resolution market narrowly missed a strict filter but is still valuable for paper evaluation.",
                    confidence_score=round(confidence_score, 4),
                    extra_context={
                        "hours_to_resolution": round(hours_to_resolution, 3),
                        "market_liquidity": market.liquidity,
                        "strategy_rationale": "Paper-only near-miss resolution candidate promoted for evaluation after discovery succeeded but a strict edge or price-floor rule narrowly rejected it.",
                        "strategy_near_miss": True,
                        "near_miss_reason": near_miss_reason,
                    },
                    stored_positions=stored_positions,
                    total_exposure=total_exposure,
                    active_market_exposure=active_market_exposure,
                    active_wallet_exposure=active_wallet_exposure,
                    age_seconds=0.0,
                )
            )
            emitted += 1
        if emitted == 0:
            extra = {"cycle_ts": cycle_ts.isoformat(), "candidate_count": len(candidates), **diagnostic_counts}
            top_candidate = best_near_miss or best_observed_failure
            if top_candidate is not None:
                near_miss_score, market, ask_price, fair_price, hours_to_resolution, near_miss_reason = top_candidate
                extra.update(
                    {
                        "top_near_miss_market_id": market.market_id,
                        "top_near_miss_token_id": market.token_id,
                        "top_near_miss_market_title": market.title,
                        "top_near_miss_ask_price": round(ask_price, 6),
                        "top_near_miss_fair_price": round(fair_price, 6),
                        "top_near_miss_hours_to_resolution": round(hours_to_resolution, 3),
                        "top_near_miss_liquidity": market.liquidity,
                        "top_near_miss_reason": near_miss_reason,
                        "top_near_miss_score": round(near_miss_score, 6),
                        "top_near_miss_qualified": bool(best_near_miss is not None and top_candidate == best_near_miss),
                    }
                )
            self._log_strategy_signal(
                strategy_name="resolution_window",
                signal_id=stable_event_key("resolution_window", "cycle-empty", cycle_ts.isoformat()),
                market_id="",
                token_id="",
                final_action=DecisionAction.SKIP.value,
                reason_code="NO_RESOLUTION_WINDOW_EDGES",
                extra=extra,
            )
        return decisions
