from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from src.clustering import cluster_detections
from src.config import AppConfig
from src.external_signals import OfficialSignalStore, mapping_confidence, parse_signal_timestamp, resolve_official_signal_market
from src.lag_signal import LagSignalConfig as _LagSignalCfg, evaluate_lag_signal
from src.live_orders import LiveOrderStore
from src.market_data import MarketDataService
from src.market_relationships import build_relationship_groups, find_best_dislocation_pair
from src.models import ApprovedWallets, DecisionAction, DetectionEvent, EntryStyle, FillEstimate, MarketInfo, OrderLifecycleStatus, OrderbookSnapshot, SourceQuality, TradeDecision, WalletMetrics
from src.orderbook import estimate_fill
from src.paper_quality import classify_trust_level, counts_as_trustworthy_approval
from src.positions import PositionStore
from src.risk_manager import RiskManager
from src.rtds_client import get_rtds_client
from src.state import AppStateStore
from src.source_quality import quality_rank
from src.utils import append_jsonl, clamp, stable_event_key
from src.logger import logger


class StrategyEngine:
    def __init__(self, config: AppConfig, data_dir: Path, state: AppStateStore) -> None:
        self.config = config
        self.data_dir = data_dir
        self.state = state
        self.market_data = MarketDataService(config, data_dir)
        self.risk = RiskManager(config)
        self.positions = PositionStore(data_dir / "positions.json")
        self.live_orders = LiveOrderStore(data_dir / "live_orders.json")
        self.trace_path = data_dir / "paper_decision_trace.jsonl"
        self.signal_log_path = data_dir / "strategy_signal_log.jsonl"
        self.official_signals = OfficialSignalStore(data_dir / Path(config.strategies.official_signal_file).name)
        # Lag signal: track first-seen Chainlink price per market_id as reference "start price"
        self._lag_start_prices: dict[str, float] = {}

    def _wallet_map(self, wallets: list[WalletMetrics]) -> dict[str, WalletMetrics]:
        return {wallet.wallet_address: wallet for wallet in wallets}

    def _stage_timeout_seconds(self, *, multiplier: float = 1.0, minimum: float = 5.0, maximum: float | None = None) -> float:
        base = float(self.config.live.bounded_execution_seconds or 20)
        if self.config.mode.value != "LIVE":
            base = max(base * 1.5, float(self.config.risk.stale_signal_seconds or 90))
        timeout_seconds = max(minimum, base * multiplier)
        if maximum is not None:
            timeout_seconds = min(timeout_seconds, maximum)
        return round(timeout_seconds, 3)

    def _max_wallet_follow_detections_per_cycle(self) -> int:
        if self.config.mode.value == "LIVE":
            live_wallet_count = max(int(self.config.env.operator_live_wallet_count or self.config.wallet_selection.approved_live_wallets), 1)
            return min(max(live_wallet_count * 2, 6), 12)
        paper_wallet_count = max(self.config.wallet_selection.approved_paper_wallets, 3)
        return min(max(paper_wallet_count * 6, 18), 48)

    def _watchlist_token_limit(self, max_detections: int) -> int:
        if self.config.mode.value == "LIVE":
            return min(max(max_detections, 4), 8)
        return min(max(max_detections * 2, 8), 24)

    async def _await_stage_timeout(
        self,
        *,
        stage_name: str,
        coro: object,
        timeout_seconds: float,
    ) -> object:
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)  # type: ignore[arg-type]
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"{stage_name} timed out after {timeout_seconds:.1f}s") from exc

    async def _run_supplemental_strategy(
        self,
        *,
        strategy_name: str,
        coro: object,
        cycle_ts: datetime,
    ) -> list[TradeDecision]:
        if self.config.mode.value == "LIVE":
            if strategy_name == "paired_binary_arb":
                timeout_seconds = self._stage_timeout_seconds(multiplier=0.5, minimum=4.0, maximum=12.0)
            else:
                timeout_seconds = self._stage_timeout_seconds(multiplier=0.4, minimum=3.0, maximum=8.0)
        else:
            timeout_seconds = self._stage_timeout_seconds(multiplier=0.75, minimum=5.0, maximum=15.0)
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)  # type: ignore[arg-type]
        except asyncio.TimeoutError:
            logger.warning(
                "Supplemental strategy timed out strategy_name={} timeout_seconds={}",
                strategy_name,
                timeout_seconds,
            )
            self._log_strategy_signal(
                strategy_name=strategy_name,
                signal_id=stable_event_key(strategy_name, "cycle-timeout", cycle_ts.isoformat()),
                market_id="",
                token_id="",
                final_action=DecisionAction.SKIP.value,
                reason_code="STAGE_TIMEOUT",
                extra={"cycle_ts": cycle_ts.isoformat(), "timeout_seconds": timeout_seconds},
            )
            return []

    def _effective_detection_category(
        self,
        detection: DetectionEvent,
        wallet: WalletMetrics,
        market_meta: dict[str, object] | None = None,
    ) -> str:
        explicit = str(detection.category or "").strip()
        if explicit and explicit != "unknown":
            return explicit
        if market_meta:
            resolved = str(market_meta.get("category") or "").strip()
            if resolved and resolved != "unknown":
                return resolved
        wallet_category = str(wallet.dominant_category or "").strip()
        if wallet_category:
            return wallet_category
        return "unknown"

    def _live_detection_priority_score(self, detection: DetectionEvent, wallet: WalletMetrics) -> float:
        category = self._effective_detection_category(detection, wallet)
        if category in self.config.live.selected_categories:
            category_score = 2.0
        elif category in {"", "unknown"}:
            category_score = 1.0
        else:
            category_score = 0.0

        price = float(detection.price or 0.0)
        preferred_min = float(self.config.live.preferred_entry_price_min)
        preferred_max = float(self.config.live.preferred_entry_price_max)
        if preferred_min <= price <= preferred_max:
            price_score = 1.0
        elif 0.05 <= price <= 0.90:
            price_score = 0.55
        else:
            price_score = 0.0

        freshness_score = clamp(
            1.0 - (float(detection.detection_latency_seconds or 0.0) / max(float(self.config.risk.stale_signal_seconds or 1), 1.0)),
            0.0,
            1.0,
        )
        notional_score = clamp(float(detection.notional or 0.0) / max(self._max_notional_for_mode(), 1.0), 0.0, 1.0)
        wallet_score = clamp(float(wallet.global_score or 0.0), 0.0, 1.0)
        return (
            category_score * 100.0
            + price_score * 10.0
            + freshness_score * 5.0
            + wallet_score * 3.0
            + notional_score
        )

    def _prioritize_detections(
        self,
        detections: list[DetectionEvent],
        wallet_map: dict[str, WalletMetrics],
    ) -> list[DetectionEvent]:
        if self.config.mode.value != "LIVE":
            return detections

        def _score(detection: DetectionEvent) -> float:
            wallet = wallet_map.get(detection.wallet_address)
            if wallet is None:
                return -1.0
            return self._live_detection_priority_score(detection, wallet)

        return sorted(detections, key=_score, reverse=True)

    def _watchlist_token_ids(
        self,
        detections: list[DetectionEvent],
        wallet_map: dict[str, WalletMetrics],
        *,
        max_detections: int,
    ) -> list[str]:
        token_ids: list[str] = []
        seen: set[str] = set()
        for detection in detections:
            token_id = str(detection.token_id or "").strip()
            if not token_id or token_id in seen:
                continue
            if self.config.mode.value == "LIVE":
                wallet = wallet_map.get(detection.wallet_address)
                if wallet is None:
                    continue
                category = self._effective_detection_category(detection, wallet)
                if category not in self.config.live.selected_categories and category not in {"", "unknown"}:
                    continue
            token_ids.append(token_id)
            seen.add(token_id)
            if len(token_ids) >= self._watchlist_token_limit(max_detections):
                break
        return token_ids

    def _count_paper_entries_last_hour(self, stored_positions: dict[str, list[dict]], now: datetime) -> int:
        cutoff = now.timestamp() - 3600.0
        count = 0
        for payload in stored_positions.get("paper", []):
            opened_at = payload.get("opened_at") or payload.get("entry_time")
            if not opened_at:
                continue
            try:
                opened_ts = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            if opened_ts >= cutoff:
                count += 1
        return count

    def _count_live_entries_last_hour(self, now: datetime) -> int:
        cutoff = now.timestamp() - 3600.0
        count = 0
        seen_order_ids: set[str] = set()
        for order in self.live_orders.load():
            order_key = order.local_order_id or order.exchange_order_id or order.client_order_id
            if order_key in seen_order_ids:
                continue
            seen_order_ids.add(order_key)
            if order.is_exit:
                continue
            if order.lifecycle_status == OrderLifecycleStatus.REJECTED:
                continue
            observed_at = order.submitted_at or order.created_at
            if observed_at.timestamp() >= cutoff:
                count += 1
        return count

    def _actual_entries_last_hour(self, stored_positions: dict[str, list[dict]], now: datetime) -> int:
        if self.config.mode.value == "LIVE":
            return self._count_live_entries_last_hour(now)
        return self._count_paper_entries_last_hour(stored_positions, now)

    def _detection_priority_key(self, detection: DetectionEvent) -> tuple[datetime, datetime, float]:
        return (
            detection.source_trade_timestamp,
            detection.local_detection_timestamp,
            float(detection.notional or 0.0),
        )

    def _dedupe_wallet_follow_detections(self, detections: list[DetectionEvent]) -> tuple[list[DetectionEvent], dict[str, int]]:
        kept_by_key: dict[tuple[str, str, str, str], DetectionEvent] = {}
        burst_sizes: dict[tuple[str, str, str, str], int] = {}
        for detection in detections:
            key = (
                detection.wallet_address,
                detection.market_id,
                detection.token_id,
                detection.side,
            )
            burst_sizes[key] = burst_sizes.get(key, 0) + 1
            current = kept_by_key.get(key)
            if current is None or self._detection_priority_key(detection) > self._detection_priority_key(current):
                kept_by_key[key] = detection
        deduped = sorted(kept_by_key.values(), key=self._detection_priority_key, reverse=True)
        burst_sizes_by_event = {
            detection.event_key: burst_sizes[key]
            for key, detection in kept_by_key.items()
        }
        return deduped, burst_sizes_by_event

    def _build_signal_consensus_maps(
        self,
        detections: list[DetectionEvent],
    ) -> tuple[
        dict[tuple[str, str, str], set[str]],
        dict[tuple[str, str, str], float],
        dict[tuple[str, str], dict[str, dict[str, object]]],
    ]:
        side_wallets: dict[tuple[str, str, str], set[str]] = {}
        side_notional: dict[tuple[str, str, str], float] = {}
        market_summary: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
        for detection in detections:
            side_key = (detection.market_id, detection.token_id, detection.side)
            market_key = (detection.market_id, detection.token_id)
            side_wallets.setdefault(side_key, set()).add(detection.wallet_address)
            side_notional[side_key] = round(side_notional.get(side_key, 0.0) + float(detection.notional or 0.0), 6)
            side_summary = market_summary.setdefault(market_key, {})
            summary = side_summary.setdefault(
                detection.side,
                {
                    "wallets": set(),
                    "notional": 0.0,
                },
            )
            summary["wallets"].add(detection.wallet_address)
            summary["notional"] = round(float(summary.get("notional", 0.0) or 0.0) + float(detection.notional or 0.0), 6)
        return side_wallets, side_notional, market_summary

    def _signal_consensus_context(
        self,
        detection: DetectionEvent,
        cluster_strength: float,
        side_wallets: dict[tuple[str, str, str], set[str]],
        side_notional: dict[tuple[str, str, str], float],
        market_summary: dict[tuple[str, str], dict[str, dict[str, object]]],
    ) -> dict[str, object]:
        side_key = (detection.market_id, detection.token_id, detection.side)
        market_key = (detection.market_id, detection.token_id)
        same_side_wallets = len(side_wallets.get(side_key, set()))
        same_side_notional = float(side_notional.get(side_key, 0.0) or 0.0)
        opposing_wallets = 0
        opposing_notional = 0.0
        for side, summary in market_summary.get(market_key, {}).items():
            if side == detection.side:
                continue
            opposing_wallets += len(summary.get("wallets", set()))
            opposing_notional += float(summary.get("notional", 0.0) or 0.0)
        total_wallets = same_side_wallets + opposing_wallets
        consensus_ratio = same_side_wallets / max(total_wallets, 1)
        notional_ratio = same_side_notional / max(same_side_notional + opposing_notional, 1e-6)
        thesis = "wallet_consensus" if same_side_wallets >= self.config.cluster.min_wallets else "single_wallet_copy"
        return {
            "same_side_wallets": same_side_wallets,
            "same_side_notional": round(same_side_notional, 6),
            "opposing_wallets": opposing_wallets,
            "opposing_notional": round(opposing_notional, 6),
            "consensus_ratio": round(consensus_ratio, 4),
            "notional_ratio": round(notional_ratio, 4),
            "cluster_strength": round(cluster_strength, 4),
            "selection_thesis": thesis,
        }

    async def process_detections(
        self,
        detections: list[DetectionEvent],
        approved_wallets: ApprovedWallets,
        wallets: list[WalletMetrics],
    ) -> list[TradeDecision]:
        wallet_map = self._wallet_map(wallets)
        effective_paper_wallets = approved_wallets.paper_wallets or approved_wallets.research_wallets[: self.config.wallet_selection.approved_paper_wallets]
        effective_live_wallets = approved_wallets.live_wallets or effective_paper_wallets[:1]
        detections, burst_sizes = self._dedupe_wallet_follow_detections(detections)
        detections = self._prioritize_detections(detections, wallet_map)
        max_detections = self._max_wallet_follow_detections_per_cycle()
        wallet_budget = len(effective_live_wallets) if self.config.mode.value == "LIVE" else len(effective_paper_wallets)
        max_detections = min(max_detections, max(wallet_budget * 2, 3))
        if len(detections) > max_detections:
            skipped_count = len(detections) - max_detections
            self._log_strategy_signal(
                strategy_name="strategy_engine",
                signal_id=stable_event_key("strategy_engine", "detection-backlog", datetime.now(timezone.utc).isoformat()),
                market_id="",
                token_id="",
                final_action=DecisionAction.SKIP.value,
                reason_code="DETECTION_BACKLOG_LIMIT",
                extra={
                    "total_detections": len(detections),
                    "processed_detections": max_detections,
                    "skipped_detections": skipped_count,
                    "mode": self.config.mode.value,
                },
            )
            detections = detections[:max_detections]
        clusters = cluster_detections(self.config, detections, self.data_dir)
        cluster_lookup = {(cluster.market_id, cluster.token_id): cluster for cluster in clusters}
        side_wallets, side_notional, market_summary = self._build_signal_consensus_maps(detections)
        decisions: list[TradeDecision] = []

        stored_positions = self.positions.load()
        cycle_now = datetime.now(timezone.utc)
        entries_last_hour = self._actual_entries_last_hour(stored_positions, cycle_now)
        category_fill_success, overall_fill_success = self._recent_live_fill_success()
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

        try:
            await self._await_stage_timeout(
                stage_name="refresh_markets",
                coro=self.market_data.refresh_markets(),
                timeout_seconds=self._stage_timeout_seconds(multiplier=0.3, minimum=4.0, maximum=10.0),
            )
        except RuntimeError as exc:
            logger.warning("Market refresh degraded reason={}", exc)
            self._log_strategy_signal(
                strategy_name="strategy_engine",
                signal_id=stable_event_key("strategy_engine", "market-refresh-timeout", cycle_now.isoformat()),
                market_id="",
                token_id="",
                final_action=DecisionAction.SKIP.value,
                reason_code="MARKET_REFRESH_TIMEOUT",
                extra={"cycle_ts": cycle_now.isoformat(), "error": str(exc)},
            )
        valid_watchlist_token_ids = self._watchlist_token_ids(
            detections,
            wallet_map,
            max_detections=max_detections,
        )
        ws_snapshots: dict[str, dict] = {}
        if valid_watchlist_token_ids:
            try:
                ws_snapshots = await self._await_stage_timeout(
                    stage_name="stream_watchlist",
                    coro=self.market_data.stream_watchlist(valid_watchlist_token_ids),
                    timeout_seconds=self._stage_timeout_seconds(multiplier=0.15, minimum=2.0, maximum=4.0),
                )
            except RuntimeError as exc:
                logger.warning("Market websocket snapshot degraded reason={}", exc)
                self._log_strategy_signal(
                    strategy_name="strategy_engine",
                    signal_id=stable_event_key("strategy_engine", "watchlist-timeout", cycle_now.isoformat()),
                    market_id="",
                    token_id="",
                    final_action=DecisionAction.SKIP.value,
                    reason_code="WATCHLIST_TIMEOUT",
                    extra={"cycle_ts": cycle_now.isoformat(), "error": str(exc)},
                )
        for detection in detections:
            wallet = wallet_map.get(detection.wallet_address)
            if not wallet:
                continue
            burst_size = burst_sizes.get(detection.event_key, 1)
            state_snapshot = self.state.read()
            decision_category = self._effective_detection_category(detection, wallet)
            discovery_state = str(state_snapshot.get("wallet_discovery_state", "UNKNOWN"))
            scoring_state = str(state_snapshot.get("wallet_scoring_state", "UNKNOWN"))
            decision_source_quality = min([wallet.source_quality, detection.source_quality], key=quality_rank)
            if (
                self.config.mode.value == "LIVE"
                and decision_category not in {"", "unknown"}
                and decision_category not in self.config.live.selected_categories
            ):
                skip = self._skip_decision(
                    detection,
                    wallet,
                    min(detection.notional * self._select_copy_fraction(wallet), self._max_notional_for_mode()),
                    "LIVE_CATEGORY_NOT_SELECTED",
                    "Category is not enabled for live trading.",
                )
                skip.context.update(
                    {
                        "category": decision_category,
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
                    market_meta = await self._await_stage_timeout(
                        stage_name="fetch_market_metadata",
                        coro=self.market_data.fetch_market_metadata(
                            detection.market_id,
                            detection.token_id,
                            detection.market_slug,
                            str(detection.market_metadata.get("outcome") or ""),
                        ),
                        timeout_seconds=self._stage_timeout_seconds(multiplier=0.18, minimum=2.5, maximum=5.0),
                    )
                except TypeError:
                    market_meta = await self._await_stage_timeout(
                        stage_name="fetch_market_metadata",
                        coro=self.market_data.fetch_market_metadata(
                            detection.market_id,
                            detection.token_id,
                        ),
                        timeout_seconds=self._stage_timeout_seconds(multiplier=0.18, minimum=2.5, maximum=5.0),
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
            decision_category = self._effective_detection_category(detection, wallet, market_meta)
            # Skip ultra-short-duration markets (e.g. "ETH Up or Down - 5 min" type markets).
            # These expire minutes after a wallet trade is detected, making them untradeable.
            _market_title_lc = str(market_meta.get("title") or detection.market_title or "").lower()
            _market_slug_lc = str(market_meta.get("slug") or detection.market_slug or "").lower()
            _short_duration_kws = ("updown-5m", "updown-1m", "updown-15m", "updown-30m",
                                   "intraday-price", "up-or-down", "up or down",
                                   "5-minute", "1-minute", "15-minute", "30-minute",
                                   "hourly-", "-hourly", "5min", "1min",
                                   "price-up", "price-down", "price up", "price down")
            if any(kw in _market_title_lc or kw in _market_slug_lc for kw in _short_duration_kws):
                skip = self._skip_decision(
                    detection,
                    wallet,
                    min(detection.notional * self._select_copy_fraction(wallet), self._max_notional_for_mode()),
                    "SHORT_DURATION_MARKET",
                    f"Signal skipped: ultra-short-duration market not suitable for copy-trading ({detection.market_title!r}).",
                )
                skip.context.update({"market_meta": market_meta, "discovery_state": discovery_state})
                self._write_decision_trace(detection, skip, False, [], discovery_state, scoring_state, decision_source_quality)
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
                tradability = await self._await_stage_timeout(
                    stage_name="get_tradability",
                    coro=self.market_data.get_tradability(resolved_market_id, resolved_token_id),
                    timeout_seconds=self._stage_timeout_seconds(multiplier=0.20, minimum=3.0, maximum=6.0),
                )
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
                orderbook = await self._await_stage_timeout(
                    stage_name="fetch_orderbook",
                    coro=self.market_data.fetch_orderbook(resolved_token_id),
                    timeout_seconds=self._stage_timeout_seconds(multiplier=0.06, minimum=0.8, maximum=1.5),
                )
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
            consensus_context = self._signal_consensus_context(
                detection,
                cluster.cluster_strength if cluster else 0.0,
                side_wallets,
                side_notional,
                market_summary,
            )
            copy_fraction = self._select_copy_fraction(wallet)
            scaled_notional = min(detection.notional * copy_fraction, self._max_notional_for_mode())

            best_decision = self._skip_decision(detection, wallet, scaled_notional, "NO_VALID_ENTRY_STYLE", "No entry style passed all checks.")
            style_evaluations: list[dict[str, object]] = []
            for entry_style in self.config.entry_styles.compare:
                fill = self._estimate_entry_fill(
                    detection=detection,
                    orderbook=orderbook,
                    target_notional=scaled_notional,
                    entry_style=entry_style,
                )
                drift_pct = abs(fill.executable_price - detection.price) / max(detection.price, 1e-6)
                hybrid_modifier = self._hybrid_confirmation_modifier(detection, ws_snapshots.get(detection.token_id, {}))
                entry_style_allowed = self._entry_style_allowed(entry_style, cluster_confirmed)
                stale_signal_limit = self._stale_signal_threshold_seconds(
                    strategy_name="wallet_follow",
                    entry_style=entry_style,
                    category=decision_category,
                    source_quality=decision_source_quality,
                )
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
                    category=decision_category,
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
                    entries_last_hour_override=entries_last_hour,
                    stale_signal_seconds_override=stale_signal_limit,
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
                live_fillability: dict[str, object] | None = None
                live_signal_quality: dict[str, object] | None = None
                if self.config.mode.value == "LIVE" and risk.reason_code in {"OK", "ENTRY_DRIFT", "WIDE_SPREAD", "FILLABILITY", "SLIPPAGE"}:
                    live_fillability = self._live_fillability_assessment(
                        category=decision_category,
                        entry_style=entry_style,
                        detection=detection,
                        orderbook=orderbook,
                        fill=fill,
                        tradability=tradability,
                        target_notional=scaled_notional,
                        drift_pct=drift_pct,
                        stale_signal_limit_seconds=stale_signal_limit,
                        category_fill_success=category_fill_success,
                        overall_fill_success=overall_fill_success,
                    )
                    if not bool(live_fillability.get("passed")):
                        risk = risk.model_copy(
                            update={
                                "allowed": False,
                                "reason_code": str(live_fillability.get("reason_code") or "LIVE_FILLABILITY_SCORE"),
                                "human_readable_reason": str(live_fillability.get("reason") or "Live fillability score below threshold."),
                                "context": {**risk.context, **live_fillability},
                            }
                        )
                if self.config.mode.value == "LIVE" and risk.allowed:
                    live_signal_quality = self._live_signal_quality_assessment(
                        category=decision_category,
                        entry_style=entry_style,
                        detection=detection,
                        wallet=wallet,
                        executable_price=fill.executable_price,
                        source_quality=decision_source_quality,
                        stale_signal_limit_seconds=stale_signal_limit,
                        consensus_context=consensus_context,
                        burst_size=burst_size,
                    )
                    if not bool(live_signal_quality.get("passed")):
                        risk = risk.model_copy(
                            update={
                                "allowed": False,
                                "reason_code": str(live_signal_quality.get("reason_code") or "LOW_SIGNAL_QUALITY"),
                                "human_readable_reason": str(live_signal_quality.get("reason") or "Live signal quality too weak."),
                                "context": {**risk.context, **live_signal_quality},
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
                        "live_fillability_score": None if live_fillability is None else live_fillability.get("score"),
                        "signal_quality_score": None if live_signal_quality is None else live_signal_quality.get("score"),
                        "stale_signal_limit_seconds": stale_signal_limit,
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
                        "signal_consensus": consensus_context,
                        "hybrid_modifier": hybrid_modifier,
                        "burst_size": burst_size,
                        "ws_snapshot": ws_snapshots.get(detection.token_id, {}),
                        "style_evaluations": style_evaluations,
                        "live_fillability_score": None if live_fillability is None else live_fillability.get("score"),
                        "live_fillability": live_fillability or {},
                        "signal_quality_score": None if live_signal_quality is None else live_signal_quality.get("score"),
                        "signal_quality": live_signal_quality or {},
                        "stale_signal_limit_seconds": stale_signal_limit,
                        "wallet_global_score": wallet.global_score,
                        "discovery_state": discovery_state,
                        "scoring_state": scoring_state,
                        "source_quality": decision_source_quality.value,
                        "selection_thesis": str(consensus_context.get("selection_thesis") or "wallet_follow"),
                    },
                )
                if self._decision_rank(decision, wallet, hybrid_modifier) > self._decision_rank(
                    best_decision,
                    wallet,
                    float(best_decision.context.get("hybrid_modifier", 0.0) or 0.0),
                ):
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
            best_decision.context["selection_score"] = self._decision_rank(
                best_decision,
                wallet,
                float(best_decision.context.get("hybrid_modifier", 0.0) or 0.0),
            )
            self._write_decision_trace(detection, best_decision, cluster_confirmed, style_evaluations, discovery_state, scoring_state, decision_source_quality)
            decisions.append(best_decision)
        decisions.extend(
            await self._run_supplemental_strategy(
                strategy_name="event_driven_official",
                coro=self._generate_event_driven_official_decisions(
                    stored_positions=active_positions,
                    total_exposure=total_exposure,
                    active_market_exposure=active_market_exposure,
                    active_wallet_exposure=active_wallet_exposure,
                ),
                cycle_ts=cycle_now,
            )
        )
        decisions.extend(
            await self._run_supplemental_strategy(
                strategy_name="paired_binary_arb",
                coro=self._generate_paired_binary_arb_decisions(
                    stored_positions=active_positions,
                    total_exposure=total_exposure,
                    active_market_exposure=active_market_exposure,
                    active_wallet_exposure=active_wallet_exposure,
                ),
                cycle_ts=cycle_now,
            )
        )
        decisions.extend(
            await self._run_supplemental_strategy(
                strategy_name="correlation_dislocation",
                coro=self._generate_correlation_dislocation_decisions(
                    stored_positions=active_positions,
                    total_exposure=total_exposure,
                    active_market_exposure=active_market_exposure,
                    active_wallet_exposure=active_wallet_exposure,
                ),
                cycle_ts=cycle_now,
            )
        )
        decisions.extend(
            await self._run_supplemental_strategy(
                strategy_name="resolution_window",
                coro=self._generate_resolution_window_decisions(
                    stored_positions=active_positions,
                    total_exposure=total_exposure,
                    active_market_exposure=active_market_exposure,
                    active_wallet_exposure=active_wallet_exposure,
                ),
                cycle_ts=cycle_now,
            )
        )
        decisions.extend(
            await self._run_supplemental_strategy(
                strategy_name="lag_signal",
                coro=self._generate_lag_signal_decisions(
                    stored_positions=active_positions,
                    total_exposure=total_exposure,
                    active_market_exposure=active_market_exposure,
                    active_wallet_exposure=active_wallet_exposure,
                ),
                cycle_ts=cycle_now,
            )
        )
        if self.config.mode.value == "LIVE":
            decisions.sort(key=self._execution_priority, reverse=True)
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

    def _recent_live_fill_success(self) -> tuple[dict[str, float], float]:
        cutoff = datetime.now(timezone.utc).timestamp() - 86_400.0
        attempts_by_category: dict[str, int] = {}
        fills_by_category: dict[str, int] = {}
        total_attempts = 0
        total_fills = 0
        for order in self.live_orders.load():
            if order.is_exit:
                continue
            observed_at = order.submitted_at or order.created_at
            if observed_at.timestamp() < cutoff:
                continue
            category = str(order.category or "unknown")
            attempts_by_category[category] = attempts_by_category.get(category, 0) + 1
            total_attempts += 1
            if order.filled_size > 0 or order.lifecycle_status in {OrderLifecycleStatus.PARTIALLY_FILLED, OrderLifecycleStatus.FILLED}:
                fills_by_category[category] = fills_by_category.get(category, 0) + 1
                total_fills += 1
        ratios = {
            category: round(fills_by_category.get(category, 0) / max(attempts, 1), 4)
            for category, attempts in attempts_by_category.items()
        }
        overall = round(total_fills / max(total_attempts, 1), 4) if total_attempts else 0.55
        return ratios, overall

    def _live_category_priority(self, category: str) -> float:
        normalized = str(category or "unknown").strip().lower()
        if normalized in {"politics", "macro / economics", "entertainment / pop culture", "regulatory / legal"}:
            return 1.0
        if normalized == "geopolitics":
            return 0.85
        if normalized == "crypto price":
            return 0.4
        if normalized == "sports":
            return 0.15
        return 0.45

    def _paired_binary_scan_limit(self) -> int:
        if self.config.mode.value == "LIVE":
            return max(int(self.config.strategies.paired_binary_max_candidates_per_cycle) * 4, 8)
        return max(int(self.config.strategies.paired_binary_max_candidates_per_cycle) * 8, 24)

    def _paired_binary_rough_priority(
        self,
        yes_market: MarketInfo,
        no_market: MarketInfo,
        category: str,
    ) -> float:
        liquidity_score = clamp(min(float(yes_market.liquidity or 0.0), float(no_market.liquidity or 0.0)) / 5000.0, 0.0, 1.0)
        volume_score = clamp(min(float(yes_market.volume or 0.0), float(no_market.volume or 0.0)) / 10000.0, 0.0, 1.0)
        return self._live_category_priority(category) * 10.0 + liquidity_score * 3.0 + volume_score

    async def _evaluate_paired_binary_candidate(
        self,
        *,
        market_id: str,
        yes_market: MarketInfo,
        no_market: MarketInfo,
        category: str,
        bundle_budget: float,
        min_leg: float,
        max_leg: float,
        now: datetime,
    ) -> dict[str, object] | None:
        try:
            yes_orderbook, no_orderbook = await asyncio.gather(
                self.market_data.fetch_orderbook(yes_market.token_id),
                self.market_data.fetch_orderbook(no_market.token_id),
            )
        except Exception:
            return None
        if not yes_orderbook.asks or not no_orderbook.asks:
            return None
        yes_ask = float(yes_orderbook.asks[0].price or 0.0)
        no_ask = float(no_orderbook.asks[0].price or 0.0)
        yes_size = float(yes_orderbook.asks[0].size or 0.0)
        no_size = float(no_orderbook.asks[0].size or 0.0)
        if yes_ask <= 0 or no_ask <= 0:
            return None
        if not (min_leg <= yes_ask <= max_leg and min_leg <= no_ask <= max_leg):
            return None
        yes_age_seconds = max((now - yes_orderbook.timestamp).total_seconds(), 0.0)
        no_age_seconds = max((now - no_orderbook.timestamp).total_seconds(), 0.0)
        if max(yes_age_seconds, no_age_seconds) > float(self.config.risk.stale_market_data_seconds):
            return None
        yes_depth = yes_ask * yes_size
        no_depth = no_ask * no_size
        min_depth = min(yes_depth, no_depth)
        if min_depth < max(self.config.risk.min_orderbook_depth_usd * 0.5, 10.0):
            return None
        bundle_sum_ask = yes_ask + no_ask
        gross_edge_pct = round(1.0 - bundle_sum_ask, 6)
        if gross_edge_pct < float(self.config.strategies.paired_binary_min_edge_pct):
            return None
        fee_yes = self._paired_binary_fee_amount(yes_ask, category)
        fee_no = self._paired_binary_fee_amount(no_ask, category)
        slippage_buffer = float(self.config.strategies.paired_binary_slippage_buffer_pct)
        net_edge_pct = round(gross_edge_pct - fee_yes - fee_no - slippage_buffer, 6)
        if net_edge_pct < float(self.config.strategies.paired_binary_min_net_edge_pct):
            return None
        max_best_level_fraction = clamp(float(self.config.strategies.paired_binary_max_best_level_fraction), 0.01, 1.0)
        max_bundle_shares_from_depth = min(yes_size, no_size) * max_best_level_fraction
        max_bundle_shares_from_budget = bundle_budget / max(bundle_sum_ask, 1e-6)
        min_bundle_shares = self._paired_binary_min_bundle_shares(yes_ask, no_ask)
        bundle_shares = round(min(max_bundle_shares_from_depth, max_bundle_shares_from_budget), 6)
        if bundle_shares < min_bundle_shares:
            return None
        yes_notional = round(bundle_shares * yes_ask, 6)
        no_notional = round(bundle_shares * no_ask, 6)
        return {
            "market_id": market_id,
            "yes_market": yes_market,
            "no_market": no_market,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "bundle_sum_ask": round(bundle_sum_ask, 6),
            "gross_edge_pct": gross_edge_pct,
            "net_edge_pct": net_edge_pct,
            "fee_yes": fee_yes,
            "fee_no": fee_no,
            "slippage_buffer": slippage_buffer,
            "min_depth": round(min_depth, 6),
            "bundle_shares": bundle_shares,
            "min_bundle_shares": min_bundle_shares,
            "yes_notional": yes_notional,
            "no_notional": no_notional,
            "category": category,
            "score": self._paired_binary_candidate_score(
                net_edge_pct=net_edge_pct,
                min_depth_usd=min_depth,
                yes_ask=yes_ask,
                no_ask=no_ask,
                category=category,
            ),
        }

    def _stale_signal_threshold_seconds(
        self,
        *,
        strategy_name: str,
        entry_style: EntryStyle,
        category: str,
        source_quality: SourceQuality,
    ) -> float | None:
        if self.config.mode.value != "LIVE":
            return None
        base = float(self.config.risk.stale_signal_seconds)
        if strategy_name == "event_driven_official":
            return max(base, float(self.config.strategies.official_signal_max_age_minutes) * 60.0)
        if (
            strategy_name == "wallet_follow"
            and entry_style in {EntryStyle.PASSIVE_LIMIT, EntryStyle.POST_ONLY_MAKER}
            and category in self.config.live.selected_categories
            and source_quality == SourceQuality.REAL_PUBLIC_DATA
        ):
            return max(base, float(self.config.live.passive_signal_ttl_seconds))
        return None

    async def _direct_resolve_official_signal_market(self, row: dict[str, object]) -> tuple[MarketInfo | None, str]:
        market_id = str(row.get("market_id") or "").strip()
        token_id = str(row.get("token_id") or "").strip()
        market_slug = str(row.get("market_slug") or row.get("slug") or "").strip()
        outcome = str(row.get("outcome") or "").strip()
        if not any([market_id, token_id, market_slug]):
            return None, "missing_mapping_hints"
        try:
            market_meta = await self.market_data.fetch_market_metadata(
                market_id,
                token_id,
                market_slug,
                outcome,
            )
        except RuntimeError:
            return None, "no_market_match"
        resolved_market_id = str(market_meta.get("market_id") or market_id).strip()
        resolved_token_id = str(market_meta.get("token_id") or token_id).strip()
        if not resolved_market_id or not resolved_token_id:
            return None, "missing_direct_market_lookup"
        market = MarketInfo(
                market_id=resolved_market_id,
                token_id=resolved_token_id,
                title=str(market_meta.get("title") or row.get("title") or resolved_market_id),
                slug=str(market_meta.get("slug") or market_slug),
                category=str(market_meta.get("category") or row.get("category") or "unknown"),
                active=bool(market_meta.get("active", True)),
                closed=bool(market_meta.get("closed", False)),
                liquidity=float(market_meta.get("liquidity", 0.0) or 0.0),
            )
        self.market_data.market_cache[market.market_id] = market
        self.market_data.token_cache[market.token_id] = market
        return (market, "direct_market_lookup")

    def _live_fillability_assessment(
        self,
        *,
        category: str,
        entry_style: EntryStyle,
        detection: DetectionEvent,
        orderbook: OrderbookSnapshot,
        fill: FillEstimate,
        tradability: dict[str, object],
        target_notional: float,
        drift_pct: float,
        stale_signal_limit_seconds: float | None,
        category_fill_success: dict[str, float],
        overall_fill_success: float,
    ) -> dict[str, object]:
        executable_price = float(fill.executable_price or 0.0)
        hard_floor = float(self.config.live.hard_skip_price_floor)
        hard_ceiling = float(self.config.live.hard_skip_price_ceiling)
        if executable_price <= hard_floor or executable_price >= hard_ceiling:
            return {
                "passed": False,
                "score": 0.0,
                "reason_code": "EXTREME_PRICE_BOOK",
                "reason": "Executable live entry sits in an extreme price band.",
                "executable_price": executable_price,
            }

        preferred_min = float(self.config.live.preferred_entry_price_min)
        preferred_max = float(self.config.live.preferred_entry_price_max)
        if preferred_min <= executable_price <= preferred_max:
            price_band_score = 1.0
        elif 0.05 <= executable_price <= 0.90:
            price_band_score = 0.65
        else:
            price_band_score = 0.25

        best_ask_depth_usd = 0.0
        if orderbook.asks:
            best_ask_depth_usd = orderbook.asks[0].price * orderbook.asks[0].size
        depth_score = clamp(best_ask_depth_usd / max(target_notional * 3.0, 1.0), 0.0, 1.0)
        spread_score = clamp(1.0 - (fill.spread_pct / max(self.config.risk.max_spread_pct, 1e-6)), 0.0, 1.0)
        drift_score = clamp(1.0 - (drift_pct / max(self.config.risk.max_entry_drift_pct, 1e-6)), 0.0, 1.0)
        freshness_limit = float(stale_signal_limit_seconds or self.config.risk.stale_signal_seconds or 1.0)
        freshness_score = clamp(1.0 - (float(detection.detection_latency_seconds or 0.0) / max(freshness_limit, 1.0)), 0.0, 1.0)
        tradability_confidence = 0.65 if bool(tradability.get("derived_from_orderbook")) else 1.0
        recent_fill_success = category_fill_success.get(category, overall_fill_success)
        category_priority = self._live_category_priority(category)
        style_bias = 1.0 if entry_style in {EntryStyle.PASSIVE_LIMIT, EntryStyle.POST_ONLY_MAKER} else 0.35
        score = clamp(
            price_band_score * 0.22
            + depth_score * 0.18
            + spread_score * 0.14
            + drift_score * 0.14
            + freshness_score * 0.12
            + tradability_confidence * 0.10
            + recent_fill_success * 0.06
            + category_priority * 0.02
            + style_bias * 0.02,
            0.0,
            1.0,
        )
        return {
            "passed": score >= float(self.config.live.minimum_fillability_score),
            "score": round(score, 4),
            "reason_code": "LIVE_FILLABILITY_SCORE",
            "reason": "Live fillability score below threshold.",
            "price_band_score": round(price_band_score, 4),
            "depth_score": round(depth_score, 4),
            "spread_score": round(spread_score, 4),
            "drift_score": round(drift_score, 4),
            "freshness_score": round(freshness_score, 4),
            "stale_signal_limit_seconds": round(freshness_limit, 4),
            "tradability_confidence": tradability_confidence,
            "recent_fill_success": round(recent_fill_success, 4),
            "category_priority": round(category_priority, 4),
            "style_bias": round(style_bias, 4),
        }

    def _live_signal_quality_assessment(
        self,
        *,
        category: str,
        entry_style: EntryStyle,
        detection: DetectionEvent,
        wallet: WalletMetrics,
        executable_price: float,
        source_quality: SourceQuality,
        stale_signal_limit_seconds: float | None,
        consensus_context: dict[str, object],
        burst_size: int,
    ) -> dict[str, object]:
        same_side_wallets = int(consensus_context.get("same_side_wallets", 1) or 1)
        opposing_wallets = int(consensus_context.get("opposing_wallets", 0) or 0)
        consensus_ratio = float(consensus_context.get("consensus_ratio", 1.0) or 1.0)
        notional_ratio = float(consensus_context.get("notional_ratio", 1.0) or 1.0)
        cluster_strength = float(consensus_context.get("cluster_strength", 0.0) or 0.0)
        if opposing_wallets > same_side_wallets:
            return {
                "passed": False,
                "score": 0.0,
                "reason_code": "SIGNAL_CONFLICT",
                "reason": "Recent wallet flow is conflicting on this market.",
                "same_side_wallets": same_side_wallets,
                "opposing_wallets": opposing_wallets,
                "consensus_ratio": round(consensus_ratio, 4),
                "notional_ratio": round(notional_ratio, 4),
            }

        price = float(executable_price or detection.price or 0.0)
        if self.config.live.preferred_entry_price_min <= price <= self.config.live.preferred_entry_price_max:
            price_band_score = 1.0
        elif 0.05 <= price <= 0.90:
            price_band_score = 0.7
        else:
            price_band_score = 0.25

        freshness_limit = float(stale_signal_limit_seconds or self.config.risk.stale_signal_seconds or 1.0)
        freshness_score = clamp(1.0 - (float(detection.detection_latency_seconds or 0.0) / max(freshness_limit, 1.0)), 0.0, 1.0)
        source_quality_score = {
            SourceQuality.REAL_PUBLIC_DATA: 1.0,
            SourceQuality.DEGRADED_PUBLIC_DATA: 0.65,
            SourceQuality.SYNTHETIC_FALLBACK: 0.0,
        }.get(source_quality, 0.5)
        category_priority = self._live_category_priority(category)
        style_score = 1.0 if entry_style in {EntryStyle.PASSIVE_LIMIT, EntryStyle.POST_ONLY_MAKER} else 0.35
        wallet_score = clamp(float(wallet.global_score or 0.0), 0.0, 1.0)
        burst_penalty = max(burst_size - 1, 0) * 0.04
        opposing_share_penalty = clamp(opposing_wallets / max(same_side_wallets + opposing_wallets, 1), 0.0, 1.0) * 0.12
        score = clamp(
            wallet_score * 0.2
            + consensus_ratio * 0.22
            + notional_ratio * 0.14
            + max(cluster_strength, 0.35 if same_side_wallets >= 2 else 0.0) * 0.16
            + freshness_score * 0.14
            + category_priority * 0.1
            + source_quality_score * 0.08
            + price_band_score * 0.06
            + style_score * 0.04
            - burst_penalty
            - opposing_share_penalty,
            0.0,
            1.0,
        )
        return {
            "passed": score >= float(self.config.live.minimum_signal_quality_score),
            "score": round(score, 4),
            "reason_code": "LOW_SIGNAL_QUALITY",
            "reason": "Live signal quality score below threshold.",
            "same_side_wallets": same_side_wallets,
            "opposing_wallets": opposing_wallets,
            "consensus_ratio": round(consensus_ratio, 4),
            "notional_ratio": round(notional_ratio, 4),
            "cluster_strength": round(cluster_strength, 4),
            "wallet_score": round(wallet_score, 4),
            "freshness_score": round(freshness_score, 4),
            "category_priority": round(category_priority, 4),
            "source_quality_score": round(source_quality_score, 4),
            "price_band_score": round(price_band_score, 4),
            "style_score": round(style_score, 4),
            "selection_thesis": str(consensus_context.get("selection_thesis") or "wallet_follow"),
        }

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

    def _estimate_entry_fill(
        self,
        detection: DetectionEvent,
        orderbook: OrderbookSnapshot,
        target_notional: float,
        entry_style: EntryStyle,
    ) -> FillEstimate:
        if entry_style == EntryStyle.FOLLOW_TAKER:
            return estimate_fill(orderbook, target_notional, self.config.risk.max_slippage_pct)
        return self._estimate_resting_fill(detection, orderbook, target_notional, entry_style)

    def _estimate_resting_fill(
        self,
        detection: DetectionEvent,
        orderbook: OrderbookSnapshot,
        target_notional: float,
        entry_style: EntryStyle,
    ) -> FillEstimate:
        best_bid = orderbook.bids[0].price if orderbook.bids else 0.0
        best_ask = orderbook.asks[0].price if orderbook.asks else 0.0
        if best_bid <= 0 and best_ask <= 0:
            return FillEstimate(
                fillable=False,
                executable_price=0.0,
                spread_pct=1.0,
                slippage_pct=1.0,
                filled_notional=0.0,
                reason="No usable orderbook levels available for resting entry.",
            )

        tick_size = 0.01
        source_price = round(clamp(detection.price, tick_size, 0.99), 6)
        limit_price = source_price
        if entry_style == EntryStyle.POST_ONLY_MAKER and best_ask > 0:
            limit_price = min(limit_price, max(round(best_ask - tick_size, 6), tick_size))
        elif entry_style == EntryStyle.PASSIVE_LIMIT and best_ask > 0 and limit_price > best_ask:
            limit_price = round(best_ask, 6)

        executable_price = round(clamp(limit_price, tick_size, 0.99), 6)
        estimated_size = round(target_notional / max(executable_price, 1e-6), 6)
        visible_depth = sum(level.price * level.size for level in (orderbook.bids or orderbook.asks))
        price_gap_to_ask = max(best_ask - executable_price, 0.0) if best_ask > 0 else 0.0
        reason = "Resting limit entry priced from source trade."
        if entry_style == EntryStyle.POST_ONLY_MAKER:
            reason = "Post-only maker entry priced below top ask."
        return FillEstimate(
            fillable=True,
            executable_price=executable_price,
            spread_pct=0.0 if price_gap_to_ask > 0 else round(max(best_ask - best_bid, 0.0) / max(best_ask, 1e-6), 6),
            slippage_pct=0.0,
            filled_notional=round(target_notional, 6),
            reason=reason,
            depth_consumed_pct=round(target_notional / max(visible_depth, target_notional, 1e-6), 6),
            max_size_within_slippage=estimated_size,
        )

    def _decision_rank(self, decision: TradeDecision, wallet: WalletMetrics, hybrid_modifier: float) -> float:
        if not decision.allowed:
            fill = decision.context.get("fill", {})
            risk_context = decision.context.get("risk_context", {})
            drift = float(risk_context.get("entry_drift_pct", 0.0) or 0.0)
            spread = float(fill.get("spread_pct", 0.0) or 0.0)
            slippage = float(fill.get("slippage_pct", 0.0) or 0.0)
            reason_bonus = {
                "EXTREME_PRICE_BOOK": 95.0,
                "ENTRY_DRIFT": 90.0,
                "LIVE_FILLABILITY_SCORE": 85.0,
                "WIDE_SPREAD": 80.0,
                "FILLABILITY": 70.0,
                "SLIPPAGE": 60.0,
                "CATEGORY_BLOCKED": 50.0,
                "ENTRY_STYLE_BLOCKED": 40.0,
                "NO_VALID_ENTRY_STYLE": -500.0,
            }.get(decision.reason_code, 10.0)
            return -1000.0 + reason_bonus - drift - spread - slippage
        fill = decision.context.get("fill", {})
        slippage = float(fill.get("slippage_pct", 1.0))
        burst_size = int(decision.context.get("burst_size", 1) or 1)
        price = float(decision.executable_price or 0.0)
        fillability_score = float(decision.context.get("live_fillability_score", 0.5) or 0.5)
        signal_quality_score = float(decision.context.get("signal_quality_score", 0.5) or 0.5)
        freshness_score = float((decision.context.get("live_fillability") or {}).get("freshness_score", 0.5) or 0.5)
        category_priority = self._live_category_priority(decision.category)
        confidence_score = float(decision.context.get("confidence_score", 0.0) or 0.0)
        selection_thesis = str(decision.context.get("selection_thesis") or "")
        strategy_bonus = 0.0
        if decision.strategy_name == "event_driven_official":
            strategy_bonus = 0.12
        elif decision.strategy_name == "paired_binary_arb":
            strategy_bonus = 0.14
        elif selection_thesis == "wallet_consensus":
            strategy_bonus = 0.06
        extreme_price_penalty = 0.0
        if price >= 0.98 or price <= 0.02:
            extreme_price_penalty = 0.6
        elif price >= 0.95 or price <= 0.05:
            extreme_price_penalty = 0.35
        elif not (self.config.live.preferred_entry_price_min <= price <= self.config.live.preferred_entry_price_max):
            extreme_price_penalty = 0.1
        burst_penalty = max(burst_size - 1, 0) * 0.05
        return round(
            wallet.global_score * 0.55
            + fillability_score * 0.55
            + signal_quality_score * 0.3
            + freshness_score * 0.15
            + category_priority * 0.2
            + confidence_score * 0.15
            + strategy_bonus
            + (0.15 if decision.cluster_confirmed else 0.0)
            + hybrid_modifier
            - slippage
            - extreme_price_penalty
            - burst_penalty
            - wallet.hedge_suspicion_score * 0.4,
            6,
        )

    def _execution_priority(self, decision: TradeDecision) -> tuple[int, float]:
        action_priority = 1 if decision.allowed and decision.action == DecisionAction.LIVE_COPY else 0
        wallet = self._strategy_wallet(decision.strategy_name)
        wallet.global_score = float(decision.context.get("wallet_global_score", wallet.global_score) or wallet.global_score)
        score = self._decision_rank(
            decision,
            wallet,
            float(decision.context.get("hybrid_modifier", 0.0) or 0.0),
        )
        return (action_priority, score)

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

    def _position_thesis_type(self, position: dict[str, object]) -> str:
        return str(position.get("thesis_type") or "directional")

    def _binary_outcome_side(self, title: str) -> str:
        normalized = str(title or "").strip().lower()
        if normalized.endswith("[yes]"):
            return "YES"
        if normalized.endswith("[no]"):
            return "NO"
        return ""

    def _is_crypto_category(self, category: str) -> bool:
        normalized = str(category or "").strip().lower()
        return normalized == "crypto price" or "crypto" in normalized

    def _paired_binary_fee_amount(self, price: float, category: str) -> float:
        if not self._is_crypto_category(category):
            return 0.0
        clipped_price = clamp(float(price or 0.0), 0.01, 0.99)
        fee_rate = 0.25
        exponent = 2
        return round(clipped_price * fee_rate * pow(clipped_price * (1.0 - clipped_price), exponent), 6)

    def _paired_binary_min_bundle_shares(self, yes_ask: float, no_ask: float) -> float:
        min_shares = 1.0
        if self.config.mode.value != "LIVE":
            return min_shares
        min_shares = max(min_shares, float(self.config.live.minimum_order_size_shares or 0.0))
        min_notional = float(self.config.live.minimum_order_notional_usd or 0.0)
        if min_notional > 0:
            min_shares = max(
                min_shares,
                min_notional / max(yes_ask, 1e-6),
                min_notional / max(no_ask, 1e-6),
            )
        return round(min_shares, 6)

    def _paired_binary_candidate_score(
        self,
        *,
        net_edge_pct: float,
        min_depth_usd: float,
        yes_ask: float,
        no_ask: float,
        category: str,
    ) -> float:
        min_net_edge = max(float(self.config.strategies.paired_binary_min_net_edge_pct), 0.001)
        edge_score = clamp(net_edge_pct / max(min_net_edge * 2.0, 0.01), 0.0, 1.0)
        depth_score = clamp(min_depth_usd / 50.0, 0.0, 1.0)
        leg_balance_score = clamp(1.0 - abs(yes_ask - no_ask), 0.0, 1.0)
        category_priority = self._live_category_priority(category) if self.config.mode.value == "LIVE" else 0.5
        return round(
            edge_score * 0.55
            + depth_score * 0.2
            + leg_balance_score * 0.15
            + category_priority * 0.1,
            6,
        )

    def _has_conflicting_market_position(
        self,
        active_positions: list[dict],
        *,
        market_id: str,
        token_id: str,
        thesis_type: str = "directional",
        bundle_id: str = "",
    ) -> bool:
        for position in active_positions:
            if position.get("closed"):
                continue
            if position.get("market_id") != market_id:
                continue
            existing_token_id = str(position.get("token_id") or "")
            existing_thesis = self._position_thesis_type(position)
            existing_bundle_id = str(position.get("bundle_id") or "")
            if thesis_type == "paired_arb":
                if existing_bundle_id == bundle_id and existing_token_id == token_id:
                    return True
                if existing_bundle_id == bundle_id and existing_token_id != token_id:
                    continue
                return True
            if existing_thesis == "paired_arb":
                return True
            if existing_token_id == token_id:
                return True
            return True
        return False

    def _has_conflicting_position(self, active_positions: list[dict], detection: DetectionEvent) -> bool:
        return self._has_conflicting_market_position(
            active_positions,
            market_id=detection.market_id,
            token_id=detection.token_id,
            thesis_type="directional",
        )

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
                "signal_quality_state": "STRONG" if float(decision.context.get("signal_quality_score", 0.0) or 0.0) >= float(self.config.live.minimum_signal_quality_score) else "WEAK",
                "risk_allowed": decision.allowed,
                "risk_reason_code": decision.reason_code,
                "risk_reason": decision.human_readable_reason,
                "human_readable_reason": decision.human_readable_reason,
                "risk_context": risk_context,
                "final_action": decision.action.value,
                "reason_code": decision.reason_code,
                "entry_style": decision.entry_style.value,
                "thesis_type": decision.thesis_type,
                "bundle_id": decision.bundle_id,
                "bundle_role": decision.bundle_role,
                "paired_token_id": decision.paired_token_id,
                "style_evaluations": style_evaluations,
                "scaled_notional": decision.scaled_notional,
            },
        )

    def _strategy_wallet(self, strategy_name: str) -> WalletMetrics:
        global_score = 0.75
        conviction_score = 0.8
        copyability_score = 0.8
        if strategy_name == "event_driven_official":
            global_score = 0.88
            conviction_score = 0.9
            copyability_score = 0.86
        elif strategy_name == "paired_binary_arb":
            global_score = 0.92
            conviction_score = 0.94
            copyability_score = 0.9
        elif strategy_name == "resolution_window":
            global_score = 0.66
            conviction_score = 0.72
            copyability_score = 0.68
        elif strategy_name == "correlation_dislocation":
            global_score = 0.62
            conviction_score = 0.68
            copyability_score = 0.64
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
            conviction_score=conviction_score,
            market_concentration=0.2,
            category_concentration=0.5,
            holding_time_estimate_hours=6.0,
            drawdown_proxy=0.1,
            copyability_score=copyability_score,
            low_velocity_score=0.8,
            delay_5s=0.8,
            delay_15s=0.8,
            delay_30s=0.8,
            delay_60s=0.8,
            delayed_viability_score=0.8,
            hedge_suspicion_score=0.0,
            global_score=global_score,
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
        if strategy_name == "paired_binary_arb":
            return self.config.strategies.paired_binary_live_enabled
        if strategy_name == "correlation_dislocation":
            return self.config.strategies.correlation_live_enabled
        if strategy_name == "resolution_window":
            return self.config.strategies.resolution_window_live_enabled
        if strategy_name == "lag_signal":
            return self.config.lag_signal.live_enabled
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
        entries_last_hour_override: int,
        age_seconds: float = 0.0,
        thesis_type: str = "directional",
        bundle_id: str = "",
        bundle_role: str = "",
        paired_token_id: str = "",
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
                    "selection_thesis": strategy_name,
                    "thesis_type": thesis_type,
                    "bundle_id": bundle_id,
                    "bundle_role": bundle_role,
                    "paired_token_id": paired_token_id,
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
            decision.context["selection_thesis"] = strategy_name
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

        fill = self._estimate_resting_fill(
            detection=detection,
            orderbook=orderbook,
            target_notional=scaled_notional,
            entry_style=EntryStyle.PASSIVE_LIMIT,
        )
        best_ask = orderbook.asks[0].price if orderbook.asks else source_price
        drift_pct = abs(fill.executable_price - source_price) / max(source_price, 1e-6)
        stale_signal_limit = self._stale_signal_threshold_seconds(
            strategy_name=strategy_name,
            entry_style=EntryStyle.PASSIVE_LIMIT,
            category=category,
            source_quality=SourceQuality.REAL_PUBLIC_DATA,
        )
        risk = self.risk.evaluate(
            detection=detection,
            wallet=wallet,
            fill=fill,
            mode=self.config.mode.value,
            total_exposure=total_exposure,
            market_exposure=active_market_exposure.get(market_id, 0.0),
            wallet_exposure=active_wallet_exposure.get(wallet.wallet_address, 0.0),
            daily_pnl=self.state.read().get("daily_pnl", 0.0),
            cluster_confirmed=True,  # Supplemental strategies have own validation; bypass wallet-cluster gate
            infra_ok=bool(market_meta.get("active", True)) and not self.state.read().get("paused", False),
            entry_style_allowed=True,
            category=category,
            market_id=market_id,
            has_conflicting_position=self._has_conflicting_market_position(
                stored_positions,
                market_id=market_id,
                token_id=token_id,
                thesis_type=thesis_type,
                bundle_id=bundle_id,
            ),
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
            entries_last_hour_override=entries_last_hour_override,
            stale_signal_seconds_override=stale_signal_limit,
        )
        live_fillability: dict[str, object] | None = None
        if self.config.mode.value == "LIVE" and risk.reason_code in {"OK", "ENTRY_DRIFT", "WIDE_SPREAD", "FILLABILITY", "SLIPPAGE"}:
            category_fill_success, overall_fill_success = self._recent_live_fill_success()
            live_fillability = self._live_fillability_assessment(
                category=category,
                entry_style=EntryStyle.PASSIVE_LIMIT,
                detection=detection,
                orderbook=orderbook,
                fill=fill,
                tradability=tradability,
                target_notional=scaled_notional,
                drift_pct=drift_pct,
                stale_signal_limit_seconds=stale_signal_limit,
                category_fill_success=category_fill_success,
                overall_fill_success=overall_fill_success,
            )
            if not bool(live_fillability.get("passed")):
                risk = risk.model_copy(
                    update={
                        "allowed": False,
                        "reason_code": str(live_fillability.get("reason_code") or "LIVE_FILLABILITY_SCORE"),
                        "human_readable_reason": str(live_fillability.get("reason") or "Live fillability score below threshold."),
                        "context": {**risk.context, **live_fillability},
                    }
                )
        action = DecisionAction.PAPER_COPY if risk.allowed and self.config.mode.value != "LIVE" else DecisionAction.SKIP
        if risk.allowed and self.config.mode.value == "LIVE":
            action = DecisionAction.LIVE_COPY
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
            scaled_notional=round(scaled_notional, 6 if thesis_type == "paired_arb" else 4),
            source_price=source_price,
            executable_price=fill.executable_price or best_ask,
            cluster_confirmed=False,
            hedge_suspicion_score=wallet.hedge_suspicion_score,
            thesis_type=thesis_type,
            bundle_id=bundle_id,
            bundle_role=bundle_role,
            paired_token_id=paired_token_id,
            context={
                "fill": fill.model_dump(),
                "risk_context": risk.context,
                "market_meta": market_meta,
                "tradability": tradability,
                "source_quality": SourceQuality.REAL_PUBLIC_DATA.value,
                "selection_thesis": strategy_name,
                "thesis_type": thesis_type,
                "bundle_id": bundle_id,
                "bundle_role": bundle_role,
                "paired_token_id": paired_token_id,
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
                "live_fillability_score": None if live_fillability is None else live_fillability.get("score"),
                "live_fillability": live_fillability or {},
                "stale_signal_limit_seconds": stale_signal_limit,
                "wallet_global_score": wallet.global_score,
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
        decision.context["selection_score"] = self._decision_rank(decision, wallet, 0.0)
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

    async def _generate_lag_signal_decisions(
        self,
        *,
        stored_positions: list[dict],
        total_exposure: float,
        active_market_exposure: dict[str, float],
        active_wallet_exposure: dict[str, float],
    ) -> list[TradeDecision]:
        """
        Oracle-aligned lag arbitrage strategy.

        Fires when the Polymarket CLOB price lags a Chainlink-consistent
        move from the Binance feed, net of fees and spread.  Only applies
        to crypto-price YES/NO markets.
        """
        if not self.config.lag_signal.enabled:
            return []

        rtds_client = get_rtds_client()
        if rtds_client is None:
            return []
        rtds = rtds_client.snapshot()

        # Quick pre-check: if RTDS is completely dead, don't bother
        if rtds.staleness_seconds() > self.config.rtds.staleness_max_seconds * 5:
            self._log_strategy_signal(
                strategy_name="lag_signal",
                signal_id=stable_event_key("lag_signal", "rtds-stale", datetime.now(timezone.utc).isoformat()),
                market_id="",
                token_id="",
                final_action=DecisionAction.SKIP.value,
                reason_code="RTDS_UNAVAILABLE",
                extra={"rtds_staleness_seconds": round(rtds.staleness_seconds(), 2)},
            )
            return []

        lag_cfg = _LagSignalCfg(
            min_price_divergence_pct=self.config.lag_signal.min_price_divergence_pct,
            min_spot_move_pct=self.config.lag_signal.min_spot_move_pct,
            rtds_staleness_max_seconds=self.config.rtds.staleness_max_seconds,
            clob_staleness_max_seconds=self.config.rtds.clob_staleness_max_seconds,
            min_net_edge_taker=self.config.lag_signal.min_net_edge_taker,
            min_net_edge_maker=self.config.lag_signal.min_net_edge_maker,
            min_lag_ms=self.config.lag_signal.min_lag_ms,
            min_time_remaining_seconds=self.config.lag_signal.min_time_remaining_seconds,
        )

        now = datetime.now(timezone.utc)
        entries_last_hour = self._actual_entries_last_hour(self.positions.load(), now)
        max_candidates = max(int(self.config.lag_signal.max_candidates_per_cycle), 1)

        # Group crypto-price markets by market_id to find YES/NO pairs
        grouped: dict[str, list[MarketInfo]] = {}
        for market in self.market_data.token_cache.values():
            if not market.active or market.closed:
                continue
            if not self._is_crypto_category(market.category or ""):
                continue
            grouped.setdefault(market.market_id, []).append(market)

        decisions: list[TradeDecision] = []
        cycle_ts = now

        for market_id, markets in grouped.items():
            if len(decisions) >= max_candidates:
                break
            if len(markets) != 2:
                continue
            outcome_map = {self._binary_outcome_side(market.title): market for market in markets}
            yes_market = outcome_map.get("YES")
            no_market = outcome_map.get("NO")
            if yes_market is None or no_market is None:
                continue
            if self.config.mode.value == "LIVE" and market_id not in (self.config.live.selected_categories or []):
                pass  # category check below is sufficient

            # Parse time remaining
            end_at = None
            for m in (yes_market, no_market):
                if m.end_date_iso:
                    try:
                        end_at = datetime.fromisoformat(m.end_date_iso.replace("Z", "+00:00"))
                        break
                    except ValueError:
                        pass
            if end_at is None or end_at <= now:
                continue
            time_remaining_seconds = (end_at - now).total_seconds()
            if time_remaining_seconds < self.config.lag_signal.min_time_remaining_seconds:
                continue

            # Track reference "start price" for this market (first-seen Chainlink price)
            start_price = self._lag_start_prices.get(market_id, 0.0)
            if start_price <= 0 and rtds.chainlink_price > 0:
                self._lag_start_prices[market_id] = rtds.chainlink_price
                start_price = rtds.chainlink_price

            # Fetch both orderbooks concurrently
            try:
                yes_ob, no_ob = await asyncio.gather(
                    self.market_data.fetch_orderbook(yes_market.token_id),
                    self.market_data.fetch_orderbook(no_market.token_id),
                )
            except RuntimeError:
                continue

            yes_ask = yes_ob.asks[0].price if yes_ob.asks else 0.0
            no_ask = no_ob.asks[0].price if no_ob.asks else 0.0
            if yes_ask <= 0 or no_ask <= 0:
                continue

            clob_ts_epoch = yes_ob.timestamp.timestamp()
            realized_vol_per_second = 0.005  # ~0.3%/min, typical for BTC in a 5-min window

            sig = evaluate_lag_signal(
                rtds=rtds,
                start_price=start_price,
                time_remaining_seconds=time_remaining_seconds,
                realized_vol_per_second=realized_vol_per_second,
                yes_ask=yes_ask,
                no_ask=no_ask,
                yes_token_id=yes_market.token_id,
                no_token_id=no_market.token_id,
                clob_timestamp_epoch=clob_ts_epoch,
                category="crypto price",
                is_taker=True,
                config=lag_cfg,
            )

            signal_id = stable_event_key("lag_signal", market_id, cycle_ts.isoformat())

            if not sig.fire:
                self._log_strategy_signal(
                    strategy_name="lag_signal",
                    signal_id=signal_id,
                    market_id=market_id,
                    token_id=sig.token_id or yes_market.token_id,
                    final_action=DecisionAction.SKIP.value,
                    reason_code=sig.skip_reason or "NO_SIGNAL",
                    extra=sig.to_dict(),
                )
                continue

            # Signal fires — build a trade decision
            fire_token_id = sig.token_id
            fire_market = yes_market if fire_token_id == yes_market.token_id else no_market
            market_title = fire_market.title
            source_price = sig.executable_price
            fair_price = sig.estimated_fair_p
            scaled_notional = min(self._max_notional_for_mode(), self.config.risk.max_single_live_trade_usd if self.config.mode.value == "LIVE" else self._max_notional_for_mode())
            confidence_score = round(clamp(sig.net_edge / max(self.config.lag_signal.min_net_edge_taker * 2.0, 0.01), 0.0, 1.0), 4)

            decision = await self._build_supplemental_decision(
                strategy_name="lag_signal",
                signal_id=signal_id,
                market_id=market_id,
                token_id=fire_token_id,
                category="crypto price",
                market_title=market_title,
                source_price=source_price,
                fair_price=fair_price,
                scaled_notional=scaled_notional,
                reason=f"Oracle-lag arb: side={sig.side} net_edge={sig.net_edge:.4f} lag_ms={sig.lag_ms:.0f}ms div_pct={sig.price_divergence_pct:.4f}",
                confidence_score=confidence_score,
                extra_context={
                    "lag_signal": sig.to_dict(),
                    "time_remaining_seconds": round(time_remaining_seconds, 1),
                },
                stored_positions=stored_positions,
                total_exposure=total_exposure,
                active_market_exposure=active_market_exposure,
                active_wallet_exposure=active_wallet_exposure,
                entries_last_hour_override=entries_last_hour,
                age_seconds=rtds.staleness_seconds(),
            )
            decisions.append(decision)

        return decisions

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
        entries_last_hour = self._actual_entries_last_hour(self.positions.load(), now)
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
            if resolved_market is None and mapping_reason in {"NO_MARKET_MATCH", "MISSING_MAPPING_HINTS", "MISSING_MARKET_MATCH"}:
                resolved_market, mapping_reason = await self._direct_resolve_official_signal_market(row)
            elif resolved_market is None and mapping_reason.lower() in {"no_market_match", "missing_mapping_hints", "missing_market_match"}:
                resolved_market, mapping_reason = await self._direct_resolve_official_signal_market(row)
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
            decision = await self._build_supplemental_decision(
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
                entries_last_hour_override=entries_last_hour,
                age_seconds=age_seconds,
                thesis_type="event_catalyst",
            )
            decisions.append(decision)
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

    async def _generate_paired_binary_arb_decisions(
        self,
        *,
        stored_positions: list[dict],
        total_exposure: float,
        active_market_exposure: dict[str, float],
        active_wallet_exposure: dict[str, float],
    ) -> list[TradeDecision]:
        if not self.config.strategies.enable_paired_binary_arb:
            return []
        now = datetime.now(timezone.utc)
        entries_last_hour = self._actual_entries_last_hour(self.positions.load(), now)
        grouped: dict[str, list[MarketInfo]] = {}
        for market in self.market_data.token_cache.values():
            if not market.active or market.closed:
                continue
            grouped.setdefault(market.market_id, []).append(market)

        min_leg = float(self.config.strategies.paired_binary_min_leg_price)
        max_leg = float(self.config.strategies.paired_binary_max_leg_price)
        bundle_budget = max(self._max_notional_for_mode(), 1.0)
        grouped_candidates: list[tuple[float, str, MarketInfo, MarketInfo, str]] = []
        for market_id, markets in grouped.items():
            if len(markets) != 2:
                continue
            outcome_map = {self._binary_outcome_side(market.title): market for market in markets}
            yes_market = outcome_map.get("YES")
            no_market = outcome_map.get("NO")
            if yes_market is None or no_market is None:
                continue
            category = str(yes_market.category or no_market.category or "unknown")
            if category == "sports":
                continue
            if self.config.mode.value == "LIVE" and category not in self.config.live.selected_categories:
                continue
            grouped_candidates.append(
                (
                    self._paired_binary_rough_priority(yes_market, no_market, category),
                    market_id,
                    yes_market,
                    no_market,
                    category,
                )
            )

        scan_candidates = sorted(grouped_candidates, key=lambda item: item[0], reverse=True)[: self._paired_binary_scan_limit()]
        evaluated_candidates = await asyncio.gather(
            *[
                self._evaluate_paired_binary_candidate(
                    market_id=market_id,
                    yes_market=yes_market,
                    no_market=no_market,
                    category=category,
                    bundle_budget=bundle_budget,
                    min_leg=min_leg,
                    max_leg=max_leg,
                    now=now,
                )
                for _, market_id, yes_market, no_market, category in scan_candidates
            ]
        )
        candidates = [candidate for candidate in evaluated_candidates if candidate is not None]

        if not candidates:
            self._log_strategy_signal(
                strategy_name="paired_binary_arb",
                signal_id=stable_event_key("paired_binary_arb", "cycle-empty", now.isoformat()),
                market_id="",
                token_id="",
                final_action=DecisionAction.SKIP.value,
                reason_code="NO_PAIRED_BINARY_ARBS",
                extra={"cycle_ts": now.isoformat()},
            )
            return []

        decisions: list[TradeDecision] = []
        ranked_candidates = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
        for index, candidate in enumerate(ranked_candidates[: max(int(self.config.strategies.paired_binary_max_candidates_per_cycle), 1)]):
            yes_market = candidate["yes_market"]
            no_market = candidate["no_market"]
            yes_market = yes_market if isinstance(yes_market, MarketInfo) else None
            no_market = no_market if isinstance(no_market, MarketInfo) else None
            if yes_market is None or no_market is None:
                continue

            pair_title = yes_market.title.rsplit(" [", 1)[0]
            bundle_id = stable_event_key("paired_binary_arb", candidate["market_id"], str(index), now.isoformat())
            confidence_score = round(
                clamp(
                    float(candidate["net_edge_pct"]) / max(float(self.config.strategies.paired_binary_min_net_edge_pct) * 2.0, 0.01),
                    0.0,
                    1.0,
                ),
                4,
            )
            common_context = {
                "strategy_rationale": f"Paired yes/no asks imply a positive executable net edge after fees and buffers ({float(candidate['net_edge_pct']):.3f}).",
                "selection_thesis": "paired_full_set_parity",
                "signal_quality_score": clamp(float(candidate["score"]), 0.0, 1.0),
                "paired_market_title": pair_title,
                "bundle_sum_ask": float(candidate["bundle_sum_ask"]),
                "bundle_gross_edge_pct": float(candidate["gross_edge_pct"]),
                "bundle_net_edge_pct": float(candidate["net_edge_pct"]),
                "bundle_fee_estimate": round(float(candidate["fee_yes"]) + float(candidate["fee_no"]), 6),
                "bundle_slippage_buffer_pct": float(candidate["slippage_buffer"]),
                "bundle_shares": float(candidate["bundle_shares"]),
                "paired_leg_prices": {
                    yes_market.token_id: float(candidate["yes_ask"]),
                    no_market.token_id: float(candidate["no_ask"]),
                },
                "paired_min_depth_usd": float(candidate["min_depth"]),
            }
            for market, role, paired_market, source_price, scaled_notional in (
                (yes_market, "paired_yes", no_market, float(candidate["yes_ask"]), float(candidate["yes_notional"])),
                (no_market, "paired_no", yes_market, float(candidate["no_ask"]), float(candidate["no_notional"])),
            ):
                decision = await self._build_supplemental_decision(
                    strategy_name="paired_binary_arb",
                    signal_id=stable_event_key(bundle_id, market.token_id, role),
                    market_id=market.market_id,
                    token_id=market.token_id,
                    category=str(candidate["category"]),
                    market_title=market.title,
                    source_price=source_price,
                    fair_price=min(0.99, max(source_price + float(candidate["net_edge_pct"]), source_price)),
                    scaled_notional=scaled_notional,
                    reason="Paired yes/no asks imply a positive net locked edge after fees and execution buffers.",
                    confidence_score=confidence_score,
                    extra_context={
                        **common_context,
                        "paired_leg_role": role,
                        "paired_token_id": paired_market.token_id,
                    },
                    stored_positions=stored_positions,
                    total_exposure=total_exposure,
                    active_market_exposure=active_market_exposure,
                    active_wallet_exposure=active_wallet_exposure,
                    entries_last_hour_override=entries_last_hour,
                    age_seconds=0.0,
                    thesis_type="paired_arb",
                    bundle_id=bundle_id,
                    bundle_role=role,
                    paired_token_id=paired_market.token_id,
                )
                decisions.append(decision)
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
        entries_last_hour = self._actual_entries_last_hour(self.positions.load(), cycle_ts)
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
            decision = await self._build_supplemental_decision(
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
                entries_last_hour_override=entries_last_hour,
                age_seconds=0.0,
            )
            decisions.append(decision)
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
            decision = await self._build_supplemental_decision(
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
                entries_last_hour_override=entries_last_hour,
                age_seconds=0.0,
            )
            decisions.append(decision)
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
        entries_last_hour = self._actual_entries_last_hour(self.positions.load(), now)
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
            decision = await self._build_supplemental_decision(
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
                entries_last_hour_override=entries_last_hour,
                age_seconds=0.0,
            )
            decisions.append(decision)
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
            decision = await self._build_supplemental_decision(
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
                entries_last_hour_override=entries_last_hour,
                age_seconds=0.0,
            )
            decisions.append(decision)
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
