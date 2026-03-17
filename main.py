from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.alerts import AlertManager
from src.analytics import AnalyticsEngine
from src.category_scoring import CategoryScorer
from src.config import AppConfig, ensure_runtime_files, load_config
from src.geoblock import GeoblockChecker
from src.live_engine import LiveTradingEngine
from src.logger import setup_logging
from src.logger import logger
from src.paper_engine import PaperTradingEngine
from src.reporting import ReportWriter
from src.rtds_client import get_rtds_client
from src.scheduler import AppScheduler
from src.state import AppStateStore
from src.strategy import StrategyEngine
from src.trade_monitor import TradeMonitor
from src.utils import append_jsonl, read_json, stable_event_key, write_json
from src.wallet_discovery import WalletDiscoveryService
from src.wallet_scoring import WalletScoringService
from src.momentum_signal_generator import generate_momentum_signals
from backtest.evaluator import BacktestEvaluator
from src.live_orders import LiveOrderStore
from src.models import ApprovedWallets, DecisionAction, DiscoveryResult, DiscoveryState, Mode, OrderLifecycleStatus, SourceQuality, TradeDecision, WalletMetrics


def _build_shadow_config(config: AppConfig) -> AppConfig:
    shadow = config.model_copy(deep=True)
    shadow.mode = Mode.PAPER
    shadow.risk.max_single_live_trade_usd = min(
        max(float(config.env.operator_live_max_trade_usd or config.risk.max_single_live_trade_usd), 1.0),
        5.0,
    )
    shadow.risk.max_new_entries_per_hour = max(config.risk.max_new_entries_per_hour, 6)
    shadow.risk.stale_signal_seconds = max(config.risk.stale_signal_seconds, 300)
    shadow.risk.max_spread_pct = max(config.risk.max_spread_pct, 0.06)
    shadow.risk.max_entry_drift_pct = max(config.risk.max_entry_drift_pct, 0.08)
    shadow.risk.min_orderbook_depth_usd = min(config.risk.min_orderbook_depth_usd, 10.0)
    shadow.risk.allow_categories = [
        category
        for category in config.categories.tracked
        if category in {"politics", "regulatory / legal", "macro / economics", "geopolitics", "crypto price"}
    ] or list(config.risk.allow_categories)
    shadow.wallet_selection.approved_live_wallets = max(config.wallet_selection.approved_live_wallets, 5)
    shadow.wallet_selection.approved_paper_wallets = max(config.wallet_selection.approved_paper_wallets, 5)
    shadow.live.enable_multi_entry_style_live = True
    shadow.live.only_cluster_confirmed = False
    shadow.live.require_paper_validation = False
    shadow.live.selected_categories = list(shadow.risk.allow_categories)
    shadow.strategies.enable_event_driven_official = False
    shadow.strategies.correlation_live_enabled = False
    shadow.strategies.resolution_window_live_enabled = True
    shadow.strategies.resolution_window_min_price = min(shadow.strategies.resolution_window_min_price, 0.6)
    shadow.strategies.resolution_window_min_edge_pct = min(shadow.strategies.resolution_window_min_edge_pct, 0.02)
    shadow.strategies.resolution_window_max_hours = max(shadow.strategies.resolution_window_max_hours, 240)
    shadow.strategies.resolution_window_min_liquidity = min(shadow.strategies.resolution_window_min_liquidity, 50.0)
    shadow.strategies.supplemental_paper_relaxed_enabled = True
    return shadow


def _decision_compare_key(decision: TradeDecision) -> str:
    return stable_event_key(
        decision.strategy_name,
        decision.wallet_address,
        decision.market_id,
        decision.token_id,
        decision.entry_style.value,
    )


def _decision_snapshot(decision: TradeDecision) -> dict[str, object]:
    return {
        "strategy_name": decision.strategy_name,
        "allowed": decision.allowed,
        "action": decision.action.value,
        "reason_code": decision.reason_code,
        "reason": decision.human_readable_reason,
        "wallet_address": decision.wallet_address,
        "market_id": decision.market_id,
        "token_id": decision.token_id,
        "entry_style": decision.entry_style.value,
        "category": decision.category,
        "scaled_notional": decision.scaled_notional,
        "source_price": decision.source_price,
        "executable_price": decision.executable_price,
        "cluster_confirmed": decision.cluster_confirmed,
        "source_quality": decision.context.get("source_quality", ""),
        "trust_level": decision.context.get("trust_level", ""),
    }


async def _run_cycle_stage(
    *,
    state: AppStateStore,
    stage_name: str,
    coro: object,
    timeout_seconds: float,
    default: object,
):
    state.update_system_status(
        last_cycle_stage=stage_name,
        last_cycle_stage_started_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)  # type: ignore[arg-type]
    except asyncio.TimeoutError:
        logger.warning("Cycle stage timed out stage={} timeout_seconds={}", stage_name, timeout_seconds)
        state.update_system_status(
            last_cycle_stage=f"{stage_name}:timeout",
            last_cycle_stage_error=f"Timed out after {timeout_seconds:.1f}s",
            last_cycle_stage_completed_at=datetime.now(timezone.utc).isoformat(),
        )
        return default
    except Exception as exc:
        logger.warning("Cycle stage failed stage={} error={}", stage_name, exc)
        state.update_system_status(
            last_cycle_stage=f"{stage_name}:error",
            last_cycle_stage_error=str(exc),
            last_cycle_stage_completed_at=datetime.now(timezone.utc).isoformat(),
        )
        return default
    state.update_system_status(
        last_cycle_stage=f"{stage_name}:complete",
        last_cycle_stage_error="",
        last_cycle_stage_completed_at=datetime.now(timezone.utc).isoformat(),
    )
    return result


async def _run_startup_stage(
    *,
    state: AppStateStore,
    stage_name: str,
    coro: object,
    timeout_seconds: float,
    default: object,
):
    state.update_system_status(
        startup_stage=stage_name,
        startup_stage_started_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)  # type: ignore[arg-type]
    except asyncio.TimeoutError:
        logger.warning("Startup stage timed out stage={} timeout_seconds={}", stage_name, timeout_seconds)
        state.update_system_status(
            startup_stage=f"{stage_name}:timeout",
            startup_stage_error=f"Timed out after {timeout_seconds:.1f}s",
            startup_stage_completed_at=datetime.now(timezone.utc).isoformat(),
        )
        return default
    except Exception as exc:
        logger.warning("Startup stage failed stage={} error={}", stage_name, exc)
        state.update_system_status(
            startup_stage=f"{stage_name}:error",
            startup_stage_error=str(exc),
            startup_stage_completed_at=datetime.now(timezone.utc).isoformat(),
        )
        return default
    state.update_system_status(
        startup_stage=f"{stage_name}:complete",
        startup_stage_error="",
        startup_stage_completed_at=datetime.now(timezone.utc).isoformat(),
    )
    return result


def _restore_cached_wallet_context(
    *,
    root: Path,
    existing_state: dict[str, object],
) -> tuple[ApprovedWallets | None, list[WalletMetrics]]:
    approved_wallets: ApprovedWallets | None = None
    approved_raw = existing_state.get("approved_wallets", {})
    if isinstance(approved_raw, dict):
        try:
            approved_wallets = ApprovedWallets.model_validate(approved_raw)
        except Exception:
            approved_wallets = None

    scored_wallets: list[WalletMetrics] = []
    raw_rows = read_json(root / "data" / "top_wallets.json", [])
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            try:
                scored_wallets.append(WalletMetrics.model_validate(row))
            except Exception:
                continue
    return approved_wallets, scored_wallets


def _write_shadow_cycle_artifacts(
    *,
    root: Path,
    cycle_ts: str,
    actual_decisions: list[TradeDecision],
    shadow_decisions: list[TradeDecision],
    shadow_state: AppStateStore,
) -> None:
    actual_map = {_decision_compare_key(decision): decision for decision in actual_decisions}
    shadow_map = {_decision_compare_key(decision): decision for decision in shadow_decisions}
    for key in sorted(actual_map.keys() | shadow_map.keys()):
        actual = actual_map.get(key)
        shadow = shadow_map.get(key)
        append_jsonl(
            root / "data" / "shadow_live_decisions.jsonl",
            {
                "ts": cycle_ts,
                "decision_key": key,
                "market_id": (shadow or actual).market_id if (shadow or actual) else "",
                "token_id": (shadow or actual).token_id if (shadow or actual) else "",
                "strategy_name": (shadow or actual).strategy_name if (shadow or actual) else "",
                "actual": _decision_snapshot(actual) if actual is not None else None,
                "shadow": _decision_snapshot(shadow) if shadow is not None else None,
                "actual_would_trade": bool(actual and actual.allowed and actual.action == DecisionAction.LIVE_COPY),
                "shadow_would_trade": bool(shadow and shadow.allowed and shadow.action == DecisionAction.PAPER_COPY),
                "comparison": (
                    "shadow_only"
                    if shadow and (not actual or not actual.allowed)
                    else "actual_only"
                    if actual and (not shadow or not shadow.allowed)
                    else "both"
                    if actual and shadow and actual.allowed and shadow.allowed
                    else "neither"
                ),
            },
        )

    paper_summary = shadow_state.read().get("paper_summary", {}) or {}
    write_json(
        root / "data" / "shadow_live_summary.json",
        {
            "updated_at": cycle_ts,
            "actual_decision_count": len(actual_decisions),
            "actual_trade_count": len(
                [decision for decision in actual_decisions if decision.allowed and decision.action == DecisionAction.LIVE_COPY]
            ),
            "shadow_decision_count": len(shadow_decisions),
            "shadow_trade_count": len(
                [decision for decision in shadow_decisions if decision.allowed and decision.action == DecisionAction.PAPER_COPY]
            ),
            "shadow_only_count": len(
                [
                    key
                    for key in shadow_map
                    if key not in actual_map or not actual_map[key].allowed or actual_map[key].action != DecisionAction.LIVE_COPY
                ]
            ),
            "paper_summary": paper_summary,
            "shadow_net_pnl_total": float(paper_summary.get("net_pnl_total", 0.0) or 0.0),
            "shadow_realized_pnl_total": float(paper_summary.get("realized_pnl_total", 0.0) or 0.0),
            "shadow_unrealized_pnl_total": float(paper_summary.get("unrealized_pnl_total", 0.0) or 0.0),
            "shadow_open_positions": int(paper_summary.get("open_positions", 0) or 0),
            "shadow_open_notional": float(paper_summary.get("open_notional", 0.0) or 0.0),
        },
    )


def _build_operator_smoke_decision(config: AppConfig) -> TradeDecision | None:
    enabled = os.getenv("POLYBOT_SMOKE_ORDER_ENABLED", "false").lower() == "true"
    if not enabled:
        return None

    market_id = os.getenv("POLYBOT_SMOKE_ORDER_MARKET_ID", "").strip()
    token_id = os.getenv("POLYBOT_SMOKE_ORDER_TOKEN_ID", "").strip()
    price_raw = os.getenv("POLYBOT_SMOKE_ORDER_PRICE", "").strip()
    if not market_id or not token_id or not price_raw:
        raise RuntimeError(
            "Explicit smoke order requires POLYBOT_SMOKE_ORDER_MARKET_ID, POLYBOT_SMOKE_ORDER_TOKEN_ID, and POLYBOT_SMOKE_ORDER_PRICE."
        )

    price = float(price_raw)
    if price <= 0:
        raise RuntimeError("POLYBOT_SMOKE_ORDER_PRICE must be greater than 0.")

    notional = float(os.getenv("POLYBOT_SMOKE_ORDER_NOTIONAL_USD", str(config.risk.max_single_live_trade_usd)))
    if notional <= 0:
        raise RuntimeError("POLYBOT_SMOKE_ORDER_NOTIONAL_USD must be greater than 0.")

    scaled_notional = round(notional, 4)
    category = os.getenv("POLYBOT_SMOKE_ORDER_CATEGORY", "").strip() or (
        config.live.selected_categories[0] if config.live.selected_categories else "politics"
    )
    wallet_label = os.getenv("POLYBOT_SMOKE_ORDER_WALLET_LABEL", "OPERATOR_SMOKE")
    decision_id = os.getenv("POLYBOT_SMOKE_ORDER_ID", "").strip() or f"operator-smoke-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    return TradeDecision(
        allowed=True,
        action=DecisionAction.LIVE_COPY,
        reason_code="OPERATOR_SMOKE_ORDER",
        human_readable_reason="Operator-approved supervised smoke order.",
        local_decision_id=decision_id,
        wallet_address=wallet_label,
        market_id=market_id,
        token_id=token_id,
        entry_style=config.entry_styles.preferred_live_entry_style,
        category=category,
        scaled_notional=round(scaled_notional, 4),
        source_price=price,
        executable_price=price,
        cluster_confirmed=True,
        hedge_suspicion_score=0.0,
        context={"operator_smoke_order": True, "requested_notional": scaled_notional},
    )


def _operator_smoke_cancel_enabled() -> bool:
    return os.getenv("POLYBOT_SMOKE_CANCEL_ENABLED", "false").lower() == "true"


def _resolve_operator_smoke_cancel_order(root: Path) -> LiveOrder | None:
    store = LiveOrderStore(root / "data" / "live_orders.json")
    orders = store.load()
    requested = os.getenv("POLYBOT_SMOKE_CANCEL_ORDER_ID", "").strip()
    if requested:
        for order in orders:
            if requested in {order.local_order_id, order.exchange_order_id, order.client_order_id}:
                return order
        raise RuntimeError(f"Unable to find smoke order for POLYBOT_SMOKE_CANCEL_ORDER_ID={requested}")
    for order in reversed(orders):
        if not str(order.local_decision_id).startswith("operator-smoke-"):
            continue
        if order.terminal_state:
            continue
        return order
    return None


def _clear_stale_operator_smoke_orders(root: Path) -> None:
    store = LiveOrderStore(root / "data" / "live_orders.json")
    orders = store.load()
    updated = False
    for order in orders:
        if order.terminal_state:
            continue
        if not str(order.local_decision_id).startswith("operator-smoke-"):
            continue
        if order.exchange_order_id:
            continue
        order.last_exchange_status = "REJECTED"
        order.lifecycle_status = OrderLifecycleStatus.REJECTED
        order.terminal_state = True
        order.last_update_at = datetime.now(timezone.utc)
        updated = True
    if updated:
        store.save(orders)


async def run() -> None:
    root = Path(__file__).resolve().parent
    config_path_value = os.getenv("POLYBOT_CONFIG_PATH", "config.yaml")
    config_path = Path(config_path_value)
    if not config_path.is_absolute():
        config_path = root / config_path
    config: AppConfig = load_config(config_path, root / ".env")
    ensure_runtime_files(root, config)
    setup_logging(root / "logs")
    preflight_only = os.getenv("POLYBOT_PREFLIGHT_ONLY", "false").lower() == "true"

    state = AppStateStore(root / "data" / "app_state.json")
    existing_state = state.read()
    cached_approved_wallets, cached_scored_wallets = _restore_cached_wallet_context(root=root, existing_state=existing_state)
    if preflight_only:
        state.clear_pause()
    alerts = AlertManager(config, root / "logs")
    geoblock = GeoblockChecker(config)
    eligibility = geoblock.preflight_status()
    state.update_system_status(
        mode=config.mode.value,
        system_status=config.mode.value,
        status=config.mode.value,
        manual_live_enable=config.live.manual_live_enable,
        manual_resume_required=bool(existing_state.get("manual_resume_required", False) and existing_state.get("paused", False)),
        eligibility=eligibility.model_dump(),
    )

    discovery = WalletDiscoveryService(config, root / "data")
    scorer = WalletScoringService(config, root / "data")
    category_scorer = CategoryScorer(config, root / "data")
    evaluator = BacktestEvaluator(config, root / "data")
    strategy = StrategyEngine(config, root / "data", state)
    paper_engine = PaperTradingEngine(config, root / "data", state)
    live_engine = LiveTradingEngine(config, root / "data", state, geoblock)
    monitor = TradeMonitor(config, root / "data", state)
    reporter = ReportWriter(config, root / "data", state)
    analytics = AnalyticsEngine(config, root / "data")
    shadow_dir = root / "data" / "shadow"
    shadow_config = _build_shadow_config(config)
    shadow_state = AppStateStore(shadow_dir / "app_state.json")
    shadow_state.update_system_status(
        mode=shadow_config.mode.value,
        system_status=shadow_config.mode.value,
        status=shadow_config.mode.value,
        paper_run_enabled=True,
        manual_live_enable=True,
        manual_resume_required=False,
        reconciliation_clean=True,
        paper_bankroll_override=float(config.env.operator_live_session_max_usd or 30.0),
        paper_trade_notional_override=float(config.env.operator_live_max_trade_usd or min(config.risk.max_single_live_trade_usd, 5.0)),
    )
    shadow_strategy = StrategyEngine(shadow_config, shadow_dir, shadow_state)
    shadow_paper_engine = PaperTradingEngine(shadow_config, shadow_dir, shadow_state)
    shadow_reporter = ReportWriter(shadow_config, shadow_dir, shadow_state)
    shadow_analytics = AnalyticsEngine(shadow_config, shadow_dir)

    has_wallet_credentials = bool(
        config.env.polymarket_private_key
        and config.env.polymarket_funder
        and config.env.polymarket_api_key
        and config.env.polymarket_api_secret
        and config.env.polymarket_api_passphrase
    )

    if config.mode.value == "LIVE" or has_wallet_credentials:
        await live_engine.refresh_live_status()
        if config.mode.value != "LIVE":
            state.update_system_status(
                mode=config.mode.value,
                system_status=config.mode.value,
                status=config.mode.value,
            )
    if config.mode.value == "LIVE" and preflight_only:
        logger.info("LIVE preflight-only mode enabled; exiting before discovery and decision loops.")
        return

    operator_smoke_decision = _build_operator_smoke_decision(config)
    if config.mode.value == "LIVE" and operator_smoke_decision is not None:
        _clear_stale_operator_smoke_orders(root)
        state.clear_pause()
        state.update_system_status(
            manual_live_enable=True,
            manual_resume_required=False,
            paused=False,
            pause_reason="",
        )
        logger.info(
            "Running explicit supervised smoke order market_id={} token_id={} notional={} price={}",
            operator_smoke_decision.market_id,
            operator_smoke_decision.token_id,
            operator_smoke_decision.scaled_notional,
            operator_smoke_decision.executable_price,
        )
        await live_engine.handle_decisions([operator_smoke_decision])
        return

    if config.mode.value == "LIVE" and _operator_smoke_cancel_enabled():
        state.clear_pause()
        state.update_system_status(
            manual_live_enable=True,
            manual_resume_required=False,
            paused=False,
            pause_reason="",
        )
        order = _resolve_operator_smoke_cancel_order(root)
        if order is None:
            logger.info("No non-terminal operator smoke order found to cancel.")
            await live_engine.refresh_live_status()
            return
        logger.info(
            "Cancelling explicit supervised smoke order local_order_id={} exchange_order_id={}",
            order.local_order_id,
            order.exchange_order_id,
        )
        live_orders = live_engine.orders.load()
        live_order = next((item for item in live_orders if item.local_order_id == order.local_order_id), None)
        if live_order is None:
            raise RuntimeError(f"Unable to load operator smoke order {order.local_order_id} for cancellation.")
        await live_engine.order_manager.cancel_open_order(live_order)
        live_engine.orders.save(live_orders)
        await live_engine.refresh_live_status()
        return

    startup_timeout_base = max(float(config.live.bounded_execution_seconds or 20), 10.0)
    discovery_result = await _run_startup_stage(
        state=state,
        stage_name="wallet_discovery",
        coro=discovery.run_discovery_cycle(),
        timeout_seconds=min(startup_timeout_base * 2.0, 45.0),
        default=DiscoveryResult(
            wallets=[],
            state=DiscoveryState.FETCH_FAILED,
            source_quality=SourceQuality.DEGRADED_PUBLIC_DATA,
            reason="Startup discovery timed out or failed before cycle start.",
            diagnostics={"startup_timeout": True},
        ),
    )
    scoring_result = await _run_startup_stage(
        state=state,
        stage_name="wallet_scoring",
        coro=asyncio.to_thread(scorer.score_wallets, discovery_result.wallets),
        timeout_seconds=min(startup_timeout_base, 20.0),
        default=scorer.score_wallets([]),
    )
    category_scorecards = await _run_startup_stage(
        state=state,
        stage_name="category_scoring",
        coro=asyncio.to_thread(category_scorer.build_scorecards, scoring_result.scored_wallets),
        timeout_seconds=min(startup_timeout_base, 20.0),
        default=[],
    )
    replay_rows = await _run_startup_stage(
        state=state,
        stage_name="wallet_backtest",
        coro=asyncio.to_thread(evaluator.evaluate_wallets, scoring_result.scored_wallets),
        timeout_seconds=min(startup_timeout_base * 2.0, 30.0),
        default=[],
    )
    approved_wallets = scorer.select_wallets(scoring_result, replay_rows)

    if (not scoring_result.scored_wallets or not approved_wallets.paper_wallets) and cached_scored_wallets:
        logger.warning(
            "Startup wallet context degraded; restoring cached scored wallets count={} approved_live_count={}",
            len(cached_scored_wallets),
            len(cached_approved_wallets.live_wallets) if cached_approved_wallets else 0,
        )
        scoring_result = scoring_result.model_copy(update={"scored_wallets": cached_scored_wallets})
        if cached_approved_wallets is not None:
            approved_wallets = cached_approved_wallets
        elif not approved_wallets.paper_wallets:
            cached_addresses = [wallet.wallet_address for wallet in cached_scored_wallets]
            fallback_count = max(int(config.env.operator_live_wallet_count or config.wallet_selection.approved_live_wallets), 1)
            approved_wallets = ApprovedWallets(
                research_wallets=cached_addresses[: config.wallet_selection.top_research_wallets],
                paper_wallets=cached_addresses[: config.wallet_selection.approved_paper_wallets],
                live_wallets=cached_addresses[:fallback_count],
            )
        state.update_system_status(
            wallet_startup_fallback_used=True,
            wallet_startup_fallback_reason="Restored cached wallet context after degraded startup discovery/scoring.",
        )
    else:
        state.update_system_status(wallet_startup_fallback_used=False, wallet_startup_fallback_reason="")

    state.set_wallets(approved_wallets, scoring_result.scored_wallets)
    state.update_system_status(
        wallet_discovery_state=discovery_result.state.value,
        wallet_discovery_reason=discovery_result.reason,
        wallet_discovery_source_quality=discovery_result.source_quality.value,
        wallet_scoring_state=scoring_result.state,
        wallet_scoring_source_quality=scoring_result.source_quality.value,
    )
    paper_quality = reporter.write_paper_quality_summary()
    if config.mode.value != "LIVE":
        logger.info(
            "Paper/Research startup discovery_state={} scoring_state={} source_quality={} paper_readiness={}",
            discovery_result.state.value,
            scoring_result.state,
            discovery_result.source_quality.value,
            paper_quality.get("paper_readiness", "UNKNOWN"),
        )
        if discovery_result.state.value != "SUCCESS":
            logger.warning("Paper/Research discovery degraded state={} reason={}", discovery_result.state.value, discovery_result.reason)
        if scoring_result.state != "SUCCESS":
            logger.warning("Paper/Research scoring degraded state={}", scoring_result.state)
        if paper_quality.get("fallback_in_use"):
            logger.warning("Paper mode is using fallback or synthetic data and is not trustworthy for live-readiness decisions.")
        elif paper_quality.get("paper_readiness") != "STRONG":
            logger.warning("Paper mode is running in a degraded state and should not be treated as strong pre-live evidence.")

    # ------------------------------------------------------------------ #
    # Start RTDS WebSocket background task for oracle-aligned lag signal
    # ------------------------------------------------------------------ #
    if config.rtds.enabled:
        _rtds_client = get_rtds_client(config.rtds.url)
        asyncio.create_task(_rtds_client.run_forever())
        logger.info("RTDS background task started url={}", config.rtds.url)

    scheduler = AppScheduler(config)
    cycle_timeout_base = max(float(config.live.bounded_execution_seconds or 20), 10.0)
    last_discovery_at: datetime = datetime.now(timezone.utc)

    async def rediscover_wallets() -> None:
        nonlocal approved_wallets, scoring_result, last_discovery_at
        logger.info(
            "Periodic wallet rediscovery starting interval_minutes={}",
            config.runtime.discovery_interval_minutes,
        )
        new_discovery = await _run_startup_stage(
            state=state,
            stage_name="wallet_discovery",
            coro=discovery.run_discovery_cycle(),
            timeout_seconds=60.0,
            default=discovery_result,
        )
        new_scoring = await _run_startup_stage(
            state=state,
            stage_name="wallet_scoring",
            coro=asyncio.to_thread(scorer.score_wallets, new_discovery.wallets),
            timeout_seconds=20.0,
            default=scoring_result,
        )
        new_replay = await _run_startup_stage(
            state=state,
            stage_name="wallet_backtest",
            coro=asyncio.to_thread(evaluator.evaluate_wallets, new_scoring.scored_wallets),
            timeout_seconds=30.0,
            default=[],
        )
        new_approved = scorer.select_wallets(new_scoring, new_replay)
        last_discovery_at = datetime.now(timezone.utc)
        if new_approved.live_wallets:
            approved_wallets = new_approved
            scoring_result = new_scoring
            state.set_wallets(approved_wallets, scoring_result.scored_wallets)
            state.update_system_status(
                wallet_discovery_state=new_discovery.state.value,
                wallet_scoring_state=new_scoring.state,
                wallet_discovery_source_quality=new_discovery.source_quality.value,
                wallet_scoring_source_quality=new_scoring.source_quality.value,
            )
            logger.info(
                "Periodic wallet rediscovery complete live_wallets={}",
                approved_wallets.live_wallets,
            )
        else:
            logger.warning("Periodic wallet rediscovery yielded no live wallets; keeping current selection")

        # ── Momentum signal refresh ────────────────────────────────────────
        if config.strategies.official_signal_live_enabled and config.mode.value == "LIVE":
            signal_path = root / config.strategies.official_signal_file
            try:
                n_new = await generate_momentum_signals(
                    config=config,
                    market_data=strategy.market_data,
                    output_path=signal_path,
                )
                if n_new:
                    logger.info("momentum_signal_generator: {} new signals written to {}", n_new, signal_path)
            except Exception as _sig_exc:
                logger.warning("momentum_signal_generator error (non-fatal): {}", _sig_exc)

    async def cycle() -> list:
        nonlocal last_discovery_at
        if config.mode.value == "LIVE":
            await live_engine.refresh_live_status()
        discovery_interval = timedelta(minutes=max(config.runtime.discovery_interval_minutes, 1))
        if datetime.now(timezone.utc) - last_discovery_at >= discovery_interval:
            await _run_cycle_stage(
                state=state,
                stage_name="wallet_rediscovery",
                coro=rediscover_wallets(),
                timeout_seconds=120.0,
                default=None,
            )
        cycle_started_at = datetime.now(timezone.utc).isoformat()
        state_snapshot = state.read()
        state.update_system_status(
            mode=config.mode.value,
            bot_loop_running=True,
            last_cycle_started_at=cycle_started_at,
        )
        if config.mode.value == "LIVE":
            watched_wallets = approved_wallets.live_wallets
        else:
            watched_wallets = approved_wallets.paper_wallets or approved_wallets.research_wallets[: config.wallet_selection.approved_paper_wallets]
            if not watched_wallets:
                watched_wallets = [
                    wallet
                    for wallet in state_snapshot.get("last_cycle_watched_wallets", [])
                    if isinstance(wallet, str) and not wallet.upper().startswith("0XWALLET")
                ]
        logger.info(
            "Cycle start mode={} paper_run_enabled={} watched_wallet_count={} watched_wallets={}",
            config.mode.value,
            state_snapshot.get("paper_run_enabled", False),
            len(watched_wallets),
            watched_wallets,
        )
        strategy_stage_timeout = min(cycle_timeout_base * 3.0, 65.0) if config.mode.value == "LIVE" else min(cycle_timeout_base * 2.0, 45.0)
        shadow_strategy_stage_timeout = min(cycle_timeout_base * 3.5, 75.0) if config.mode.value == "LIVE" else min(cycle_timeout_base * 2.0, 45.0)
        detections = await _run_cycle_stage(
            state=state,
            stage_name="poll_wallets",
            coro=monitor.poll_wallets(watched_wallets),
            timeout_seconds=min(cycle_timeout_base, 15.0),
            default=[],
        )
        decisions = await _run_cycle_stage(
            state=state,
            stage_name="strategy_process",
            coro=strategy.process_detections(detections, approved_wallets, scoring_result.scored_wallets),
            timeout_seconds=strategy_stage_timeout,
            default=[],
        )
        shadow_decisions = await _run_cycle_stage(
            state=shadow_state,
            stage_name="shadow_strategy_process",
            coro=shadow_strategy.process_detections(detections, approved_wallets, scoring_result.scored_wallets),
            timeout_seconds=shadow_strategy_stage_timeout,
            default=[],
        )
        if config.mode.value != "PAPER" or state_snapshot.get("paper_run_enabled", False):
            await _run_cycle_stage(
                state=state,
                stage_name="paper_engine",
                coro=paper_engine.handle_decisions(decisions),
                timeout_seconds=min(cycle_timeout_base * 1.5, 30.0),
                default=None,
            )
        await _run_cycle_stage(
            state=shadow_state,
            stage_name="shadow_paper_engine",
            coro=shadow_paper_engine.handle_decisions(shadow_decisions),
            timeout_seconds=min(cycle_timeout_base * 1.5, 30.0),
            default=None,
        )
        await _run_cycle_stage(
            state=state,
            stage_name="live_engine",
            coro=live_engine.handle_decisions(decisions),
            timeout_seconds=min(cycle_timeout_base * 3.0, 90.0),  # increased: 6+ positions need 5s each for orderbooks
            default=None,
        )
        reporter.write_daily_summary(scoring_result.scored_wallets, decisions)
        paper_quality = reporter.write_paper_quality_summary(decisions)
        shadow_reporter.write_daily_summary(scoring_result.scored_wallets, shadow_decisions)
        shadow_reporter.write_paper_quality_summary(shadow_decisions)
        analytics.write_strategy_comparison()
        shadow_analytics.write_strategy_comparison()
        _write_shadow_cycle_artifacts(
            root=root,
            cycle_ts=datetime.now(timezone.utc).isoformat(),
            actual_decisions=decisions,
            shadow_decisions=shadow_decisions,
            shadow_state=shadow_state,
        )
        alerts.emit_health_alerts(state.read())
        state.update_system_status(
            mode=config.mode.value,
            bot_loop_running=True,
            last_cycle_stage="cycle_complete",
            last_cycle_completed_at=datetime.now(timezone.utc).isoformat(),
            last_cycle_detection_count=len(detections),
            last_cycle_decision_count=len(decisions),
            last_cycle_watched_wallets=watched_wallets,
        )
        logger.info(
            "Cycle complete mode={} detections={} decisions={} paper_run_enabled={} paper_readiness={} source_quality={}",
            config.mode.value,
            len(detections),
            len(decisions),
            state_snapshot.get("paper_run_enabled", False),
            paper_quality.get("paper_readiness", "UNKNOWN"),
            paper_quality.get("dominant_source_quality", "UNKNOWN"),
        )
        return decisions

    first_cycle_decisions = await cycle()

    if config.mode.value == "RESEARCH":
        reporter.write_research_snapshot(
            scoring_result.scored_wallets,
            category_scorecards.rows,
            replay_rows,
            decisions=first_cycle_decisions,
        )
        return

    cycle_count = 1
    async for _ in scheduler.ticks():
        await cycle()
        cycle_count += 1
        if config.runtime.max_runtime_cycles and cycle_count >= config.runtime.max_runtime_cycles:
            break


if __name__ == "__main__":
    asyncio.run(run())
