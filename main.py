from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
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
from src.scheduler import AppScheduler
from src.state import AppStateStore
from src.strategy import StrategyEngine
from src.trade_monitor import TradeMonitor
from src.wallet_discovery import WalletDiscoveryService
from src.wallet_scoring import WalletScoringService
from backtest.evaluator import BacktestEvaluator
from src.live_orders import LiveOrderStore
from src.models import DecisionAction, OrderLifecycleStatus, TradeDecision


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
        manual_resume_required=config.live.manual_resume_required,
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

    discovery_result = await discovery.run_discovery_cycle()
    scoring_result = scorer.score_wallets(discovery_result.wallets)
    category_scorecards = category_scorer.build_scorecards(scoring_result.scored_wallets)
    replay_rows = evaluator.evaluate_wallets(scoring_result.scored_wallets)
    approved_wallets = scorer.select_wallets(scoring_result, replay_rows)

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

    scheduler = AppScheduler(config)

    async def cycle() -> list:
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
        detections = await monitor.poll_wallets(watched_wallets)
        decisions = await strategy.process_detections(detections, approved_wallets, scoring_result.scored_wallets)
        if config.mode.value != "PAPER" or state_snapshot.get("paper_run_enabled", False):
            await paper_engine.handle_decisions(decisions)
        await live_engine.handle_decisions(decisions)
        reporter.write_daily_summary(scoring_result.scored_wallets, decisions)
        paper_quality = reporter.write_paper_quality_summary(decisions)
        analytics.write_strategy_comparison()
        alerts.emit_health_alerts(state.read())
        state.update_system_status(
            mode=config.mode.value,
            bot_loop_running=True,
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
