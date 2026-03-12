from __future__ import annotations

import asyncio
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


async def run() -> None:
    root = Path(__file__).resolve().parent
    config: AppConfig = load_config(root / "config.yaml", root / ".env")
    ensure_runtime_files(root, config)
    setup_logging(root / "logs")

    state = AppStateStore(root / "data" / "app_state.json")
    alerts = AlertManager(config, root / "logs")
    geoblock = GeoblockChecker(config)
    eligibility = geoblock.preflight_status()
    state.update_system_status(
        mode=config.mode.value,
        system_status=config.mode.value,
        status=config.mode.value,
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
        state.update_system_status(bot_loop_running=True, last_cycle_started_at=cycle_started_at)
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
