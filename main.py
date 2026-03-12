from __future__ import annotations

import asyncio
from pathlib import Path

from src.alerts import AlertManager
from src.analytics import AnalyticsEngine
from src.category_scoring import CategoryScorer
from src.config import AppConfig, ensure_runtime_files, load_config
from src.geoblock import GeoblockChecker
from src.live_engine import LiveTradingEngine
from src.logger import setup_logging
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
    state.update_system_status(mode=config.mode.value, eligibility=eligibility.model_dump())

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

    if config.mode.value == "LIVE":
        await live_engine.refresh_live_status()

    wallets = await discovery.run_discovery_cycle()
    scored_wallets = scorer.score_wallets(wallets)
    category_scorecards = category_scorer.build_scorecards(scored_wallets)
    replay_rows = evaluator.evaluate_wallets(scored_wallets)
    approved_wallets = scorer.select_wallets(scored_wallets, replay_rows)

    state.set_wallets(approved_wallets, scored_wallets)

    scheduler = AppScheduler(config)

    async def cycle() -> None:
        watched_wallets = approved_wallets.live_wallets if config.mode.value == "LIVE" else approved_wallets.paper_wallets
        detections = await monitor.poll_wallets(watched_wallets)
        decisions = await strategy.process_detections(detections, approved_wallets, scored_wallets)
        paper_engine.handle_decisions(decisions)
        await live_engine.handle_decisions(decisions)
        reporter.write_daily_summary(scored_wallets, decisions)
        analytics.write_strategy_comparison()
        alerts.emit_health_alerts(state.read())

    await cycle()

    if config.mode.value == "RESEARCH":
        reporter.write_research_snapshot(scored_wallets, category_scorecards, replay_rows)
        return

    cycle_count = 1
    async for _ in scheduler.ticks():
        await cycle()
        cycle_count += 1
        if config.runtime.max_runtime_cycles and cycle_count >= config.runtime.max_runtime_cycles:
            break


if __name__ == "__main__":
    asyncio.run(run())
