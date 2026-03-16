from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import load_config
from src.models import (
    ApprovedWallets,
    DecisionAction,
    DetectionEvent,
    EntryStyle,
    MarketInfo,
    OrderbookLevel,
    OrderbookSnapshot,
    SourceQuality,
    TradeDecision,
    WalletMetrics,
)
from src.state import AppStateStore
from src.strategy import StrategyEngine


def _paper_engine(tmp_path: Path) -> StrategyEngine:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write({"paper_run_enabled": True, "manual_live_enable": True, "reconciliation_clean": True})
    return StrategyEngine(config, tmp_path, state)


def _live_engine(tmp_path: Path) -> StrategyEngine:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "manual_live_enable": True,
            "manual_resume_required": False,
            "live_readiness_last_result": {"ready": True, "checks": []},
            "reconciliation_clean": True,
            "live_health_state": "HEALTHY",
            "heartbeat_ok": True,
            "balance_visible": True,
            "allowance_sufficient": True,
            "kill_switch": False,
        }
    )
    return StrategyEngine(config, tmp_path, state)


def _snapshot(token_id: str, bid: float, ask: float) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        token_id=token_id,
        bids=[OrderbookLevel(price=bid, size=100.0)],
        asks=[OrderbookLevel(price=ask, size=100.0)],
        timestamp=datetime.now(timezone.utc),
    )


def _wallet() -> WalletMetrics:
    return WalletMetrics(
        wallet_address="0xabc",
        evaluation_window_days=30,
        trade_count=20,
        trades_per_day=0.5,
        buy_count=14,
        sell_count=6,
        estimated_pnl_percent=0.12,
        win_rate=0.61,
        average_trade_size=55,
        conviction_score=0.7,
        market_concentration=0.3,
        category_concentration=0.5,
        holding_time_estimate_hours=12,
        drawdown_proxy=0.08,
        copyability_score=0.7,
        low_velocity_score=0.75,
        delay_5s=0.72,
        delay_15s=0.7,
        delay_30s=0.68,
        delay_60s=0.6,
        dominant_category="politics",
        delayed_viability_score=0.675,
        hedge_suspicion_score=0.1,
        source_quality=SourceQuality.REAL_PUBLIC_DATA,
    )


def _detection(*, price: float = 0.52, category: str = "politics") -> DetectionEvent:
    return DetectionEvent(
        event_key="evt-1",
        wallet_address="0xabc",
        market_title="Test market",
        market_slug="test-market",
        market_id="m1",
        token_id="t1",
        side="BUY",
        price=price,
        size=10,
        notional=price * 10,
        transaction_hash="tx-1",
        detection_latency_seconds=5,
        category=category,
        source_quality=SourceQuality.REAL_PUBLIC_DATA,
    )


def test_event_driven_official_generates_paper_signal(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    signal_path = tmp_path / "official_event_signals.json"
    signal_path.write_text(
        json.dumps(
            [
                {
                    "event_id": "official-1",
                    "title": "Court filing resolved",
                    "market_id": "m1",
                    "token_id": "t1",
                    "category": "politics",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "fair_price": 0.75,
                    "source_price": 0.55,
                    "source_name": "Official Source",
                    "rationale": "Official filing materially improved resolution odds.",
                }
            ]
        ),
        encoding="utf-8",
    )

    async def _refresh_markets():
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return {"market_id": market_id, "token_id": token_id, "title": "Court filing resolved", "category": "politics", "active": True}

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        return _snapshot(token_id, 0.54, 0.55)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert len(decisions) == 1
    assert decisions[0].strategy_name == "event_driven_official"
    assert decisions[0].action == DecisionAction.PAPER_COPY
    assert decisions[0].context["source_name"] == "Official Source"


def test_correlation_dislocation_generates_paper_signal(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will reform pass by June 2026?",
            slug="will-reform-pass-by-june-2026",
            category="politics",
            end_date_iso="2026-06-30T00:00:00Z",
        ),
        "m2": MarketInfo(
            market_id="m2",
            token_id="t2",
            title="Will reform pass by December 2026?",
            slug="will-reform-pass-by-december-2026",
            category="politics",
            end_date_iso="2026-12-31T00:00:00Z",
        ),
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        market = engine.market_data.market_cache[market_id]
        return market.model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        if token_id == "t1":
            return _snapshot(token_id, 0.59, 0.61)
        return _snapshot(token_id, 0.44, 0.46)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert len(decisions) == 1
    assert decisions[0].strategy_name == "correlation_dislocation"
    assert decisions[0].action == DecisionAction.PAPER_COPY
    assert decisions[0].market_id == "m2"


def test_paired_binary_arb_generates_two_live_bundle_decisions(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    engine.config.strategies.paired_binary_live_enabled = True
    yes_market = MarketInfo(
        market_id="oscars-m1",
        token_id="yes-token",
        title="Will Anora win Best Picture at the 98th Academy Awards? [Yes]",
        slug="will-anora-win-best-picture-at-the-98th-academy-awards",
        category="entertainment / pop culture",
        end_date_iso="2026-03-16T06:00:00Z",
    )
    no_market = MarketInfo(
        market_id="oscars-m1",
        token_id="no-token",
        title="Will Anora win Best Picture at the 98th Academy Awards? [No]",
        slug="will-anora-win-best-picture-at-the-98th-academy-awards",
        category="entertainment / pop culture",
        end_date_iso="2026-03-16T06:00:00Z",
    )
    engine.market_data.token_cache = {
        yes_market.token_id: yes_market,
        no_market.token_id: no_market,
    }
    engine.market_data.market_cache = {yes_market.market_id: yes_market}

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return engine.market_data.token_cache[token_id].model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        if token_id == "yes-token":
            return _snapshot(token_id, 0.45, 0.46)
        return _snapshot(token_id, 0.46, 0.47)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    paired = [decision for decision in decisions if decision.strategy_name == "paired_binary_arb"]
    assert len(paired) == 2
    assert all(decision.action == DecisionAction.LIVE_COPY for decision in paired)
    assert all(decision.thesis_type == "paired_arb" for decision in paired)
    assert len({decision.bundle_id for decision in paired}) == 1
    assert {decision.bundle_role for decision in paired} == {"paired_yes", "paired_no"}


def test_event_driven_official_stays_live_disabled_by_default(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    engine.config.strategies.official_signal_live_enabled = False
    signal_path = tmp_path / "official_event_signals.json"
    signal_path.write_text(
        json.dumps(
            [
                {
                    "event_id": "official-live-1",
                    "title": "Court filing resolved",
                    "market_id": "m1",
                    "token_id": "t1",
                    "category": "politics",
                    "published_at": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
                    "fair_price": 0.75,
                    "source_price": 0.55,
                }
            ]
        ),
        encoding="utf-8",
    )

    async def _refresh_markets():
        return {}

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert len(decisions) == 1
    assert decisions[0].strategy_name == "event_driven_official"
    assert decisions[0].action == DecisionAction.SKIP
    assert decisions[0].reason_code == "STRATEGY_LIVE_DISABLED"


def test_event_driven_official_can_generate_live_signal_when_enabled(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    engine.config.strategies.official_signal_live_enabled = True
    signal_path = tmp_path / "official_event_signals.json"
    signal_path.write_text(
        json.dumps(
            [
                {
                    "event_id": "oscars-live-1",
                    "title": "Will Michael B. Jordan win Best Actor at the 98th Academy Awards?",
                    "market_slug": "will-michael-b-jordan-win-best-actor-at-the-98th-academy-awards",
                    "outcome": "Yes",
                    "category": "entertainment / pop culture",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "fair_price": 0.68,
                    "source_price": 0.59,
                    "source_reliability": 0.84,
                    "confidence_score": 0.85,
                    "source_name": "Academy Awards",
                }
            ]
        ),
        encoding="utf-8",
    )
    engine.market_data.market_cache = {
        "m-oscars": MarketInfo(
            market_id="m-oscars",
            token_id="t-oscars-yes",
            title="Will Michael B. Jordan win Best Actor at the 98th Academy Awards? [Yes]",
            slug="will-michael-b-jordan-win-best-actor-at-the-98th-academy-awards",
            category="entertainment / pop culture",
        )
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        market = engine.market_data.market_cache[market_id]
        return market.model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True, "derived_from_orderbook": False}

    async def _orderbook(token_id: str):
        return _snapshot(token_id, 0.58, 0.6)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert len(decisions) == 1
    assert decisions[0].strategy_name == "event_driven_official"
    assert decisions[0].action == DecisionAction.LIVE_COPY
    assert decisions[0].reason_code == "OK"
    assert decisions[0].entry_style == EntryStyle.PASSIVE_LIMIT
    assert decisions[0].executable_price == 0.59


def test_event_driven_official_live_smoke_accepts_near_threshold_oscars_signal(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    engine.config.strategies.official_signal_live_enabled = True
    signal_path = tmp_path / "official_event_signals.json"
    signal_path.write_text(
        json.dumps(
            [
                {
                    "event_id": "oscars-live-near-threshold",
                    "title": "Will Michael B. Jordan win Best Actor at the 98th Academy Awards?",
                    "market_slug": "will-michael-b-jordan-win-best-actor-at-the-98th-academy-awards",
                    "outcome": "Yes",
                    "category": "entertainment / pop culture",
                    "published_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
                    "fair_price": 0.68,
                    "source_price": 0.59,
                    "source_reliability": 0.84,
                    "confidence_score": 0.85,
                    "source_name": "Academy Awards",
                }
            ]
        ),
        encoding="utf-8",
    )
    engine.market_data.market_cache = {
        "m-oscars": MarketInfo(
            market_id="m-oscars",
            token_id="t-oscars-yes",
            title="Will Michael B. Jordan win Best Actor at the 98th Academy Awards? [Yes]",
            slug="will-michael-b-jordan-win-best-actor-at-the-98th-academy-awards",
            category="entertainment / pop culture",
        )
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        market = engine.market_data.market_cache[market_id]
        return market.model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True, "derived_from_orderbook": False}

    async def _orderbook(token_id: str):
        return _snapshot(token_id, 0.58, 0.6)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert len(decisions) == 1
    assert decisions[0].strategy_name == "event_driven_official"
    assert decisions[0].action == DecisionAction.LIVE_COPY
    assert decisions[0].reason_code == "OK"


def test_execution_priority_prefers_official_over_noisier_crypto_copy(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    official = TradeDecision(
        strategy_name="event_driven_official",
        allowed=True,
        action=DecisionAction.LIVE_COPY,
        reason_code="OK",
        human_readable_reason="Risk checks passed.",
        local_decision_id="official-1",
        wallet_address="strategy:event_driven_official",
        market_id="m-official",
        token_id="t-official",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        category="entertainment / pop culture",
        scaled_notional=3.0,
        source_price=0.6,
        executable_price=0.6,
        cluster_confirmed=False,
        hedge_suspicion_score=0.0,
        context={
            "wallet_global_score": 0.88,
            "live_fillability_score": 0.72,
            "signal_quality_score": 0.72,
            "confidence_score": 0.67,
            "live_fillability": {"freshness_score": 0.9},
            "selection_thesis": "event_driven_official",
        },
    )
    crypto = TradeDecision(
        strategy_name="wallet_follow",
        allowed=True,
        action=DecisionAction.LIVE_COPY,
        reason_code="OK",
        human_readable_reason="Risk checks passed.",
        local_decision_id="crypto-1",
        wallet_address="0xabc",
        market_id="m-crypto",
        token_id="t-crypto",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        category="crypto price",
        scaled_notional=1.0,
        source_price=0.44,
        executable_price=0.44,
        cluster_confirmed=False,
        hedge_suspicion_score=0.1,
        context={
            "wallet_global_score": 0.82,
            "live_fillability_score": 0.76,
            "signal_quality_score": 0.74,
            "confidence_score": 0.0,
            "live_fillability": {"freshness_score": 0.9},
            "selection_thesis": "single_wallet_copy",
        },
    )

    assert engine._execution_priority(official) > engine._execution_priority(crypto)


def test_live_signal_quality_rewards_consensus_mid_price(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    detection = _detection(price=0.46, category="politics")

    quality = engine._live_signal_quality_assessment(
        category="politics",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        detection=detection,
        wallet=_wallet(),
        executable_price=0.46,
        source_quality=SourceQuality.REAL_PUBLIC_DATA,
        stale_signal_limit_seconds=900.0,
        consensus_context={
            "same_side_wallets": 3,
            "opposing_wallets": 0,
            "consensus_ratio": 1.0,
            "notional_ratio": 1.0,
            "cluster_strength": 1.0,
            "selection_thesis": "wallet_consensus",
        },
        burst_size=1,
    )

    assert quality["passed"] is True
    assert quality["score"] >= engine.config.live.minimum_signal_quality_score
    assert quality["selection_thesis"] == "wallet_consensus"


def test_live_signal_quality_blocks_conflicting_wallet_flow(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    detection = _detection(price=0.46, category="politics")

    quality = engine._live_signal_quality_assessment(
        category="politics",
        entry_style=EntryStyle.PASSIVE_LIMIT,
        detection=detection,
        wallet=_wallet(),
        executable_price=0.46,
        source_quality=SourceQuality.REAL_PUBLIC_DATA,
        stale_signal_limit_seconds=900.0,
        consensus_context={
            "same_side_wallets": 1,
            "opposing_wallets": 2,
            "consensus_ratio": 0.3333,
            "notional_ratio": 0.25,
            "cluster_strength": 0.0,
            "selection_thesis": "single_wallet_copy",
        },
        burst_size=1,
    )

    assert quality["passed"] is False
    assert quality["reason_code"] == "SIGNAL_CONFLICT"


def test_live_wallet_follow_hard_skips_extreme_price_books(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    engine.config.live.enable_multi_entry_style_live = False
    engine.config.entry_styles.preferred_live_entry_style = EntryStyle.FOLLOW_TAKER
    detection = _detection(price=0.65, category="entertainment / pop culture").model_copy(
        update={
            "market_id": "m-oscars",
            "token_id": "t-oscars-yes",
            "market_title": "Will Michael B. Jordan win Best Actor at the 98th Academy Awards?",
            "market_slug": "will-michael-b-jordan-win-best-actor-at-the-98th-academy-awards",
            "size": 20.0,
            "notional": 13.0,
        }
    )

    async def _refresh_markets():
        return {}

    async def _stream_watchlist(_token_ids: list[str]):
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return {
            "market_id": market_id,
            "token_id": token_id,
            "title": detection.market_title,
            "category": "entertainment / pop culture",
            "active": True,
            "closed": False,
        }

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True, "derived_from_orderbook": False}

    async def _orderbook(token_id: str):
        return OrderbookSnapshot(
            token_id=token_id,
            bids=[OrderbookLevel(price=0.65, size=200.0)],
            asks=[OrderbookLevel(price=0.99, size=500.0)],
            timestamp=datetime.now(timezone.utc),
        )

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(
        engine.process_detections(
            [detection],
            ApprovedWallets(research_wallets=["0xabc"], paper_wallets=["0xabc"], live_wallets=["0xabc"]),
            [_wallet().model_copy(update={"dominant_category": "entertainment / pop culture"})],
        )
    )

    assert len(decisions) == 1
    assert decisions[0].action == DecisionAction.SKIP
    assert decisions[0].reason_code == "EXTREME_PRICE_BOOK"


def test_live_entry_style_gate_can_allow_multiple_styles_when_enabled(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)

    engine.config.live.enable_multi_entry_style_live = False
    assert engine._entry_style_allowed(EntryStyle.FOLLOW_TAKER, cluster_confirmed=False) is False

    engine.config.live.enable_multi_entry_style_live = True
    assert engine._entry_style_allowed(EntryStyle.FOLLOW_TAKER, cluster_confirmed=False) is True


def test_live_passive_limit_uses_resting_price_not_taker_ask(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    detection = _detection(price=0.52)
    wallet = _wallet()
    orderbook = _snapshot("t1", 0.05, 0.99)

    fill = engine._estimate_entry_fill(
        detection=detection,
        orderbook=orderbook,
        target_notional=1.0,
        entry_style=EntryStyle.PASSIVE_LIMIT,
    )

    assert fill.fillable is True
    assert fill.executable_price == 0.52
    assert fill.spread_pct == 0.0

    decision = engine._skip_decision(detection, wallet, 1.0, "NO_VALID_ENTRY_STYLE", "generic")
    decision = decision.model_copy(
        update={
            "reason_code": "ENTRY_DRIFT",
            "human_readable_reason": "Executable entry drift exceeds threshold.",
            "context": {
                "fill": fill.model_dump(),
                "risk_context": {"entry_drift_pct": 0.01},
            },
        }
    )
    generic = engine._skip_decision(detection, wallet, 1.0, "NO_VALID_ENTRY_STYLE", "generic")
    assert engine._decision_rank(decision, wallet, 0.0) > engine._decision_rank(generic, wallet, 0.0)


def test_live_wallet_follow_can_use_passive_entry_for_trusted_stale_signal(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    detection = _detection(price=0.52, category="politics").model_copy(
        update={
            "detection_latency_seconds": 400,
            "size": 30.0,
            "notional": 15.6,
            "market_id": "m-politics",
            "token_id": "t-politics",
        }
    )

    async def _refresh_markets():
        return {}

    async def _stream_watchlist(_token_ids: list[str]):
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return {
            "market_id": market_id,
            "token_id": token_id,
            "title": "Will reform pass?",
            "category": "politics",
            "active": True,
            "closed": False,
        }

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True, "derived_from_orderbook": False}

    async def _orderbook(token_id: str):
        return OrderbookSnapshot(
            token_id=token_id,
            bids=[OrderbookLevel(price=0.51, size=250.0)],
            asks=[OrderbookLevel(price=0.99, size=250.0)],
            timestamp=datetime.now(timezone.utc),
        )

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(
        engine.process_detections(
            [detection],
            ApprovedWallets(research_wallets=["0xabc"], paper_wallets=["0xabc"], live_wallets=["0xabc"]),
            [_wallet()],
        )
    )

    assert len(decisions) == 1
    assert decisions[0].action == DecisionAction.LIVE_COPY
    assert decisions[0].entry_style == EntryStyle.PASSIVE_LIMIT
    assert decisions[0].reason_code == "OK"
    assert decisions[0].context["stale_signal_limit_seconds"] == 900.0


def test_live_style_comparison_does_not_consume_entry_rate_limit_before_final_choice(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    engine.config.risk.max_new_entries_per_hour = 1
    detection = _detection(price=0.523, category="politics").model_copy(update={"size": 200, "notional": 104.6})

    async def _refresh_markets():
        return {}

    async def _stream_watchlist(_token_ids: list[str]):
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return {
            "market_id": market_id,
            "token_id": token_id,
            "title": "Test market",
            "category": "politics",
            "active": True,
            "closed": False,
        }

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        return OrderbookSnapshot(
            token_id=token_id,
            bids=[OrderbookLevel(price=0.52, size=100.0)],
            asks=[
                OrderbookLevel(price=0.535, size=2.0),
                OrderbookLevel(price=0.544, size=200.0),
            ],
            timestamp=datetime.now(timezone.utc),
        )

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(
        engine.process_detections(
            [detection],
            ApprovedWallets(research_wallets=["0xabc"], paper_wallets=["0xabc"], live_wallets=["0xabc"]),
            [_wallet()],
        )
    )

    assert len(decisions) == 1
    assert decisions[0].allowed is True
    assert decisions[0].action == DecisionAction.LIVE_COPY
    assert decisions[0].entry_style == EntryStyle.PASSIVE_LIMIT
    assert decisions[0].reason_code == "OK"
    assert engine.risk.entries_last_hour() == 0


def test_wallet_follow_dedupes_repeated_market_bursts_to_latest_detection(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    older = _detection(price=0.52, category="crypto price").model_copy(
        update={
            "event_key": "evt-older",
            "transaction_hash": "tx-older",
            "notional": 5.2,
            "source_trade_timestamp": datetime.now(timezone.utc) - timedelta(seconds=30),
            "local_detection_timestamp": datetime.now(timezone.utc) - timedelta(seconds=30),
        }
    )
    newer = _detection(price=0.54, category="crypto price").model_copy(
        update={
            "event_key": "evt-newer",
            "transaction_hash": "tx-newer",
            "notional": 10.8,
            "source_trade_timestamp": datetime.now(timezone.utc),
            "local_detection_timestamp": datetime.now(timezone.utc),
        }
    )

    deduped, burst_sizes = engine._dedupe_wallet_follow_detections([older, newer])

    assert len(deduped) == 1
    assert deduped[0].event_key == "evt-newer"
    assert burst_sizes["evt-newer"] == 2


def test_live_decision_rank_penalizes_extreme_price_allowed_entries(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    wallet = _wallet()
    base = engine._skip_decision(_detection(), wallet, 1.0, "OK", "ok")
    moderate = base.model_copy(
        update={
            "allowed": True,
            "action": DecisionAction.LIVE_COPY,
            "entry_style": EntryStyle.PASSIVE_LIMIT,
            "executable_price": 0.46,
            "context": {"fill": {"slippage_pct": 0.0}, "burst_size": 1},
        }
    )
    extreme = moderate.model_copy(update={"executable_price": 0.99})

    assert engine._decision_rank(moderate, wallet, 0.0) > engine._decision_rank(extreme, wallet, 0.0)


def test_event_driven_official_can_map_market_from_title_and_outcome(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    signal_path = tmp_path / "official_event_signals.json"
    signal_path.write_text(
        json.dumps(
            [
                {
                    "event_id": "official-map-1",
                    "title": "Will reform pass by June 2026?",
                    "outcome": "Yes",
                    "category": "politics",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "fair_price": 0.78,
                    "source_price": 0.55,
                    "source_reliability": 0.9,
                    "confidence_score": 0.9,
                    "source_name": "Official Calendar",
                }
            ]
        ),
        encoding="utf-8",
    )
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t-yes",
            title="Will reform pass by June 2026? [Yes]",
            slug="will-reform-pass-by-june-2026",
            category="politics",
        ),
        "m2": MarketInfo(
            market_id="m2",
            token_id="t-no",
            title="Will reform pass by June 2026? [No]",
            slug="will-reform-pass-by-june-2026",
            category="politics",
        ),
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        market = next(m for m in engine.market_data.market_cache.values() if m.market_id == market_id and m.token_id == token_id)
        return market.model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        return _snapshot(token_id, 0.54, 0.56)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert len(decisions) == 1
    assert decisions[0].strategy_name == "event_driven_official"
    assert decisions[0].market_id == "m1"
    assert decisions[0].token_id == "t-yes"
    assert decisions[0].context["mapping_reason"] == "title_match"
    assert decisions[0].context["source_reliability"] == 0.9


def test_event_driven_official_can_use_direct_market_lookup_when_cache_misses(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
    engine.config.strategies.official_signal_live_enabled = True
    signal_path = tmp_path / "official_event_signals.json"
    signal_path.write_text(
        json.dumps(
            [
                {
                    "event_id": "oscars-direct-1",
                    "title": "Will Michael B. Jordan win Best Actor at the 98th Academy Awards?",
                    "market_slug": "will-michael-b-jordan-win-best-actor-at-the-98th-academy-awards",
                    "outcome": "Yes",
                    "category": "entertainment / pop culture",
                    "published_at": (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat(),
                    "fair_price": 0.68,
                    "source_price": 0.59,
                    "source_reliability": 0.84,
                    "confidence_score": 0.85,
                }
            ]
        ),
        encoding="utf-8",
    )

    async def _refresh_markets():
        engine.market_data.market_cache = {}
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = "", market_slug: str = "", outcome: str = ""):
        if market_slug:
            assert market_slug == "will-michael-b-jordan-win-best-actor-at-the-98th-academy-awards"
            assert outcome == "Yes"
        else:
            assert market_id == "m-oscars"
            assert token_id == "t-oscars-yes"
        return {
            "market_id": "m-oscars",
            "token_id": "t-oscars-yes",
            "title": "Will Michael B. Jordan win Best Actor at the 98th Academy Awards? [Yes]",
            "slug": "will-michael-b-jordan-win-best-actor-at-the-98th-academy-awards",
            "category": "entertainment / pop culture",
            "active": True,
            "closed": False,
            "liquidity": 5000.0,
        }

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True, "derived_from_orderbook": False}

    async def _orderbook(token_id: str):
        return _snapshot(token_id, 0.58, 0.6)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert len(decisions) == 1
    assert decisions[0].strategy_name == "event_driven_official"
    assert decisions[0].action == DecisionAction.LIVE_COPY
    assert decisions[0].reason_code == "OK"
    assert decisions[0].context["mapping_reason"] == "direct_market_lookup"


def test_event_driven_official_skips_low_source_reliability(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    signal_path = tmp_path / "official_event_signals.json"
    signal_path.write_text(
        json.dumps(
            [
                {
                    "event_id": "official-low-rel-1",
                    "title": "Will reform pass by June 2026?",
                    "market_slug": "will-reform-pass-by-june-2026",
                    "outcome": "Yes",
                    "category": "politics",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "fair_price": 0.75,
                    "source_price": 0.55,
                    "source_reliability": 0.45,
                }
            ]
        ),
        encoding="utf-8",
    )
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t-yes",
            title="Will reform pass by June 2026? [Yes]",
            slug="will-reform-pass-by-june-2026",
            category="politics",
        )
    }
    async def _refresh_markets():
        return engine.market_data.market_cache

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert decisions == []
    signal_log = (tmp_path / "strategy_signal_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert signal_log
    payloads = [json.loads(line) for line in signal_log]
    assert any(
        item["strategy_name"] == "event_driven_official" and item["reason_code"] == "LOW_SOURCE_RELIABILITY"
        for item in payloads
    )


def test_event_driven_official_skips_low_computed_confidence(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    signal_path = tmp_path / "official_event_signals.json"
    signal_path.write_text(
        json.dumps(
            [
                {
                    "event_id": "official-low-conf-1",
                    "title": "Will reform pass by June 2026?",
                    "market_slug": "will-reform-pass-by-june-2026",
                    "outcome": "Yes",
                    "category": "politics",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "fair_price": 0.64,
                    "source_price": 0.55,
                    "source_reliability": 0.72,
                    "confidence_score": 0.55,
                }
            ]
        ),
        encoding="utf-8",
    )
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t-yes",
            title="Will reform pass by June 2026? [Yes]",
            slug="will-reform-pass-by-june-2026",
            category="politics",
        )
    }
    async def _refresh_markets():
        return engine.market_data.market_cache

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert decisions == []
    signal_log = (tmp_path / "strategy_signal_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert signal_log
    payloads = [json.loads(line) for line in signal_log]
    assert any(
        item["strategy_name"] == "event_driven_official" and item["reason_code"] == "LOW_SIGNAL_CONFIDENCE"
        for item in payloads
    )


def test_event_driven_official_skips_ambiguous_market_mapping(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    signal_path = tmp_path / "official_event_signals.json"
    signal_path.write_text(
        json.dumps(
            [
                {
                    "event_id": "official-ambiguous-1",
                    "title": "Will reform pass?",
                    "category": "politics",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "fair_price": 0.71,
                    "source_price": 0.55,
                }
            ]
        ),
        encoding="utf-8",
    )
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will reform pass by June 2026? [Yes]",
            slug="will-reform-pass-by-june-2026",
            category="politics",
        ),
        "m2": MarketInfo(
            market_id="m2",
            token_id="t2",
            title="Will reform pass by December 2026? [Yes]",
            slug="will-reform-pass-by-december-2026",
            category="politics",
        ),
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert decisions == []
    signal_log = (tmp_path / "strategy_signal_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert signal_log
    payloads = [json.loads(line) for line in signal_log]
    assert any(
        item["strategy_name"] == "event_driven_official" and item["reason_code"] == "AMBIGUOUS_TITLE_MATCH"
        for item in payloads
    )


def test_correlation_dislocation_uses_best_pair_not_just_extremes(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will reform pass by June 2026?",
            slug="will-reform-pass-by-june-2026",
            category="politics",
            end_date_iso="2026-06-30T00:00:00Z",
        ),
        "m2": MarketInfo(
            market_id="m2",
            token_id="t2",
            title="Will reform pass by September 2026?",
            slug="will-reform-pass-by-september-2026",
            category="politics",
            end_date_iso="2026-09-30T00:00:00Z",
        ),
        "m3": MarketInfo(
            market_id="m3",
            token_id="t3",
            title="Will reform pass by December 2026?",
            slug="will-reform-pass-by-december-2026",
            category="politics",
            end_date_iso="2026-12-31T00:00:00Z",
        ),
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        market = engine.market_data.market_cache[market_id]
        return market.model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        if token_id == "t1":
            return _snapshot(token_id, 0.68, 0.70)
        if token_id == "t2":
            return _snapshot(token_id, 0.39, 0.41)
        return _snapshot(token_id, 0.66, 0.68)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert len(decisions) == 1
    assert decisions[0].strategy_name == "correlation_dislocation"
    assert decisions[0].market_id == "m2"
    assert decisions[0].context["reference_market_id"] == "m1"


def test_correlation_dislocation_groups_markets_with_variant_wording(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will the Fed cut rates by June 2026?",
            slug="will-the-fed-cut-rates-by-june-2026",
            category="macro / economics",
            end_date_iso="2026-06-30T00:00:00Z",
            liquidity=5000.0,
        ),
        "m2": MarketInfo(
            market_id="m2",
            token_id="t2",
            title="Will the Fed reduce rates before September 2026?",
            slug="will-the-fed-reduce-rates-before-september-2026",
            category="macro / economics",
            end_date_iso="2026-09-30T00:00:00Z",
            liquidity=5000.0,
        ),
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return engine.market_data.market_cache[market_id].model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        if token_id == "t1":
            return _snapshot(token_id, 0.66, 0.68)
        return _snapshot(token_id, 0.48, 0.50)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert any(decision.strategy_name == "correlation_dislocation" for decision in decisions)


def test_correlation_dislocation_ignores_weak_same_entity_different_competition_pairs(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    engine.config.strategies.supplemental_paper_relaxed_enabled = False
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will Arsenal win the 2025 Carabao Cup? [Yes]",
            slug="will-arsenal-win-the-2025-carabao-cup",
            category="sports",
            end_date_iso="2026-03-10T00:00:00Z",
            liquidity=5000.0,
        ),
        "m2": MarketInfo(
            market_id="m2",
            token_id="t2",
            title="Will Arsenal win the 2025-26 English Premier League? [No]",
            slug="will-arsenal-win-the-2025-26-english-premier-league",
            category="sports",
            end_date_iso="2026-05-10T00:00:00Z",
            liquidity=5000.0,
        ),
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return engine.market_data.market_cache[market_id].model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        if token_id == "t1":
            return _snapshot(token_id, 0.49, 0.51)
        return _snapshot(token_id, 0.49, 0.51)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert all(decision.strategy_name != "correlation_dislocation" for decision in decisions)
    payloads = [
        json.loads(line)
        for line in (tmp_path / "strategy_signal_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    match = next(item for item in payloads if item["strategy_name"] == "correlation_dislocation")
    assert match["reason_code"] == "NO_RELATIONSHIP_GROUPS"


def test_correlation_dislocation_logs_empty_cycle_when_no_groups(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)

    async def _refresh_markets():
        return {}

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert decisions == []
    signal_log = (tmp_path / "strategy_signal_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payloads = [json.loads(line) for line in signal_log]
    assert any(
        item["strategy_name"] == "correlation_dislocation" and item["reason_code"] == "NO_RELATIONSHIP_GROUPS"
        for item in payloads
    )


def test_resolution_window_generates_paper_signal(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    near_end = datetime.now(timezone.utc) + timedelta(hours=8)
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will bill pass by this weekend? [Yes]",
            slug="will-bill-pass-by-this-weekend",
            category="politics",
            end_date_iso=near_end.isoformat(),
            liquidity=2500.0,
        )
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return engine.market_data.market_cache[market_id].model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        return _snapshot(token_id, 0.85, 0.86)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    resolution = [decision for decision in decisions if decision.strategy_name == "resolution_window"]
    assert len(resolution) == 1
    assert resolution[0].action == DecisionAction.PAPER_COPY


def test_resolution_window_logs_empty_cycle_when_no_near_resolution_markets(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert decisions == []
    signal_log = (tmp_path / "strategy_signal_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payloads = [json.loads(line) for line in signal_log]
    assert any(
        item["strategy_name"] == "resolution_window" and item["reason_code"] == "NO_NEAR_RESOLUTION_MARKETS"
        for item in payloads
    )


def test_resolution_window_ignores_sports_markets(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    near_end = datetime.now(timezone.utc) + timedelta(hours=10)
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will Santos FC vs. SC Corinthians Paulista end in a draw? [Yes]",
            slug="will-santos-fc-vs-sc-corinthians-paulista-end-in-a-draw",
            category="sports",
            end_date_iso=near_end.isoformat(),
            liquidity=3000.0,
        )
    }

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert all(decision.strategy_name != "resolution_window" for decision in decisions)
    payloads = [
        json.loads(line)
        for line in (tmp_path / "strategy_signal_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    match = next(item for item in payloads if item["strategy_name"] == "resolution_window")
    assert match["reason_code"] == "NO_NEAR_RESOLUTION_MARKETS"


def test_correlation_dislocation_logs_top_near_miss_details_when_no_dislocation(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    engine.config.strategies.supplemental_paper_relaxed_enabled = False
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will reform pass by June 2026?",
            slug="will-reform-pass-by-june-2026",
            category="politics",
            end_date_iso="2026-06-30T00:00:00Z",
            liquidity=5000.0,
        ),
        "m2": MarketInfo(
            market_id="m2",
            token_id="t2",
            title="Will reform pass by December 2026?",
            slug="will-reform-pass-by-december-2026",
            category="politics",
            end_date_iso="2026-12-31T00:00:00Z",
            liquidity=5000.0,
        ),
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return engine.market_data.market_cache[market_id].model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        if token_id == "t1":
            return _snapshot(token_id, 0.60, 0.62)
        return _snapshot(token_id, 0.578, 0.594)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert all(decision.strategy_name != "correlation_dislocation" for decision in decisions)
    payloads = [
        json.loads(line)
        for line in (tmp_path / "strategy_signal_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    match = next(item for item in payloads if item["strategy_name"] == "correlation_dislocation" and item["reason_code"] == "NO_PRICE_DISLOCATIONS")
    assert match["top_near_miss_market_id"] == "m2"
    assert match["top_near_miss_reference_market_id"] == "m1"
    assert "top_near_miss_gap_ratio" in match


def test_resolution_window_logs_top_near_miss_details_when_edge_too_small(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    engine.config.strategies.supplemental_paper_relaxed_enabled = False
    near_end = datetime.now(timezone.utc) + timedelta(hours=10)
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will bill pass by this weekend? [Yes]",
            slug="will-bill-pass-by-this-weekend",
            category="politics",
            end_date_iso=near_end.isoformat(),
            liquidity=3000.0,
        )
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return engine.market_data.market_cache[market_id].model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        return _snapshot(token_id, 0.885, 0.905)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert all(decision.strategy_name != "resolution_window" for decision in decisions)
    payloads = [
        json.loads(line)
        for line in (tmp_path / "strategy_signal_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    match = next(item for item in payloads if item["strategy_name"] == "resolution_window" and item["reason_code"] == "NO_RESOLUTION_WINDOW_EDGES")
    assert match["top_near_miss_market_id"] == "m1"
    assert match["top_near_miss_reason"] == "EDGE_NEAR_MISS"
    assert "top_near_miss_fair_price" in match


def test_resolution_window_logs_above_fair_anchor_for_very_high_ask(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    engine.config.strategies.supplemental_paper_relaxed_enabled = False
    near_end = datetime.now(timezone.utc) + timedelta(hours=12)
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will Trump say Hormuz this week? [Yes]",
            slug="will-trump-say-hormuz-this-week",
            category="politics",
            end_date_iso=near_end.isoformat(),
            liquidity=1300.0,
        )
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return engine.market_data.market_cache[market_id].model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        return _snapshot(token_id, 0.98, 0.99)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert all(decision.strategy_name != "resolution_window" for decision in decisions)
    payloads = [
        json.loads(line)
        for line in (tmp_path / "strategy_signal_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    match = next(item for item in payloads if item["strategy_name"] == "resolution_window" and item["reason_code"] == "NO_RESOLUTION_WINDOW_EDGES")
    assert match["top_near_miss_market_id"] == "m1"
    assert match["top_near_miss_reason"] == "ABOVE_FAIR_ANCHOR"


def test_correlation_dislocation_promotes_near_miss_pair_for_paper(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will reform pass by June 2026?",
            slug="will-reform-pass-by-june-2026",
            category="politics",
            end_date_iso="2026-06-30T00:00:00Z",
            liquidity=5000.0,
        ),
        "m2": MarketInfo(
            market_id="m2",
            token_id="t2",
            title="Will reform pass by December 2026?",
            slug="will-reform-pass-by-december-2026",
            category="politics",
            end_date_iso="2026-12-31T00:00:00Z",
            liquidity=5000.0,
        ),
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return engine.market_data.market_cache[market_id].model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        if token_id == "t1":
            return _snapshot(token_id, 0.60, 0.62)
        return _snapshot(token_id, 0.575, 0.595)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    correlation = [decision for decision in decisions if decision.strategy_name == "correlation_dislocation"]
    assert len(correlation) == 1
    assert correlation[0].action == DecisionAction.PAPER_COPY
    assert correlation[0].context["strategy_near_miss"] is True


def test_resolution_window_promotes_near_miss_candidate_for_paper(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)
    near_end = datetime.now(timezone.utc) + timedelta(hours=10)
    engine.market_data.market_cache = {
        "m1": MarketInfo(
            market_id="m1",
            token_id="t1",
            title="Will bill pass by this weekend? [Yes]",
            slug="will-bill-pass-by-this-weekend",
            category="politics",
            end_date_iso=near_end.isoformat(),
            liquidity=3000.0,
        )
    }

    async def _refresh_markets():
        return engine.market_data.market_cache

    async def _fetch_market_metadata(market_id: str, token_id: str = ""):
        return engine.market_data.market_cache[market_id].model_dump()

    async def _tradability(_market_id: str, _token_id: str):
        return {"tradable": True, "orderbook_enabled": True}

    async def _orderbook(token_id: str):
        return _snapshot(token_id, 0.885, 0.905)

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    resolution = [decision for decision in decisions if decision.strategy_name == "resolution_window"]
    assert len(resolution) == 1
    assert resolution[0].action == DecisionAction.PAPER_COPY
    assert resolution[0].context["strategy_near_miss"] is True


def test_event_driven_official_logs_empty_cycle_when_no_curated_signals(tmp_path: Path) -> None:
    engine = _paper_engine(tmp_path)

    decisions = asyncio.run(engine.process_detections([], ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=[]), []))

    assert decisions == []
    signal_log = (tmp_path / "strategy_signal_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payloads = [json.loads(line) for line in signal_log]
    assert any(
        item["strategy_name"] == "event_driven_official" and item["reason_code"] == "NO_OFFICIAL_SIGNALS_CONFIGURED"
        for item in payloads
    )
