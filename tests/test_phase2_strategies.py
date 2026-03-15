from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import load_config
from src.models import (
    ApprovedWallets,
    DecisionAction,
    EntryStyle,
    MarketInfo,
    OrderbookLevel,
    OrderbookSnapshot,
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


def test_event_driven_official_stays_live_disabled_by_default(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)
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


def test_live_entry_style_gate_can_allow_multiple_styles_when_enabled(tmp_path: Path) -> None:
    engine = _live_engine(tmp_path)

    engine.config.live.enable_multi_entry_style_live = False
    assert engine._entry_style_allowed(EntryStyle.FOLLOW_TAKER, cluster_confirmed=False) is False

    engine.config.live.enable_multi_entry_style_live = True
    assert engine._entry_style_allowed(EntryStyle.FOLLOW_TAKER, cluster_confirmed=False) is True


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
