from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.config import load_config
from src.market_data import MarketDataService
from src.models import ApprovedWallets, DetectionEvent, DiscoveryState, MarketInfo, SourceQuality, WalletMetrics
from src.state import AppStateStore
from src.strategy import StrategyEngine


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
        hedge_suspicion_score=0.4,
        source_quality=SourceQuality.REAL_PUBLIC_DATA,
    )


def _detection() -> DetectionEvent:
    return DetectionEvent(
        event_key="evt-1",
        wallet_address="0xabc",
        market_title="Test market",
        market_slug="test-market",
        market_id="missing-market",
        token_id="token-1",
        side="BUY",
        price=0.5,
        size=10,
        notional=5,
        transaction_hash="tx-1",
        detection_latency_seconds=5,
        category="politics",
        source_quality=SourceQuality.REAL_PUBLIC_DATA,
    )


def test_live_missing_market_metadata_skips_instead_of_crashing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "LIVE",
            "system_status": "LIVE_READY",
            "wallet_discovery_state": DiscoveryState.SUCCESS.value,
            "wallet_scoring_state": "SUCCESS",
            "live_readiness_last_result": {"ready": True},
            "approved_wallets": {"paper_wallets": [], "research_wallets": [], "live_wallets": ["0xabc"]},
        }
    )
    engine = StrategyEngine(config, tmp_path, state)

    async def _refresh_markets() -> dict[str, object]:
        return {}

    async def _stream_watchlist(token_ids: list[str]) -> dict[str, dict]:
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = "") -> dict[str, object]:
        raise RuntimeError(f"Missing market metadata for {market_id} in LIVE mode.")

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]

    decisions = asyncio.run(
        engine.process_detections(
            [_detection().model_copy(update={"token_id": "unknown-token"})],
            ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=["0xabc"]),
            [_wallet()],
        )
    )

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.allowed is False
    assert decision.reason_code == "MISSING_MARKET_METADATA"
    assert decision.context["market_meta_error"].startswith("Missing market metadata")
    trace_rows = (tmp_path / "paper_decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(trace_rows[-1])
    assert payload["reason_code"] == "MISSING_MARKET_METADATA"
    assert payload["final_action"] == "SKIP"


def test_live_missing_token_id_skips_without_touching_ws(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "LIVE",
            "system_status": "LIVE_READY",
            "wallet_discovery_state": DiscoveryState.SUCCESS.value,
            "wallet_scoring_state": "SUCCESS",
            "live_readiness_last_result": {"ready": True},
            "approved_wallets": {"paper_wallets": [], "research_wallets": [], "live_wallets": ["0xabc"]},
        }
    )
    engine = StrategyEngine(config, tmp_path, state)
    detection = _detection().model_copy(update={"token_id": ""})

    async def _refresh_markets() -> dict[str, object]:
        return {}

    async def _stream_watchlist(token_ids: list[str]) -> dict[str, dict]:
        raise AssertionError(f"websocket watchlist should not be called for blank token ids: {token_ids}")

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]

    decisions = asyncio.run(
        engine.process_detections(
            [detection],
            ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=["0xabc"]),
            [_wallet()],
        )
    )

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.allowed is False
    assert decision.reason_code == "MISSING_TOKEN_ID"
    trace_rows = (tmp_path / "paper_decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(trace_rows[-1])
    assert payload["reason_code"] == "MISSING_TOKEN_ID"
    assert payload["final_action"] == "SKIP"


def test_market_data_uses_direct_lookup_when_cache_misses(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    service = MarketDataService(config, tmp_path)

    async def _refresh_markets(limit: int = 500) -> list[MarketInfo]:
        return []

    async def _fetch_market_lookup(market_id: str, token_id: str = "") -> MarketInfo | None:
        assert market_id == "missing-market"
        assert token_id == "token-1"
        return MarketInfo(
            market_id="missing-market",
            token_id="token-1",
            title="Recovered market",
            slug="recovered-market",
            category="politics",
            active=True,
            closed=False,
            liquidity=120.0,
            volume=350.0,
            source_quality=SourceQuality.REAL_PUBLIC_DATA,
        )

    service.client.fetch_markets = _refresh_markets  # type: ignore[method-assign]
    service.client.fetch_market_lookup = _fetch_market_lookup  # type: ignore[method-assign]

    payload = asyncio.run(service.fetch_market_metadata("missing-market", "token-1"))
    assert payload["market_id"] == "missing-market"
    assert payload["token_id"] == "token-1"
    assert payload["title"] == "Recovered market"


def test_live_direct_lookup_prevents_missing_market_skip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "LIVE",
            "system_status": "LIVE_READY",
            "wallet_discovery_state": DiscoveryState.SUCCESS.value,
            "wallet_scoring_state": "SUCCESS",
            "live_readiness_last_result": {"ready": True},
            "approved_wallets": {"paper_wallets": [], "research_wallets": [], "live_wallets": ["0xabc"]},
            "allowance_sufficient": True,
            "balance_visible": True,
            "manual_live_enable": True,
            "reconciliation_clean": True,
            "heartbeat_ok": True,
            "tradability_ok": True,
            "live_health_state": "HEALTHY",
        }
    )
    engine = StrategyEngine(config, tmp_path, state)

    async def _refresh_markets() -> dict[str, object]:
        return {}

    async def _stream_watchlist(token_ids: list[str]) -> dict[str, dict]:
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = "") -> dict[str, object]:
        return {
            "market_id": market_id,
            "token_id": token_id or "token-1",
            "title": "Resolved market",
            "slug": "resolved-market",
            "category": "politics",
            "active": True,
            "closed": False,
        }

    async def _tradability(market_id: str, token_id: str) -> dict[str, object]:
        return {
            "market_id": market_id,
            "token_id": token_id,
            "tradable": False,
            "orderbook_enabled": False,
            "category": "politics",
            "title": "Resolved market",
            "liquidity": 0.0,
        }

    async def _orderbook(token_id: str):
        from src.models import OrderbookLevel, OrderbookSnapshot

        return OrderbookSnapshot(
            token_id=token_id,
            bids=[OrderbookLevel(price=0.49, size=10.0)],
            asks=[OrderbookLevel(price=0.51, size=10.0)],
        )

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(
        engine.process_detections(
            [_detection()],
            ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=["0xabc"]),
            [_wallet()],
        )
    )

    assert len(decisions) == 1
    assert decisions[0].reason_code != "MISSING_MARKET_METADATA"


def test_live_allowed_decision_uses_live_wallet_list_not_paper_wallet_list(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config = config.model_copy(
        update={
            "risk": config.risk.model_copy(update={"require_cluster_confirmation_live": False}),
            "live": config.live.model_copy(update={"only_cluster_confirmed": False}),
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "LIVE",
            "system_status": "LIVE_READY",
            "wallet_discovery_state": DiscoveryState.SUCCESS.value,
            "wallet_scoring_state": "SUCCESS",
            "live_readiness_last_result": {"ready": True},
            "allowance_sufficient": True,
            "balance_visible": True,
            "manual_live_enable": True,
            "reconciliation_clean": True,
            "heartbeat_ok": True,
            "live_health_state": "HEALTHY",
        }
    )
    engine = StrategyEngine(config, tmp_path, state)

    async def _refresh_markets() -> dict[str, object]:
        return {}

    async def _stream_watchlist(token_ids: list[str]) -> dict[str, dict]:
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = "") -> dict[str, object]:
        return {
            "market_id": market_id,
            "token_id": token_id,
            "title": "Resolved market",
            "slug": "resolved-market",
            "category": "politics",
            "active": True,
            "closed": False,
        }

    async def _tradability(market_id: str, token_id: str) -> dict[str, object]:
        return {
            "market_id": market_id,
            "token_id": token_id,
            "tradable": True,
            "orderbook_enabled": True,
            "category": "politics",
            "title": "Resolved market",
            "liquidity": 1000.0,
        }

    async def _orderbook(token_id: str):
        from src.models import OrderbookLevel, OrderbookSnapshot

        return OrderbookSnapshot(
            token_id=token_id,
            bids=[OrderbookLevel(price=0.50, size=1000.0)],
            asks=[OrderbookLevel(price=0.50, size=1000.0)],
        )

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    # Use original detection with price=0.5 — politics category has fee=0 so fee gate doesn't apply
    decisions = asyncio.run(
        engine.process_detections(
            [_detection()],
            ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=["0xabc"]),
            [_wallet()],
        )
    )

    assert len(decisions) == 1
    assert decisions[0].allowed is True
    assert decisions[0].action.value == "LIVE_COPY"


def test_live_unknown_detection_category_uses_market_metadata_category(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config = config.model_copy(
        update={
            "risk": config.risk.model_copy(update={"require_cluster_confirmation_live": False}),
            "live": config.live.model_copy(update={"only_cluster_confirmed": False}),
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "LIVE",
            "system_status": "LIVE_READY",
            "wallet_discovery_state": DiscoveryState.SUCCESS.value,
            "wallet_scoring_state": "SUCCESS",
            "live_readiness_last_result": {"ready": True},
            "allowance_sufficient": True,
            "balance_visible": True,
            "manual_live_enable": True,
            "manual_resume_required": False,
            "reconciliation_clean": True,
            "heartbeat_ok": True,
            "live_health_state": "HEALTHY",
        }
    )
    engine = StrategyEngine(config, tmp_path, state)
    detection = _detection().model_copy(
        update={
            "category": "unknown",
            "market_title": "Will the Senate pass the budget bill?",
            "market_slug": "will-the-senate-pass-the-budget-bill",
            "price": 0.55,
            "size": 20.0,
            "notional": 20.0,
        }
    )

    async def _refresh_markets() -> dict[str, object]:
        return {}

    async def _stream_watchlist(token_ids: list[str]) -> dict[str, dict]:
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = "") -> dict[str, object]:
        return {
            "market_id": market_id,
            "token_id": token_id,
            "title": "Will the Senate pass the budget bill?",
            "slug": "will-the-senate-pass-the-budget-bill",
            "category": "politics",
            "active": True,
            "closed": False,
        }

    async def _tradability(market_id: str, token_id: str) -> dict[str, object]:
        return {
            "market_id": market_id,
            "token_id": token_id,
            "tradable": True,
            "orderbook_enabled": True,
            "category": "politics",
            "title": "Will the Senate pass the budget bill?",
            "liquidity": 1000.0,
        }

    async def _orderbook(token_id: str):
        from src.models import OrderbookLevel, OrderbookSnapshot

        return OrderbookSnapshot(
            token_id=token_id,
            bids=[OrderbookLevel(price=0.54, size=1000.0)],
            asks=[OrderbookLevel(price=0.56, size=1000.0)],
        )

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(
        engine.process_detections(
            [detection],
            ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=["0xabc"]),
            [_wallet()],
        )
    )

    assert len(decisions) == 1
    assert decisions[0].allowed is True
    assert decisions[0].action.value == "LIVE_COPY"
    assert decisions[0].category == "politics"


def test_live_metadata_fallback_uses_detection_fields_when_token_is_present(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config = config.model_copy(
        update={
            "risk": config.risk.model_copy(update={"require_cluster_confirmation_live": False}),
            "live": config.live.model_copy(update={"only_cluster_confirmed": False}),
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "LIVE",
            "system_status": "LIVE_READY",
            "wallet_discovery_state": DiscoveryState.SUCCESS.value,
            "wallet_scoring_state": "SUCCESS",
            "live_readiness_last_result": {"ready": True},
            "allowance_sufficient": True,
            "balance_visible": True,
            "manual_live_enable": True,
            "reconciliation_clean": True,
            "heartbeat_ok": True,
            "live_health_state": "HEALTHY",
        }
    )
    engine = StrategyEngine(config, tmp_path, state)

    async def _refresh_markets() -> dict[str, object]:
        return {}

    async def _stream_watchlist(token_ids: list[str]) -> dict[str, dict]:
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = "") -> dict[str, object]:
        raise RuntimeError("gamma miss")

    async def _tradability(market_id: str, token_id: str) -> dict[str, object]:
        return {
            "market_id": market_id,
            "token_id": token_id,
            "tradable": True,
            "orderbook_enabled": True,
            "category": "politics",
            "title": "Recovered from token",
            "liquidity": 500.0,
            "derived_from_orderbook": True,
        }

    async def _orderbook(token_id: str):
        from src.models import OrderbookLevel, OrderbookSnapshot

        return OrderbookSnapshot(
            token_id=token_id,
            bids=[OrderbookLevel(price=0.50, size=1000.0)],
            asks=[OrderbookLevel(price=0.50, size=1000.0)],
        )

    detection = _detection().model_copy(
        update={
            "market_title": "Will the US officially declare war on Iran by December 31, 2026?",
            "market_slug": "war-on-iran-2026",
            "category": "politics",
        }
    )

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(
        engine.process_detections(
            [detection],
            ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=["0xabc"]),
            [_wallet()],
        )
    )

    assert len(decisions) == 1
    assert decisions[0].reason_code != "MISSING_MARKET_METADATA"


def test_live_tradability_failure_skips_instead_of_crashing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "LIVE",
            "system_status": "LIVE_READY",
            "wallet_discovery_state": DiscoveryState.SUCCESS.value,
            "wallet_scoring_state": "SUCCESS",
            "live_readiness_last_result": {"ready": True},
            "approved_wallets": {"paper_wallets": [], "research_wallets": [], "live_wallets": ["0xabc"]},
        }
    )
    engine = StrategyEngine(config, tmp_path, state)

    async def _refresh_markets() -> dict[str, object]:
        return {}

    async def _stream_watchlist(token_ids: list[str]) -> dict[str, dict]:
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = "") -> dict[str, object]:
        return {
            "market_id": market_id,
            "token_id": token_id or "token-1",
            "title": "Resolved market",
            "slug": "resolved-market",
            "category": "politics",
            "active": True,
            "closed": False,
        }

    async def _tradability(market_id: str, token_id: str) -> dict[str, object]:
        raise RuntimeError("tradability miss")

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]

    decisions = asyncio.run(
        engine.process_detections(
            [_detection()],
            ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=["0xabc"]),
            [_wallet()],
        )
    )

    assert len(decisions) == 1
    assert decisions[0].allowed is False
    assert decisions[0].reason_code == "MISSING_TRADABILITY"
    assert decisions[0].context["tradability_error"] == "tradability miss"


def test_live_unknown_token_can_be_resolved_from_market_lookup(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    config = config.model_copy(
        update={
            "risk": config.risk.model_copy(update={"require_cluster_confirmation_live": False}),
            "live": config.live.model_copy(update={"only_cluster_confirmed": False}),
        }
    )
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "LIVE",
            "system_status": "LIVE_READY",
            "wallet_discovery_state": DiscoveryState.SUCCESS.value,
            "wallet_scoring_state": "SUCCESS",
            "live_readiness_last_result": {"ready": True},
            "allowance_sufficient": True,
            "balance_visible": True,
            "manual_live_enable": True,
            "reconciliation_clean": True,
            "heartbeat_ok": True,
            "live_health_state": "HEALTHY",
        }
    )
    engine = StrategyEngine(config, tmp_path, state)
    detection = _detection().model_copy(
        update={
            "token_id": "unknown-token",
            "market_slug": "resolved-market",
            "market_metadata": {"outcome": "Yes"},
        }
    )

    async def _refresh_markets() -> dict[str, object]:
        return {}

    async def _stream_watchlist(token_ids: list[str]) -> dict[str, dict]:
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = "", market_slug: str = "", outcome: str = "") -> dict[str, object]:
        assert market_slug == "resolved-market"
        assert outcome == "Yes"
        return {
            "market_id": market_id,
            "token_id": "resolved-token",
            "title": "Resolved market",
            "slug": market_slug,
            "category": "politics",
            "active": True,
            "closed": False,
        }

    async def _tradability(market_id: str, token_id: str) -> dict[str, object]:
        assert token_id == "resolved-token"
        return {
            "market_id": market_id,
            "token_id": token_id,
            "tradable": True,
            "orderbook_enabled": True,
            "category": "politics",
            "title": "Resolved market",
            "liquidity": 1000.0,
        }

    async def _orderbook(token_id: str):
        from src.models import OrderbookLevel, OrderbookSnapshot

        return OrderbookSnapshot(
            token_id=token_id,
            bids=[OrderbookLevel(price=0.50, size=1000.0)],
            asks=[OrderbookLevel(price=0.50, size=1000.0)],
        )

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(
        engine.process_detections(
            [detection],
            ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=["0xabc"]),
            [_wallet()],
        )
    )

    assert len(decisions) == 1
    assert decisions[0].reason_code != "MISSING_TOKEN_ID"
    assert decisions[0].token_id == "resolved-token"


def test_live_orderbook_failure_skips_instead_of_crashing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    state = AppStateStore(tmp_path / "app_state.json")
    state.write(
        {
            "mode": "LIVE",
            "system_status": "LIVE_READY",
            "wallet_discovery_state": DiscoveryState.SUCCESS.value,
            "wallet_scoring_state": "SUCCESS",
            "live_readiness_last_result": {"ready": True},
            "approved_wallets": {"paper_wallets": [], "research_wallets": [], "live_wallets": ["0xabc"]},
        }
    )
    engine = StrategyEngine(config, tmp_path, state)

    async def _refresh_markets() -> dict[str, object]:
        return {}

    async def _stream_watchlist(token_ids: list[str]) -> dict[str, dict]:
        return {}

    async def _fetch_market_metadata(market_id: str, token_id: str = "") -> dict[str, object]:
        return {
            "market_id": market_id,
            "token_id": token_id or "token-1",
            "title": "Resolved market",
            "slug": "resolved-market",
            "category": "politics",
            "active": True,
            "closed": False,
        }

    async def _tradability(market_id: str, token_id: str) -> dict[str, object]:
        return {
            "market_id": market_id,
            "token_id": token_id,
            "tradable": True,
            "orderbook_enabled": True,
            "category": "politics",
            "title": "Resolved market",
            "liquidity": 1000.0,
        }

    async def _orderbook(token_id: str):
        raise RuntimeError("orderbook miss")

    engine.market_data.refresh_markets = _refresh_markets  # type: ignore[method-assign]
    engine.market_data.stream_watchlist = _stream_watchlist  # type: ignore[method-assign]
    engine.market_data.fetch_market_metadata = _fetch_market_metadata  # type: ignore[method-assign]
    engine.market_data.get_tradability = _tradability  # type: ignore[method-assign]
    engine.market_data.fetch_orderbook = _orderbook  # type: ignore[method-assign]

    decisions = asyncio.run(
        engine.process_detections(
            [_detection()],
            ApprovedWallets(research_wallets=[], paper_wallets=[], live_wallets=["0xabc"]),
            [_wallet()],
        )
    )

    assert len(decisions) == 1
    assert decisions[0].allowed is False
    assert decisions[0].reason_code == "ORDERBOOK_UNAVAILABLE"
    assert decisions[0].context["orderbook_error"] == "orderbook miss"
