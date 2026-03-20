from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from src.bracket_executor import BracketExecutor, BracketPhase
from src.config import CryptoDirectionConfig
from src.fees import max_profitable_opposite_price
from src.models import BracketSignalEvent, OrderbookLevel, OrderbookSnapshot


class _DummyClient:
    def __init__(self) -> None:
        self.buy_calls: list[dict] = []
        self.sell_calls: list[dict] = []
        self.allowance_refresh_calls: list[str] = []
        self.refresh_live_order_session_calls = 0
        self.open_orders: list[dict] = []
        self.positions: list[dict] = []
        self.next_buy_errors: list[str] = []
        self.next_buy_results: list[dict] = []
        self.next_buy_result: dict = {
            "exchange_order_id": "test-order-1",
            "status": "MATCHED",
            "filled_size": 10.0,
            "average_fill_price": 0.58,
            "remaining_size": 0.0,
        }
        self.next_sell_result: dict = {
            "exchange_order_id": "test-sell-1",
            "status": "MATCHED",
            "filled_size": 10.0,
            "average_fill_price": 0.48,
            "remaining_size": 0.0,
        }
        self.next_sell_results: list[dict] = []
        self.next_sell_errors: list[str] = []
        self.orderbook = OrderbookSnapshot(
            token_id="yes-1",
            bids=[OrderbookLevel(price=0.58, size=20.0)],
            asks=[OrderbookLevel(price=0.58, size=20.0)],
        )

    async def place_buy_order(
        self,
        token_id: str,
        price: float,
        size: float,
        entry_style: str,
        client_order_id: str | None = None,
    ) -> dict:
        self.buy_calls.append(
            {
                "token_id": token_id,
                "price": price,
                "size": size,
                "entry_style": entry_style,
                "client_order_id": client_order_id,
            }
        )
        if self.next_buy_errors:
            raise RuntimeError(self.next_buy_errors.pop(0))
        if self.next_buy_results:
            result = dict(self.next_buy_results.pop(0))
            self.next_buy_result = dict(result)
            return result
        return dict(self.next_buy_result)

    async def place_sell_order(
        self,
        token_id: str,
        price: float,
        size: float,
        entry_style: str,
        client_order_id: str | None = None,
    ) -> dict:
        self.sell_calls.append(
            {
                "token_id": token_id,
                "price": price,
                "size": size,
                "entry_style": entry_style,
                "client_order_id": client_order_id,
            }
        )
        if self.next_sell_errors:
            raise RuntimeError(self.next_sell_errors.pop(0))
        if self.next_sell_results:
            result = dict(self.next_sell_results.pop(0))
            self.next_sell_result = dict(result)
            return result
        return dict(self.next_sell_result)

    async def get_order_status(self, exchange_order_id: str) -> dict:
        if exchange_order_id.startswith("test-sell"):
            return dict(self.next_sell_result)
        return dict(self.next_buy_result)

    async def get_orderbook(self, token_id: str) -> OrderbookSnapshot:
        return self.orderbook

    async def get_open_orders(self) -> list[dict]:
        return list(self.open_orders)

    async def get_positions(self) -> list[dict]:
        return list(self.positions)

    async def ensure_token_sell_allowance(self, token_id: str) -> dict:
        self.allowance_refresh_calls.append(token_id)
        return {"ok": True}

    async def refresh_live_order_session(self) -> dict:
        self.refresh_live_order_session_calls += 1
        return {"refreshed": True, "sdk_ready": True}


def _signal_event() -> BracketSignalEvent:
    return BracketSignalEvent(
        event_id="evt-1",
        fired_at=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
        asset="BTC",
        window_open_ts=1_700_000_000,
        window_close_ts=1_700_000_300,
        minutes_remaining=3.5,
        mid_window_start=False,
        asset_open=84_000.0,
        asset_current=84_420.0,
        asset_move_pct=0.005,
        momentum_side="YES",
        momentum_price=0.58,
        opposite_price=0.41,
        implied_momentum_price=0.63,
        lag_gap=0.05,
        chop_score=0.95,
        checkpoints=[],
        fee_at_momentum_price=0.01,
        fee_at_target_y=0.01,
        net_bracket_at_target=0.06,
        market_id="mkt-1",
        yes_token_id="yes-1",
        no_token_id="no-1",
        market_liquidity=1000.0,
        market_volume=500.0,
    )


def test_phase1_uses_fixed_configured_share_count(tmp_path) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)

    submitted = asyncio.run(executor.on_signal(_signal_event()))

    assert submitted is True
    assert len(client.buy_calls) == 1
    assert client.buy_calls[0]["size"] == 10.0

    positions = executor.active_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert pos.p1_shares == 10.0
    assert pos.p1_notional_usd == 5.8


def test_phase1_uses_exact_visible_depth_when_between_min_and_target(tmp_path) -> None:
    client = _DummyClient()
    client.orderbook = OrderbookSnapshot(
        token_id="yes-1",
        bids=[OrderbookLevel(price=0.58, size=20.0)],
        asks=[OrderbookLevel(price=0.59, size=7.4)],
    )
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        min_bracket_shares=5.0,
        phase1_max_chase_cents=0.01,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)

    submitted = asyncio.run(executor.on_signal(_signal_event()))

    assert submitted is True
    assert len(client.buy_calls) == 1
    assert client.buy_calls[0]["size"] == 7.4
    pos = executor.active_positions()[0]
    assert pos.p1_shares == 7.4
    assert pos.p1_initial_shares == 7.4


def test_phase1_skips_when_visible_depth_below_minimum(tmp_path) -> None:
    client = _DummyClient()
    client.orderbook = OrderbookSnapshot(
        token_id="yes-1",
        bids=[OrderbookLevel(price=0.58, size=20.0)],
        asks=[OrderbookLevel(price=0.59, size=4.9)],
    )
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        min_bracket_shares=5.0,
        phase1_max_chase_cents=0.01,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)

    submitted = asyncio.run(executor.on_signal(_signal_event()))

    assert submitted is False
    assert client.buy_calls == []
    summary = executor.take_window_summary(1_700_000_000, "BTC")
    assert summary is not None
    assert summary["phase"] == "PHASE1_NOT_FILLED"


def test_phase1_follow_taker_uses_configured_chase_limit(tmp_path) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.01,
        min_bracket_shares=1.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)

    submitted = asyncio.run(executor.on_signal(_signal_event()))

    assert submitted is True
    assert len(client.buy_calls) == 1
    assert client.buy_calls[0]["price"] == 0.59


def test_phase1_follow_taker_does_not_chase_above_entry_band_high(tmp_path) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.01,
        entry_range_high=0.63,
        min_bracket_shares=1.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(update={"momentum_price": 0.63})

    submitted = asyncio.run(executor.on_signal(event))

    assert submitted is True
    assert len(client.buy_calls) == 1
    assert client.buy_calls[0]["price"] == 0.63


def test_band_touch_phase1_chase_stays_inside_band_cap(tmp_path) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.02,
        entry_range_high=0.68,
        immediate_band_entry_enabled=True,
        immediate_band_entry_high=0.61,
        min_bracket_shares=1.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(update={"entry_model": "band_touch", "momentum_price": 0.59})

    submitted = asyncio.run(executor.on_signal(event))

    assert submitted is True
    assert len(client.buy_calls) == 1
    assert client.buy_calls[0]["price"] == 0.61


def test_phase1_is_not_credited_when_follow_taker_does_not_fill(tmp_path) -> None:
    client = _DummyClient()
    client.next_buy_result = {
        "exchange_order_id": "test-order-2",
        "status": "CANCELLED",
        "filled_size": 0.0,
        "average_fill_price": 0.0,
        "remaining_size": 10.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)

    submitted = asyncio.run(executor.on_signal(_signal_event()))

    assert submitted is False
    assert executor.active_positions() == []
    summary = executor.take_window_summary(1_700_000_000, "BTC")
    assert summary is not None
    assert summary["phase"] == "PHASE1_NOT_FILLED"
    assert summary["phase1_filled"] is False
    assert summary["actual_pnl_usd"] == 0.0
    assert summary["p1_shares"] == 10.0


def test_phase1_follow_taker_retries_once_after_fok_error_and_fills(tmp_path) -> None:
    client = _DummyClient()
    client.next_buy_errors = [
        "Real live order placement failed: order couldn't be fully filled. FOK orders are fully filled or killed.",
    ]
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.01,
        phase1_follow_taker_retry_attempts=1,
        phase1_follow_taker_retry_delay_seconds=0.0,
        phase1_follow_taker_retry_to_strategy_cap=True,
        entry_range_high=0.60,
        min_bracket_shares=1.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)

    submitted = asyncio.run(executor.on_signal(_signal_event()))

    assert submitted is True
    assert len(client.buy_calls) == 2
    assert client.buy_calls[0]["price"] == 0.59
    assert client.buy_calls[1]["price"] == 0.60
    positions = executor.active_positions()
    assert len(positions) == 1
    assert positions[0].phase == BracketPhase.PHASE1_FILLED


def test_phase1_follow_taker_request_exception_fails_safe_without_blind_retry(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_buy_errors = [
        "Real live order placement failed: signature_type=1:PolyApiException[status_code=None, error_message=Request exception!]",
    ]
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.01,
        phase1_follow_taker_retry_attempts=1,
        phase1_follow_taker_retry_delay_seconds=5.0,
        phase1_follow_taker_retry_to_strategy_cap=True,
        entry_range_high=0.60,
        min_bracket_shares=1.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("src.bracket_executor.asyncio.sleep", _fake_sleep)

    submitted = asyncio.run(executor.on_signal(_signal_event()))

    assert submitted is False
    assert len(client.buy_calls) == 1
    assert client.refresh_live_order_session_calls == 1
    assert sleep_calls == []
    assert executor.active_positions() == []


def test_phase1_request_exception_recovers_from_reconciled_position(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_buy_errors = [
        "Real live order placement failed: signature_type=1:PolyApiException[status_code=None, error_message=Request exception!]",
    ]
    client.positions = [
        {
            "market_id": "mkt-1",
            "token_id": "yes-1",
            "quantity": 5.0,
            "avg_price": 0.59,
        }
    ]
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.01,
        phase1_follow_taker_retry_attempts=1,
        phase1_follow_taker_retry_delay_seconds=5.0,
        phase1_follow_taker_retry_to_strategy_cap=True,
        entry_range_high=0.60,
        min_bracket_shares=1.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("src.bracket_executor.asyncio.sleep", _fake_sleep)

    submitted = asyncio.run(executor.on_signal(_signal_event()))

    assert submitted is True
    assert len(client.buy_calls) == 1
    assert client.refresh_live_order_session_calls == 1
    assert sleep_calls == []
    positions = executor.active_positions()
    assert len(positions) == 1
    assert positions[0].phase == BracketPhase.PHASE1_FILLED
    assert positions[0].p1_fill_price == 0.59


def test_phase1_follow_taker_skips_when_fresh_book_is_no_longer_marketable(tmp_path) -> None:
    client = _DummyClient()
    client.next_buy_errors = [
        "Real live order placement failed: order couldn't be fully filled. FOK orders are fully filled or killed.",
    ]
    client.orderbook = OrderbookSnapshot(
        token_id="yes-1",
        bids=[OrderbookLevel(price=0.58, size=20.0)],
        asks=[OrderbookLevel(price=0.62, size=2.0)],
    )
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.01,
        phase1_follow_taker_retry_attempts=1,
        phase1_follow_taker_retry_delay_seconds=0.0,
        phase1_follow_taker_retry_to_strategy_cap=True,
        entry_range_high=0.60,
        min_bracket_shares=1.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)

    submitted = asyncio.run(executor.on_signal(_signal_event()))

    assert submitted is False
    assert len(client.buy_calls) == 0
    summary = executor.take_window_summary(1_700_000_000, "BTC")
    assert summary is not None
    assert summary["phase"] == "PHASE1_NOT_FILLED"


def test_phase1_follow_taker_retry_uses_continuation_entry_ceiling(tmp_path) -> None:
    client = _DummyClient()
    client.next_buy_errors = [
        "Real live order placement failed: order couldn't be fully filled. FOK orders are fully filled or killed.",
    ]
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.01,
        phase1_follow_taker_retry_attempts=1,
        phase1_follow_taker_retry_delay_seconds=0.0,
        phase1_follow_taker_retry_to_strategy_cap=True,
        entry_range_high=0.63,
        continuation_max_momentum_price=0.61,
        min_bracket_shares=1.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(update={"entry_model": "continuation"})

    submitted = asyncio.run(executor.on_signal(event))

    assert submitted is True
    assert len(client.buy_calls) == 2
    assert client.buy_calls[0]["price"] == 0.59
    assert client.buy_calls[1]["price"] == 0.61


def test_phase1_follow_taker_retries_after_thin_depth_if_best_ask_is_within_retry_ceiling(tmp_path) -> None:
    client = _DummyClient()
    client.orderbook = OrderbookSnapshot(
        token_id="yes-1",
        bids=[OrderbookLevel(price=0.58, size=20.0)],
        asks=[OrderbookLevel(price=0.60, size=10.0)],
    )
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        min_bracket_shares=5.0,
        phase1_max_chase_cents=0.01,
        phase1_follow_taker_retry_attempts=1,
        phase1_follow_taker_retry_delay_seconds=0.0,
        phase1_follow_taker_retry_to_strategy_cap=True,
        entry_range_high=0.60,
        continuation_max_momentum_price=0.60,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(update={"entry_model": "continuation"})

    submitted = asyncio.run(executor.on_signal(event))

    assert submitted is True
    assert len(client.buy_calls) == 1
    assert client.buy_calls[0]["price"] == 0.60


def test_hard_exit_uses_fee_aware_realized_pnl(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-1",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.50,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.50, no_ask=0.50))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.actual_pnl_usd < -0.85
    assert pos.hard_exit_reason == "STOP_50C"
    assert pos.hard_exit_fill_price == 0.50
    assert client.sell_calls

    asyncio.run(executor.on_window_close(1_700_000_000, "BTC", yes_won=False))
    summary = executor.take_window_summary(1_700_000_000, "BTC")
    assert summary is not None
    assert summary["hard_exited"] is True
    assert summary["hard_exit_reason"] == "STOP_50C"
    assert summary["hard_exit_fill_price"] == 0.50


def test_hard_exit_prefers_bid_side_for_trigger_and_exit_model(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-bid-trigger",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.50,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_trigger_buffer_cents=0.02,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.60, no_ask=0.60, yes_bid=0.51, no_bid=0.51))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert client.sell_calls[0]["price"] == 0.50


def test_hard_exit_tries_stop_price_before_market_through(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-stop-first",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.50,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_trigger_buffer_cents=0.02,
        hard_exit_market_through_cents=0.02,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.52, no_ask=0.52))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_reason == "STOP_50C"
    assert pos.hard_exit_fill_price == 0.50
    assert [call["price"] for call in client.sell_calls] == [0.50]


def test_hard_exit_retries_lower_if_stop_price_misses(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_results = [
        {
            "exchange_order_id": "test-sell-stop-miss",
            "status": "CANCELLED",
            "filled_size": 0.0,
            "average_fill_price": 0.0,
            "remaining_size": 10.0,
        },
        {
            "exchange_order_id": "test-sell-stop-fallback",
            "status": "MATCHED",
            "filled_size": 10.0,
            "average_fill_price": 0.48,
            "remaining_size": 0.0,
        },
    ]
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_trigger_buffer_cents=0.02,
        hard_exit_market_through_cents=0.02,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.50, no_ask=0.50))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_fill_price == 0.48
    assert [call["price"] for call in client.sell_calls] == [0.50, 0.48]


def test_hard_exit_stays_at_stop_price_when_market_through_disabled(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_results = [
        {
            "exchange_order_id": "test-sell-stop-miss-1",
            "status": "CANCELLED",
            "filled_size": 0.0,
            "average_fill_price": 0.0,
            "remaining_size": 10.0,
        },
        {
            "exchange_order_id": "test-sell-stop-miss-2",
            "status": "CANCELLED",
            "filled_size": 0.0,
            "average_fill_price": 0.0,
            "remaining_size": 10.0,
        },
    ]
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_trigger_buffer_cents=0.02,
        hard_exit_market_through_cents=0.0,
        hard_exit_fallback_step_cents=0.0,
        hard_exit_retry_cooldown_seconds=2.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000

    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)
    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.48, no_ask=0.48))
    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert [call["price"] for call in client.sell_calls] == [0.50]

    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_053.0)
    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.47, no_ask=0.47))
    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert [call["price"] for call in client.sell_calls] == [0.50, 0.50]


def test_hard_exit_does_not_run_after_window_close(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 1_050.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.01, no_ask=0.01))

    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert client.sell_calls == []


def test_hard_exit_retries_on_later_tick_after_unfilled_stop(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_results = [
        {
            "exchange_order_id": "test-sell-stop-miss-1",
            "status": "CANCELLED",
            "filled_size": 0.0,
            "average_fill_price": 0.0,
            "remaining_size": 10.0,
        },
        {
            "exchange_order_id": "test-sell-stop-miss-2",
            "status": "CANCELLED",
            "filled_size": 0.0,
            "average_fill_price": 0.0,
            "remaining_size": 10.0,
        },
        {
            "exchange_order_id": "test-sell-stop-fill",
            "status": "MATCHED",
            "filled_size": 10.0,
            "average_fill_price": 0.48,
            "remaining_size": 0.0,
        },
    ]
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_trigger_buffer_cents=0.02,
        hard_exit_market_through_cents=0.02,
        hard_exit_fallback_step_cents=0.0,
        hard_exit_retry_cooldown_seconds=2.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000

    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)
    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.50, no_ask=0.50))
    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert pos.hard_exit_fill_price == 0.0

    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_051.0)
    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.49, no_ask=0.49))
    assert len(client.sell_calls) == 2
    assert pos.phase == BracketPhase.PHASE1_FILLED

    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_053.5)
    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.49, no_ask=0.49))
    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_fill_price == 0.48
    assert len(client.sell_calls) == 3


def test_hard_exit_steps_down_within_same_tick_when_50c_does_not_fill(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_results = [
        {
            "exchange_order_id": "test-sell-stop-miss-50",
            "status": "CANCELLED",
            "filled_size": 0.0,
            "average_fill_price": 0.0,
            "remaining_size": 10.0,
        },
        {
            "exchange_order_id": "test-sell-stop-fill-48",
            "status": "MATCHED",
            "filled_size": 10.0,
            "average_fill_price": 0.48,
            "remaining_size": 0.0,
        },
    ]
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_market_through_cents=0.0,
        hard_exit_fallback_step_cents=0.02,
        hard_exit_min_sell_price=0.40,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.48, no_ask=0.48))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_fill_price == 0.48
    assert [call["price"] for call in client.sell_calls] == [0.50, 0.48]


def test_hard_exit_can_fill_across_visible_depth_fok_chunks(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.orderbook = OrderbookSnapshot(
        token_id="yes-1",
        bids=[
            OrderbookLevel(price=0.50, size=2.0),
            OrderbookLevel(price=0.48, size=3.0),
        ],
        asks=[OrderbookLevel(price=0.58, size=20.0)],
    )
    client.next_sell_results = [
        {
            "exchange_order_id": "test-sell-partial-50",
            "status": "MATCHED",
            "filled_size": 2.0,
            "average_fill_price": 0.50,
            "remaining_size": 0.0,
        },
        {
            "exchange_order_id": "test-sell-partial-48",
            "status": "MATCHED",
            "filled_size": 3.0,
            "average_fill_price": 0.48,
            "remaining_size": 0.0,
        },
    ]
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_market_through_cents=0.0,
        hard_exit_fallback_step_cents=0.02,
        hard_exit_min_sell_price=0.40,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.48, no_ask=0.48))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.p1_shares == 0.0
    assert pos.p1_initial_shares == 5.0
    assert pos.hard_exit_filled_shares == 5.0
    assert round(pos.hard_exit_fill_price, 3) == 0.488
    assert [call["price"] for call in client.sell_calls] == [0.50, 0.48]
    assert all(call["entry_style"] == "FOLLOW_TAKER_PARTIAL" for call in client.sell_calls)
    assert client.allowance_refresh_calls == ["yes-1", "yes-1"]


def test_hard_exit_retries_allowance_failures_then_sells(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.positions = [
        {
            "token_id": "yes-1",
            "quantity": 10.0,
            "avg_price": 0.58,
        }
    ]
    client.next_sell_errors = [
        "Real live order placement failed: PolyApiException[status_code=400, error_message={'error': 'not enough balance / allowance'}]"
    ]
    client.next_sell_result = {
        "exchange_order_id": "test-sell-after-allowance",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.50,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.50, no_ask=0.50))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_fill_price == 0.50
    assert len(client.sell_calls) == 2
    assert client.refresh_live_order_session_calls == 1
    assert client.allowance_refresh_calls == ["yes-1", "yes-1", "yes-1"]


def test_hard_exit_caps_sell_size_to_visible_position_after_allowance_failure(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.positions = [
        {
            "token_id": "yes-1",
            "quantity": 3.2,
            "avg_price": 0.60,
        }
    ]
    client.next_sell_errors = [
        "Real live order placement failed: PolyApiException[status_code=400, error_message={'error': 'not enough balance / allowance'}]"
    ]
    client.next_sell_result = {
        "exchange_order_id": "test-sell-after-position-reconcile",
        "status": "MATCHED",
        "filled_size": 3.2,
        "average_fill_price": 0.50,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.50, no_ask=0.50))

    assert len(client.sell_calls) == 2
    assert client.sell_calls[0]["size"] == 3.2
    assert client.sell_calls[1]["size"] == 3.2
    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert round(pos.hard_exit_filled_shares, 3) == 3.2
    assert round(pos.p1_shares, 3) == 6.8
    assert round(pos.hard_exit_fill_price, 3) == 0.50


def test_hard_exit_uses_visible_position_when_book_precheck_shows_no_depth(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.orderbook = OrderbookSnapshot(
        token_id="yes-1",
        bids=[],
        asks=[OrderbookLevel(price=0.58, size=20.0)],
    )
    client.positions = [
        {
            "token_id": "yes-1",
            "quantity": 10.0,
            "avg_price": 0.58,
        }
    ]
    client.next_sell_result = {
        "exchange_order_id": "test-sell-blind-visible-position",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.48,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.50, no_ask=0.50, yes_bid=0.52, no_bid=0.52))

    assert len(client.sell_calls) == 1
    assert client.sell_calls[0]["size"] == 10.0
    assert client.sell_calls[0]["entry_style"] == "FOLLOW_TAKER_PARTIAL"
    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_fill_price == 0.48


def test_hard_exit_uses_actual_reported_partial_fill_size(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.orderbook = OrderbookSnapshot(
        token_id="yes-1",
        bids=[OrderbookLevel(price=0.50, size=10.0)],
        asks=[OrderbookLevel(price=0.58, size=20.0)],
    )
    client.next_sell_result = {
        "exchange_order_id": "test-sell-partial-size",
        "status": "MATCHED",
        "filled_size": 2.4,
        "average_fill_price": 0.50,
        "remaining_size": 0.6,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_market_through_cents=0.0,
        hard_exit_fallback_step_cents=0.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000.0
    pos.p1_shares = 3.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.50, no_ask=0.50))

    assert round(pos.hard_exit_filled_shares, 3) == 2.4
    assert round(pos.p1_shares, 3) == 0.6
    assert pos.phase == BracketPhase.PHASE1_FILLED


def test_hard_exit_high_entry_bucket_uses_tighter_stop(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_buy_result = {
        "exchange_order_id": "test-order-high-entry",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.68,
        "remaining_size": 0.0,
    }
    client.next_sell_result = {
        "exchange_order_id": "test-sell-tight-stop",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.54,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_high_entry_price=0.64,
        hard_exit_high_entry_stop_price=0.54,
        hard_exit_trigger_buffer_cents=0.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(
        update={"entry_model": "continuation", "momentum_price": 0.63}
    )
    asyncio.run(executor.on_signal(event))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_040.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.54, no_ask=0.54))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_reason == "STOP_50C"
    assert client.sell_calls[0]["price"] == 0.54


def test_hard_exit_high_entry_bucket_ignores_continuation_grace(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_buy_result = {
        "exchange_order_id": "test-order-high-entry-grace",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.68,
        "remaining_size": 0.0,
    }
    client.next_sell_result = {
        "exchange_order_id": "test-sell-tight-stop-grace",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.54,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_high_entry_price=0.64,
        hard_exit_high_entry_stop_price=0.54,
        continuation_hard_exit_grace_seconds=30.0,
        continuation_catastrophic_stop_price=0.45,
        hard_exit_trigger_buffer_cents=0.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(update={"entry_model": "continuation", "momentum_price": 0.63})
    asyncio.run(executor.on_signal(event))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_010.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.54, no_ask=0.54))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_fill_price == 0.54
    assert client.sell_calls[0]["price"] == 0.54


def test_continuation_phase1_retry_ceiling_respects_continuation_cap(tmp_path) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_follow_taker_retry_to_strategy_cap=True,
        entry_range_high=0.68,
        continuation_max_momentum_price=0.66,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(update={"entry_model": "continuation"})

    assert executor._phase1_retry_ceiling(event, 0.65) == 0.66


def test_hard_exit_ladder_stops_at_controlled_floor(tmp_path) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        hard_exit_stop_price=0.50,
        hard_exit_fallback_step_cents=0.02,
        hard_exit_min_sell_price=0.40,
        hard_exit_emergency_min_sell_price=0.40,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)

    assert executor._build_hard_exit_attempt_prices(0.50, include_emergency=True) == [
        0.50,
        0.48,
        0.46,
        0.44,
        0.42,
        0.40,
    ]


def test_continuation_stop_grace_skips_early_stop_hit(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        continuation_hard_exit_grace_seconds=30.0,
        continuation_catastrophic_stop_price=0.45,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(update={"entry_model": "continuation"})
    asyncio.run(executor.on_signal(event))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_010.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.50, no_ask=0.50))

    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert pos.hard_exit_reason == ""
    assert client.sell_calls == []


def test_continuation_catastrophic_stop_still_fires_during_grace(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-2",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.43,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        continuation_hard_exit_grace_seconds=30.0,
        continuation_catastrophic_stop_price=0.45,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(update={"entry_model": "continuation"})
    asyncio.run(executor.on_signal(event))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_010.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.44, no_ask=0.44))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_reason == "STOP_CATASTROPHIC"
    assert pos.hard_exit_fill_price == 0.43
    assert len(client.sell_calls) == 1


def test_continuation_grace_with_catastrophic_disabled_does_not_force_deep_exit(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        continuation_hard_exit_grace_seconds=30.0,
        continuation_catastrophic_stop_price=0.0,
        hard_exit_market_through_cents=0.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(update={"entry_model": "continuation"})
    asyncio.run(executor.on_signal(event))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_010.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.44, no_ask=0.44))

    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert pos.hard_exit_reason == ""
    assert client.sell_calls == []


def test_continuation_stop_fires_normally_after_grace(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-3",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.49,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        continuation_hard_exit_grace_seconds=30.0,
        continuation_catastrophic_stop_price=0.45,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(update={"entry_model": "continuation"})
    asyncio.run(executor.on_signal(event))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_031.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.50, no_ask=0.50))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_reason == "STOP_50C"
    assert pos.hard_exit_fill_price == 0.49


def test_safe_arm_protection_skips_ordinary_stop_for_lag_entries(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        safe_arm_suspend_stop=True,
        safe_arm_catastrophic_stop_price=0.45,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event().model_copy(update={"entry_model": "lag"})))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    pos.safe_opposite_price = 0.34
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.49, no_ask=0.49))

    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert pos.hard_exit_reason == ""
    assert client.sell_calls == []


def test_safe_arm_protection_expires_after_safe_breach(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-safe-breach",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.50,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        safe_arm_suspend_stop=True,
        safe_arm_catastrophic_stop_price=0.45,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event().model_copy(update={"entry_model": "lag"})))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    pos.safe_opposite_price = 0.34
    pos.dipped_below_safe_price = True
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.49, no_ask=0.49))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_reason == "STOP_50C"
    assert pos.hard_exit_fill_price == 0.50
    assert len(client.sell_calls) == 1


def test_phase2_arms_from_dynamic_profitable_ceiling_before_fixed_target(tmp_path) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        min_bracket_shares=1.0,
        phase1_max_chase_cents=0.0,
        phase2_min_locked_profit_per_share=0.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    asyncio.run(executor._check_phase2(pos, yes_ask=0.60, no_ask=0.41))

    expected = max_profitable_opposite_price(0.58, min_net_margin=0.0, category="crypto price")
    assert pos.safe_opposite_price == pytest.approx(expected, abs=1e-4)
    assert pos.safe_opposite_price > 0.41


def test_phase2_tightens_safe_anchor_when_better_target_is_reached(tmp_path) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        min_bracket_shares=1.0,
        phase1_max_chase_cents=0.0,
        phase2_min_locked_profit_per_share=0.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    asyncio.run(executor._check_phase2(pos, yes_ask=0.60, no_ask=0.41))
    assert pos.safe_opposite_price > 0.40

    asyncio.run(executor._check_phase2(pos, yes_ask=0.60, no_ask=0.34))
    assert pos.safe_opposite_price == pytest.approx(0.34)


def test_safe_arm_catastrophic_stop_still_fires_for_lag_entries(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-safe-arm",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.43,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        safe_arm_suspend_stop=True,
        safe_arm_catastrophic_stop_price=0.45,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event().model_copy(update={"entry_model": "lag"})))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 2_000
    pos.safe_opposite_price = 0.34
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.44, no_ask=0.44))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_reason == "STOP_CATASTROPHIC"
    assert pos.hard_exit_fill_price == 0.43
    assert len(client.sell_calls) == 1


def test_safe_arm_still_allows_final_window_loss_exit(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-final-window",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.53,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_final_seconds=30.0,
        safe_arm_suspend_stop=True,
        safe_arm_catastrophic_stop_price=0.45,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event().model_copy(update={"entry_model": "lag"})))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.window_close_ts = 1_070.0
    pos.safe_opposite_price = 0.34
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_045.0)

    asyncio.run(executor._check_hard_exit(pos, yes_ask=0.55, no_ask=0.55))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_reason == "FINAL_30S_LOSS"
    assert pos.hard_exit_fill_price == 0.53
