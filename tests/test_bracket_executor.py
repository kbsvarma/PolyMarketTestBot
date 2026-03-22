from __future__ import annotations

import asyncio
import json
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
        self.cancelled_order_ids: list[str] = []
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

    async def cancel_order(self, order_id: str) -> dict:
        self.cancelled_order_ids.append(order_id)
        return {"cancelled": True, "order_id": order_id}


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


def test_phase1_follow_taker_lag_capture_can_lift_two_cents_but_stays_below_62(tmp_path) -> None:
    client = _DummyClient()
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=5.0,
        phase1_max_chase_cents=0.02,
        entry_range_high=0.61,
        min_bracket_shares=1.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    event = _signal_event().model_copy(update={"momentum_price": 0.59, "entry_model": "lag"})

    submitted = asyncio.run(executor.on_signal(event))

    assert submitted is True
    assert len(client.buy_calls) == 1
    assert client.buy_calls[0]["price"] == 0.61


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


def test_phase1_follow_taker_steps_down_to_min_block_after_fok_error(tmp_path) -> None:
    client = _DummyClient()
    client.next_buy_errors = [
        "Real live order placement failed: order couldn't be fully filled. FOK orders are fully filled or killed.",
    ]
    client.next_buy_result = {
        "exchange_order_id": "test-order-stepdown",
        "status": "MATCHED",
        "filled_size": 5.0,
        "average_fill_price": 0.59,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        min_bracket_shares=5.0,
        phase1_max_chase_cents=0.01,
        phase1_follow_taker_retry_attempts=0,
        phase1_follow_taker_retry_delay_seconds=0.0,
        phase1_follow_taker_retry_to_strategy_cap=False,
        entry_range_high=0.63,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)

    submitted = asyncio.run(executor.on_signal(_signal_event()))

    assert submitted is True
    assert len(client.buy_calls) == 2
    assert client.buy_calls[0]["size"] == 10.0
    assert client.buy_calls[1]["size"] == 5.0
    assert client.buy_calls[0]["price"] == 0.59
    assert client.buy_calls[1]["price"] == 0.59
    positions = executor.active_positions()
    assert len(positions) == 1
    assert positions[0].phase == BracketPhase.PHASE1_FILLED
    assert positions[0].p1_shares == 5.0
    assert positions[0].p1_fill_price == 0.59


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


def test_hard_exit_steps_down_within_same_tick_when_fak_no_match_error_occurs(
    tmp_path, monkeypatch
) -> None:
    client = _DummyClient()
    client.orderbook = OrderbookSnapshot(
        token_id="yes-1",
        bids=[OrderbookLevel(price=0.49, size=20.0)],
        asks=[OrderbookLevel(price=0.58, size=20.0)],
    )
    client.positions = [
        {
            "token_id": "yes-1",
            "quantity": 10.0,
            "avg_price": 0.58,
        }
    ]
    client.next_sell_errors = [
        "PolyApiException[status_code=400, error_message={'error': 'no orders found to match with FAK order. FAK orders are partially filled or killed if no match is found.'}]"
    ]
    client.next_sell_results = [
        {
            "exchange_order_id": "test-sell-stop-fill-48-after-fak-miss",
            "status": "MATCHED",
            "filled_size": 10.0,
            "average_fill_price": 0.48,
            "remaining_size": 0.0,
        }
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


def test_hard_exit_treats_tiny_residual_as_dust_and_stops_retrying(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.positions = [
        {
            "token_id": "yes-1",
            "quantity": 10.0,
            "avg_price": 0.58,
        }
    ]
    client.next_sell_result = {
        "exchange_order_id": "test-sell-dust",
        "status": "MATCHED",
        "filled_size": 9.86,
        "average_fill_price": 0.50,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_dust_shares=0.25,
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
    assert pos.p1_shares == 0.0
    assert pos.hard_exit_filled_shares == pytest.approx(9.86)
    assert pos.hard_exit_fill_price == 0.50
    assert len(client.sell_calls) == 1


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


def test_safe_arm_does_not_suspend_stop_in_default_live_profile(tmp_path, monkeypatch) -> None:
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-safe-arm-default-off",
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
        safe_arm_suspend_stop=False,
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


# ---------------------------------------------------------------------------
# Layer 1: Shallow reversal sell — fill confirmation contract
# ---------------------------------------------------------------------------


def test_shallow_reversal_sell_full_fill_closes_position(tmp_path, monkeypatch) -> None:
    """
    A confirmed fill from _check_shallow_reversal_sell should mark the position
    HARD_EXITED with the confirmed average_fill_price (not just the attempt price),
    zero pos.p1_shares to prevent on_window_close double-counting, and set a
    fee-aware actual_pnl_usd.
    """
    client = _DummyClient()
    # Sell result: exchange confirms full fill at 0.56
    client.next_sell_result = {
        "exchange_order_id": "test-sell-shallow",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.56,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        shallow_reversal_drop_threshold=0.02,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    # Simulate post-fill state: Phase 2 armed (safe_opposite_price set) but NOT
    # tightened. Peak was 0.60, price dropped 0.03 → reversal triggered.
    pos.p1_filled_at = 1_000.0
    pos.p1_fill_price = 0.58
    pos.peak_momentum_price = 0.60
    pos.safe_opposite_price = 0.38   # Phase 2 was armed
    pos.phase2_tightened = False     # but never tightened — shallow reversal fires
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    # Price is 0.57 (below 0.60 - 0.02 = 0.58 threshold) → reversal fires
    asyncio.run(executor._check_shallow_reversal_sell(
        pos, yes_ask=0.57, no_ask=0.43, yes_bid=0.56, no_bid=0.44
    ))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_reason == "SHALLOW_REVERSAL_SELL"
    # Confirmed fill price from average_fill_price, not just attempt_price
    assert pos.hard_exit_fill_price == 0.56
    assert pos.hard_exit_filled_shares == 10.0
    # p1_shares zeroed to prevent on_window_close double-counting
    assert pos.p1_shares == 0.0
    # Fee-aware PnL: entry cost > exit proceeds → small loss (sold at entry)
    assert pos.actual_pnl_usd is not None
    assert pos.closed_at is not None


def test_shallow_reversal_sell_no_fill_leaves_hard_exit_backstop(tmp_path, monkeypatch) -> None:
    """
    If _confirm_follow_taker_fill returns filled=False for all attempts, the
    position must NOT be marked HARD_EXITED — the hard exit remains as backstop.
    """
    client = _DummyClient()
    # Sell result: exchange returns a CANCELLED / no fill
    client.next_sell_result = {
        "exchange_order_id": "test-sell-shallow-miss",
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
        hard_exit_stop_price=0.50,
        shallow_reversal_drop_threshold=0.02,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.p1_fill_price = 0.58
    pos.peak_momentum_price = 0.60
    pos.safe_opposite_price = 0.38   # Phase 2 armed
    pos.phase2_tightened = False
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_shallow_reversal_sell(
        pos, yes_ask=0.57, no_ask=0.43, yes_bid=0.56, no_bid=0.44
    ))

    # Position stays PHASE1_FILLED — hard exit is still the backstop
    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert pos.hard_exit_reason == ""
    # p1_shares still intact so hard exit can execute
    assert pos.p1_shares == 10.0


# ---------------------------------------------------------------------------
# Layer 3: Crossback stop — fill confirmation contract
# ---------------------------------------------------------------------------


def test_crossback_stop_full_fill_closes_position(tmp_path, monkeypatch) -> None:
    """
    A confirmed fill from _check_crossback_stop should mark the position
    HARD_EXITED, zero p1_shares, set a fee-aware actual_pnl_usd, and record
    the confirmed fill price — not the attempt price.
    """
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-crossback",
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
        crossback_stop_buffer=0.05,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    # Never-armed (Phase 2 was never armed): safe_opposite_price == 0
    pos.p1_filled_at = 1_000.0
    pos.p1_fill_price = 0.58
    pos.safe_opposite_price = 0.0
    pos.window_close_ts = 2_000.0
    # Time = 1_010 so seconds_since_fill = 10 > 3s grace
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_010.0)

    # Price dropped to 0.52 (below 0.58 - 0.05 = 0.53 threshold)
    asyncio.run(executor._check_crossback_stop(
        pos, yes_ask=0.52, no_ask=0.48, yes_bid=0.51, no_bid=0.49
    ))

    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_reason == "CROSSBACK_STOP"
    assert pos.hard_exit_fill_price == 0.54
    assert pos.hard_exit_filled_shares == 10.0
    assert pos.p1_shares == 0.0
    assert pos.actual_pnl_usd is not None
    assert pos.closed_at is not None


def test_crossback_stop_no_fill_leaves_hard_exit_backstop(tmp_path, monkeypatch) -> None:
    """
    If _confirm_follow_taker_fill returns filled=False for all crossback-stop
    attempts, the position must stay PHASE1_FILLED so the hard exit remains as
    the true backstop.
    """
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-crossback-miss",
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
        hard_exit_stop_price=0.50,
        crossback_stop_buffer=0.05,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.p1_fill_price = 0.58
    pos.safe_opposite_price = 0.0
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_010.0)

    asyncio.run(executor._check_crossback_stop(
        pos, yes_ask=0.52, no_ask=0.48, yes_bid=0.51, no_bid=0.49
    ))

    # Position stays PHASE1_FILLED — hard exit is still the backstop
    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert pos.hard_exit_reason == ""
    assert pos.p1_shares == 10.0


# ---------------------------------------------------------------------------
# Layer 1/3: Partial-fill residual handling
# ---------------------------------------------------------------------------


def test_shallow_reversal_partial_fill_reduces_p1_shares_for_hard_exit(tmp_path, monkeypatch) -> None:
    """
    When the exchange confirms a partial fill (filled_size < p1_shares, residual
    > dust), the position must NOT be marked HARD_EXITED.  p1_shares must be
    reduced to the unfilled residual so that the hard exit can sell the remainder.
    The PnL and hard_exit_filled_shares must reflect only the confirmed shares.
    """
    client = _DummyClient()
    # Exchange confirms only 4 out of 10 shares filled
    client.next_sell_result = {
        "exchange_order_id": "test-sell-partial",
        "status": "MATCHED",
        "filled_size": 4.0,
        "average_fill_price": 0.57,
        "remaining_size": 6.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_dust_shares=0.5,
        shallow_reversal_drop_threshold=0.02,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.p1_fill_price = 0.58
    pos.peak_momentum_price = 0.60
    pos.safe_opposite_price = 0.38
    pos.phase2_tightened = False
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_shallow_reversal_sell(
        pos, yes_ask=0.57, no_ask=0.43, yes_bid=0.56, no_bid=0.44
    ))

    # Partial fill — position stays PHASE1_FILLED so hard exit handles residual
    assert pos.phase == BracketPhase.PHASE1_FILLED
    # p1_shares reduced to unfilled residual
    assert pos.p1_shares == pytest.approx(6.0, abs=1e-5)
    # Already-filled portion is tracked
    assert pos.hard_exit_filled_shares == pytest.approx(4.0, abs=1e-5)
    assert pos.hard_exit_fill_price == pytest.approx(0.57, abs=1e-4)
    # PnL reflects only the 4 filled shares
    assert pos.actual_pnl_usd is not None
    # hard_exit_reason set so Layer 1 cannot fire again and overwrite accounting
    assert pos.early_exit_partial_reason == "SHALLOW_REVERSAL_PARTIAL"


def test_crossback_stop_partial_fill_reduces_p1_shares_for_hard_exit(tmp_path, monkeypatch) -> None:
    """
    Same partial-residual contract for Layer 3: partial fill must leave
    pos.phase == PHASE1_FILLED and p1_shares at the unfilled residual.
    """
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-cb-partial",
        "status": "MATCHED",
        "filled_size": 3.0,
        "average_fill_price": 0.54,
        "remaining_size": 7.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_dust_shares=0.5,
        crossback_stop_buffer=0.05,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.p1_fill_price = 0.58
    pos.safe_opposite_price = 0.0
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_010.0)

    asyncio.run(executor._check_crossback_stop(
        pos, yes_ask=0.52, no_ask=0.48, yes_bid=0.51, no_bid=0.49
    ))

    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert pos.p1_shares == pytest.approx(7.0, abs=1e-5)
    assert pos.hard_exit_filled_shares == pytest.approx(3.0, abs=1e-5)
    assert pos.actual_pnl_usd is not None
    # hard_exit_reason set so Layer 3 cannot fire again and overwrite accounting
    assert pos.early_exit_partial_reason == "CROSSBACK_PARTIAL"


# ---------------------------------------------------------------------------
# Layer 1/3: Allowance failure does not abort the sell attempt
# ---------------------------------------------------------------------------


def test_shallow_reversal_allowance_failure_still_attempts_sell(tmp_path, monkeypatch) -> None:
    """
    If ensure_token_sell_allowance raises, Layer 1 must warn and proceed with the
    sell attempt rather than aborting.  A confirmed fill after allowance failure
    must still close the position correctly.
    """
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-after-allowance-fail",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.56,
        "remaining_size": 0.0,
    }
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        shallow_reversal_drop_threshold=0.02,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.p1_fill_price = 0.58
    pos.peak_momentum_price = 0.60
    pos.safe_opposite_price = 0.38
    pos.phase2_tightened = False
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    # Install the failing allowance AFTER on_signal (Phase 1 readiness also calls it)
    allowance_calls: list[str] = []

    async def failing_allowance(token_id: str) -> dict:
        allowance_calls.append(token_id)
        raise RuntimeError("allowance service unavailable")

    client.ensure_token_sell_allowance = failing_allowance  # type: ignore[method-assign]

    asyncio.run(executor._check_shallow_reversal_sell(
        pos, yes_ask=0.57, no_ask=0.43, yes_bid=0.56, no_bid=0.44
    ))

    # Allowance was attempted (and raised)
    assert allowance_calls == ["yes-1"]
    # Despite allowance failure, sell was still attempted and fill confirmed
    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_reason == "SHALLOW_REVERSAL_SELL"
    assert pos.p1_shares == 0.0


def test_crossback_stop_allowance_failure_still_attempts_sell(tmp_path, monkeypatch) -> None:
    """
    Same allowance-failure resilience contract for Layer 3.
    """
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-cb-after-allowance-fail",
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
        crossback_stop_buffer=0.05,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.p1_fill_price = 0.58
    pos.safe_opposite_price = 0.0
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_010.0)

    # Install the failing allowance AFTER on_signal (Phase 1 readiness also calls it)
    allowance_calls: list[str] = []

    async def failing_allowance(token_id: str) -> dict:
        allowance_calls.append(token_id)
        raise RuntimeError("allowance service unavailable")

    client.ensure_token_sell_allowance = failing_allowance  # type: ignore[method-assign]

    asyncio.run(executor._check_crossback_stop(
        pos, yes_ask=0.52, no_ask=0.48, yes_bid=0.51, no_bid=0.49
    ))

    assert allowance_calls == ["yes-1"]
    assert pos.phase == BracketPhase.HARD_EXITED
    assert pos.hard_exit_reason == "CROSSBACK_STOP"
    assert pos.p1_shares == 0.0


# ---------------------------------------------------------------------------
# Layer 1/3: Second-tick firing blocked after partial fill (overwrite regression)
# ---------------------------------------------------------------------------


def test_shallow_reversal_second_tick_blocked_after_partial_fill(tmp_path, monkeypatch) -> None:
    """
    Regression: after a partial Layer 1 fill, hard_exit_reason is set to
    "SHALLOW_REVERSAL_PARTIAL".  A second call to _check_shallow_reversal_sell
    on the next tick must be a no-op — the accounting fields written by the first
    partial fill must not be overwritten.
    """
    client = _DummyClient()
    # First call: partial fill — 4 of 10 shares
    client.next_sell_results = [
        {
            "exchange_order_id": "test-sell-partial-1",
            "status": "MATCHED",
            "filled_size": 4.0,
            "average_fill_price": 0.57,
            "remaining_size": 6.0,
        },
        # This result should never be consumed — the second call must be blocked
        {
            "exchange_order_id": "test-sell-partial-2",
            "status": "MATCHED",
            "filled_size": 3.0,
            "average_fill_price": 0.56,
            "remaining_size": 3.0,
        },
    ]
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_dust_shares=0.5,
        shallow_reversal_drop_threshold=0.02,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.p1_fill_price = 0.58
    pos.peak_momentum_price = 0.60
    pos.safe_opposite_price = 0.38
    pos.phase2_tightened = False
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    # First tick: partial fill
    asyncio.run(executor._check_shallow_reversal_sell(
        pos, yes_ask=0.57, no_ask=0.43, yes_bid=0.56, no_bid=0.44
    ))
    assert pos.early_exit_partial_reason == "SHALLOW_REVERSAL_PARTIAL"
    assert pos.hard_exit_filled_shares == pytest.approx(4.0, abs=1e-5)
    assert pos.p1_shares == pytest.approx(6.0, abs=1e-5)
    first_pnl = pos.actual_pnl_usd

    # Second tick: must be a complete no-op (guard fires on hard_exit_reason)
    asyncio.run(executor._check_shallow_reversal_sell(
        pos, yes_ask=0.56, no_ask=0.44, yes_bid=0.55, no_bid=0.45
    ))

    # Accounting fields must be unchanged from after the first partial
    assert pos.hard_exit_filled_shares == pytest.approx(4.0, abs=1e-5)
    assert pos.p1_shares == pytest.approx(6.0, abs=1e-5)
    assert pos.actual_pnl_usd == pytest.approx(first_pnl, abs=1e-6)
    # The second sell result was never consumed
    assert len(client.next_sell_results) == 1


def test_crossback_stop_second_tick_blocked_after_partial_fill(tmp_path, monkeypatch) -> None:
    """
    Same regression guard for Layer 3: partial fill sets hard_exit_reason to
    "CROSSBACK_PARTIAL", blocking any subsequent tick from overwriting state.
    """
    client = _DummyClient()
    client.next_sell_results = [
        {
            "exchange_order_id": "test-sell-cb-partial-1",
            "status": "MATCHED",
            "filled_size": 3.0,
            "average_fill_price": 0.54,
            "remaining_size": 7.0,
        },
        # Should never be consumed
        {
            "exchange_order_id": "test-sell-cb-partial-2",
            "status": "MATCHED",
            "filled_size": 2.0,
            "average_fill_price": 0.53,
            "remaining_size": 5.0,
        },
    ]
    cfg = CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        hard_exit_stop_price=0.50,
        hard_exit_dust_shares=0.5,
        crossback_stop_buffer=0.05,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    executor = BracketExecutor(cfg, client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.p1_fill_price = 0.58
    pos.safe_opposite_price = 0.0
    pos.window_close_ts = 2_000.0
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_010.0)

    # First tick: partial fill
    asyncio.run(executor._check_crossback_stop(
        pos, yes_ask=0.52, no_ask=0.48, yes_bid=0.51, no_bid=0.49
    ))
    assert pos.early_exit_partial_reason == "CROSSBACK_PARTIAL"
    assert pos.hard_exit_filled_shares == pytest.approx(3.0, abs=1e-5)
    assert pos.p1_shares == pytest.approx(7.0, abs=1e-5)
    first_pnl = pos.actual_pnl_usd

    # Second tick: must be blocked
    asyncio.run(executor._check_crossback_stop(
        pos, yes_ask=0.51, no_ask=0.49, yes_bid=0.50, no_bid=0.50
    ))

    assert pos.hard_exit_filled_shares == pytest.approx(3.0, abs=1e-5)
    assert pos.p1_shares == pytest.approx(7.0, abs=1e-5)
    assert pos.actual_pnl_usd == pytest.approx(first_pnl, abs=1e-6)
    assert len(client.next_sell_results) == 1


# ---------------------------------------------------------------------------
# Phase 2 resting limit — full lifecycle
# ---------------------------------------------------------------------------


def _resting_cfg(tmp_path) -> CryptoDirectionConfig:
    """Base config with resting limit enabled and all required fields."""
    return CryptoDirectionConfig(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        phase2_enabled=True,
        phase2_reversal_threshold=0.01,
        phase2_min_locked_profit_per_share=0.0,
        phase2_resting_limit_enabled=True,
        hard_exit_stop_price=0.50,
        hard_exit_final_seconds=30.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )


def test_phase2_resting_posts_gtc_on_breach(tmp_path) -> None:
    """
    When phase2_resting_limit_enabled=True, the first time the opposite-side
    price breaches safe_y by reversal_threshold, a GTC buy order must be posted
    at safe_y and the position must transition to PHASE2_PENDING.
    """
    client = _DummyClient()
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    # Phase 1 uses the default next_buy_result (MATCHED)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_fill_price = 0.58
    pos.p2_token_id = "no-1"

    # ARM: no_ask drops to safe_y territory
    asyncio.run(executor._check_phase2(pos, yes_ask=0.60, no_ask=0.41))
    assert pos.safe_opposite_price > 0   # armed

    # TIGHTEN: drops to target_y zone (must be <= target_y_price=0.34)
    asyncio.run(executor._check_phase2(pos, yes_ask=0.60, no_ask=0.33))
    assert pos.phase2_tightened is True   # safe_y now tightened to 0.33

    # Override next_buy_result for the GTC post (Phase 1 already consumed the default)
    client.next_buy_result = {
        "exchange_order_id": "test-p2-resting-1",
        "status": "LIVE",
        "filled_size": 0.0,
        "average_fill_price": 0.0,
        "remaining_size": 10.0,
    }

    # BREACH: drops past tightened safe_y - reversal_threshold (0.33 - 0.01 = 0.32)
    asyncio.run(executor._check_phase2(pos, yes_ask=0.60, no_ask=0.31))

    assert pos.phase == BracketPhase.PHASE2_PENDING
    assert pos.p2_resting_order_id == "test-p2-resting-1"
    assert pos.p2_order_id == "test-p2-resting-1"
    assert pos.p2_price == pytest.approx(pos.safe_opposite_price, abs=1e-4)
    # Confirm the order was posted as a GTC buy
    gtc_calls = [c for c in client.buy_calls if c.get("entry_style") == "GTC"]
    assert len(gtc_calls) == 1
    assert gtc_calls[0]["token_id"] == "no-1"


def test_phase2_resting_fill_completes_bracket(tmp_path, monkeypatch) -> None:
    """
    When _poll_phase2_resting_order finds the GTC order is MATCHED/FILLED,
    the position must transition to BRACKET_COMPLETE with correct fill price
    and locked PnL.
    """
    client = _DummyClient()
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_fill_price = 0.58
    pos.p2_token_id = "no-1"
    pos.p2_shares = 10.0
    pos.p2_price = 0.35
    pos.p2_resting_order_id = "test-p2-resting-fill"
    pos.p2_order_id = "test-p2-resting-fill"
    pos.phase = BracketPhase.PHASE2_PENDING

    # Exchange reports the resting order filled
    client.next_buy_result = {
        "exchange_order_id": "test-p2-resting-fill",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.35,
        "remaining_size": 0.0,
    }
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._poll_phase2_resting_order(pos))

    assert pos.phase == BracketPhase.BRACKET_COMPLETE
    assert pos.p2_fill_price == pytest.approx(0.35, abs=1e-4)
    assert pos.p2_resting_order_id == ""
    # PnL is intentionally 0.0 here — on_window_close() is the sole PnL
    # authority (prevents double-counting).  The guaranteed margin is logged
    # and audited; actual_pnl_usd is finalized at settlement only.
    assert pos.actual_pnl_usd == pytest.approx(0.0, abs=1e-6)


def test_phase2_resting_cancelled_falls_back_to_phase1_filled(tmp_path) -> None:
    """
    If the GTC resting order is CANCELLED on the exchange (e.g. market admin
    cancel), _poll_phase2_resting_order must fall back to PHASE1_FILLED so
    hard exit can act as backstop.
    """
    client = _DummyClient()
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    asyncio.run(executor.on_signal(_signal_event()))
    # Override get_order_status to return CANCELLED for the resting order
    client.next_buy_result = {
        "exchange_order_id": "test-p2-resting-cancel",
        "status": "CANCELLED",
        "filled_size": 0.0,
        "average_fill_price": 0.0,
        "remaining_size": 10.0,
    }

    pos = executor.active_positions()[0]
    pos.p1_fill_price = 0.58
    pos.p2_shares = 10.0
    pos.p2_price = 0.35
    pos.p2_resting_order_id = "test-p2-resting-cancel"
    pos.p2_order_id = "test-p2-resting-cancel"
    pos.phase = BracketPhase.PHASE2_PENDING

    asyncio.run(executor._poll_phase2_resting_order(pos))

    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert pos.p2_resting_order_id == ""
    assert pos.p2_order_id == ""


def test_phase2_resting_hard_exit_cancels_p2_then_sells_p1(tmp_path, monkeypatch) -> None:
    """
    When phase is PHASE2_PENDING and the hard exit stop is hit, _check_hard_exit
    must cancel the resting P2 order, restore PHASE1_FILLED, then execute the
    Phase 1 sell.  The cancelled order ID must appear in client.cancelled_order_ids.
    """
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-hard-exit-p2",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.50,
        "remaining_size": 0.0,
    }
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_filled_at = 1_000.0
    pos.p1_fill_price = 0.58
    pos.p2_resting_order_id = "test-p2-resting-to-cancel"
    pos.p2_order_id = "test-p2-resting-to-cancel"
    pos.p2_shares = 10.0
    pos.p2_price = 0.35
    pos.window_close_ts = 2_000.0
    pos.phase = BracketPhase.PHASE2_PENDING
    # Simulate clean cancel (no partial fill) returned by get_order_status
    # after _cancel_resting_p2_and_reconcile sends the cancel request.
    client.next_buy_result = {
        "exchange_order_id": "test-p2-resting-to-cancel",
        "status": "CANCELLED",
        "filled_size": 0.0,
        "average_fill_price": 0.0,
        "remaining_size": 10.0,
    }
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    # Mark price at 0.50 — triggers hard exit stop
    asyncio.run(executor._check_hard_exit(
        pos, yes_ask=0.50, no_ask=0.50, yes_bid=0.50, no_bid=0.50
    ))

    # Resting order was cancelled before the Phase 1 sell
    assert "test-p2-resting-to-cancel" in client.cancelled_order_ids
    assert pos.p2_resting_order_id == ""
    # Phase 1 was sold and position is hard-exited
    assert pos.phase == BracketPhase.HARD_EXITED
    assert client.sell_calls


def test_phase2_resting_window_close_cancels_order_and_settles_phase1(tmp_path) -> None:
    """
    on_window_close with PHASE2_PENDING must cancel the resting GTC order and
    settle the position as Phase1-only, exactly as it does for FOK-pending orders.
    """
    client = _DummyClient()
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_fill_price = 0.58
    pos.p2_resting_order_id = "test-p2-resting-wc"
    pos.p2_order_id = "test-p2-resting-wc"
    pos.p2_shares = 10.0
    pos.p2_price = 0.35
    pos.phase = BracketPhase.PHASE2_PENDING
    # Simulate clean cancel returned by get_order_status inside the reconcile helper.
    client.next_buy_result = {
        "exchange_order_id": "test-p2-resting-wc",
        "status": "CANCELLED",
        "filled_size": 0.0,
        "average_fill_price": 0.0,
        "remaining_size": 10.0,
    }

    asyncio.run(executor.on_window_close(1_700_000_000, "BTC", yes_won=True))

    # Resting order was cancelled
    assert "test-p2-resting-wc" in client.cancelled_order_ids
    # Summary stored, position settled as Phase1-only
    summary = executor.take_window_summary(1_700_000_000, "BTC")
    assert summary is not None
    assert summary["phase2_filled"] is False
    assert summary["hard_exited"] is False


# ---------------------------------------------------------------------------
# Entry filter — _check_entry_filter + on_signal integration
# ---------------------------------------------------------------------------

def _filter_cfg(tmp_path, **overrides) -> CryptoDirectionConfig:
    """Config with entry_filter_enabled=True and all filter rules active."""
    defaults = dict(
        execute_enabled=True,
        phase1_shares=10.0,
        phase1_max_chase_cents=0.0,
        min_bracket_shares=1.0,
        entry_filter_enabled=True,
        entry_filter_skip_lag_model=True,
        entry_filter_max_gap_to_ceiling=0.025,
        entry_filter_min_minutes_remaining=7.0,
        bracket_audit_log_path=str(tmp_path / "bracket_trades.jsonl"),
    )
    defaults.update(overrides)
    return CryptoDirectionConfig(**defaults)


def _continuation_signal(**overrides) -> BracketSignalEvent:
    """A passing continuation signal (all filter rules satisfied by default)."""
    base = _signal_event()
    # Override entry_model and minutes_remaining to values that pass all rules.
    # entry_model is not a constructor arg on the default helper, so patch it.
    sig = base.model_copy(update={"entry_model": "continuation", "minutes_remaining": 12.0})
    if overrides:
        sig = sig.model_copy(update=overrides)
    return sig


def test_entry_filter_disabled_passes_lag_signal(tmp_path) -> None:
    """When entry_filter_enabled=False, lag signals are NOT skipped."""
    client = _DummyClient()
    cfg = _filter_cfg(tmp_path, entry_filter_enabled=False)
    executor = BracketExecutor(cfg, client)

    lag_signal = _signal_event()  # default entry_model="lag"
    result = asyncio.run(executor.on_signal(lag_signal))

    # Filter is off → Phase 1 should be submitted
    assert result is True
    assert len(client.buy_calls) == 1


def test_entry_filter_skips_lag_model(tmp_path) -> None:
    """entry_filter_skip_lag_model=True must skip entry_model='lag' signals."""
    client = _DummyClient()
    executor = BracketExecutor(_filter_cfg(tmp_path), client)

    lag_signal = _signal_event()  # default entry_model="lag"
    result = asyncio.run(executor.on_signal(lag_signal))

    assert result is False
    assert len(client.buy_calls) == 0
    # Audit event written
    audit_log = tmp_path / "bracket_trades.jsonl"
    entries = [json.loads(l) for l in audit_log.read_text().splitlines() if l]
    filtered = [e for e in entries if e.get("type") == "SIGNAL_FILTERED"]
    assert len(filtered) == 1
    assert "LAG_MODEL" in filtered[0]["reason"]


def test_entry_filter_skips_large_gap(tmp_path) -> None:
    """Opposite-side price too far above profitable ceiling must be skipped."""
    client = _DummyClient()
    executor = BracketExecutor(_filter_cfg(tmp_path), client)

    # momentum_price=0.58 → profitable_ceiling ≈ 0.4125
    # opposite_price=0.46 → gap = 0.46 - 0.4125 = 0.0475 > 0.025 → skip
    big_gap_signal = _continuation_signal(opposite_price=0.46)
    result = asyncio.run(executor.on_signal(big_gap_signal))

    assert result is False
    assert len(client.buy_calls) == 0
    audit_log = tmp_path / "bracket_trades.jsonl"
    entries = [json.loads(l) for l in audit_log.read_text().splitlines() if l]
    filtered = [e for e in entries if e.get("type") == "SIGNAL_FILTERED"]
    assert len(filtered) == 1
    assert "GAP_TO_CEILING" in filtered[0]["reason"]


def test_entry_filter_skips_low_minutes(tmp_path) -> None:
    """Signals with too few minutes remaining must be skipped."""
    client = _DummyClient()
    executor = BracketExecutor(_filter_cfg(tmp_path), client)

    low_time_signal = _continuation_signal(minutes_remaining=5.0)
    result = asyncio.run(executor.on_signal(low_time_signal))

    assert result is False
    assert len(client.buy_calls) == 0
    audit_log = tmp_path / "bracket_trades.jsonl"
    entries = [json.loads(l) for l in audit_log.read_text().splitlines() if l]
    filtered = [e for e in entries if e.get("type") == "SIGNAL_FILTERED"]
    assert len(filtered) == 1
    assert "LOW_TIME" in filtered[0]["reason"]


def test_entry_filter_passes_good_continuation_signal(tmp_path) -> None:
    """A continuation signal with good gap and plenty of time must NOT be skipped."""
    client = _DummyClient()
    executor = BracketExecutor(_filter_cfg(tmp_path), client)

    # momentum_price=0.58 → ceiling ≈ 0.4125; opposite_price=0.43 → gap ≈ 0.0175 < 0.025
    # minutes_remaining=12.0 > 7.0; entry_model="continuation"
    good_signal = _continuation_signal(opposite_price=0.43, minutes_remaining=12.0)
    result = asyncio.run(executor.on_signal(good_signal))

    assert result is True
    assert len(client.buy_calls) == 1


def test_entry_filter_check_all_three_rules_independently(tmp_path) -> None:
    """Verify each rule triggers independently when others are satisfied."""
    def _check(signal, extra_cfg=None) -> tuple[bool, str]:
        """Run filter standalone without placing orders."""
        kw = {} if extra_cfg is None else extra_cfg
        cfg = _filter_cfg(tmp_path, **kw)
        executor = BracketExecutor(cfg, _DummyClient())
        skip, reason = executor._check_entry_filter(signal)
        return skip, reason

    # Rule 1: lag model
    skip, reason = _check(_signal_event())  # entry_model="lag" by default
    assert skip is True and "LAG_MODEL" in reason

    # Rule 2: large gap (continuation, gap > 0.025)
    skip, reason = _check(_continuation_signal(opposite_price=0.46))
    assert skip is True and "GAP_TO_CEILING" in reason

    # Rule 3: low time (continuation, normal gap, low minutes)
    skip, reason = _check(_continuation_signal(minutes_remaining=4.0))
    assert skip is True and "LOW_TIME" in reason

    # All rules pass
    skip, reason = _check(_continuation_signal(opposite_price=0.43, minutes_remaining=12.0))
    assert skip is False and reason == ""


# ---------------------------------------------------------------------------
# Resting P2 correctness — double-count, partial fill, arm→breach direct
# ---------------------------------------------------------------------------


def test_resting_fill_then_window_close_pnl_not_doubled(tmp_path, monkeypatch) -> None:
    """
    Full end-to-end: resting P2 fills during the window, window then closes.
    actual_pnl_usd must be written exactly once (at settlement in
    on_window_close), never pre-written by the poll path.
    """
    from src.fees import taker_fee as _tf

    client = _DummyClient()
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_fill_price = 0.58
    pos.p2_token_id = "no-1"
    pos.p2_shares = 10.0
    pos.p2_price = 0.41
    pos.p2_resting_order_id = "test-p2-fill-wc"
    pos.p2_order_id = "test-p2-fill-wc"
    pos.phase = BracketPhase.PHASE2_PENDING
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    # Step 1: resting order fills during window
    client.next_buy_result = {
        "exchange_order_id": "test-p2-fill-wc",
        "status": "MATCHED",
        "filled_size": 10.0,
        "average_fill_price": 0.41,
        "remaining_size": 0.0,
    }
    asyncio.run(executor._poll_phase2_resting_order(pos))
    assert pos.phase == BracketPhase.BRACKET_COMPLETE

    # Poll path must NOT write PnL — it should still be 0.0
    assert pos.actual_pnl_usd == pytest.approx(0.0, abs=1e-6), (
        f"Poll path must not pre-write PnL; got {pos.actual_pnl_usd}"
    )

    # Step 2: window closes — this is where PnL must be written exactly once
    asyncio.run(executor.on_window_close(1_700_000_000, "BTC", yes_won=True))
    summary = executor.take_window_summary(1_700_000_000, "BTC")
    assert summary is not None

    guaranteed = round(
        1.0
        - 0.58 * (1.0 + _tf(0.58))
        - 0.41 * (1.0 + _tf(0.41)),
        6,
    )
    expected_pnl = round(guaranteed * 10.0, 4)
    assert summary["actual_pnl_usd"] == pytest.approx(expected_pnl, abs=1e-3), (
        f"PnL doubled: expected {expected_pnl}, got {summary['actual_pnl_usd']}"
    )


def test_phase2_resting_partial_fill_cancel_records_partial_hedge(tmp_path) -> None:
    """
    Exchange partially fills (3/10) then cancels a GTC resting order.
    _poll_phase2_resting_order must record p2_partial_filled_shares=3,
    clear the order IDs, and fall back to PHASE1_FILLED.
    """
    client = _DummyClient()
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_fill_price = 0.58
    pos.p2_shares = 10.0
    pos.p2_price = 0.41
    pos.p2_resting_order_id = "test-partial-cancel"
    pos.p2_order_id = "test-partial-cancel"
    pos.phase = BracketPhase.PHASE2_PENDING

    client.next_buy_result = {
        "exchange_order_id": "test-partial-cancel",
        "status": "CANCELLED",
        "filled_size": 3.0,
        "average_fill_price": 0.41,
        "remaining_size": 7.0,
    }
    asyncio.run(executor._poll_phase2_resting_order(pos))

    assert pos.phase == BracketPhase.PHASE1_FILLED
    assert pos.p2_partial_filled_shares == pytest.approx(3.0, abs=1e-6)
    assert pos.p2_partial_fill_price == pytest.approx(0.41, abs=1e-4)
    assert pos.p2_resting_order_id == ""
    assert pos.p2_order_id == ""


def test_hard_exit_skips_hedged_shares_after_partial_p2_fill(tmp_path, monkeypatch) -> None:
    """
    With 3 of 10 Phase 1 shares hedged by a partial resting P2 fill,
    hard exit must sell only the 7 unhedged shares, not the full 10.
    Selling the hedged 3 would destroy guaranteed bracket profit.
    """
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-he-partial",
        "status": "MATCHED",
        "filled_size": 7.0,
        "average_fill_price": 0.50,
        "remaining_size": 0.0,
    }
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_fill_price = 0.58
    pos.p1_filled_at = 1_000.0
    pos.p1_shares = 10.0
    pos.window_close_ts = 2_000.0
    pos.phase = BracketPhase.PHASE1_FILLED
    pos.p2_partial_filled_shares = 3.0
    pos.p2_partial_fill_price = 0.41
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    asyncio.run(executor._check_hard_exit(
        pos, yes_ask=0.50, no_ask=0.50, yes_bid=0.50, no_bid=0.50
    ))

    assert client.sell_calls, "Expected hard exit sell to be placed"
    # Each individual sell attempt must target at most 7 unhedged shares
    sell_sizes = [c["size"] for c in client.sell_calls]
    assert all(s <= 7.0 + 1e-6 for s in sell_sizes), (
        f"Hard exit targeted more than 7 unhedged shares: {sell_sizes}"
    )


def test_window_close_splits_pnl_between_bracket_and_directional(tmp_path) -> None:
    """
    After 3/10 partial P2 fill + cancel with no hard exit:
    settlement PnL = guaranteed_per_share * 3  (bracketed)
                   + pnl_per_share * 7          (directional, YES wins)
    """
    from src.fees import taker_fee as _tf

    client = _DummyClient()
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_fill_price = 0.58
    pos.p1_shares = 10.0
    pos.p2_partial_filled_shares = 3.0
    pos.p2_partial_fill_price = 0.41
    pos.phase = BracketPhase.PHASE1_FILLED

    asyncio.run(executor.on_window_close(1_700_000_000, "BTC", yes_won=True))
    summary = executor.take_window_summary(1_700_000_000, "BTC")
    assert summary is not None

    x, y = 0.58, 0.41
    cost_x = x * (1.0 + _tf(x))
    guaranteed = round(1.0 - cost_x - y * (1.0 + _tf(y)), 6)
    bracket_pnl = round(guaranteed * 3.0, 4)
    directional_pnl = (1.0 - cost_x) * 7.0   # YES wins
    expected = round(bracket_pnl + directional_pnl, 4)

    assert summary["actual_pnl_usd"] == pytest.approx(expected, abs=1e-3)


def test_phase2_check_arm_to_breach_without_tighten(tmp_path) -> None:
    """
    The resting order must post on BREACH of the original arm-level safe_price
    even when the TIGHTEN step never fired (market went ARM → BREACH directly,
    skipping target_y_price).  This locks in the documented behavior.
    """
    client = _DummyClient()
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_fill_price = 0.58
    pos.p2_token_id = "no-1"

    # ARM: no_ask enters the profitable ceiling zone; safe_y ≈ 0.4125
    asyncio.run(executor._check_phase2(pos, yes_ask=0.60, no_ask=0.41))
    assert pos.safe_opposite_price > 0
    assert pos.phase2_tightened is False   # TIGHTEN has NOT fired

    client.next_buy_result = {
        "exchange_order_id": "test-direct-breach",
        "status": "LIVE",
        "filled_size": 0.0,
        "average_fill_price": 0.0,
        "remaining_size": 10.0,
    }

    # BREACH: drop past safe_y - reversal (0.4125 - 0.01 = 0.4025) without
    # passing through target_y_price=0.34 first
    asyncio.run(executor._check_phase2(pos, yes_ask=0.60, no_ask=0.40))

    assert pos.phase == BracketPhase.PHASE2_PENDING
    assert pos.p2_resting_order_id == "test-direct-breach"
    assert pos.phase2_tightened is False   # confirmed: tighten still never fired
    gtc_calls = [c for c in client.buy_calls if c.get("entry_style") == "GTC"]
    assert len(gtc_calls) == 1


# ---------------------------------------------------------------------------
# _cancel_resting_p2_and_reconcile — cancel-time partial fill reconciliation
# ---------------------------------------------------------------------------


def test_hard_exit_reconciles_partial_resting_fill_on_cancel(
    tmp_path, monkeypatch
) -> None:
    """
    When hard exit fires against a live resting P2 GTC order that was already
    partially filled at the exchange, _cancel_resting_p2_and_reconcile must
    fetch the post-cancel status, record the partial hedge in p2_partial_*,
    and hard exit must sell only the unhedged remainder (7 of 10 shares).

    This closes the gap where _cancel_order alone (no status fetch) silently
    discards partial fills on the hard-exit cancel path.
    """
    client = _DummyClient()
    client.next_sell_result = {
        "exchange_order_id": "test-sell-he-reconcile",
        "status": "MATCHED",
        "filled_size": 7.0,
        "average_fill_price": 0.50,
        "remaining_size": 0.0,
    }
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_fill_price = 0.58
    pos.p1_filled_at = 1_000.0
    pos.p1_shares = 10.0
    pos.window_close_ts = 2_000.0
    # Simulate a live resting P2 order that has been partially filled (3/10 shares)
    pos.p2_resting_order_id = "test-p2-resting-he"
    pos.p2_order_id = "test-p2-resting-he"
    pos.p2_shares = 10.0
    pos.p2_price = 0.40
    pos.phase = BracketPhase.PHASE2_PENDING

    # get_order_status returns: CANCELLED with 3 partial shares filled
    client.next_buy_result = {
        "exchange_order_id": "test-p2-resting-he",
        "status": "CANCELLED",
        "filled_size": 3.0,
        "average_fill_price": 0.40,
        "remaining_size": 7.0,
    }
    monkeypatch.setattr("src.bracket_executor.time.time", lambda: 1_050.0)

    # mark=0.48 triggers stop (≤ 0.50) from PHASE2_PENDING
    asyncio.run(executor._check_hard_exit(
        pos, yes_ask=0.48, no_ask=0.48, yes_bid=0.48, no_bid=0.48,
    ))

    # Reconcile must have captured the partial hedge
    assert pos.p2_partial_filled_shares == pytest.approx(3.0), (
        f"Expected partial hedge of 3.0; got {pos.p2_partial_filled_shares}"
    )
    assert pos.p2_partial_fill_price == pytest.approx(0.40)

    # Resting order IDs must be cleared after reconcile
    assert pos.p2_resting_order_id == ""
    assert pos.p2_order_id == ""

    # Hard exit must have sold only the 7 unhedged shares
    assert client.sell_calls, "Expected hard exit sell to be placed"
    sell_sizes = [c["size"] for c in client.sell_calls]
    assert all(s <= 7.0 + 1e-6 for s in sell_sizes), (
        f"Hard exit targeted more than 7 unhedged shares: {sell_sizes}"
    )
    total_sold = sum(sell_sizes)
    assert total_sold == pytest.approx(7.0, abs=1e-6), (
        f"Expected 7 shares sold (unhedged); got {total_sold}"
    )


def test_window_close_reconciles_partial_resting_fill_on_cancel(
    tmp_path,
) -> None:
    """
    When the window closes against a live resting P2 GTC order that was already
    partially filled at the exchange, _cancel_resting_p2_and_reconcile must
    fetch the post-cancel status, record the partial hedge, and settlement
    must split PnL between the bracketed (3 shares) and directional (7 shares)
    portions.

    This closes the symmetric gap on the window-close cancel path.
    """
    from src.fees import taker_fee as _tf

    client = _DummyClient()
    executor = BracketExecutor(_resting_cfg(tmp_path), client)
    asyncio.run(executor.on_signal(_signal_event()))

    pos = executor.active_positions()[0]
    pos.p1_fill_price = 0.58
    pos.p1_shares = 10.0
    # Simulate a live resting P2 order that has been partially filled (3/10 shares)
    pos.p2_resting_order_id = "test-p2-resting-wc"
    pos.p2_order_id = "test-p2-resting-wc"
    pos.p2_shares = 10.0
    pos.p2_price = 0.41
    pos.phase = BracketPhase.PHASE2_PENDING

    # get_order_status returns: CANCELLED with 3 partial shares filled
    client.next_buy_result = {
        "exchange_order_id": "test-p2-resting-wc",
        "status": "CANCELLED",
        "filled_size": 3.0,
        "average_fill_price": 0.41,
        "remaining_size": 7.0,
    }

    asyncio.run(executor.on_window_close(1_700_000_000, "BTC", yes_won=True))
    summary = executor.take_window_summary(1_700_000_000, "BTC")
    assert summary is not None

    # Verify partial hedge was recorded by reconcile
    assert pos.p2_partial_filled_shares == pytest.approx(3.0), (
        f"Reconcile must record partial hedge; got {pos.p2_partial_filled_shares}"
    )
    assert pos.p2_resting_order_id == ""
    assert pos.p2_order_id == ""

    # Settlement must split: 3 bracketed shares + 7 directional shares (YES wins)
    x, y = 0.58, 0.41
    cost_x = x * (1.0 + _tf(x))
    guaranteed = round(1.0 - cost_x - y * (1.0 + _tf(y)), 6)
    bracket_pnl = round(guaranteed * 3.0, 4)
    directional_pnl = (1.0 - cost_x) * 7.0   # YES wins: receive $1/share
    expected = round(bracket_pnl + directional_pnl, 4)

    assert summary["actual_pnl_usd"] == pytest.approx(expected, abs=1e-3), (
        f"Settlement PnL split wrong: expected {expected}, got {summary['actual_pnl_usd']}"
    )
