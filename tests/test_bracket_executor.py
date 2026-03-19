from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from src.bracket_executor import BracketExecutor, BracketPhase
from src.config import CryptoDirectionConfig
from src.models import BracketSignalEvent


class _DummyClient:
    def __init__(self) -> None:
        self.buy_calls: list[dict] = []
        self.sell_calls: list[dict] = []
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
        if self.next_sell_results:
            result = dict(self.next_sell_results.pop(0))
            self.next_sell_result = dict(result)
            return result
        return dict(self.next_sell_result)

    async def get_order_status(self, exchange_order_id: str) -> dict:
        if exchange_order_id.startswith("test-sell"):
            return dict(self.next_sell_result)
        return dict(self.next_buy_result)


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
