from __future__ import annotations

from pathlib import Path

from main import _build_operator_smoke_decision, _clear_stale_operator_smoke_orders
from src.config import load_config
from src.live_orders import LiveOrderStore
from src.models import EntryStyle, LiveOrder, OrderLifecycleStatus
from src.state import AppStateStore


def test_operator_smoke_order_disabled_by_default(monkeypatch) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    monkeypatch.delenv("POLYBOT_SMOKE_ORDER_ENABLED", raising=False)
    assert _build_operator_smoke_decision(config) is None


def test_operator_smoke_order_requires_core_fields(monkeypatch) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    monkeypatch.setenv("POLYBOT_SMOKE_ORDER_ENABLED", "true")
    monkeypatch.delenv("POLYBOT_SMOKE_ORDER_MARKET_ID", raising=False)
    monkeypatch.delenv("POLYBOT_SMOKE_ORDER_TOKEN_ID", raising=False)
    monkeypatch.delenv("POLYBOT_SMOKE_ORDER_PRICE", raising=False)
    try:
        _build_operator_smoke_decision(config)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "POLYBOT_SMOKE_ORDER_MARKET_ID" in str(exc)


def test_operator_smoke_order_uses_requested_notional(monkeypatch) -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.live_smoke.yaml", root / ".env.example")
    monkeypatch.setenv("POLYBOT_SMOKE_ORDER_ENABLED", "true")
    monkeypatch.setenv("POLYBOT_SMOKE_ORDER_MARKET_ID", "market-1")
    monkeypatch.setenv("POLYBOT_SMOKE_ORDER_TOKEN_ID", "token-1")
    monkeypatch.setenv("POLYBOT_SMOKE_ORDER_PRICE", "0.52")
    monkeypatch.setenv("POLYBOT_SMOKE_ORDER_NOTIONAL_USD", "9.0")

    decision = _build_operator_smoke_decision(config)

    assert decision is not None
    assert decision.action.value == "LIVE_COPY"
    assert decision.market_id == "market-1"
    assert decision.token_id == "token-1"
    assert decision.executable_price == 0.52
    assert decision.scaled_notional == 9.0
    assert decision.context["operator_smoke_order"] is True
    assert decision.context["requested_notional"] == 9.0


def test_operator_smoke_run_can_clear_stale_manual_resume(tmp_path: Path) -> None:
    store = AppStateStore(tmp_path / "app_state.json")
    store.write(
        {
            "mode": "LIVE",
            "paused": True,
            "pause_reason": "old pause",
            "manual_live_enable": False,
            "manual_resume_required": True,
        }
    )

    store.clear_pause()
    store.update_system_status(
        manual_live_enable=True,
        manual_resume_required=False,
        paused=False,
        pause_reason="",
    )

    payload = store.read()
    assert payload["paused"] is False
    assert payload["manual_resume_required"] is False
    assert payload["manual_live_enable"] is True
    assert payload["pause_reason"] == ""


def test_clear_stale_operator_smoke_orders_only_marks_local_pending_smoke_orders(tmp_path: Path) -> None:
    store = LiveOrderStore(tmp_path / "data" / "live_orders.json")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    store.save(
        [
            LiveOrder(
                local_decision_id="operator-smoke-123",
                local_order_id="o1",
                client_order_id="c1",
                market_id="m1",
                token_id="t1",
                side="BUY",
                intended_price=0.5,
                intended_size=2.0,
                entry_style=EntryStyle.PASSIVE_LIMIT,
                lifecycle_status=OrderLifecycleStatus.SUBMITTING,
                exchange_order_id="",
            ),
            LiveOrder(
                local_decision_id="normal-live-123",
                local_order_id="o2",
                client_order_id="c2",
                market_id="m2",
                token_id="t2",
                side="BUY",
                intended_price=0.5,
                intended_size=2.0,
                entry_style=EntryStyle.PASSIVE_LIMIT,
                lifecycle_status=OrderLifecycleStatus.SUBMITTING,
                exchange_order_id="",
            ),
        ]
    )

    _clear_stale_operator_smoke_orders(tmp_path)

    orders = store.load()
    assert orders[0].lifecycle_status == OrderLifecycleStatus.REJECTED
    assert orders[0].terminal_state is True
    assert orders[1].lifecycle_status == OrderLifecycleStatus.SUBMITTING
