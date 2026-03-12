from pathlib import Path

from src.config import load_config
from src.trade_monitor import TradeMonitor


def test_trade_event_key_is_stable() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    monitor = TradeMonitor(config, root / "data", object())
    key_one = monitor.make_event_key("0x1", "m1", "t1", "tx1", "BUY")
    key_two = monitor.make_event_key("0x1", "m1", "t1", "tx1", "BUY")
    assert key_one == key_two
