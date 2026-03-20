from pathlib import Path

from src.config import apply_crypto_direction_profile, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_btc_15m_profile_overrides_crypto_lane_without_mutating_base() -> None:
    config = load_config(ROOT / "config.yaml")

    assert config.crypto_direction.window_duration_seconds == 300
    assert config.crypto_direction.btc.slug_prefix == "btc-updown-5m"
    assert config.crypto_direction.window_report_path == "logs/window_report.md"


def test_polymarket_btc_5m_instance_profile_has_isolated_logs() -> None:
    config = load_config(ROOT / "config.yaml")
    profiled = apply_crypto_direction_profile(config, "polymarket_btc_5m")

    assert profiled.crypto_direction.venue == "polymarket"
    assert profiled.crypto_direction.instance_name == "polymarket_btc_5m"
    assert profiled.crypto_direction.window_duration_seconds == 300
    assert profiled.crypto_direction.signal_event_log_path == "logs/crypto_signal_events_pm_btc_5m.jsonl"
    assert profiled.crypto_direction.window_report_path == "logs/window_report_pm_btc_5m.md"
    assert profiled.crypto_direction.bracket_audit_log_path == "logs/bracket_trades_pm_btc_5m.jsonl"


def test_kalshi_15m_profile_is_ready_for_runtime() -> None:
    config = load_config(ROOT / "config.yaml")
    profiled = apply_crypto_direction_profile(config, "kalshi_btc_15m")

    assert profiled.crypto_direction.venue == "kalshi"
    assert profiled.crypto_direction.instance_name == "kalshi_btc_15m"
    assert profiled.crypto_direction.window_duration_seconds == 900
    assert profiled.crypto_direction.btc.slug_prefix == "KXBTC15M"
    assert profiled.crypto_direction.time_gate_minutes == 10.0
    assert profiled.crypto_direction.entry_range_high == 0.61
    assert profiled.crypto_direction.track_eth is False
    assert profiled.crypto_direction.signal_event_log_path == "logs/crypto_signal_events_kalshi_btc_15m.jsonl"

    profiled = apply_crypto_direction_profile(config, "btc_15m")

    assert profiled.crypto_direction.window_duration_seconds == 900
    assert profiled.crypto_direction.btc.slug_prefix == "btc-updown-15m"
    assert profiled.crypto_direction.track_btc is True
    assert profiled.crypto_direction.track_eth is False
    assert profiled.crypto_direction.time_gate_minutes == 10.0
    assert profiled.crypto_direction.entry_range_high == 0.61
    assert profiled.crypto_direction.signal_event_log_path == "logs/crypto_signal_events_15m.jsonl"
    assert profiled.crypto_direction.window_report_path == "logs/window_report_15m.md"
    assert profiled.crypto_direction.bracket_audit_log_path == "logs/bracket_trades_15m.jsonl"

    # Base 5m lane remains unchanged.
    assert config.crypto_direction.window_duration_seconds == 300
    assert config.crypto_direction.btc.slug_prefix == "btc-updown-5m"
    assert config.crypto_direction.window_report_path == "logs/window_report.md"
