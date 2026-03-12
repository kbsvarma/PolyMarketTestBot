from pathlib import Path

from src.config import ensure_runtime_files, load_config


def test_load_config_defaults() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    assert config.mode.value == "RESEARCH"
    assert config.bankroll.paper_starting_bankroll == 200.0


def test_ensure_runtime_files() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    ensure_runtime_files(root, config)
    assert (root / "data" / "top_wallets.json").exists()
