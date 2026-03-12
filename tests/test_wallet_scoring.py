from pathlib import Path

from src.config import load_config
from src.wallet_discovery import WalletDiscoveryService
from src.wallet_scoring import WalletScoringService


def test_wallet_scoring_orders_by_global_score() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    wallets = __import__("asyncio").run(WalletDiscoveryService(config, root / "data").run_discovery_cycle())
    scored = WalletScoringService(config, root / "data").score_wallets(wallets)
    assert scored[0].global_score >= scored[-1].global_score
    assert scored[0].copyability_score >= 0.4
