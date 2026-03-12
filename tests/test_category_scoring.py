from pathlib import Path

from src.category_scoring import CategoryScorer
from src.config import load_config
from src.wallet_discovery import WalletDiscoveryService
from src.wallet_scoring import WalletScoringService


def test_category_scorecards_generated() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    wallets = __import__("asyncio").run(WalletDiscoveryService(config, root / "data").run_discovery_cycle())
    scored = WalletScoringService(config, root / "data").score_wallets(wallets)
    rows = CategoryScorer(config, root / "data").build_scorecards(scored)
    assert rows
    assert rows[0].category
