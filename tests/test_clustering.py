from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.clustering import cluster_detections
from src.config import load_config
from src.models import DetectionEvent


def test_cluster_detects_multi_wallet_agreement() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml", root / ".env.example")
    base = datetime.now(timezone.utc)
    detections = [
        DetectionEvent(
            event_key=f"k{i}",
            wallet_address=f"0x{i}",
            market_title="M",
            market_slug="m",
            market_id="market-1",
            token_id="token-1",
            side="BUY",
            price=0.5,
            size=10,
            notional=5,
            transaction_hash=f"tx{i}",
            detection_latency_seconds=5,
            category="politics",
            source_trade_timestamp=base + timedelta(seconds=i * 10),
        )
        for i in range(2)
    ]
    clusters = cluster_detections(config, detections, root / "data")
    assert len(clusters) == 1
    assert clusters[0]["wallet_count"] == 2
