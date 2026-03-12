from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from src.config import AppConfig
from src.models import ClusterSignal, DetectionEvent
from src.utils import stable_event_key, write_csv


def cluster_detections(config: AppConfig, detections: list[DetectionEvent], data_dir: Path) -> list[ClusterSignal]:
    grouped: dict[tuple[str, str, str], list[DetectionEvent]] = defaultdict(list)
    for detection in detections:
        grouped[(detection.market_id, detection.token_id, detection.side)].append(detection)

    clusters: list[ClusterSignal] = []
    window = timedelta(seconds=config.cluster.confirmation_window_seconds)
    for (market_id, token_id, side), group in grouped.items():
        group = sorted(group, key=lambda item: item.source_trade_timestamp)
        first = group[0]
        last = group[-1]
        unique_wallets = {item.wallet_address for item in group if last.source_trade_timestamp - item.source_trade_timestamp <= window}
        wallet_count = len(unique_wallets)
        if wallet_count >= config.cluster.min_wallets:
            strength = round(min(wallet_count / max(config.cluster.strong_cluster_wallets, 1), 1.0), 4)
            clusters.append(
                ClusterSignal(
                    cluster_id=stable_event_key(market_id, token_id, side, first.source_trade_timestamp.isoformat()),
                    market_id=market_id,
                    token_id=token_id,
                    side=side,
                    wallet_count=wallet_count,
                    cluster_strength=strength,
                    first_seen=first.source_trade_timestamp,
                    last_seen=last.source_trade_timestamp,
                    category=first.category,
                    wallets=sorted(unique_wallets),
                )
            )
    if clusters:
        write_csv(data_dir / "clustered_signals.csv", [cluster.model_dump(mode="json") for cluster in clusters])
    return clusters
