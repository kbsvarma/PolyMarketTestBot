"""Test the momentum signal generator."""
import asyncio, json
from pathlib import Path
from src.config import load_config
from src.market_data import MarketDataService
from src.momentum_signal_generator import generate_momentum_signals


async def main():
    cfg = load_config(Path('config.yaml'))
    svc = MarketDataService(cfg, Path('data'))
    await svc.refresh_markets()

    test_path = Path('/tmp/test_signals.json')
    test_path.write_text('[]')

    n = await generate_momentum_signals(cfg, svc, test_path)
    print(f'New signals generated: {n}')

    signals = json.loads(test_path.read_text())
    print(f'Total in file: {len(signals)}')
    for s in signals[:5]:
        print(f"\n  title: {s.get('title', '')[:60]}")
        print(f"  market_id: {s.get('market_id', '')[:20]}...")
        print(f"  avg_price={s.get('source_price')} fair={s.get('fair_price')} conf={s.get('confidence_score')}")
        print(f"  rationale: {s.get('rationale', '')[:100]}")

    await svc.client.close()


asyncio.run(main())
