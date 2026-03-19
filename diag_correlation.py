"""Diagnose correlation_dislocation gap distribution with real orderbook data."""
import asyncio
from pathlib import Path
from src.config import load_config
from src.market_data import MarketDataService
from src.market_relationships import build_relationship_groups, find_best_dislocation_pair


async def main():
    cfg = load_config(Path('config.yaml'))
    svc = MarketDataService(cfg, Path('data'))
    markets = await svc.refresh_markets()
    print(f'Market cache: {len(markets)} markets')

    groups = build_relationship_groups(list(markets.values()), 2)
    print(f'Relationship groups: {len(groups)}')

    positive_gaps = []
    negative_gaps = []
    no_price_groups = 0

    for group in groups[:30]:  # sample first 30 groups
        orderbooks = {}
        for m in group:
            try:
                ob = await svc.fetch_orderbook(m.token_id)
                orderbooks[m.market_id] = ob
            except Exception:
                continue

        mid_prices = {}
        for m in group:
            ob = orderbooks.get(m.market_id)
            if ob and ob.bids and ob.asks:
                mid_prices[m.market_id] = round((ob.bids[0].price + ob.asks[0].price) / 2.0, 4)

        if len(mid_prices) < 2:
            no_price_groups += 1
            continue

        for i, earlier in enumerate(group):
            emid = mid_prices.get(earlier.market_id)
            if emid is None:
                continue
            for later in group[i+1:]:
                lmid = mid_prices.get(later.market_id)
                if lmid is None:
                    continue
                gap = emid - lmid
                if gap > 0:
                    positive_gaps.append((gap, earlier.title[:50], later.title[:50]))
                else:
                    negative_gaps.append((gap, earlier.title[:50], later.title[:50]))

    print(f'\nGroups with no price data: {no_price_groups}')
    print(f'Positive gaps (earlier > later): {len(positive_gaps)}')
    print(f'Negative gaps (earlier < later): {len(negative_gaps)}')

    if positive_gaps:
        print('\nTop positive gaps:')
        for gap, e, l in sorted(positive_gaps, reverse=True)[:10]:
            print(f'  +{gap:.4f}  earlier={e}')
    else:
        print('\nNO positive gaps found! All earlier markets priced <= later markets.')
        print('Sample negative gaps:')
        for gap, e, l in sorted(negative_gaps)[:5]:
            print(f'  {gap:.4f}  earlier={e}')

    await svc.client.close()


asyncio.run(main())
