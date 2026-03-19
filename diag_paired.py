import asyncio
from pathlib import Path
from src.config import load_config
from src.market_data import MarketDataService

async def main():
    cfg = load_config(Path('config.yaml'))
    svc = MarketDataService(cfg, Path('data'))
    markets = await svc.refresh_markets()

    grouped = {}
    for m in svc.token_cache.values():
        if not m.active or m.closed:
            continue
        grouped.setdefault(m.market_id, []).append(m)

    pairs = []
    for mid, mlist in grouped.items():
        if len(mlist) != 2:
            continue
        yes_m = next((m for m in mlist if '[yes]' in m.title.lower()), None)
        no_m = next((m for m in mlist if '[no]' in m.title.lower()), None)
        if yes_m and no_m:
            pairs.append((yes_m, no_m))

    print(f'Total YES/NO pairs found: {len(pairs)}')

    best_edges = []
    errors = 0
    for yes_m, no_m in pairs[:60]:
        try:
            yes_ob = await svc.fetch_orderbook(yes_m.token_id)
            no_ob = await svc.fetch_orderbook(no_m.token_id)
            if yes_ob.asks and no_ob.asks:
                ya = yes_ob.asks[0].price
                na = no_ob.asks[0].price
                bundle = ya + na
                edge = round(1.0 - bundle, 4)
                best_edges.append((edge, ya, na, yes_m.title[:50]))
        except Exception:
            errors += 1

    print(f'Evaluated: {len(best_edges)}, errors: {errors}')
    positive = [(e, ya, na, t) for e, ya, na, t in best_edges if e > 0]
    print(f'Pairs with positive edge: {len(positive)}')
    print(f'Pairs with edge >= 1.8%: {sum(1 for e,_,_,_ in positive if e >= 0.018)}')
    print(f'Pairs with edge >= 0.5%: {sum(1 for e,_,_,_ in positive if e >= 0.005)}')
    print()
    print('Top edges:')
    for edge, ya, na, title in sorted(best_edges, key=lambda x: -x[0])[:15]:
        print(f'  edge={edge:+.4f} yes={ya:.3f} no={na:.3f} sum={ya+na:.3f} | {title}')

    await svc.client.close()

asyncio.run(main())
