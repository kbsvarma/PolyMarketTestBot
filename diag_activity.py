"""Check what recent public activity actually looks like."""
import asyncio
from pathlib import Path
from src.config import load_config
from src.polymarket_client import PolymarketClient


async def main():
    cfg = load_config(Path('config.yaml'))
    client = PolymarketClient(cfg)

    print('Fetching recent public activity...')
    try:
        activity = await client.fetch_recent_public_activity(limit=50)
        print(f'Got {len(activity)} activity records')
        if activity:
            print('\nSample record keys:', list(activity[0].keys()))
            print('\nFirst 10 records:')
            for rec in activity[:10]:
                print(f'  {dict(list(rec.items())[:8])}')
    except Exception as e:
        print(f'Error: {e}')

    # Also try the Gamma API market endpoint directly
    import httpx
    print('\n\nChecking Gamma API market with price data...')
    async with httpx.AsyncClient(timeout=10) as http:
        # Try to get a market with price data
        url = 'https://gamma-api.polymarket.com/markets'
        resp = await http.get(url, params={'limit': 3, 'active': 'true'})
        data = resp.json()
        markets = data if isinstance(data, list) else data.get('data', data.get('markets', []))
        if markets:
            print(f'Market fields available: {list(markets[0].keys())[:20]}')
            m = markets[0]
            # Look for price-related fields
            price_fields = {k: v for k, v in m.items() if any(x in k.lower() for x in ['price', 'mid', 'last', 'bid', 'ask', 'outcome', 'prob'])}
            print(f'Price fields: {price_fields}')

    await client.close()


asyncio.run(main())
