"""Check bid/ask structure to understand true market prices."""
import asyncio
from pathlib import Path
from src.config import load_config
from src.market_data import MarketDataService
from datetime import datetime, timezone


async def main():
    cfg = load_config(Path('config.yaml'))
    svc = MarketDataService(cfg, Path('data'))
    markets = await svc.refresh_markets()
    now = datetime.now(timezone.utc)

    # Get the resolution_window candidates
    candidates = []
    for m in markets.values():
        if m.closed or not m.active or not m.end_date_iso:
            continue
        try:
            end = datetime.fromisoformat(m.end_date_iso.replace('Z', '+00:00'))
        except Exception:
            continue
        h = (end - now).total_seconds() / 3600.0
        if 0 < h <= 720 and m.liquidity >= 200:
            candidates.append((h, m))

    print(f'Candidates (<=720h, liq>=200): {len(candidates)}')

    print(f'\n{"Hours":>7}  {"Bid":>6}  {"Ask":>6}  {"Mid":>6}  {"Spread":>7}  {"Liq":>9}  Title')
    count = 0
    for h, m in sorted(candidates, key=lambda x: x[0])[:25]:
        try:
            ob = await svc.fetch_orderbook(m.token_id)
        except Exception:
            continue
        bid = ob.bids[0].price if ob.bids else 0
        ask = ob.asks[0].price if ob.asks else 0
        mid = (bid + ask) / 2 if (bid and ask) else (bid or ask)
        spread = ask - bid if (bid and ask) else 0
        in_range = 0.20 <= mid <= 0.65
        marker = '***' if in_range else '   '
        print(f'{marker}{h:5.0f}h  {bid:.4f}  {ask:.4f}  {mid:.4f}  {spread:.4f}  ${m.liquidity:8.0f}  {m.title[:55]}')
        count += 1

    await svc.client.close()


asyncio.run(main())
