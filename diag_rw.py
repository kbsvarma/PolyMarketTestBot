"""Diagnose resolution_window candidates - check actual ask prices."""
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

    candidates = []
    for m in markets.values():
        if m.closed or not m.active or not m.end_date_iso:
            continue
        try:
            end = datetime.fromisoformat(m.end_date_iso.replace('Z', '+00:00'))
        except Exception:
            continue
        if end <= now:
            continue
        h = (end - now).total_seconds() / 3600.0
        if h <= 720 and m.liquidity >= 50:
            candidates.append((h, m))

    print(f'Candidates (<=720h, liq>=50): {len(candidates)}')

    qualifying = []
    no_orderbook = 0
    for h, m in sorted(candidates, key=lambda x: x[0])[:30]:
        try:
            ob = await svc.fetch_orderbook(m.token_id)
        except Exception:
            no_orderbook += 1
            continue
        if not ob.asks:
            no_orderbook += 1
            continue
        ask = ob.asks[0].price
        bid = ob.bids[0].price if ob.bids else 0
        mid = (ask + bid) / 2 if bid else ask
        fair_price = 0.93
        edge = fair_price - ask
        in_window = 0.35 <= ask <= 0.65
        status = 'QUALIFY' if (in_window and edge >= 0.02) else ('PRICE_FLOOR' if ask < 0.35 else ('CEILING' if ask > 0.65 else 'EDGE_SMALL'))
        qualifying.append((h, ask, edge, status, m.liquidity, m.title[:55]))

    print(f'\nOrders fetched: {len(qualifying)}, no_orderbook: {no_orderbook}')
    print(f'QUALIFYING (in 0.35-0.65 with edge>=2%): {sum(1 for _,_,_,s,_,_ in qualifying if s == "QUALIFY")}')
    print()
    print(f'{"Hours":>7}  {"Ask":>6}  {"Edge":>6}  {"Status":>15}  {"Liq":>9}  Title')
    for h, ask, edge, status, liq, title in sorted(qualifying):
        marker = '***' if status == 'QUALIFY' else '   '
        print(f'{marker}{h:5.0f}h  {ask:.4f}  {edge:+.4f}  {status:>15}  ${liq:8.0f}  {title}')

    await svc.client.close()


asyncio.run(main())
