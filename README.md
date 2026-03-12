# Polymarket Wallet-Following Bot

Production-style local Python project for researching, paper-trading, and eventually cautiously live-testing a Polymarket wallet-following strategy with a small bankroll.

## Overview

This bot is designed to answer one question honestly:

`Does copying selected wallets on Polymarket still work after delay, slippage, fees, and liquidity constraints?`

The system starts in `RESEARCH` mode by default, ranks wallets by copyability instead of raw PnL alone, simulates realistic execution, compares entry styles, and only allows `LIVE` trading when explicit configuration and safety gates permit it.

## Architecture

- `main.py`: startup orchestration for discovery, scoring, replay, monitoring, and engine routing
- `src/config.py`: YAML and environment loading plus runtime path bootstrap
- `src/polymarket_client.py`: isolated exchange-specific wrapper for public data and authenticated CLOB trading
- `src/wallet_discovery.py`, `src/wallet_scoring.py`, `src/category_scoring.py`: wallet research pipeline
- `src/trade_monitor.py`, `src/market_data.py`, `src/orderbook.py`: trade detection and execution-quality estimation
- `src/risk_manager.py`, `src/strategy.py`, `src/paper_engine.py`, `src/live_engine.py`: decisioning and execution
- `src/health.py`, `src/live_readiness.py`, `src/live_orders.py`, `src/state_machine.py`, `src/audit.py`: live readiness, health, lifecycle, and audit subsystems
- `backtest/`: replay and evaluation modules used for delayed-copy viability checks
- `app/dashboard.py`: Streamlit operator UI
- `tests/`: focused coverage for config, scoring, risk, dedup, exits, clustering, and fillability

## Setup

1. Create a virtual environment.
2. Install requirements.
3. Copy `.env.example` to `.env` and fill only the secrets you actually need.

## Install Commands

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment Configuration

- Secrets belong in `.env`
- Non-secret system settings belong in `config.yaml`
- `MODE` is controlled via `config.yaml`
- Live trading remains disabled unless `config.yaml`, `.env`, and persisted state all enable it
- Live mode requires `py-clob-client` plus valid Polymarket private-key/funder credentials

## Run Commands

Run research mode:

```bash
python main.py
```

Run paper mode:

1. Set `mode: PAPER` in `config.yaml`
2. Run:

```bash
python main.py
```

Run the dashboard:

```bash
streamlit run app/dashboard.py
```

Enable live mode safely:

1. Validate paper results first
2. Set `mode: LIVE` in `config.yaml`
3. Set `live.enabled: true` in `config.yaml`
4. Set `LIVE_TRADING_ENABLED=true` in `.env`
5. Set `live.manual_live_enable: true` and ensure kill switch is off
6. Verify geoblock eligibility and API credentials
7. Confirm startup validation reaches `LIVE_READY`
6. Run:

```bash
python main.py
```

## Strategy Design

### Category specialization

Wallets receive both global and category-specific scores. This helps separate a wallet that is strong in `politics` from one that is noisy outside its core area. The dashboard and downstream strategy selection can filter by category.

### Clustering

The clustering layer detects when two or more approved wallets enter the same side of the same market inside a configurable time window. This supports single-wallet, cluster-only, or hybrid signal selection.

### Entry style comparison

The system compares:

- `FOLLOW_TAKER`
- `PASSIVE_LIMIT`
- `POST_ONLY_MAKER`

It tracks fill rate, missed trades, slippage, fees, and PnL summaries in `data/strategy_comparison.csv`.

### Replay / backtest

The replay engine evaluates wallets across delay buckets, entry styles, and copyability thresholds. A wallet should not graduate to paper following if delayed-copy expectancy does not clear configured thresholds.

## Warnings

- This project is experimental.
- Paper results do not guarantee live profits.
- Slippage, latency, and depth constraints can destroy wallet-following edge.
- You are responsible for confirming legal eligibility to trade in your jurisdiction and operating environment.
- LIVE mode fails closed when readiness, health, heartbeat, or reconciliation checks fail.
- The included Polymarket endpoint wrapper is isolated, but some SDK/endpoint details may still need manual verification against current official docs before unattended live deployment.

## Runtime Outputs

The bot auto-creates and maintains:

- `data/top_wallets.json`
- `data/wallet_scorecard.csv`
- `data/category_wallet_scorecard.csv`
- `data/detected_wallet_trades.csv`
- `data/clustered_signals.csv`
- `data/strategy_comparison.csv`
- `data/paper_trade_history.csv`
- `data/live_trade_history.csv`
- `data/positions.json`
- `data/daily_summary.json`
- `data/app_state.json`
- `data/live_orders.json`
- `data/live_audit.jsonl`
- `data/live_decisions.jsonl`
- `data/health_status.json`
- `data/wallet_backtest_summary.csv`
- `logs/system.log`
- `logs/errors.log`

## Remaining TODOs For Manual Verification

- Verify the current `py-clob-client` method signatures and returned field names used in [`/Users/varmakammili/Documents/GitHub/PolyMarketTestBot/src/polymarket_client.py`](/Users/varmakammili/Documents/GitHub/PolyMarketTestBot/src/polymarket_client.py)
- Confirm current public orderbook, wallet activity, and leaderboard endpoint shapes used in [`/Users/varmakammili/Documents/GitHub/PolyMarketTestBot/src/polymarket_client.py`](/Users/varmakammili/Documents/GitHub/PolyMarketTestBot/src/polymarket_client.py)
- Replace non-LIVE synthetic research fallbacks in [`/Users/varmakammili/Documents/GitHub/PolyMarketTestBot/src/wallet_discovery.py`](/Users/varmakammili/Documents/GitHub/PolyMarketTestBot/src/wallet_discovery.py) once exact public leaderboard/activity/profile polling is fully verified
