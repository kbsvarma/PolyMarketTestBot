from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_: object, **__: object) -> bool:
        return False
from pydantic import BaseModel, Field

from src.models import EntryStyle, Mode


class RuntimeConfig(BaseModel):
    ui_refresh_seconds: int = 15
    discovery_interval_minutes: int = 45
    polling_interval_seconds: int = 20
    summary_interval_minutes: int = 60
    wallet_evaluation_window_days: int = 30
    max_runtime_cycles: int = 0


class BankrollConfig(BaseModel):
    paper_starting_bankroll: float = 200.0
    live_bankroll_reference: float = 200.0


class RiskConfig(BaseModel):
    max_total_exposure_pct: float = 0.15
    max_market_exposure_pct: float = 0.04
    max_wallet_exposure_pct: float = 0.06
    copy_fraction_min: float = 0.03
    copy_fraction_max: float = 0.05
    max_single_live_trade_usd: float = 6.0
    daily_stop_loss_pct: float = 0.05
    max_new_entries_per_hour: int = 4
    stale_signal_seconds: int = 90
    stale_market_data_seconds: int = 15
    max_spread_pct: float = 0.025
    max_entry_drift_pct: float = 0.03
    max_slippage_pct: float = 0.03
    min_orderbook_depth_usd: float = 50.0
    max_hedge_suspicion_score: float = 0.72
    require_cluster_confirmation_live: bool = True
    allow_categories: list[str] = Field(default_factory=list)
    heartbeat_timeout_seconds: int = 30
    order_timeout_seconds: int = 60
    max_reprice_attempts: int = 2


class ClusterConfig(BaseModel):
    enabled: bool = True
    confirmation_window_seconds: int = 180
    min_wallets: int = 2
    strong_cluster_wallets: int = 3
    mode: str = "HYBRID"


class EntryStyleConfig(BaseModel):
    compare: list[EntryStyle] = Field(default_factory=lambda: list(EntryStyle))
    preferred_live_entry_style: EntryStyle = EntryStyle.PASSIVE_LIMIT


class BacktestConfig(BaseModel):
    delay_buckets_seconds: list[int] = Field(default_factory=lambda: [5, 15, 30, 60])
    min_wallet_replay_expectancy: float = 0.01
    min_wallet_copyability_score: float = 0.55


class WalletSelectionConfig(BaseModel):
    top_research_wallets: int = 5
    approved_paper_wallets: int = 3
    approved_live_wallets: int = 1
    min_trade_count: int = 12
    min_copyability_score: float = 0.55
    min_delay_viability_score: float = 0.5


class CategoriesConfig(BaseModel):
    tracked: list[str] = Field(default_factory=list)


class AlertsConfig(BaseModel):
    enable_console_alerts: bool = True
    health_pause_on_live_mismatch: bool = True


class LiveConfig(BaseModel):
    enabled: bool = False
    require_paper_validation: bool = True
    only_cluster_confirmed: bool = True
    only_low_hedge_suspicion: bool = True
    selected_categories: list[str] = Field(default_factory=list)
    manual_live_enable: bool = False
    manual_resume_required: bool = False
    global_kill_switch: bool = False
    emergency_flatten_flag: bool = False
    heartbeat_required: bool = True
    bounded_execution_seconds: int = 20
    enable_multi_entry_style_live: bool = False
    minimum_order_size_shares: float = 5.0
    minimum_order_notional_usd: float = 1.0
    hard_skip_price_floor: float = 0.03
    hard_skip_price_ceiling: float = 0.95
    preferred_entry_price_min: float = 0.15
    preferred_entry_price_max: float = 0.80
    minimum_fillability_score: float = 0.58
    minimum_signal_quality_score: float = 0.52
    passive_signal_ttl_seconds: int = 600
    adaptive_profit_arm_pct: float = 0.08
    adaptive_profit_min_lock_pct: float = 0.03
    trailing_profit_retrace_pct: float = 0.35
    strong_winner_profit_pct: float = 0.35
    strong_winner_retrace_pct: float = 0.20
    paired_arb_time_stop_hours: int = 168
    # Binary prediction market exit rules — wide by design since price moves
    # are noise before resolution; premature exits were the #1 PnL destroyer.
    binary_stop_loss_pct: float = 0.45   # only exit on a genuine directional collapse
    binary_time_stop_hours: int = 168    # 7 days; most markets resolve within that window


class StrategyConfig(BaseModel):
    enable_event_driven_official: bool = True
    enable_correlation_dislocation: bool = True
    enable_resolution_window: bool = True
    enable_paired_binary_arb: bool = True
    strategy_market_page_size: int = 250
    strategy_market_max_pages_live: int = 4
    strategy_market_max_pages_research: int = 2
    official_signal_file: str = "data/official_event_signals.json"
    official_signal_max_age_minutes: int = 240
    official_signal_min_edge_pct: float = 0.08
    official_signal_min_source_reliability: float = 0.7
    official_signal_min_confidence: float = 0.68
    official_signal_live_enabled: bool = False
    correlation_min_gap_pct: float = 0.03
    correlation_min_group_size: int = 2
    correlation_max_groups: int = 16
    correlation_near_miss_gap_ratio: float = 0.8
    correlation_live_enabled: bool = False
    supplemental_paper_relaxed_enabled: bool = True
    supplemental_paper_relaxed_min_confidence: float = 0.72
    paired_binary_min_edge_pct: float = 0.035
    paired_binary_min_net_edge_pct: float = 0.012
    paired_binary_min_leg_price: float = 0.06
    paired_binary_max_leg_price: float = 0.94
    paired_binary_slippage_buffer_pct: float = 0.003
    paired_binary_max_candidates_per_cycle: int = 2
    paired_binary_max_best_level_fraction: float = 0.15
    paired_binary_live_enabled: bool = False
    resolution_window_max_hours: int = 168
    resolution_window_min_price: float = 0.68
    resolution_window_min_edge_pct: float = 0.04
    resolution_window_near_miss_edge_ratio: float = 0.5
    resolution_window_near_miss_price_buffer: float = 0.05
    resolution_window_target_fair_price: float = 0.94
    resolution_window_min_liquidity: float = 100.0
    resolution_window_live_enabled: bool = False


class RTDSConfig(BaseModel):
    """Real-Time Data Socket (RTDS) configuration."""
    enabled: bool = True
    url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/price"
    ping_interval_seconds: float = 5.0
    staleness_max_seconds: float = 1.5      # hard skip if RTDS older than this
    clob_staleness_max_seconds: float = 0.8 # hard skip if CLOB book older than this


class FeeConfig(BaseModel):
    """Fee model configuration."""
    # Minimum net edge after taker fees to allow a signal through
    min_edge_after_fees_taker: float = 0.020   # 2.0 ¢
    # Minimum net edge after maker rebate (net cost ~0) for maker fills
    min_edge_after_fees_maker: float = 0.010   # 1.0 ¢
    # Enable fee gate in risk_manager for all strategies (not just paired arb)
    enable_global_fee_gate: bool = True


class LagSignalConfig(BaseModel):
    """Oracle-aligned lag signal configuration."""
    enabled: bool = True
    live_enabled: bool = False              # gate for live trading
    # Minimum Binance-Chainlink price divergence (fraction) to consider a lag
    min_price_divergence_pct: float = 0.003
    # Minimum Binance move (fraction) over short window
    min_spot_move_pct: float = 0.003
    # Minimum Binance-Chainlink timestamp lag (ms)
    min_lag_ms: float = 50.0
    # Minimum net edge after fees to fire a signal
    min_net_edge_taker: float = 0.020
    min_net_edge_maker: float = 0.010
    # Don't lag-arb if < N seconds remain in the window
    min_time_remaining_seconds: float = 30.0
    # Max candidates to emit per cycle
    max_candidates_per_cycle: int = 2


class CompletionTrackerConfig(BaseModel):
    """Bayesian completion probability tracker configuration."""
    enabled: bool = True
    # Only enter leg-1 if P(complete) > this
    entry_threshold: float = 0.70
    # Abort (flatten) if posterior P(complete) drops below this
    abort_threshold: float = 0.40
    # Switch to management mode after this fraction of time-to-expiry has elapsed
    deadline_fraction: float = 0.60
    # Tick the tracker every N seconds during live monitoring
    tick_interval_seconds: float = 5.0


class EndpointConfig(BaseModel):
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    data_base_url: str = "https://data-api.polymarket.com"
    user_agent: str = "polymarket-wallet-follow-bot/0.2"


class EnvConfig(BaseModel):
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""
    polymarket_private_key: str = ""
    polymarket_funder: str = ""
    polymarket_signature_type: int | None = None
    kalshi_api_key_id: str = ""
    kalshi_private_key: str = ""
    kalshi_private_key_path: str = ""
    kalshi_api_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    live_trading_enabled: bool = False
    allowed_country_codes: list[str] = Field(default_factory=list)
    country_code: str = "US"
    state_code: str = ""
    simulate_geoblock_status: str = "eligible"
    polymarket_chain_id: int = 137
    polygon_rpc_url: str = "https://polygon-bor-rpc.publicnode.com"
    polygon_usdc_address: str = "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359"
    polygon_usdce_address: str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    operator_live_session_max_usd: float | None = None
    operator_live_max_trade_usd: float | None = None
    operator_live_max_positions: int | None = None
    operator_live_wallet_count: int | None = None


class AssetVolConfig(BaseModel):
    """
    Per-asset parameters for the bracket direction signal.

    Each tracked asset (BTC, ETH) has its own price feed URL, Polymarket
    slug prefix, and volatility estimate. Volatility drives the GBM model
    used to compute the 'implied fair value' of the momentum side — the
    larger the lag between implied price and actual Polymarket price, the
    stronger the entry signal.

    Volatility calibration notes:
      - BTC  ~69% annualized → 0.37% per 15m window std dev
      - ETH ~103% annualized → 0.55% per 15m window std dev
      These are starting estimates. Calibrate from observed data over time.

    The vol_per_second used by _estimate_fair_p_up is derived as:
      annual_vol_pct / sqrt(365 * 24 * 3600)
    """
    # Binance aggTrade WebSocket stream — RTDSClient connects here for live price
    binance_ws_url: str = "wss://stream.binance.com/ws/btcusdt@aggTrade"

    # Polymarket 15m market slug prefix.
    # IMPORTANT: Verify the actual format on first run — the code will log the
    # real market title so you can confirm the slug is correct.
    # Common formats: "btc-15m-updown", "btc-15-minute-up-or-down", etc.
    slug_prefix: str = "btc-15m-updown"

    # Annualized volatility fraction (e.g. 0.69 = 69%).
    # Used to compute vol_per_second for the GBM fair-probability model.
    annual_vol_pct: float = 0.69

    # Minimum asset price move (fraction) from window open before we consider
    # the market "directional". Below this, the market is still too near 50/50.
    min_asset_move_pct: float = 0.002   # 0.2% for BTC


class CryptoDirectionConfig(BaseModel):
    """
    Configuration for the bracket strategy direction signal observer.

    This module watches live BTC/ETH Polymarket markets and fires signals
    when all quality gates pass.

    It supports three practical modes using the same signal logic:
      1. Observe-only      → no order state machine
      2. Shadow execution  → same bracket executor, simulated fills from live books
      3. Live execution    → same bracket executor, real wallet + exchange orders

    Gate summary (all must pass for a signal to fire):
      1. Time gate:      >= time_gate_minutes remaining in window
      2. Asset move:     asset moved >= min_asset_move_pct from window open
      3. Price range:    momentum_side ask in [entry_range_low, entry_range_high]
      4. Price sanity:   both YES and NO ask prices are non-zero
      5. Chop filter:    :00-second checkpoints show a clean directional move
      6. Lag gap:        GBM implied price exceeds actual price by >= lag_threshold
      7. Cooldown:       same side hasn't already fired a signal this window
    """
    enabled: bool = False   # must be explicitly enabled in config.yaml
    venue: str = "polymarket"
    instance_name: str = "polymarket_btc_5m"

    # ---- Asset configs ----
    btc: AssetVolConfig = Field(default_factory=lambda: AssetVolConfig(
        binance_ws_url="wss://stream.binance.com/ws/btcusdt@aggTrade",
        slug_prefix="btc-15m-updown",   # VERIFY on first run
        annual_vol_pct=0.69,
        min_asset_move_pct=0.002,
    ))
    eth: AssetVolConfig = Field(default_factory=lambda: AssetVolConfig(
        binance_ws_url="wss://stream.binance.com/ws/ethusdt@aggTrade",
        slug_prefix="eth-15m-updown",   # VERIFY on first run
        annual_vol_pct=1.03,
        min_asset_move_pct=0.003,       # ETH is noisier, require slightly more move
    ))

    # Which assets to monitor
    track_btc: bool = True
    track_eth: bool = True

    # Asset selection strategy when both assets are available:
    #   True  → pick the asset with higher Polymarket liquidity each window
    #   False → evaluate and emit signals for both independently
    prefer_higher_liquidity: bool = True

    # ---- Window duration ----
    # Seconds per Polymarket window. 900 = 15-min, 300 = 5-min.
    # Change this (and btc.slug_prefix) to switch between window sizes.
    window_duration_seconds: int = 900

    # ---- Gate 0: Window settle gate ----
    # Seconds to wait after window open before evaluating signals.
    # The AMM seeds liquidity in the first ~60s; prices are noisy until
    # real market makers enter and the BTC direction becomes clearer.
    window_settle_seconds: float = 60.0

    # ---- Gate 1: Time gate ----
    # Minimum minutes remaining in the window to allow Phase 1 entry.
    # Below this, there's not enough time for Phase 2 to develop.
    time_gate_minutes: float = 9.0

    # ---- Gate 3: Momentum side price range ----
    # Signal only fires when the momentum-side ask is inside the approved band.
    # Current safer 5m approach loosens the low side slightly while keeping a
    # tighter ceiling, then treats the top of the band as a stricter "stretch"
    # lane rather than a normal entry.
    entry_range_low: float = 0.55
    entry_range_high: float = 0.61
    entry_core_range_high: float = 0.60
    stretch_entry_range_high: float = 0.61
    stretch_min_asset_move_pct: float = 0.0
    stretch_min_chop_score: float = 0.0
    stretch_min_lag_gap: float = 0.0
    stretch_min_consecutive_polls: int = 1

    # ---- Gate 5: Chop filter ----
    # How many recent :00-second checkpoints to evaluate for directional cleanliness.
    chop_window: int = 4
    # Maximum allowed counter-move between consecutive checkpoints (as a price fraction).
    # A reversal larger than this counts as "chop" and penalises the chop score.
    chop_max_reversal_pct: float = 0.0005   # 0.05%
    # Minimum chop score to pass the filter (0.0 = all reversals, 1.0 = perfectly clean)
    chop_min_score: float = 0.6

    # ---- Gate 6: Lag gap ----
    # Minimum difference between GBM-implied fair value and actual Polymarket price.
    # A larger lag_gap means the market hasn't priced in the BTC move yet — better entry.
    lag_threshold: float = 0.04   # 4¢

    # ---- Controlled continuation override ----
    # Optional second entry family for clean, early momentum windows where the
    # market is no longer lagging but the bracket structure is still attractive.
    continuation_enabled: bool = False
    continuation_min_minutes_remaining: float = 1.5
    continuation_min_asset_move_pct: float = 0.00035
    continuation_min_chop_score: float = 0.66
    continuation_max_momentum_price: float = 0.61
    continuation_max_opposite_price: float = 0.44
    continuation_max_negative_lag_gap: float = 0.03
    continuation_ignore_lag_veto: bool = False
    # Third entry lane: as soon as the momentum side is cleanly trading inside
    # the original 57-61c thesis band, buy it without waiting for lag.
    immediate_band_entry_enabled: bool = False
    immediate_band_entry_low: float = 0.57
    immediate_band_entry_high: float = 0.61
    continuation_hard_exit_grace_seconds: float = 0.0
    continuation_catastrophic_stop_price: float = 0.0
    safe_arm_suspend_stop: bool = False
    safe_arm_catastrophic_stop_price: float = 0.0

    # ---- Post-signal bracket tracking ----
    # Target price for the opposite-side (second leg of the bracket).
    # If x=0.58 is paid for leg 1, we want x+y+fees < $1.00.
    # With fees ~1.5-2% total, y_target ~0.34 gives ~6% net bracket margin.
    target_y_price: float = 0.34
    # Smallest locked profit/share we still consider worth arming for Phase 2.
    # 0.00 means "strictly profitable after fees"; raise later if we want a
    # larger safety cushion before committing the opposite leg.
    phase2_min_locked_profit_per_share: float = 0.00
    # How much the opposite side must recover from its floor to confirm reversal.
    # E.g., if NO bottoms at 0.30 and recovers to 0.31, that's a 1¢ reversal confirmation.
    phase2_reversal_threshold: float = 0.01   # 1¢

    # ---- Runtime ----
    # How often to run the signal evaluation loop (seconds)
    poll_interval_seconds: float = 0.5
    # How often to refresh YES/NO orderbook prices (seconds)
    # Can be slightly less frequent than the eval loop since prices change slowly
    price_refresh_interval_seconds: float = 0.5
    # If market resolution fails at a window rollover, keep retrying within the
    # same window instead of staying unresolved until the next rollover.
    market_resolve_retry_seconds: float = 5.0
    # Maximum age of YES/NO prices before they're considered stale (seconds)
    price_staleness_max_seconds: float = 10.0

    # ---- Logging ----
    # Full signal events (one JSON line per signal fire + one per outcome)
    signal_event_log_path: str = "logs/crypto_signal_events.jsonl"
    # Compact evaluation log (one JSON line per eval cycle per asset)
    evaluation_log_path: str = "logs/crypto_signal_evaluations.jsonl"
    # Human-readable per-window Markdown report (updated every 15 min)
    window_report_path: str = "logs/window_report.md"
    # Fixed share count used by the observer report for hypothetical PnL.
    # A value of 10.0 means "pretend each window used 10 shares", not "$10 notional".
    report_shares: float = 10.0

    # ---- Shadow-live execution ----
    # When true, --observe-crypto still runs without touching the wallet, but it
    # uses the same BracketExecutor state machine as live mode. Orders are
    # simulated against fresh live orderbooks at submission time, so the report
    # is much closer to what --execute-crypto would have done.
    shadow_execute_enabled: bool = True
    shadow_execution_log_path: str = "logs/bracket_trades.jsonl"

    # ---- Immediate fill confirmation ----
    # FOLLOW_TAKER/FOK orders should not be credited as filled just because the
    # submission succeeded. Confirm the fill via placement payload or order
    # status polling before marking a leg complete.
    fill_confirmation_attempts: int = 2
    fill_confirmation_delay_seconds: float = 0.05

    # ---- Live execution (bracket_executor.py) ----
    # SAFETY: execute_enabled MUST be set true in config AND LIVE_TRADING_ENABLED
    # must be set in .env before any real orders are placed.
    # To go live: set execute_enabled: true in config.yaml and run:
    #   python main.py --execute-crypto
    execute_enabled: bool = False               # master switch — must be explicit
    phase2_enabled: bool = True                 # both phases run together by default
    max_concurrent_brackets: int = 1            # max open positions across all assets
    # Adaptive Phase 1 sizing. `phase1_shares` is the target size on liquid
    # windows; if visible depth inside our cap is thinner, we can still trade
    # the exact marketable size as long as it stays above min_bracket_shares.
    # Phase 2 mirrors the actual Phase 1 filled shares so the bracket remains
    # symmetric.
    phase1_shares: float = 10.0
    min_bracket_shares: float = 5.0             # minimum executable shares for Phase 1
    # FOLLOW_TAKER (FOK) for both phases — immediate fill, locks bracket in real-time
    # after the safe-y reclaim condition has been satisfied.
    phase1_entry_style: str = "FOLLOW_TAKER"
    # Phase 1 catch-up window for FOLLOW_TAKER. This is intentionally a bit
    # aggressive in 5m mode so we catch more real windows instead of missing
    # them on a one-tick lift.
    phase1_max_chase_cents: float = 0.01
    # Give live/shadow FOK entry a couple of bounded retries when the first
    # immediate cross misses because the book shifted between signal snapshot
    # and submission.
    phase1_follow_taker_retry_attempts: int = 1
    phase1_follow_taker_retry_delay_seconds: float = 0.10
    phase1_follow_taker_retry_to_strategy_cap: bool = False
    # Skip the pre-submission orderbook fetch on attempt 0 to reduce latency.
    # The FOK exchange rejection is the protection; saves ~100-300ms on hot path.
    phase1_skip_depth_precheck: bool = False
    # Lag-conditioned retry ceiling multiplier.  On retry, the ceiling is
    # signal_price + lag_gap * multiplier (capped at entry_range_high).
    # 0.0 = disabled (same price as attempt 0).
    phase1_lag_retry_multiplier: float = 0.0
    # Phase 2 FOK entry buffer: lift the initial attempt price this many cents
    # above the reclaim detection price to absorb submission latency.  The
    # reclaim is detected at y_price but the ask keeps moving; without a buffer
    # the FOK arrives at the exchange marginally below the ask and is killed.
    # Capped at profitable_y_ceiling so the bracket margin is never blown.
    phase2_fok_entry_buffer: float = 0.01
    # Phase 2 FOK retry: on FOK rejection, retry once at y_price + slippage
    # capped at profitable_y_ceiling.  0.0 = disabled.
    phase2_fok_retry_slippage: float = 0.02
    phase2_entry_style: str = "FOLLOW_TAKER"
    bracket_audit_log_path: str = "logs/bracket_trades.jsonl"
    # Hard exit: sell Phase-1 leg mid-window to cap losses.
    # Triggers if mark drops to stop_price OR we're in the final N seconds losing.
    hard_exit_stop_price: float = 0.50
    hard_exit_high_entry_price: float = 0.64
    hard_exit_high_entry_stop_price: float = 0.54
    hard_exit_trigger_buffer_cents: float = 0.02
    hard_exit_market_through_cents: float = 0.0
    hard_exit_fallback_step_cents: float = 0.02
    hard_exit_min_sell_price: float = 0.40
    hard_exit_emergency_min_sell_price: float = 0.40
    hard_exit_retry_cooldown_seconds: float = 2.0
    hard_exit_dust_shares: float = 0.25
    hard_exit_final_seconds: int = 30


class AppConfig(BaseModel):
    mode: Mode = Mode.RESEARCH
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    bankroll: BankrollConfig = Field(default_factory=BankrollConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    entry_styles: EntryStyleConfig = Field(default_factory=EntryStyleConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    wallet_selection: WalletSelectionConfig = Field(default_factory=WalletSelectionConfig)
    categories: CategoriesConfig = Field(default_factory=CategoriesConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)
    strategies: StrategyConfig = Field(default_factory=StrategyConfig)
    endpoints: EndpointConfig = Field(default_factory=EndpointConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    # New modules
    rtds: RTDSConfig = Field(default_factory=RTDSConfig)
    fees: FeeConfig = Field(default_factory=FeeConfig)
    lag_signal: LagSignalConfig = Field(default_factory=LagSignalConfig)
    completion_tracker: CompletionTrackerConfig = Field(default_factory=CompletionTrackerConfig)
    # Bracket strategy direction signal observer (crypto 15m markets)
    crypto_direction: CryptoDirectionConfig = Field(default_factory=CryptoDirectionConfig)
    # Optional named crypto profiles that override the base crypto_direction
    # block. This lets us keep 5m stable while adding a clean 15m lane.
    crypto_direction_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)


RUNTIME_FILES: dict[str, str] = {
    "data/top_wallets.json": "[]",
    "data/wallet_scorecard.csv": "wallet_address,global_score,copyability_score,delayed_viability_score,dominant_category\n",
    "data/category_wallet_scorecard.csv": "wallet_address,category,score,trade_count,win_rate,copyability_score,delay_viability_score\n",
    "data/detected_wallet_trades.csv": "event_key,local_detection_timestamp,source_trade_timestamp,wallet_address,market_title,market_slug,market_id,token_id,side,price,size,notional,transaction_hash,detection_latency_seconds,best_bid,best_ask,depth_available,category\n",
    "data/clustered_signals.csv": "cluster_id,market_id,token_id,side,wallet_count,cluster_strength,first_seen,last_seen,category\n",
    "data/strategy_comparison.csv": "strategy_name,entry_style,signal_count,fill_rate,missed_trade_rate,avg_slippage,avg_fees,realized_pnl,unrealized_pnl,pnl_by_wallet,pnl_by_category,pnl_by_delay_bucket\n",
    "data/paper_trade_history.csv": "position_id,market_id,token_id,strategy_name,wallet_address,category,entry_style,entry_price,quantity,notional,fees_paid,realized_pnl,unrealized_pnl,entry_reason,exit_reason,cluster_confirmed,hedge_suspicion_score,opened_at,closed,status,reason_code\n",
    "data/live_trade_history.csv": "position_id,market_id,token_id,wallet_address,category,entry_style,entry_price,quantity,notional,fees_paid,realized_pnl,unrealized_pnl,entry_reason,exit_reason,cluster_confirmed,hedge_suspicion_score,opened_at,closed,status,reason_code\n",
    "data/positions.json": "{\"paper\": [], \"live\": []}",
    "data/daily_summary.json": "{}",
    "data/app_state.json": "{\"status\": \"INIT\", \"system_status\": \"INIT\", \"paused\": false, \"manual_resume_required\": false}",
    "data/wallet_backtest_summary.csv": "wallet_address,delay_bucket,entry_style,expectancy,fill_rate,net_pnl,copyable\n",
    "data/live_orders.json": "[]",
    "data/live_audit.jsonl": "",
    "data/live_decisions.jsonl": "",
    "data/shadow_live_decisions.jsonl": "",
    "data/shadow_live_summary.json": "{}",
    "data/paper_audit.jsonl": "",
    "data/paper_decision_trace.jsonl": "",
    "data/wallet_discovery_diagnostics.json": "{}",
    "data/wallet_scoring_diagnostics.json": "{}",
    "data/paper_quality_summary.json": "{}",
    "data/source_quality_summary.json": "{}",
    "data/health_status.json": "{}",
    "data/official_event_signals.json": "[]",
    "data/strategy_signal_log.jsonl": "",
    "data/shadow/positions.json": "{\"paper\": [], \"live\": []}",
    "data/shadow/app_state.json": "{\"status\": \"INIT\", \"system_status\": \"INIT\", \"paused\": false, \"manual_resume_required\": false}",
    "data/shadow/paper_trade_history.csv": "position_id,market_id,token_id,strategy_name,wallet_address,category,entry_style,entry_price,quantity,notional,fees_paid,realized_pnl,unrealized_pnl,entry_reason,exit_reason,cluster_confirmed,hedge_suspicion_score,opened_at,closed,status,reason_code\n",
    "data/shadow/paper_audit.jsonl": "",
    "data/shadow/paper_quality_summary.json": "{}",
    "data/shadow/daily_summary.json": "{}",
    "data/shadow/source_quality_summary.json": "{}",
    "data/shadow/strategy_comparison.csv": "strategy_name,entry_style,signal_count,fill_rate,missed_trade_rate,avg_slippage,avg_fees,realized_pnl,unrealized_pnl,pnl_by_wallet,pnl_by_category,pnl_by_delay_bucket\n",
    "data/shadow/strategy_signal_log.jsonl": "",
    "logs/system.log": "",
    "logs/errors.log": "",
    # Bracket strategy direction signal observer logs
    "logs/crypto_signal_events.jsonl": "",
    "logs/crypto_signal_evaluations.jsonl": "",
    "logs/crypto_signal_events_15m.jsonl": "",
    "logs/crypto_signal_evaluations_15m.jsonl": "",
    "logs/bracket_trades_15m.jsonl": "",
    "logs/window_report_15m.md": "",
    "logs/crypto_signal_events_pm_btc_5m.jsonl": "",
    "logs/crypto_signal_evaluations_pm_btc_5m.jsonl": "",
    "logs/window_report_pm_btc_5m.md": "",
    "logs/bracket_trades_pm_btc_5m.jsonl": "",
    "logs/crypto_signal_events_pm_btc_15m.jsonl": "",
    "logs/crypto_signal_evaluations_pm_btc_15m.jsonl": "",
    "logs/window_report_pm_btc_15m.md": "",
    "logs/bracket_trades_pm_btc_15m.jsonl": "",
    "logs/crypto_signal_events_kalshi_btc_5m.jsonl": "",
    "logs/crypto_signal_evaluations_kalshi_btc_5m.jsonl": "",
    "logs/window_report_kalshi_btc_5m.md": "",
    "logs/bracket_trades_kalshi_btc_5m.jsonl": "",
    "logs/crypto_signal_events_kalshi_btc_15m.jsonl": "",
    "logs/crypto_signal_evaluations_kalshi_btc_15m.jsonl": "",
    "logs/window_report_kalshi_btc_15m.md": "",
    "logs/bracket_trades_kalshi_btc_15m.jsonl": "",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_env() -> EnvConfig:
    import os

    allowed = os.getenv("ALLOWED_COUNTRY_CODES", "US").split(",")
    return EnvConfig(
        polymarket_api_key=os.getenv("POLYMARKET_API_KEY", ""),
        polymarket_api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
        polymarket_api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
        polymarket_private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
        polymarket_funder=os.getenv("POLYMARKET_FUNDER", ""),
        polymarket_signature_type=(
            int(os.getenv("POLYMARKET_SIGNATURE_TYPE", ""))
            if os.getenv("POLYMARKET_SIGNATURE_TYPE", "").strip()
            else None
        ),
        kalshi_api_key_id=os.getenv("KALSHI_API_KEY_ID", ""),
        kalshi_private_key=os.getenv("KALSHI_PRIVATE_KEY", ""),
        kalshi_private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
        kalshi_api_base_url=os.getenv("KALSHI_API_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"),
        live_trading_enabled=os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true",
        allowed_country_codes=[code.strip() for code in allowed if code.strip()],
        country_code=os.getenv("COUNTRY_CODE", "US"),
        state_code=os.getenv("STATE_CODE", ""),
        simulate_geoblock_status=os.getenv("SIMULATE_GEOBLOCK_STATUS", "eligible"),
        polymarket_chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
        polygon_rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com"),
        polygon_usdc_address=os.getenv("POLYGON_USDC_ADDRESS", "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359"),
        polygon_usdce_address=os.getenv("POLYGON_USDCE_ADDRESS", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
        operator_live_session_max_usd=(
            float(os.getenv("POLYBOT_LIVE_SESSION_MAX_USD", ""))
            if os.getenv("POLYBOT_LIVE_SESSION_MAX_USD", "").strip()
            else None
        ),
        operator_live_max_trade_usd=(
            float(os.getenv("POLYBOT_LIVE_MAX_TRADE_USD", ""))
            if os.getenv("POLYBOT_LIVE_MAX_TRADE_USD", "").strip()
            else None
        ),
        operator_live_max_positions=(
            int(os.getenv("POLYBOT_LIVE_MAX_POSITIONS", ""))
            if os.getenv("POLYBOT_LIVE_MAX_POSITIONS", "").strip()
            else None
        ),
        operator_live_wallet_count=(
            int(os.getenv("POLYBOT_LIVE_WALLET_COUNT", ""))
            if os.getenv("POLYBOT_LIVE_WALLET_COUNT", "").strip()
            else None
        ),
    )


def load_config(config_path: Path, env_path: Path | None = None) -> AppConfig:
    if env_path and env_path.exists():
        load_dotenv(env_path)
    elif Path(".env").exists():
        load_dotenv()
    data = _load_yaml(config_path)
    config = AppConfig.model_validate(data)
    config.env = _load_env()
    return config


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def apply_crypto_direction_profile(config: AppConfig, profile_name: str) -> AppConfig:
    """
    Return a copy of AppConfig with crypto_direction overridden by a named profile.
    """
    profiles = config.crypto_direction_profiles or {}
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise KeyError(
            f"Unknown crypto profile '{profile_name}'. Available profiles: {available}"
        )

    merged = _deep_merge_dicts(
        config.crypto_direction.model_dump(),
        profiles[profile_name],
    )
    profiled_crypto = CryptoDirectionConfig.model_validate(merged)
    return config.model_copy(update={"crypto_direction": profiled_crypto})


def ensure_runtime_files(root: Path, config: AppConfig) -> None:
    for relative_path, default_content in RUNTIME_FILES.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(default_content, encoding="utf-8")
    app_state = root / "data" / "app_state.json"
    if app_state.exists():
        current = json.loads(app_state.read_text(encoding="utf-8") or "{}")
        current.setdefault("mode", config.mode.value)
        current.setdefault("system_status", "INIT")
        current.setdefault("kill_switch", config.live.global_kill_switch)
        current.setdefault("manual_live_enable", config.live.manual_live_enable)
        current.setdefault("manual_resume_required", False)
        app_state.write_text(json.dumps(current, indent=2), encoding="utf-8")
