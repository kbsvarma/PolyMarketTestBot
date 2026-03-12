from __future__ import annotations

import json
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
    max_new_entries_per_hour: int = 2
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
    order_timeout_seconds: int = 20
    max_reprice_attempts: int = 1


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
    manual_resume_required: bool = True
    global_kill_switch: bool = False
    emergency_flatten_flag: bool = False
    heartbeat_required: bool = True
    bounded_execution_seconds: int = 20
    enable_multi_entry_style_live: bool = False


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
    live_trading_enabled: bool = False
    allowed_country_codes: list[str] = Field(default_factory=list)
    country_code: str = "US"
    state_code: str = ""
    simulate_geoblock_status: str = "eligible"
    polymarket_chain_id: int = 137
    polygon_rpc_url: str = "https://polygon-bor-rpc.publicnode.com"
    polygon_usdc_address: str = "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359"
    polygon_usdce_address: str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"


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
    endpoints: EndpointConfig = Field(default_factory=EndpointConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)


RUNTIME_FILES: dict[str, str] = {
    "data/top_wallets.json": "[]",
    "data/wallet_scorecard.csv": "wallet_address,global_score,copyability_score,delayed_viability_score,dominant_category\n",
    "data/category_wallet_scorecard.csv": "wallet_address,category,score,trade_count,win_rate,copyability_score,delay_viability_score\n",
    "data/detected_wallet_trades.csv": "event_key,local_detection_timestamp,source_trade_timestamp,wallet_address,market_title,market_slug,market_id,token_id,side,price,size,notional,transaction_hash,detection_latency_seconds,best_bid,best_ask,depth_available,category\n",
    "data/clustered_signals.csv": "cluster_id,market_id,token_id,side,wallet_count,cluster_strength,first_seen,last_seen,category\n",
    "data/strategy_comparison.csv": "entry_style,signal_count,fill_rate,missed_trade_rate,avg_slippage,avg_fees,realized_pnl,unrealized_pnl,pnl_by_wallet,pnl_by_category,pnl_by_delay_bucket\n",
    "data/paper_trade_history.csv": "position_id,market_id,token_id,wallet_address,category,entry_style,entry_price,quantity,notional,fees_paid,realized_pnl,unrealized_pnl,entry_reason,exit_reason,cluster_confirmed,hedge_suspicion_score,opened_at,closed,status,reason_code\n",
    "data/live_trade_history.csv": "position_id,market_id,token_id,wallet_address,category,entry_style,entry_price,quantity,notional,fees_paid,realized_pnl,unrealized_pnl,entry_reason,exit_reason,cluster_confirmed,hedge_suspicion_score,opened_at,closed,status,reason_code\n",
    "data/positions.json": "{\"paper\": [], \"live\": []}",
    "data/daily_summary.json": "{}",
    "data/app_state.json": "{\"status\": \"INIT\", \"system_status\": \"INIT\", \"paused\": false, \"manual_resume_required\": false}",
    "data/wallet_backtest_summary.csv": "wallet_address,delay_bucket,entry_style,expectancy,fill_rate,net_pnl,copyable\n",
    "data/live_orders.json": "[]",
    "data/live_audit.jsonl": "",
    "data/live_decisions.jsonl": "",
    "data/paper_audit.jsonl": "",
    "data/health_status.json": "{}",
    "logs/system.log": "",
    "logs/errors.log": "",
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
        live_trading_enabled=os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true",
        allowed_country_codes=[code.strip() for code in allowed if code.strip()],
        country_code=os.getenv("COUNTRY_CODE", "US"),
        state_code=os.getenv("STATE_CODE", ""),
        simulate_geoblock_status=os.getenv("SIMULATE_GEOBLOCK_STATUS", "eligible"),
        polymarket_chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
        polygon_rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com"),
        polygon_usdc_address=os.getenv("POLYGON_USDC_ADDRESS", "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359"),
        polygon_usdce_address=os.getenv("POLYGON_USDCE_ADDRESS", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
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
