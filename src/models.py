from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Mode(str, Enum):
    RESEARCH = "RESEARCH"
    PAPER = "PAPER"
    LIVE = "LIVE"


class SystemStatus(str, Enum):
    INIT = "INIT"
    RESEARCH = "RESEARCH"
    PAPER = "PAPER"
    LIVE_READY = "LIVE_READY"
    LIVE_ACTIVE = "LIVE_ACTIVE"
    RECONCILING = "RECONCILING"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"


class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class DiscoveryState(str, Enum):
    SUCCESS = "SUCCESS"
    NO_DATA = "NO_DATA"
    FETCH_FAILED = "FETCH_FAILED"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    FILTERED_TO_ZERO = "FILTERED_TO_ZERO"
    SYNTHETIC_FALLBACK_USED = "SYNTHETIC_FALLBACK_USED"


class SourceQuality(str, Enum):
    REAL_PUBLIC_DATA = "REAL_PUBLIC_DATA"
    DEGRADED_PUBLIC_DATA = "DEGRADED_PUBLIC_DATA"
    SYNTHETIC_FALLBACK = "SYNTHETIC_FALLBACK"


class TrustLevel(str, Enum):
    TRUSTWORTHY = "TRUSTWORTHY"
    DEGRADED = "DEGRADED"
    NOT_TRUSTWORTHY = "NOT_TRUSTWORTHY"


class PaperReadiness(str, Enum):
    STRONG = "STRONG"
    DEGRADED = "DEGRADED"
    NOT_TRUSTWORTHY = "NOT_TRUSTWORTHY"


class ValidationMode(str, Enum):
    VALIDATION_GRADE = "VALIDATION_GRADE"
    DEGRADED_VALIDATION = "DEGRADED_VALIDATION"
    DEV_ONLY = "DEV_ONLY"


class EntryStyle(str, Enum):
    FOLLOW_TAKER = "FOLLOW_TAKER"
    PASSIVE_LIMIT = "PASSIVE_LIMIT"
    POST_ONLY_MAKER = "POST_ONLY_MAKER"


class DecisionAction(str, Enum):
    SKIP = "SKIP"
    PAPER_COPY = "PAPER_COPY"
    LIVE_COPY = "LIVE_COPY"


class OrderLifecycleStatus(str, Enum):
    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESTING = "RESTING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"
    RECONCILING = "RECONCILING"


class WalletMetrics(BaseModel):
    wallet_address: str
    evaluation_window_days: int
    trade_count: int
    trades_per_day: float
    buy_count: int
    sell_count: int
    estimated_pnl_percent: float
    win_rate: float
    average_trade_size: float
    conviction_score: float
    market_concentration: float
    category_concentration: float
    holding_time_estimate_hours: float
    drawdown_proxy: float
    copyability_score: float
    low_velocity_score: float
    delay_5s: float
    delay_15s: float
    delay_30s: float
    delay_60s: float
    performance_score: float = 0.0
    consistency_score: float = 0.0
    sample_size_score: float = 0.0
    market_quality_score: float = 0.0
    delayed_viability_score: float = 0.0
    category_strength_score: float = 0.0
    hedge_suspicion_score: float = 0.0
    complexity_penalty: float = 0.0
    copied_performance_score: float = 0.0
    global_score: float = 0.0
    dominant_category: str = "unknown"
    source_quality: SourceQuality = SourceQuality.REAL_PUBLIC_DATA


class DiscoveryResult(BaseModel):
    wallets: list[WalletMetrics] = Field(default_factory=list)
    state: DiscoveryState = DiscoveryState.NO_DATA
    source_quality: SourceQuality = SourceQuality.DEGRADED_PUBLIC_DATA
    reason: str = ""
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    candidate_wallets: list[dict[str, Any]] = Field(default_factory=list)
    filtered_wallets: list[dict[str, Any]] = Field(default_factory=list)
    rejected_wallets: list[dict[str, Any]] = Field(default_factory=list)


class WalletScoringResult(BaseModel):
    scored_wallets: list[WalletMetrics] = Field(default_factory=list)
    skipped_wallets: list[dict[str, Any]] = Field(default_factory=list)
    rejected_wallets: list[dict[str, Any]] = Field(default_factory=list)
    state: str = "EMPTY"
    source_quality: SourceQuality = SourceQuality.DEGRADED_PUBLIC_DATA
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class CategoryScoringResult(BaseModel):
    rows: list["CategoryScore"] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class CategoryScore(BaseModel):
    wallet_address: str
    category: str
    score: float
    trade_count: int
    win_rate: float
    copyability_score: float
    delay_viability_score: float


class ApprovedWallets(BaseModel):
    research_wallets: list[str]
    paper_wallets: list[str]
    live_wallets: list[str]


class DetectionEvent(BaseModel):
    event_key: str
    local_detection_timestamp: datetime = Field(default_factory=utcnow)
    source_trade_timestamp: datetime = Field(default_factory=utcnow)
    wallet_address: str
    market_title: str
    market_slug: str
    market_id: str
    token_id: str
    side: str
    price: float
    size: float
    notional: float
    transaction_hash: str
    detection_latency_seconds: float
    best_bid: float | None = None
    best_ask: float | None = None
    depth_available: float | None = None
    category: str = "unknown"
    source_alias: str = ""
    market_metadata: dict[str, Any] = Field(default_factory=dict)
    source_quality: SourceQuality = SourceQuality.REAL_PUBLIC_DATA


class OrderbookLevel(BaseModel):
    price: float
    size: float


class OrderbookSnapshot(BaseModel):
    token_id: str
    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]
    timestamp: datetime = Field(default_factory=utcnow)


class FillEstimate(BaseModel):
    fillable: bool
    executable_price: float
    spread_pct: float
    slippage_pct: float
    filled_notional: float
    reason: str
    depth_consumed_pct: float = 0.0
    max_size_within_slippage: float = 0.0


class RiskResult(BaseModel):
    allowed: bool
    reason_code: str
    human_readable_reason: str
    context: dict[str, Any] = Field(default_factory=dict)


class TradeDecision(BaseModel):
    allowed: bool
    action: DecisionAction
    reason_code: str
    human_readable_reason: str
    local_decision_id: str = ""
    wallet_address: str
    market_id: str
    token_id: str
    entry_style: EntryStyle
    category: str
    scaled_notional: float
    source_price: float
    executable_price: float
    cluster_confirmed: bool
    hedge_suspicion_score: float
    context: dict[str, Any] = Field(default_factory=dict)


class ClusterSignal(BaseModel):
    cluster_id: str
    market_id: str
    token_id: str
    side: str
    wallet_count: int
    cluster_strength: float
    first_seen: datetime
    last_seen: datetime
    category: str
    wallets: list[str] = Field(default_factory=list)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class MarketInfo(BaseModel):
    market_id: str
    token_id: str
    title: str
    slug: str
    category: str
    active: bool = True
    closed: bool = False
    liquidity: float = 0.0
    volume: float = 0.0
    end_date_iso: str | None = None
    source_quality: SourceQuality = SourceQuality.REAL_PUBLIC_DATA


class Position(BaseModel):
    position_id: str
    mode: Mode
    wallet_address: str
    market_id: str
    token_id: str
    category: str
    entry_style: EntryStyle
    entry_price: float
    current_mark_price: float
    quantity: float
    notional: float
    fees_paid: float
    source_trade_timestamp: datetime
    opened_at: datetime = Field(default_factory=utcnow)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    entry_reason: str = ""
    exit_reason: str = ""
    cluster_confirmed: bool = False
    hedge_suspicion_score: float = 0.0
    closed: bool = False
    side: str = "BUY"
    entry_order_ids: list[str] = Field(default_factory=list)
    entry_price_estimated: float = 0.0
    entry_price_actual: float = 0.0
    stop_loss_rule: str = ""
    take_profit_rule: str = ""
    time_stop_rule: str = ""
    source_exit_following_enabled: bool = False
    exit_state: str = ""
    last_reconciled_at: datetime | None = None
    closed_at: datetime | None = None
    source_wallet: str = ""
    entry_time: datetime | None = None
    entry_size: float = 0.0
    exited_size: float = 0.0
    remaining_size: float = 0.0
    exit_order_ids: list[str] = Field(default_factory=list)


class GeoblockStatus(BaseModel):
    eligible: bool
    status: str
    detail: str
    checked_at: datetime = Field(default_factory=utcnow)


class HealthStatus(BaseModel):
    ok: bool
    detail: str
    checked_at: datetime = Field(default_factory=utcnow)


class HealthComponent(BaseModel):
    name: str
    state: HealthState
    detail: str
    observed_at: datetime = Field(default_factory=utcnow)
    age_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemHealth(BaseModel):
    overall: HealthState
    components: list[HealthComponent] = Field(default_factory=list)
    summary: str = ""
    checked_at: datetime = Field(default_factory=utcnow)


class ReadinessCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class LiveReadinessResult(BaseModel):
    ready: bool
    checks: list[ReadinessCheck] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=utcnow)


class LiveOrder(BaseModel):
    local_decision_id: str
    local_order_id: str
    client_order_id: str
    exchange_order_id: str = ""
    market_id: str
    token_id: str
    side: str
    intended_price: float
    intended_size: float
    entry_style: EntryStyle
    created_at: datetime = Field(default_factory=utcnow)
    submitted_at: datetime | None = None
    last_exchange_status: str = ""
    lifecycle_status: OrderLifecycleStatus = OrderLifecycleStatus.CREATED
    filled_size: float = 0.0
    average_fill_price: float = 0.0
    remaining_size: float = 0.0
    cancel_requested: bool = False
    cancel_confirmed: bool = False
    terminal_state: bool = False
    linked_position_id: str = ""
    audit_log_ref: str = ""
    raw_exchange_response_refs: list[str] = Field(default_factory=list)
    repriced_once: bool = False
    timeout_at: datetime | None = None
    is_exit: bool = False
    linked_parent_order_id: str = ""
    last_update_at: datetime = Field(default_factory=utcnow)


class ReconciliationIssue(BaseModel):
    severity: str
    issue_type: str
    detail: str
    local_ref: str = ""
    exchange_ref: str = ""


class ReconciliationSummary(BaseModel):
    clean: bool
    severity: str
    issues: list[ReconciliationIssue] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=utcnow)
