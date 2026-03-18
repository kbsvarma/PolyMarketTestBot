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
    strategy_name: str = "wallet_follow"
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
    side: str = "BUY"
    thesis_type: str = "directional"
    bundle_id: str = ""
    bundle_role: str = ""
    paired_token_id: str = ""
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
    strategy_name: str = "wallet_follow"
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
    thesis_type: str = "directional"
    bundle_id: str = ""
    bundle_role: str = ""
    paired_token_id: str = ""
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
    peak_mark_price: float = 0.0
    peak_pnl_pct: float = 0.0
    peak_mark_seen_at: datetime | None = None
    profit_lock_armed: bool = False
    trailing_stop_price: float = 0.0
    partial_profit_taken: bool = False


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
    strategy_name: str = "wallet_follow"
    wallet_address: str = ""
    category: str = "unknown"
    source_price: float = 0.0
    exchange_order_id: str = ""
    market_id: str
    token_id: str
    side: str
    intended_price: float
    intended_size: float
    entry_style: EntryStyle
    thesis_type: str = "directional"
    bundle_id: str = ""
    bundle_role: str = ""
    paired_token_id: str = ""
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
    reprice_attempts: int = 0
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


# ---------------------------------------------------------------------------
# Bracket strategy — direction signal models
# ---------------------------------------------------------------------------

class WindowCheckpoint(BaseModel):
    """
    A single BTC or ETH price sample recorded at a :00-second boundary.

    We only record prices at :00-second marks (e.g., 14:03:00, 14:04:00)
    because those are the reference points Chainlink uses for settlement.
    Using these as checkpoints means our chop filter is evaluating the
    same price series that will determine the outcome.
    """
    ts: float       # Unix epoch seconds — always a :00-second boundary
    price: float    # BTC/USD or ETH/USD price at this moment


class PostSignalObservation(BaseModel):
    """
    Tracks what actually happened AFTER a direction signal fired.

    This is filled in continuously on every poll cycle until the 15m window
    closes. It answers the key calibration questions:
      - Did the bracket actually become achievable?
      - Would Phase 2 (opposite-side buy) have triggered?
      - What profit would the early-exit path have captured?

    No orders are placed — this is purely observation data for strategy
    validation and parameter calibration.
    """
    # Links back to the BracketSignalEvent that spawned this observation
    event_id: str

    # Bracket margin tracking: best spread seen after signal (positive = profitable)
    # bracket_margin = 1.0 - (x_cost*(1+fee_x) + y_cost*(1+fee_y))
    peak_bracket_margin: float = 0.0

    # Lowest price seen for the opposite side after the signal fired
    # (the floor we'd wait for before buying the second leg)
    min_opposite_price: float = 999.0

    # Would the bracket equation have been solvable at any point?
    # True if x + y + fees < $1.00 was satisfied after signal fired
    bracket_would_have_formed: bool = False

    # Phase 2 check: did the opposite side reverse upward from its floor
    # by more than the reversal_threshold? This is the trigger for buying
    # the second leg in the real strategy.
    phase2_would_have_triggered: bool = False
    phase2_trigger_price: float | None = None  # y price at the reversal point

    # Momentum side tracking: how far did it travel after signal?
    momentum_side_peak: float = 0.0       # highest momentum_side ask seen
    momentum_side_at_close: float = 0.0   # momentum_side ask at window close

    # Resolution outcome (filled in when window resolves)
    outcome: str = ""                     # "YES_WINS", "NO_WINS", "UNKNOWN"
    asset_close_price: float = 0.0        # BTC/ETH price at window close

    # Estimated P&L if Phase 1 was exited at the momentum_side peak
    # (the early-exit path: sell momentum_side when bracket never formed)
    estimated_phase1_exit_pnl: float = 0.0   # in dollars per 10-share position


class BracketSignalEvent(BaseModel):
    """
    Emitted when all direction signal gates pass for a 15m BTC/ETH market.

    This records the complete state at signal-fire time so we can:
      1. Replay and audit exactly why the signal fired
      2. Compare predicted vs actual market movement
      3. Calibrate gate thresholds (lag_threshold, entry_range, chop_window)

    IMPORTANT: This module is observation-only. No orders are placed.
    The 'observation' field is populated after the window closes.
    """
    event_id: str           # uuid4 — unique identifier for this signal
    fired_at: datetime      # UTC timestamp when signal fired
    asset: str              # "BTC" or "ETH"

    # --- Window context ---
    window_open_ts: int     # Unix ts of 15m window start (:00/:15/:30/:45 minute)
    window_close_ts: int    # window_open_ts + 900
    minutes_remaining: float    # minutes left when signal fired (should be > 9.0)
    mid_window_start: bool = False  # True if bot started mid-window (less reliable baseline)

    # --- Asset price at signal time ---
    asset_open: float       # Asset price at window open (BTC/USD or ETH/USD)
    asset_current: float    # Asset price when signal fired
    asset_move_pct: float   # (asset_current - asset_open) / asset_open — direction + magnitude

    # --- Signal quality metrics ---
    momentum_side: str      # "YES" (BTC going up) or "NO" (BTC going down)
    momentum_price: float   # Best ask price of momentum_side token at signal time
    opposite_price: float   # Best ask price of opposite_side token at signal time
    implied_momentum_price: float  # GBM-model fair value for momentum_side
    lag_gap: float          # implied_momentum_price - momentum_price (larger = better entry)
    chop_score: float       # 0.0 (choppy) to 1.0 (perfectly clean directional move)

    # --- :00-second BTC/ETH price history this window ---
    checkpoints: list[WindowCheckpoint] = Field(default_factory=list)

    # --- Fee context (using the actual Polymarket crypto fee curve) ---
    fee_at_momentum_price: float   # taker_fee(momentum_price, "crypto price")
    fee_at_target_y: float         # taker_fee(target_y_price, "crypto price")
    # Net bracket margin if second leg enters at target_y (e.g. 0.34):
    # 1.0 - momentum_price*(1+fee_x) - target_y*(1+fee_y)
    net_bracket_at_target: float

    # --- Market identifiers ---
    market_id: str
    yes_token_id: str
    no_token_id: str
    market_liquidity: float = 0.0   # From Polymarket market metadata
    market_volume: float = 0.0      # 24h volume if available

    # --- Post-signal outcome (populated after window closes) ---
    observation: PostSignalObservation | None = None
