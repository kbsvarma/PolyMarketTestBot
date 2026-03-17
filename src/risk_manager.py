from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import AppConfig
from src.fees import taker_fee, meets_min_edge
from src.models import DecisionAction, DetectionEvent, FillEstimate, HealthState, RiskResult, WalletMetrics


class RiskManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.hourly_entries: list[datetime] = []

    def _prune_hourly_entries(self, now: datetime | None = None) -> None:
        current_time = now or datetime.now(timezone.utc)
        self.hourly_entries = [
            timestamp
            for timestamp in self.hourly_entries
            if current_time - timestamp <= timedelta(hours=1)
        ]

    def entries_last_hour(self, now: datetime | None = None) -> int:
        current_time = now or datetime.now(timezone.utc)
        self._prune_hourly_entries(current_time)
        return len(self.hourly_entries)

    def register_entry(self, timestamp: datetime | None = None) -> None:
        current_time = timestamp or datetime.now(timezone.utc)
        self._prune_hourly_entries(current_time)
        self.hourly_entries.append(current_time)

    def evaluate(
        self,
        detection: DetectionEvent,
        wallet: WalletMetrics,
        fill: FillEstimate,
        mode: str,
        total_exposure: float,
        market_exposure: float,
        wallet_exposure: float,
        daily_pnl: float,
        cluster_confirmed: bool,
        infra_ok: bool,
        entry_style_allowed: bool,
        category: str | None = None,
        market_id: str | None = None,
        has_conflicting_position: bool = False,
        data_age_seconds: float = 0.0,
        entry_drift_pct: float = 0.0,
        live_ready: bool = True,
        kill_switch: bool = False,
        manual_live_enable: bool = True,
        manual_resume_required: bool = False,
        health_state: str = HealthState.HEALTHY.value,
        reconciliation_clean: bool = True,
        heartbeat_ok: bool = True,
        balance_visible: bool = True,
        allowance_sufficient: bool = True,
        tradable: bool = True,
        bankroll_override: float | None = None,
        entries_last_hour_override: int | None = None,
        stale_signal_seconds_override: float | None = None,
        live_category_bypass: bool = False,
    ) -> RiskResult:
        bankroll = self.config.bankroll.paper_starting_bankroll if mode != "LIVE" else self.config.bankroll.live_bankroll_reference
        if mode != "LIVE" and bankroll_override and bankroll_override > 0:
            bankroll = bankroll_override
        spread_limit = self.config.risk.max_spread_pct if mode == "LIVE" else max(self.config.risk.max_spread_pct, 0.05)
        stale_signal_limit = (
            float(stale_signal_seconds_override)
            if stale_signal_seconds_override is not None and stale_signal_seconds_override > 0
            else float(self.config.risk.stale_signal_seconds)
        )
        now = datetime.now(timezone.utc)
        entries_last_hour = entries_last_hour_override if entries_last_hour_override is not None else self.entries_last_hour(now)

        if mode == "LIVE" and not live_ready:
            return RiskResult(allowed=False, reason_code="LIVE_NOT_READY", human_readable_reason="Live readiness gate is not satisfied.", context={})
        if mode == "LIVE" and kill_switch:
            return RiskResult(allowed=False, reason_code="KILL_SWITCH", human_readable_reason="Global kill switch is enabled.", context={})
        if mode == "LIVE" and not manual_live_enable:
            return RiskResult(allowed=False, reason_code="MANUAL_ENABLE_REQUIRED", human_readable_reason="Manual live enable flag is required.", context={})
        if mode == "LIVE" and manual_resume_required:
            return RiskResult(allowed=False, reason_code="MANUAL_RESUME_REQUIRED", human_readable_reason="Manual resume is required after a pause.", context={})
        if mode == "LIVE" and health_state == HealthState.UNHEALTHY.value:
            return RiskResult(allowed=False, reason_code="HEALTH_NOT_HEALTHY", human_readable_reason="Live health is not healthy.", context={"health_state": health_state})
        if mode == "LIVE" and not reconciliation_clean:
            return RiskResult(allowed=False, reason_code="RECONCILIATION_DIRTY", human_readable_reason="Reconciliation is unresolved.", context={})
        if mode == "LIVE" and not heartbeat_ok:
            return RiskResult(allowed=False, reason_code="HEARTBEAT_BAD", human_readable_reason="Heartbeat is stale or failed.", context={})
        if mode == "LIVE" and not balance_visible:
            return RiskResult(allowed=False, reason_code="BALANCE_UNAVAILABLE", human_readable_reason="Balance is unavailable.", context={})
        if mode == "LIVE" and not allowance_sufficient:
            return RiskResult(allowed=False, reason_code="ALLOWANCE_UNAVAILABLE", human_readable_reason="Allowance/spendability is insufficient.", context={})
        if mode == "LIVE" and not tradable:
            return RiskResult(allowed=False, reason_code="MARKET_NOT_TRADABLE", human_readable_reason="Market is not explicitly tradable.", context={"market_id": market_id})
        if daily_pnl <= -(bankroll * self.config.risk.daily_stop_loss_pct):
            return RiskResult(allowed=False, reason_code="DAILY_STOP", human_readable_reason="Daily stop loss reached.", context={"daily_pnl": daily_pnl})
        if total_exposure > bankroll * self.config.risk.max_total_exposure_pct:
            return RiskResult(allowed=False, reason_code="TOTAL_EXPOSURE", human_readable_reason="Total exposure limit exceeded.", context={"total_exposure": total_exposure})
        if market_exposure > bankroll * self.config.risk.max_market_exposure_pct:
            return RiskResult(allowed=False, reason_code="MARKET_EXPOSURE", human_readable_reason="Per-market exposure limit exceeded.", context={"market_exposure": market_exposure})
        if wallet_exposure > bankroll * self.config.risk.max_wallet_exposure_pct:
            return RiskResult(allowed=False, reason_code="WALLET_EXPOSURE", human_readable_reason="Per-wallet exposure limit exceeded.", context={"wallet_exposure": wallet_exposure})
        if entries_last_hour >= self.config.risk.max_new_entries_per_hour:
            return RiskResult(
                allowed=False,
                reason_code="ENTRY_RATE_LIMIT",
                human_readable_reason="Hourly entry limit reached.",
                context={"entries_last_hour": entries_last_hour},
            )
        if detection.detection_latency_seconds > stale_signal_limit:
            return RiskResult(
                allowed=False,
                reason_code="STALE_SIGNAL",
                human_readable_reason="Signal is too stale.",
                context={
                    "latency": detection.detection_latency_seconds,
                    "stale_signal_limit_seconds": stale_signal_limit,
                },
            )
        if detection.depth_available is not None and detection.depth_available < self.config.risk.min_orderbook_depth_usd:
            return RiskResult(allowed=False, reason_code="LOW_LIQUIDITY", human_readable_reason="Orderbook depth below threshold.", context={"depth": detection.depth_available})
        if data_age_seconds > self.config.risk.stale_market_data_seconds:
            return RiskResult(allowed=False, reason_code="STALE_MARKET_DATA", human_readable_reason="Market data is too stale.", context={"data_age_seconds": data_age_seconds})
        if fill.spread_pct > spread_limit:
            return RiskResult(allowed=False, reason_code="WIDE_SPREAD", human_readable_reason="Spread is too wide.", context={"spread_pct": fill.spread_pct, "spread_limit": spread_limit})
        if not fill.fillable:
            return RiskResult(allowed=False, reason_code="FILLABILITY", human_readable_reason=fill.reason, context=fill.model_dump())
        if fill.slippage_pct > self.config.risk.max_slippage_pct:
            return RiskResult(allowed=False, reason_code="SLIPPAGE", human_readable_reason="Expected slippage too high.", context={"slippage_pct": fill.slippage_pct})
        if entry_drift_pct > self.config.risk.max_entry_drift_pct:
            return RiskResult(allowed=False, reason_code="ENTRY_DRIFT", human_readable_reason="Entry drift exceeds threshold.", context={"entry_drift_pct": entry_drift_pct})
        selected_category = category or wallet.dominant_category
        if selected_category not in self.config.risk.allow_categories:
            return RiskResult(allowed=False, reason_code="CATEGORY_BLOCKED", human_readable_reason="Category not allowed.", context={"category": selected_category})
        if mode == "LIVE" and not live_category_bypass and selected_category not in self.config.live.selected_categories:
            return RiskResult(
                allowed=False,
                reason_code="LIVE_CATEGORY_NOT_SELECTED",
                human_readable_reason="Category is not enabled for live trading.",
                context={"category": selected_category},
            )
        if wallet.hedge_suspicion_score > self.config.risk.max_hedge_suspicion_score:
            return RiskResult(allowed=False, reason_code="HEDGE_SUSPECT", human_readable_reason="Hedge suspicion too high.", context={"hedge_suspicion_score": wallet.hedge_suspicion_score})
        if has_conflicting_position:
            return RiskResult(allowed=False, reason_code="CONFLICTING_POSITION", human_readable_reason="Conflicting or duplicate position already exists.", context={"market_id": market_id})
        if mode == "LIVE" and self.config.risk.require_cluster_confirmation_live and not cluster_confirmed:
            return RiskResult(allowed=False, reason_code="CLUSTER_REQUIRED", human_readable_reason="Cluster confirmation required for live trading.", context={})
        if not infra_ok:
            return RiskResult(allowed=False, reason_code="INFRA_DEGRADED", human_readable_reason="Infrastructure health degraded.", context={})
        if not entry_style_allowed:
            return RiskResult(allowed=False, reason_code="ENTRY_STYLE_BLOCKED", human_readable_reason="Entry style not approved.", context={})
        # Fee gate — only applies to crypto markets where taker fees are non-zero.
        # Placed last so higher-priority reason codes (category, drift, etc.) take precedence.
        if self.config.fees.enable_global_fee_gate and fill.fillable and detection.price > 0:
            _exec_p = float(fill.executable_price)
            _fc = category or wallet.dominant_category
            _fee = taker_fee(_exec_p, category=_fc)
            if _fee > 0:
                _gross_edge = detection.price - _exec_p
                _net_edge = _gross_edge - _fee
                _min_net = self.config.fees.min_edge_after_fees_taker
                if _net_edge < _min_net:
                    return RiskResult(
                        allowed=False,
                        reason_code="NET_EDGE_AFTER_FEES",
                        human_readable_reason=f"Net edge after fees ({_net_edge:.4f}) below minimum ({_min_net:.4f}).",
                        context={"gross_edge": round(_gross_edge, 6), "fee": round(_fee, 6), "net_edge": round(_net_edge, 6), "min_net_edge": _min_net},
                    )
        return RiskResult(
            allowed=True,
            reason_code="OK",
            human_readable_reason="Risk checks passed.",
            context={
                "action": DecisionAction.PAPER_COPY.value if mode != "LIVE" else DecisionAction.LIVE_COPY.value,
                "entries_last_hour": entries_last_hour,
            },
        )
