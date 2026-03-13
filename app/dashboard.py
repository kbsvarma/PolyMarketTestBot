from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime, timezone
import asyncio

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dashboard_helpers import (
    bot_runtime_status,
    currency_text,
    health_component,
    load_csv,
    load_json,
    load_jsonl_tail,
    load_log_tail,
    paper_positions_frame,
    placeholder_wallets,
    render_jsonl_lines,
    wallet_balance_text,
)
from src.state import AppStateStore
from src.config import load_config
from src.geoblock import GeoblockChecker
from src.live_engine import LiveTradingEngine


DATA = ROOT / "data"
LOGS = ROOT / "logs"
STATE_STORE = AppStateStore(DATA / "app_state.json")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Polymarket Live Operator Console", layout="wide")
    st.title("Polymarket Wallet-Following Operator Console")
    st.caption("Live-readiness, health, reconciliation, and risk console for a tightly gated wallet-following bot.")
    flash_message = st.session_state.pop("paper_flash_message", "")
    flash_level = st.session_state.pop("paper_flash_level", "info")
    control_cols = st.columns([0.2, 0.25, 0.55])
    if control_cols[0].button("Refresh Dashboard", use_container_width=True):
        st.rerun()
    if control_cols[1].button("Refresh Wallet Status", use_container_width=True):
        config_obj = load_config(ROOT / "config.yaml", ROOT / ".env")
        engine = LiveTradingEngine(config_obj, DATA, STATE_STORE, GeoblockChecker(config_obj))
        asyncio.run(engine.refresh_live_status())
        st.session_state["paper_flash_message"] = "Wallet status refreshed."
        st.session_state["paper_flash_level"] = "success"
        st.rerun()
    control_cols[2].caption(f"Dashboard loaded at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    state = load_json("app_state.json")
    summary = load_json("daily_summary.json")
    health = load_json("health_status.json")
    discovery = load_json("wallet_discovery_diagnostics.json")
    scoring = load_json("wallet_scoring_diagnostics.json")
    paper_quality = load_json("paper_quality_summary.json")
    live_orders = load_json("live_orders.json")
    wallet_scores = load_csv("wallet_scorecard.csv")
    category_scores = load_csv("category_wallet_scorecard.csv")
    trade_feed = load_csv("detected_wallet_trades.csv")
    strategy = load_csv("strategy_comparison.csv")
    paper = load_csv("paper_trade_history.csv")
    live = load_csv("live_trade_history.csv")
    positions_payload = load_json("positions.json")
    paper_positions = paper_positions_frame(positions_payload)
    paper_summary = state.get("paper_summary", {}) if isinstance(state, dict) else {}
    paper_events = pd.DataFrame(load_jsonl_tail("paper_audit.jsonl", lines=30))
    decision_traces = pd.DataFrame(load_jsonl_tail("paper_decision_trace.jsonl", lines=40))
    runtime_state, runtime_detail = bot_runtime_status(state, int(config.get("runtime", {}).get("polling_interval_seconds", 20)))
    watched_wallets = state.get("last_cycle_watched_wallets", [])
    using_placeholder_wallets = placeholder_wallets(watched_wallets)
    no_real_wallets = not watched_wallets

    with st.sidebar:
        st.subheader("System")
        st.json(state)
        st.subheader("Live Readiness")
        st.json(state.get("live_readiness_last_result", {}))
        st.subheader("Config")
        st.code(yaml.safe_dump(config, sort_keys=False), language="yaml")

    top = st.columns(6)
    top[0].metric("Mode", state.get("mode", config.get("mode", "RESEARCH")))
    top[1].metric("System Status", state.get("system_status", "INIT"))
    top[2].metric("Health", state.get("live_health_state", "UNKNOWN"))
    top[3].metric("Paused", "YES" if state.get("paused") else "NO")
    top[4].metric("Kill Switch", "ON" if state.get("kill_switch") else "OFF")
    top[5].metric("Heartbeat", "OK" if state.get("heartbeat_ok") else "BAD")

    balance_component = health_component(health, "balance")
    allowance_component = health_component(health, "allowance")
    wallet_row = st.columns(7)
    wallet_row[0].metric("Wallet Auth", "OK" if state.get("auth_detail") else "UNKNOWN")
    wallet_row[1].metric("Wallet Stablecoins", currency_text(state.get("wallet_total_stablecoins")))
    wallet_row[2].metric("Spendable USDC", wallet_balance_text(state))
    wallet_row[3].metric("Position Value", currency_text(state.get("portfolio_position_value")))
    wallet_row[4].metric("Balance Visible", "YES" if state.get("balance_visible") else "NO")
    wallet_row[5].metric("Allowance", "OK" if state.get("allowance_sufficient") else "NOT READY")
    wallet_row[6].metric("Exchange Positions", str(state.get("positions_detail", "0 positions visible")).split(" ")[0] if state.get("positions_detail") else "0")

    row1_left, row1_right = st.columns([1.1, 0.9])
    with row1_left:
        st.subheader("Live Readiness Checklist")
        readiness_checks = state.get("live_readiness_last_result", {}).get("checks", [])
        st.dataframe(pd.DataFrame(readiness_checks), use_container_width=True, height=260)
    with row1_right:
        st.subheader("Health Summary")
        st.json(health)

    diag_left, diag_mid, diag_right = st.columns(3)
    diag_left.metric("Discovery State", str(discovery.get("diagnostics", {}).get("discovery_state") or discovery.get("state") or "UNKNOWN"))
    diag_mid.metric("Scoring State", str(scoring.get("state") or "UNKNOWN"))
    diag_right.metric("Paper Readiness", str(paper_quality.get("paper_readiness") or "UNKNOWN"))
    trust_cols = st.columns(4)
    trust_cols[0].metric("Trust Level", str(paper_quality.get("trust_level") or "UNKNOWN"))
    trust_cols[1].metric("Trustworthy Approvals", int(paper_quality.get("total_approved_decisions_trustworthy", 0)))
    trust_cols[2].metric("Degraded Approvals", int(paper_quality.get("total_approved_decisions_degraded", 0)))
    trust_cols[3].metric("Synthetic Wallets", int(paper_quality.get("synthetic_wallet_count", 0)))
    approval_cols = st.columns(4)
    approval_cols[0].metric("Approvals Total", int(paper_quality.get("total_approved_decisions", 0)))
    approval_cols[1].metric("Non-Validation Approvals", int(paper_quality.get("total_approved_decisions_non_validation", 0)))
    approval_cols[2].metric("Dominant Source Quality", str(paper_quality.get("dominant_source_quality") or "UNKNOWN"))
    approval_cols[3].metric("Validation Mode", str(paper_quality.get("validation_mode") or "UNKNOWN"))

    dominant_source_quality = str(paper_quality.get("dominant_source_quality") or "UNKNOWN")
    fallback_in_use = bool(paper_quality.get("fallback_in_use", False))
    if fallback_in_use or dominant_source_quality == "SYNTHETIC_FALLBACK":
        st.error("Paper run is using synthetic fallback data. This run is not trustworthy for live-readiness decisions.")
    elif dominant_source_quality == "DEGRADED_PUBLIC_DATA":
        st.warning("Paper run is using degraded public data. Treat results as development-only unless source quality improves.")

    st.subheader("Paper Trust Diagnostics")
    trust_left, trust_mid, trust_right = st.columns(3)
    with trust_left:
        st.caption("Wallet discovery")
        st.json(discovery)
    with trust_mid:
        st.caption("Wallet scoring")
        st.json(scoring.get("diagnostics", {}))
        rejected_wallets = pd.DataFrame(scoring.get("rejected_wallets", []))
        if not rejected_wallets.empty:
            st.dataframe(rejected_wallets, use_container_width=True, height=180)
    with trust_right:
        st.caption("Paper quality / source quality")
        st.json(paper_quality)

    wallet_left, wallet_right = st.columns(2)
    with wallet_left:
        st.subheader("Wallet Connection")
        st.json(
            {
                "auth_detail": state.get("auth_detail", ""),
                "connected_funder_wallet": state.get("connected_funder_wallet", ""),
                "connected_proxy_wallet": state.get("connected_proxy_wallet", ""),
                "wallet_balance_visible": state.get("wallet_balance_visible", False),
                "wallet_balance_detail": state.get("wallet_balance_detail", ""),
                "wallet_usdc_balance": currency_text(state.get("wallet_usdc_balance")),
                "wallet_usdce_balance": currency_text(state.get("wallet_usdce_balance")),
                "wallet_total_stablecoins": currency_text(state.get("wallet_total_stablecoins")),
                "balance_visible": state.get("balance_visible", False),
                "spendable_usdc": wallet_balance_text(state),
                "position_value": currency_text(state.get("portfolio_position_value")),
                "position_cost_basis": currency_text(state.get("portfolio_cost_basis")),
                "balance_detail": state.get("balance_detail", ""),
                "allowance_sufficient": state.get("allowance_sufficient", False),
                "allowance_detail": state.get("allowance_detail", ""),
                "open_orders_visible": state.get("open_orders_visible", False),
                "open_orders_detail": state.get("open_orders_detail", ""),
                "positions_visible": state.get("positions_visible", False),
                "positions_detail": state.get("positions_detail", ""),
            }
        )
        st.caption("Wallet Stablecoins is read directly from the Polygon wallet. Spendable USDC comes from Polymarket's authenticated CLOB balance/allowance endpoint and represents collateral available for new orders.")
    with wallet_right:
        st.subheader("Wallet Health Metadata")
        st.json(
            {
                "balance_component": balance_component,
                "allowance_component": allowance_component,
            }
        )

    st.subheader("Paper Trading Controls")
    paper_left, paper_mid, paper_right = st.columns([1.1, 1.1, 1.8])
    with paper_left:
        paper_bankroll = st.number_input(
            "Paper bankroll",
            min_value=10.0,
            value=float(state.get("paper_bankroll_override", 0.0) or config.get("bankroll", {}).get("paper_starting_bankroll", 200.0)),
            step=10.0,
            help="Total paper bankroll used by the strategy sizing and risk checks.",
        )
    with paper_mid:
        paper_trade_notional = st.number_input(
            "Max paper trade",
            min_value=1.0,
            value=float(state.get("paper_trade_notional_override", 0.0) or 5.0),
            step=1.0,
            help="Hard cap for each paper trade notional.",
        )
        if st.button("Save Paper Settings", use_container_width=True):
            STATE_STORE.update_system_status(
                paper_bankroll_override=float(paper_bankroll),
                paper_trade_notional_override=float(paper_trade_notional),
            )
            st.session_state["paper_flash_message"] = "Paper settings saved. They will be used on the next bot cycle."
            st.session_state["paper_flash_level"] = "success"
            st.rerun()
        run_cols = st.columns(2)
        if run_cols[0].button("Run Paper Bot", use_container_width=True):
            STATE_STORE.update_system_status(
                paper_bankroll_override=float(paper_bankroll),
                paper_trade_notional_override=float(paper_trade_notional),
                paper_run_enabled=True,
            )
            st.session_state["paper_flash_message"] = "Paper trading started."
            st.session_state["paper_flash_level"] = "success"
            st.rerun()
        if run_cols[1].button("Pause Paper Bot", use_container_width=True):
            STATE_STORE.update_system_status(paper_run_enabled=False)
            st.session_state["paper_flash_message"] = "Paper trading paused."
            st.session_state["paper_flash_level"] = "warning"
            st.rerun()
    with paper_right:
        if flash_message:
            getattr(st, flash_level, st.info)(flash_message)
        pnl_cols = st.columns(4)
        pnl_cols[0].metric("Last Hour PnL", currency_text(paper_summary.get("last_hour_net_pnl")))
        pnl_cols[1].metric("Last 24h PnL", currency_text(paper_summary.get("last_24h_net_pnl")))
        pnl_cols[2].metric("Last 7d PnL", currency_text(paper_summary.get("last_7d_net_pnl")))
        pnl_cols[3].metric("Net PnL", currency_text(paper_summary.get("net_pnl_total")))
        st.caption("Paper PnL is updated from live Polymarket orderbook marks for open paper positions during bot cycles.")
        if runtime_state == "RUNNING":
            st.success(f"Worker running. {runtime_detail}")
        elif runtime_state == "IDLE":
            st.warning(f"Worker idle. {runtime_detail}")
        else:
            st.error(f"Worker stopped. {runtime_detail}")
        if no_real_wallets:
            st.warning("No real public wallets have been approved for paper following yet, so no paper trades can fire.")
        if using_placeholder_wallets:
            st.warning(
                "The current paper cycle is watching placeholder fallback wallets (for example `0xWALLET00`). "
                "That means you should not expect real public-wallet detections or meaningful paper trades yet."
            )
        st.json(
            {
                "bot_runtime": runtime_state,
                "bot_runtime_detail": runtime_detail,
                "paper_run_enabled": state.get("paper_run_enabled", False),
                "paper_bankroll": currency_text(paper_summary.get("paper_bankroll")),
                "paper_trade_notional_override": currency_text(paper_summary.get("paper_trade_notional_override")),
                "open_positions": paper_summary.get("open_positions", 0),
                "open_notional": currency_text(paper_summary.get("open_notional")),
                "realized_pnl_total": currency_text(paper_summary.get("realized_pnl_total")),
                "unrealized_pnl_total": currency_text(paper_summary.get("unrealized_pnl_total")),
                "last_cycle_detection_count": state.get("last_cycle_detection_count", 0),
                "last_cycle_decision_count": state.get("last_cycle_decision_count", 0),
                "last_cycle_completed_at": state.get("last_cycle_completed_at", ""),
                "last_cycle_watched_wallets": state.get("last_cycle_watched_wallets", []),
                "placeholder_wallets": using_placeholder_wallets,
                "updated_at": paper_summary.get("updated_at", ""),
                "discovery_state": discovery.get("diagnostics", {}).get("discovery_state", ""),
                "scoring_state": scoring.get("state", ""),
                "paper_readiness": paper_quality.get("paper_readiness", ""),
                "trust_level": paper_quality.get("trust_level", ""),
                "validation_mode": paper_quality.get("validation_mode", ""),
                "dominant_source_quality": dominant_source_quality,
                "fallback_in_use": fallback_in_use,
                "approved_decisions_trustworthy": paper_quality.get("total_approved_decisions_trustworthy", 0),
                "approved_decisions_degraded": paper_quality.get("total_approved_decisions_degraded", 0),
            }
        )

    funnel_cols = st.columns(5)
    funnel = paper_quality.get("funnel", {}) if isinstance(paper_quality, dict) else {}
    funnel_cols[0].metric("Detected", funnel.get("detected", 0))
    funnel_cols[1].metric("Candidates", funnel.get("candidates", funnel.get("detected", 0)))
    funnel_cols[2].metric("Approved", funnel.get("approved", funnel.get("entered", 0)))
    funnel_cols[3].metric("Skipped", funnel.get("skipped", 0))
    funnel_cols[4].metric("Entered", funnel.get("entered", 0))
    trust_funnel_cols = st.columns(3)
    trust_funnel_cols[0].metric("Decision Count (Trustworthy)", int(paper_quality.get("decision_count_trustworthy", 0)))
    trust_funnel_cols[1].metric("Decision Count (Degraded)", int(paper_quality.get("decision_count_degraded", 0)))
    trust_funnel_cols[2].metric("Warnings", len(paper_quality.get("warnings", [])) if isinstance(paper_quality, dict) else 0)
    skip_reason_distribution = paper_quality.get("skip_reason_distribution", {}) if isinstance(paper_quality, dict) else {}
    if skip_reason_distribution:
        st.caption("Skip reason distribution")
        st.dataframe(pd.DataFrame([{"reason_code": key, "count": value} for key, value in skip_reason_distribution.items()]), use_container_width=True, height=160)
    warnings = paper_quality.get("warnings", []) if isinstance(paper_quality, dict) else []
    if warnings:
        st.caption("Paper trust warnings")
        for warning in warnings:
            st.warning(str(warning))

    st.subheader("Paper Activity Monitor")
    monitor_left, monitor_right = st.columns(2)
    with monitor_left:
        st.caption("Paper audit events")
        st.code(render_jsonl_lines(load_jsonl_tail("paper_audit.jsonl", lines=25)), language="json")
    with monitor_right:
        st.caption("Worker cycle log")
        st.code(load_log_tail("system.log", lines=30) or "No recent system log entries.", language="text")

    st.subheader("Paper Blotter")
    blotter_left, blotter_mid, blotter_right = st.columns(3)
    if not paper_positions.empty:
        if "closed" in paper_positions.columns:
            open_paper = paper_positions[paper_positions["closed"] != True].copy()
            closed_paper = paper_positions[paper_positions["closed"] == True].copy()
        else:
            open_paper = paper_positions.copy()
            closed_paper = pd.DataFrame()
        if "opened_at" in open_paper.columns:
            open_paper = open_paper.sort_values("opened_at", ascending=False)
        if "closed_at" in closed_paper.columns:
            closed_paper = closed_paper.sort_values("closed_at", ascending=False)
    else:
        open_paper = pd.DataFrame()
        closed_paper = pd.DataFrame()
    with blotter_left:
        st.metric("Realized PnL", currency_text(paper_summary.get("realized_pnl_total")))
        st.subheader("Open Paper Positions")
        st.dataframe(
            open_paper[
                [
                    column
                    for column in [
                        "market_id",
                        "token_id",
                        "category",
                        "entry_style",
                        "entry_price",
                        "current_mark_price",
                        "quantity",
                        "notional",
                        "unrealized_pnl",
                        "opened_at",
                    ]
                    if column in open_paper.columns
                ]
            ],
            use_container_width=True,
            height=280,
        )
    with blotter_mid:
        st.metric("Unrealized PnL", currency_text(paper_summary.get("unrealized_pnl_total")))
        st.subheader("Closed Paper Trades")
        st.dataframe(
            closed_paper[
                [
                    column
                    for column in [
                        "market_id",
                        "token_id",
                        "category",
                        "entry_style",
                        "entry_price",
                        "current_mark_price",
                        "quantity",
                        "realized_pnl",
                        "exit_reason",
                        "closed_at",
                    ]
                    if column in closed_paper.columns
                ]
            ],
            use_container_width=True,
            height=280,
        )
    with blotter_right:
        st.metric("Open Notional", currency_text(paper_summary.get("open_notional")))
        st.subheader("Recent Paper Events")
        if paper_events.empty:
            st.info("No paper events recorded yet.")
        st.dataframe(paper_events, use_container_width=True, height=280)

    st.subheader("Paper Decision Trace Feed")
    if decision_traces.empty:
        st.info("No paper decision traces recorded yet.")
    else:
        st.dataframe(
            decision_traces[
                [
                    column
                    for column in [
                        "detected_at",
                        "wallet_address",
                        "category",
                        "discovery_state",
                        "scoring_state",
                        "source_quality",
                        "cluster_state",
                        "freshness_state",
                        "fillability_state",
                        "risk_reason_code",
                        "final_action",
                        "reason_code",
                        "scaled_notional",
                    ]
                    if column in decision_traces.columns
                ]
            ],
            use_container_width=True,
            height=240,
        )

    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.subheader("Reconciliation / Pause State")
        st.json(
            {
                "pause_reason": state.get("pause_reason", ""),
                "manual_resume_required": state.get("manual_resume_required", False),
                "reconciliation_clean": state.get("reconciliation_clean", False),
                "reconciliation_summary": state.get("reconciliation_summary", {}),
                "live_last_reconciled_at": state.get("live_last_reconciled_at", ""),
                "unresolved_live_order_ids": state.get("unresolved_live_order_ids", []),
            }
        )
    with row2_right:
        st.subheader("Live Eligibility / Tradability")
        allowed = set(config.get("live", {}).get("selected_categories", []))
        categories = config.get("categories", {}).get("tracked", [])
        st.dataframe(pd.DataFrame([{"category": category, "live_allowed": category in allowed} for category in categories]), use_container_width=True, height=120)
        st.json(state.get("watched_market_tradability_cache", {}))

    row3_left, row3_right = st.columns(2)
    with row3_left:
        st.subheader("Open Live Orders")
        st.dataframe(pd.DataFrame(live_orders if isinstance(live_orders, list) else []), use_container_width=True, height=260)
    with row3_right:
        st.subheader("Live Positions")
        st.dataframe(live, use_container_width=True, height=260)

    row4_left, row4_right = st.columns(2)
    with row4_left:
        st.subheader("Recent Live Decisions")
        st.dataframe(pd.DataFrame(summary.get("recent_live_decisions", [])), use_container_width=True, height=240)
    with row4_right:
        st.subheader("Recent Audit Events")
        st.dataframe(pd.DataFrame(load_jsonl_tail("live_audit.jsonl")), use_container_width=True, height=240)

    row5_left, row5_right = st.columns(2)
    with row5_left:
        st.subheader("Wallet Rankings")
        st.dataframe(wallet_scores, use_container_width=True, height=240)
    with row5_right:
        st.subheader("Category Scorecards")
        st.dataframe(category_scores, use_container_width=True, height=240)

    row6_left, row6_right = st.columns(2)
    with row6_left:
        st.subheader("Source Trade Feed")
        st.dataframe(trade_feed, use_container_width=True, height=240)
    with row6_right:
        st.subheader("Strategy Comparison")
        st.dataframe(strategy, use_container_width=True, height=240)

    row7_left, row7_right = st.columns(2)
    with row7_left:
        st.subheader("Paper History (Raw)")
        st.dataframe(paper, use_container_width=True, height=220)
    with row7_right:
        st.subheader("Alert / Error Log Tail")
        error_log = load_log_tail("errors.log")
        if not error_log.strip():
            error_log = "No recent errors logged."
        st.code(error_log, language="text")

    st.subheader("System Log Tail")
    system_log = load_log_tail("system.log")
    if not system_log.strip():
        system_log = "No recent system log entries. Use the runtime status and paper audit feed above to verify activity."
    st.code(system_log, language="text")


if __name__ == "__main__":
    main()
