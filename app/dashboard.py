from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"


def load_csv(name: str) -> pd.DataFrame:
    path = DATA / name
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def load_json(name: str) -> dict | list:
    path = DATA / name
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else {}


def load_jsonl_tail(name: str, lines: int = 10) -> list[dict]:
    path = DATA / name
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[-lines:]:
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_log_tail(name: str, lines: int = 25) -> str:
    path = LOGS / name
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])


def main() -> None:
    st.set_page_config(page_title="Polymarket Live Operator Console", layout="wide")
    st.title("Polymarket Wallet-Following Operator Console")
    st.caption("Live-readiness, health, reconciliation, and risk console for a tightly gated wallet-following bot.")

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    state = load_json("app_state.json")
    summary = load_json("daily_summary.json")
    health = load_json("health_status.json")
    live_orders = load_json("live_orders.json")
    wallet_scores = load_csv("wallet_scorecard.csv")
    category_scores = load_csv("category_wallet_scorecard.csv")
    trade_feed = load_csv("detected_wallet_trades.csv")
    strategy = load_csv("strategy_comparison.csv")
    paper = load_csv("paper_trade_history.csv")
    live = load_csv("live_trade_history.csv")

    with st.sidebar:
        st.subheader("System")
        st.json(state)
        st.subheader("Live Readiness")
        st.json(state.get("live_readiness_last_result", {}))
        st.subheader("Config")
        st.yaml(config)

    top = st.columns(6)
    top[0].metric("Mode", state.get("mode", config.get("mode", "RESEARCH")))
    top[1].metric("System Status", state.get("system_status", "INIT"))
    top[2].metric("Health", state.get("live_health_state", "UNKNOWN"))
    top[3].metric("Paused", "YES" if state.get("paused") else "NO")
    top[4].metric("Kill Switch", "ON" if state.get("kill_switch") else "OFF")
    top[5].metric("Heartbeat", "OK" if state.get("heartbeat_ok") else "BAD")

    row1_left, row1_right = st.columns([1.1, 0.9])
    with row1_left:
        st.subheader("Live Readiness Checklist")
        readiness_checks = state.get("live_readiness_last_result", {}).get("checks", [])
        st.dataframe(pd.DataFrame(readiness_checks), use_container_width=True, height=260)
    with row1_right:
        st.subheader("Health Summary")
        st.json(health)

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
        st.subheader("Paper Portfolio")
        st.dataframe(paper, use_container_width=True, height=220)
    with row7_right:
        st.subheader("Alert / Error Log Tail")
        st.code(load_log_tail("errors.log"), language="text")

    st.subheader("System Log Tail")
    st.code(load_log_tail("system.log"), language="text")


if __name__ == "__main__":
    main()
