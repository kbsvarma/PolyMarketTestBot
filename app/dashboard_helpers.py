from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"


def load_csv(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    base_dir = data_dir or DATA
    path = base_dir / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def load_json(name: str, data_dir: Path | None = None) -> dict | list:
    base_dir = data_dir or DATA
    path = base_dir / name
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    try:
        return json.loads(text) if text else {}
    except json.JSONDecodeError:
        return {}


def load_jsonl_tail(name: str, lines: int = 10, data_dir: Path | None = None) -> list[dict]:
    base_dir = data_dir or DATA
    path = base_dir / name
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[-lines:]:
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def render_jsonl_lines(rows: list[dict]) -> str:
    if not rows:
        return "No recent activity."
    return "\n".join(json.dumps(row, default=str) for row in rows)


def load_log_tail(name: str, lines: int = 25, logs_dir: Path | None = None) -> str:
    base_dir = logs_dir or LOGS
    path = base_dir / name
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])


def health_component(health: dict, name: str) -> dict:
    components = health.get("components", []) if isinstance(health, dict) else []
    for component in components:
        if component.get("name") == name:
            return component
    return {}


def parse_detail_mapping(detail: object) -> dict:
    if isinstance(detail, dict):
        return detail
    if not detail:
        return {}
    text = str(detail).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}


def wallet_balance_text(state: dict) -> str:
    payload = parse_detail_mapping(state.get("balance_detail", ""))
    raw_balance = payload.get("cash_usd", payload.get("available", payload.get("balance", "")))
    if raw_balance in ("", None):
        return "N/A"
    try:
        return f"${float(raw_balance):,.2f}"
    except (TypeError, ValueError):
        return str(raw_balance)


def currency_text(value: object) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def parse_iso_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def bot_runtime_status(state: dict, refresh_seconds: int) -> tuple[str, str]:
    last_started = parse_iso_datetime(state.get("last_cycle_started_at"))
    last_cycle = parse_iso_datetime(state.get("last_cycle_completed_at"))
    if state.get("bot_loop_running") and last_started is not None:
        started_age = (datetime.now(timezone.utc) - last_started).total_seconds()
        if started_age <= max(refresh_seconds * 4, 180):
            return ("RUNNING", f"Cycle in progress, started {int(started_age)}s ago")
    if last_cycle is None:
        return ("STOPPED", "No completed bot cycle recorded yet.")
    age = (datetime.now(timezone.utc) - last_cycle).total_seconds()
    if age <= max(refresh_seconds * 2, 45):
        return ("RUNNING", f"Last cycle {int(age)}s ago")
    return ("IDLE", f"Last cycle {int(age)}s ago")


def paper_positions_frame(positions_payload: dict | list) -> pd.DataFrame:
    if isinstance(positions_payload, dict):
        rows = positions_payload.get("paper", [])
        return pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()
    return pd.DataFrame()


def placeholder_wallets(wallets: list[object]) -> bool:
    return bool(wallets) and all(str(wallet).startswith("0xWALLET") for wallet in wallets)
