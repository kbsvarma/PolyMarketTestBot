from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import ParserError


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_event_key(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def append_csv_row(path: Path, row: dict[str, object]) -> None:
    existing = path.exists() and path.stat().st_size > 0
    if not existing:
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_fieldnames = list(reader.fieldnames or [])
        existing_rows = list(reader)

    new_keys = [key for key in row.keys() if key not in existing_fieldnames]
    if new_keys:
        fieldnames = existing_fieldnames + new_keys
        normalized_rows = [{name: existing_row.get(name, "") for name in fieldnames} for existing_row in existing_rows]
        normalized_rows.append({name: row.get(name, "") for name in fieldnames})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized_rows)
        return

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=existing_fieldnames)
        writer.writerow({name: row.get(name, "") for name in existing_fieldnames})


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else default


def write_json(path: Path, payload: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
