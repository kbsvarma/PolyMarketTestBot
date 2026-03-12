from __future__ import annotations

from pathlib import Path

from src.utils import append_jsonl


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, event_type: str, payload: dict) -> str:
        record = {"event_type": event_type, **payload}
        append_jsonl(self.path, record)
        return f"{event_type}:{payload.get('ts', '')}"
