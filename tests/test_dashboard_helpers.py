from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_dashboard_module():
    root = Path(__file__).resolve().parent.parent
    module_path = root / "app" / "dashboard.py"
    spec = importlib.util.spec_from_file_location("dashboard_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dashboard_load_json_handles_malformed_file(tmp_path: Path) -> None:
    dashboard = _load_dashboard_module()
    original_data = dashboard.DATA
    try:
        dashboard.DATA = tmp_path
        (tmp_path / "bad.json").write_text("{bad json", encoding="utf-8")
        assert dashboard.load_json("bad.json") == {}
    finally:
        dashboard.DATA = original_data


def test_dashboard_load_jsonl_tail_skips_bad_lines(tmp_path: Path) -> None:
    dashboard = _load_dashboard_module()
    original_data = dashboard.DATA
    try:
        dashboard.DATA = tmp_path
        (tmp_path / "events.jsonl").write_text('{"ok": 1}\nnot-json\n{"ok": 2}\n', encoding="utf-8")
        rows = dashboard.load_jsonl_tail("events.jsonl", lines=10)
        assert rows == [{"ok": 1}, {"ok": 2}]
    finally:
        dashboard.DATA = original_data
