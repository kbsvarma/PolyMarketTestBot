from __future__ import annotations

from pathlib import Path

from app.dashboard_helpers import load_json, load_jsonl_tail


def test_dashboard_helpers_import_without_streamlit_dependency() -> None:
    assert callable(load_json)
    assert callable(load_jsonl_tail)


def test_dashboard_load_json_handles_malformed_file(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{bad json", encoding="utf-8")
    assert load_json("bad.json", data_dir=tmp_path) == {}


def test_dashboard_load_jsonl_tail_skips_bad_lines(tmp_path: Path) -> None:
    (tmp_path / "events.jsonl").write_text('{"ok": 1}\nnot-json\n{"ok": 2}\n', encoding="utf-8")
    rows = load_jsonl_tail("events.jsonl", lines=10, data_dir=tmp_path)
    assert rows == [{"ok": 1}, {"ok": 2}]
