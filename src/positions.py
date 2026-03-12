from __future__ import annotations

from pathlib import Path

from src.models import Mode, Position
from src.utils import read_json, write_json


class PositionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, list[dict]]:
        return read_json(self.path, {"paper": [], "live": []})  # type: ignore[return-value]

    def save_positions(self, paper: list[Position], live: list[Position]) -> None:
        write_json(
            self.path,
            {
                "paper": [position.model_dump(mode="json") for position in paper],
                "live": [position.model_dump(mode="json") for position in live],
            },
        )

    def positions_for_mode(self, mode: Mode) -> list[Position]:
        payload = self.load()
        key = "paper" if mode == Mode.PAPER else "live"
        return [Position.model_validate(item) for item in payload.get(key, [])]
