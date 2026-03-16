from __future__ import annotations

from pathlib import Path

from src.models import Mode, Position
from src.utils import read_json, write_json


class PositionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, list[dict]]:
        payload = read_json(self.path, {"paper": [], "live": []})  # type: ignore[assignment]
        paper = self._dedupe_positions([Position.model_validate(item) for item in payload.get("paper", [])])
        live = self._dedupe_positions([Position.model_validate(item) for item in payload.get("live", [])])
        return {
            "paper": [position.model_dump(mode="json") for position in paper],
            "live": [position.model_dump(mode="json") for position in live],
        }

    def save_positions(self, paper: list[Position], live: list[Position]) -> None:
        write_json(
            self.path,
            {
                "paper": [position.model_dump(mode="json") for position in self._dedupe_positions(paper)],
                "live": [position.model_dump(mode="json") for position in self._dedupe_positions(live)],
            },
        )

    def positions_for_mode(self, mode: Mode) -> list[Position]:
        payload = self.load()
        key = "paper" if mode == Mode.PAPER else "live"
        return [Position.model_validate(item) for item in payload.get(key, [])]

    def _dedupe_positions(self, positions: list[Position]) -> list[Position]:
        deduped: dict[str, Position] = {}
        for position in positions:
            position = self._normalize_position(position)
            key = position.position_id
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = position.model_copy(deep=True)
                continue
            deduped[key] = self._merge_position_versions(existing, position)
        return sorted(
            deduped.values(),
            key=lambda position: (
                position.entry_time or position.opened_at,
                position.opened_at,
                position.position_id,
            ),
        )

    def _normalize_position(self, position: Position) -> Position:
        normalized = position.model_copy(deep=True)
        # Once we know a token has no exit orderbook, keep it quarantined across reloads.
        if normalized.exit_state == "EXIT_DATA_UNAVAILABLE":
            normalized.source_exit_following_enabled = False
        return normalized

    def _merge_position_versions(self, current: Position, candidate: Position) -> Position:
        current_key = self._position_sort_key(current)
        candidate_key = self._position_sort_key(candidate)
        winner = candidate if candidate_key >= current_key else current
        loser = current if winner is candidate else candidate

        merged = winner.model_copy(deep=True)
        merged.opened_at = min(current.opened_at, candidate.opened_at)
        if current.entry_time and candidate.entry_time:
            merged.entry_time = min(current.entry_time, candidate.entry_time)
        else:
            merged.entry_time = current.entry_time or candidate.entry_time
        if current.closed_at and candidate.closed_at:
            merged.closed_at = max(current.closed_at, candidate.closed_at)
        else:
            merged.closed_at = current.closed_at or candidate.closed_at
        if current.last_reconciled_at and candidate.last_reconciled_at:
            merged.last_reconciled_at = max(current.last_reconciled_at, candidate.last_reconciled_at)
        else:
            merged.last_reconciled_at = current.last_reconciled_at or candidate.last_reconciled_at
        merged.entry_order_ids = list(dict.fromkeys(current.entry_order_ids + candidate.entry_order_ids))
        merged.exit_order_ids = list(dict.fromkeys(current.exit_order_ids + candidate.exit_order_ids))
        merged.remaining_size = max(current.remaining_size, candidate.remaining_size) if not merged.closed else 0.0
        merged.entry_size = max(current.entry_size, candidate.entry_size, merged.remaining_size + merged.exited_size)
        if merged.current_mark_price <= 0:
            merged.current_mark_price = max(current.current_mark_price, candidate.current_mark_price)
        if merged.entry_price_actual <= 0:
            merged.entry_price_actual = max(current.entry_price_actual, candidate.entry_price_actual)
        if not merged.source_wallet:
            merged.source_wallet = loser.source_wallet
        if "EXIT_DATA_UNAVAILABLE" in {current.exit_state, candidate.exit_state} and not merged.closed:
            merged.exit_state = "EXIT_DATA_UNAVAILABLE"
        elif not merged.exit_state:
            merged.exit_state = loser.exit_state
        # Prefer the more restrictive value so quarantined positions stay quarantined.
        merged.source_exit_following_enabled = current.source_exit_following_enabled and candidate.source_exit_following_enabled
        if merged.exit_state == "EXIT_DATA_UNAVAILABLE":
            merged.source_exit_following_enabled = False
        return merged

    def _position_sort_key(self, position: Position) -> tuple[object, int, float, float]:
        observed_at = position.last_reconciled_at or position.closed_at or position.entry_time or position.opened_at
        active_rank = 1 if (not position.closed and max(position.remaining_size, position.quantity) > 0) else 0
        return (
            observed_at,
            active_rank,
            max(position.remaining_size, position.quantity),
            position.current_mark_price,
        )
