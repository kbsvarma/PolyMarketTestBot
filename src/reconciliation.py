from __future__ import annotations

from datetime import datetime, timezone

from src.models import ReconciliationIssue, ReconciliationSummary


RECENT_EXCHANGE_VISIBILITY_GRACE_SECONDS = 90.0


def _normalize_order_status(value: object) -> str:
    status = str(value or "").upper()
    mapping = {
        "LIVE": "RESTING",
        "OPEN": "RESTING",
        "MATCHED": "FILLED",
        "CANCELED": "CANCELLED",
        "CANCELED_MARKET_RESOLVED": "CANCELLED",
    }
    return mapping.get(status, status)


def _normalize_local_order_status(item: dict) -> str:
    lifecycle = str(item.get("lifecycle_status") or "")
    if lifecycle.upper() == "UNKNOWN" and item.get("last_exchange_status"):
        return _normalize_order_status(item.get("last_exchange_status"))
    return _normalize_order_status(lifecycle or item.get("last_exchange_status"))


def _normalize_exchange_remaining(item: dict) -> float:
    remaining = item.get("remaining_size")
    if remaining in (None, ""):
        remaining = item.get("remaining")
    try:
        numeric = float(remaining or 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric > 0:
        return round(numeric, 6)

    try:
        original_size = float(item.get("original_size") or 0.0)
        size_matched = float(item.get("size_matched") or 0.0)
    except (TypeError, ValueError):
        original_size = 0.0
        size_matched = 0.0
    if original_size > 0:
        return round(max(original_size - size_matched, 0.0), 6)
    return 0.0


def _normalize_exchange_filled(item: dict) -> float:
    try:
        numeric = float(item.get("filled_size") or item.get("filled") or item.get("size_matched") or 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    return round(numeric, 6)


def _normalize_order_quantity(value: object) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    scaled = int(numeric * 100)
    return scaled / 100.0


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_recent_exchange_lag(value: object, now: datetime) -> bool:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return False
    return (now - parsed).total_seconds() <= RECENT_EXCHANGE_VISIBILITY_GRACE_SECONDS


def _local_position_key(item: dict) -> tuple[str, str, str, float]:
    quantity_source = item.get("remaining_size")
    if quantity_source in (None, "", 0, 0.0):
        quantity_source = item.get("quantity", 0.0)
    return (
        str(item.get("market_id")),
        str(item.get("token_id")),
        str(item.get("side", "BUY")),
        round(float(quantity_source or 0.0), 6),
    )


def _local_position_open_quantity(item: dict) -> float:
    quantity_source = item.get("remaining_size")
    if quantity_source in (None, ""):
        quantity_source = item.get("quantity", 0.0)
    try:
        return round(float(quantity_source or 0.0), 6)
    except (TypeError, ValueError):
        return 0.0


def _exchange_position_key(item: dict) -> tuple[str, str, str, float]:
    return (
        str(item.get("market_id") or item.get("conditionId") or item.get("market")),
        str(item.get("token_id") or item.get("asset") or item.get("asset_id") or item.get("tokenId")),
        str(item.get("side") or "BUY"),
        round(float(item.get("quantity") or item.get("size") or item.get("shares") or 0.0), 6),
    )


def _local_order_key(item: dict) -> tuple[str, str, str, float, float, str]:
    return (
        str(item.get("exchange_order_id") or item.get("client_order_id") or item.get("local_order_id")),
        "",
        _normalize_local_order_status(item),
        _normalize_order_quantity(item.get("filled_size")),
        _normalize_order_quantity(item.get("remaining_size")),
        "",
    )


def _exchange_order_key(item: dict) -> tuple[str, str, str, float, float, str]:
    return (
        str(item.get("id") or item.get("orderID") or item.get("exchange_order_id") or item.get("client_order_id")),
        "",
        _normalize_order_status(item.get("status") or item.get("state") or ""),
        _normalize_order_quantity(_normalize_exchange_filled(item)),
        _normalize_order_quantity(_normalize_exchange_remaining(item)),
        "",
    )


def _should_grace_local_position(item: dict, now: datetime) -> bool:
    return any(
        _is_recent_exchange_lag(item.get(field), now)
        for field in ("opened_at", "entry_time", "last_reconciled_at")
    )


def _should_grace_local_order(item: dict, now: datetime) -> bool:
    status = _normalize_local_order_status(item)
    if status not in {"DELAYED", "SUBMITTED", "ACKNOWLEDGED", "RESTING", "PARTIALLY_FILLED"}:
        return False
    return any(
        _is_recent_exchange_lag(item.get(field), now)
        for field in ("submitted_at", "created_at", "last_update_at")
    )


def reconcile_live_state(local_positions: list[dict], exchange_positions: list[dict], local_orders: list[dict], exchange_orders: list[dict]) -> ReconciliationSummary:
    issues: list[ReconciliationIssue] = []
    now = datetime.now(timezone.utc)

    local_open_positions = [
        item
        for item in local_positions
        if not item.get("closed") and _local_position_open_quantity(item) > 0
    ]
    local_position_keys = {_local_position_key(item) for item in local_open_positions}
    exchange_position_keys = {_exchange_position_key(item) for item in exchange_positions}
    unmatched_local_positions = [
        item for item in local_open_positions if _local_position_key(item) not in exchange_position_keys
    ]
    unmatched_exchange_positions = [
        item for item in exchange_positions if _exchange_position_key(item) not in local_position_keys
    ]
    significant_unmatched_local_positions = [
        item for item in unmatched_local_positions if not _should_grace_local_position(item, now)
    ]
    if significant_unmatched_local_positions or unmatched_exchange_positions:
        issues.append(
            ReconciliationIssue(
                severity="SEVERE",
                issue_type="POSITION_MISMATCH",
                detail=f"local={sorted(local_position_keys)} exchange={sorted(exchange_position_keys)}",
            )
        )

    local_open_orders = [item for item in local_orders if not item.get("terminal_state")]
    local_order_keys = {_local_order_key(item) for item in local_open_orders}
    exchange_order_keys = {_exchange_order_key(item) for item in exchange_orders}
    unmatched_local_orders = [
        item for item in local_open_orders if _local_order_key(item) not in exchange_order_keys
    ]
    unmatched_exchange_orders = [
        item for item in exchange_orders if _exchange_order_key(item) not in local_order_keys
    ]
    significant_unmatched_local_orders = [
        item for item in unmatched_local_orders if not _should_grace_local_order(item, now)
    ]
    if significant_unmatched_local_orders or unmatched_exchange_orders:
        issues.append(
            ReconciliationIssue(
                severity="SEVERE",
                issue_type="ORDER_MISMATCH",
                detail=f"local={sorted(local_order_keys)} exchange={sorted(exchange_order_keys)}",
                local_ref=str(sorted(local_order_keys)),
                exchange_ref=str(sorted(exchange_order_keys)),
            )
        )

    unresolved_unknowns = [
        item
        for item in local_orders
        if _normalize_local_order_status(item) == "UNKNOWN"
        and not item.get("terminal_state")
    ]
    if unresolved_unknowns:
        issues.append(
            ReconciliationIssue(
                severity="SEVERE",
                issue_type="UNKNOWN_ORDER_STATE",
                detail=f"Unresolved unknown order states: {[item.get('local_order_id') for item in unresolved_unknowns]}",
            )
        )

    severity = "NONE" if not issues else ("SEVERE" if any(issue.severity == "SEVERE" for issue in issues) else "WARN")
    return ReconciliationSummary(clean=not issues, severity=severity, issues=issues)
