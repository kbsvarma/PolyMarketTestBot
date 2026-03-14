from __future__ import annotations

from src.models import ReconciliationIssue, ReconciliationSummary


def _normalize_order_status(value: object) -> str:
    status = str(value or "").upper()
    mapping = {
        "LIVE": "RESTING",
        "OPEN": "RESTING",
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


def _normalize_order_quantity(value: object) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    scaled = int(numeric * 100)
    return scaled / 100.0


def reconcile_live_state(local_positions: list[dict], exchange_positions: list[dict], local_orders: list[dict], exchange_orders: list[dict]) -> ReconciliationSummary:
    issues: list[ReconciliationIssue] = []

    local_position_keys = {
        (
            str(item.get("market_id")),
            str(item.get("token_id")),
            str(item.get("side", "BUY")),
            round(float(item.get("quantity", 0.0)), 6),
        )
        for item in local_positions
        if not item.get("closed")
    }
    exchange_position_keys = {
        (
            str(item.get("market_id") or item.get("conditionId") or item.get("market")),
            str(item.get("token_id") or item.get("asset") or item.get("tokenId")),
            str(item.get("side") or "BUY"),
            round(float(item.get("quantity") or item.get("size") or item.get("shares") or 0.0), 6),
        )
        for item in exchange_positions
    }
    if local_position_keys != exchange_position_keys:
        issues.append(
            ReconciliationIssue(
                severity="SEVERE",
                issue_type="POSITION_MISMATCH",
                detail=f"local={sorted(local_position_keys)} exchange={sorted(exchange_position_keys)}",
            )
        )

    local_order_keys = {
        (
            str(item.get("exchange_order_id") or item.get("client_order_id") or item.get("local_order_id")),
            "",
            _normalize_local_order_status(item),
            _normalize_order_quantity(item.get("filled_size")),
            _normalize_order_quantity(item.get("remaining_size")),
            str(item.get("linked_position_id") or ""),
        )
        for item in local_orders
        if not item.get("terminal_state")
    }
    exchange_order_keys = {
        (
            str(item.get("id") or item.get("orderID") or item.get("exchange_order_id") or item.get("client_order_id")),
            "",
            _normalize_order_status(item.get("status") or item.get("state") or ""),
            _normalize_order_quantity(item.get("filled_size") or item.get("filled")),
            _normalize_order_quantity(_normalize_exchange_remaining(item)),
            "",
        )
        for item in exchange_orders
    }
    if local_order_keys != exchange_order_keys:
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
