"""
Momentum signal generator for the event_driven_official strategy.

Scans recent public Polymarket trades to find markets with strong directional
momentum, then writes qualified signals to official_event_signals.json so the
event_driven_official strategy can act on them.

Key design: works directly from public activity data (conditionId + asset tokens)
WITHOUT requiring markets to be in the bot's 1000-market cache.  This catches the
short-term crypto, weather, and sports-total markets that are actively traded
but not surfaced by the standard Gamma markets endpoint.

Runs automatically inside the periodic wallet-rediscovery phase (every 5 minutes).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Tuning ────────────────────────────────────────────────────────────────────
_WINDOW_MINUTES = 15          # rolling trade window (shorter = fewer stale signals)
_MIN_TRADE_USD = 10.0         # filter dust trades
_MIN_BUY_COUNT = 2            # need ≥2 qualifying buys
_MIN_BUY_VOLUME_USD = 30.0    # total buy volume in window
_PRICE_FLOOR = 0.20           # don't signal near-0 outcomes
_PRICE_CEILING = 0.62         # stay under hard_skip_price_ceiling=0.65
_BASE_CONFIDENCE = 0.62
_SOURCE_RELIABILITY = 0.75
_FAIR_PRICE_UPLIFT = 0.08     # bullish momentum → fair price 8 cents above
_MAX_FAIR_PRICE = 0.90
# Short-term markets (5-minute BTC up/down) resolve quickly. Only emit a signal
# if the newest qualifying buy happened within this many minutes — otherwise the
# market may already have closed by the time the bot processes the signal.
_MAX_SIGNAL_LATENCY_MINUTES = 4
# Momentum signals should expire quickly since the underlying market is short-lived.
_MOMENTUM_SIGNAL_MAX_AGE_MINUTES = 8
# ─────────────────────────────────────────────────────────────────────────────


async def generate_momentum_signals(
    config: Any,
    market_data: Any,
    output_path: Path,
) -> int:
    """
    Generate momentum signals from recent public activity and persist them.
    Returns the number of NEW signals written this call.
    """
    now = datetime.now(timezone.utc)
    client = market_data.client

    # ── Load + prune existing signals ────────────────────────────────────────
    existing: list[dict] = []
    if output_path.exists():
        try:
            raw = json.loads(output_path.read_text())
            existing = [r for r in (raw if isinstance(raw, list) else []) if isinstance(r, dict)]
        except Exception:
            existing = []
    # Momentum signals expire quickly — use the shorter per-generator max age,
    # not the (potentially long) config value meant for slow-moving external signals.
    max_age_s = _MOMENTUM_SIGNAL_MAX_AGE_MINUTES * 60
    fresh = [r for r in existing if _signal_age_seconds(r, now) < max_age_s]

    # ── Fetch recent public trades ────────────────────────────────────────────
    try:
        activity: list[dict] = await client.fetch_recent_public_activity(limit=400)
    except Exception as exc:
        logger.warning("momentum_signal_generator: activity fetch failed – %s", exc)
        return 0

    if not activity:
        return 0

    # ── Group BUY trades by (conditionId, asset) in the rolling window ───────
    window_start = now - timedelta(minutes=_WINDOW_MINUTES)
    # Key: (condition_id, token_id)
    token_buys: dict[tuple[str, str], list[dict]] = {}
    # Also keep a lookup: token_id → metadata from the activity record
    token_meta: dict[tuple[str, str], dict] = {}

    for trade in activity:
        ts = _parse_ts(trade)
        if ts is None or ts < window_start:
            continue

        # Accept buy / yes side only
        side = str(trade.get("side") or trade.get("outcomeIndex") or "").lower()
        if side not in {"buy", "yes", "0", "maker_buy"}:
            continue

        usd = _to_float(
            trade.get("usd_amount")
            or trade.get("amount")
            or trade.get("size")
            or trade.get("value")
        )
        if usd < _MIN_TRADE_USD:
            continue

        price = _to_float(trade.get("price") or trade.get("avg_price"))
        if not (_PRICE_FLOOR <= price <= _PRICE_CEILING):
            continue

        cond_id = str(trade.get("conditionId") or trade.get("market_id") or "").strip()
        token_id = str(trade.get("asset") or trade.get("token_id") or trade.get("asset_id") or "").strip()
        if not cond_id or not token_id:
            continue

        key = (cond_id, token_id)
        token_buys.setdefault(key, []).append({"ts": ts, "usd": usd, "price": price})
        if key not in token_meta:
            token_meta[key] = {
                "title": str(trade.get("title") or ""),
                "slug": str(trade.get("slug") or ""),
                "category": _infer_category(trade),
            }

    if not token_buys:
        logger.debug("momentum_signal_generator: no qualifying buy activity in window")
        return 0

    # ── Score and emit signals ────────────────────────────────────────────────
    new_signals: list[dict] = []
    existing_keys = {(r.get("market_id", ""), r.get("token_id", "")) for r in fresh}

    latency_cutoff = now - timedelta(minutes=_MAX_SIGNAL_LATENCY_MINUTES)

    for (cond_id, token_id), buys in token_buys.items():
        if len(buys) < _MIN_BUY_COUNT:
            continue

        total_vol = sum(b["usd"] for b in buys)
        if total_vol < _MIN_BUY_VOLUME_USD:
            continue

        if (cond_id, token_id) in existing_keys:
            continue

        # Only emit if the most recent qualifying buy is fresh — short-term markets
        # (BTC up/down 5-min windows) may have already closed if newest trade is old.
        newest_buy_ts = max(b["ts"] for b in buys)
        if newest_buy_ts < latency_cutoff:
            logger.debug(
                "momentum_signal_generator: skip stale market %s (newest buy %s)",
                cond_id[:24],
                newest_buy_ts.isoformat(),
            )
            continue

        meta = token_meta.get((cond_id, token_id), {})
        buy_count = len(buys)
        avg_price = sum(b["price"] * b["usd"] for b in buys) / max(total_vol, 1e-9)
        fair_price = min(avg_price + _FAIR_PRICE_UPLIFT, _MAX_FAIR_PRICE)
        edge = fair_price - avg_price

        confidence = min(
            0.95,
            _BASE_CONFIDENCE + buy_count * 0.04 + min(total_vol / 500.0, 0.15),
        )

        if confidence < config.strategies.official_signal_min_confidence:
            continue
        if _SOURCE_RELIABILITY < config.strategies.official_signal_min_source_reliability:
            continue
        if edge < config.strategies.official_signal_min_edge_pct:
            continue

        title = meta.get("title") or cond_id[:32]
        signal: dict[str, Any] = {
            "event_id": f"momentum_{cond_id[:16]}_{now.strftime('%Y%m%d%H%M')}",
            "published_at": now.isoformat(),
            "market_id": cond_id,
            "token_id": token_id,
            "title": title,
            "market_title": title,
            "market_slug": meta.get("slug") or "",
            "category": meta.get("category") or "unknown",
            "outcome": "Yes",
            "fair_price": round(fair_price, 4),
            "source_price": round(avg_price, 4),
            "source_reliability": round(_SOURCE_RELIABILITY, 3),
            "confidence_score": round(confidence, 3),
            "notional_usd": min(float(config.risk.max_single_live_trade_usd), 5.0),
            "rationale": (
                f"On-chain momentum: {buy_count} buys / ${total_vol:.0f} "
                f"in last {_WINDOW_MINUTES}min at avg {avg_price:.3f}. "
                f"fair={fair_price:.3f} edge={edge:.3f}"
            ),
        }
        new_signals.append(signal)
        existing_keys.add((cond_id, token_id))

    # ── Persist ───────────────────────────────────────────────────────────────
    all_signals = fresh + new_signals
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_signals, indent=2, default=str))

    if new_signals:
        logger.info(
            "momentum_signal_generator: +%d new signals (%d total active)",
            len(new_signals),
            len(all_signals),
        )
    else:
        logger.debug(
            "momentum_signal_generator: 0 new this pass (%d fresh existing)",
            len(fresh),
        )

    return len(new_signals)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_category(trade: dict) -> str:
    title = str(trade.get("title") or "").lower()
    if any(x in title for x in ["bitcoin", "btc", "eth", "xrp", "crypto", "up or down"]):
        return "crypto price"
    if any(x in title for x in ["nba", "nfl", "nhl", "mlb", "soccer", "tennis", "golf",
                                  "lakers", "celtics", "warriors", "knicks"]):
        return "sports"
    if any(x in title for x in ["temperature", "weather", "rain", "snow"]):
        return "macro / economics"
    if any(x in title for x in ["election", "president", "congress", "senate", "vote", "poll"]):
        return "politics"
    return "unknown"


def _signal_age_seconds(row: dict, now: datetime) -> float:
    from src.external_signals import parse_signal_timestamp
    ts = parse_signal_timestamp(row.get("published_at") or row.get("updated_at"))
    if ts is None:
        return float("inf")
    return max((now - ts).total_seconds(), 0.0)


def _parse_ts(trade: dict) -> datetime | None:
    for key in ("timestamp", "created_at", "ts", "time", "date", "updatedAt", "createdAt"):
        val = trade.get(key)
        if not val:
            continue
        try:
            if isinstance(val, (int, float)):
                t = float(val)
                if t > 1e12:
                    t /= 1000.0
                return datetime.fromtimestamp(t, tz=timezone.utc)
            text = str(val).strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (ValueError, OSError):
            continue
    return None


def _to_float(val: Any) -> float:
    try:
        return float(val or 0.0)
    except (TypeError, ValueError):
        return 0.0
