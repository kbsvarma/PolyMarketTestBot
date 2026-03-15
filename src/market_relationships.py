from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

from src.models import MarketInfo


_STOPWORDS = {
    "will",
    "the",
    "a",
    "an",
    "be",
    "before",
    "by",
    "in",
    "on",
    "of",
    "to",
    "for",
    "this",
    "that",
    "and",
    "or",
    "is",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "quarter",
    "q1",
    "q2",
    "q3",
    "q4",
    "2024",
    "2025",
    "2026",
    "2027",
}

_TOKEN_SYNONYMS = {
    "reduce": "cut",
    "reduction": "cut",
    "decrease": "cut",
    "cuts": "cut",
    "raises": "raise",
    "increase": "raise",
    "hike": "raise",
    "hikes": "raise",
}


def _base_text(value: str) -> str:
    text = re.sub(r"\[[^\]]+\]", " ", value.lower())
    text = re.sub(r"\b(20\d{2}|q[1-4])\b", " ", text)
    return text


def relationship_tokens(market: MarketInfo) -> list[str]:
    base = _base_text(market.slug or market.title)
    tokens = re.findall(r"[a-z0-9]+", base)
    normalized = [_TOKEN_SYNONYMS.get(token, token) for token in tokens if token not in _STOPWORDS and not token.isdigit()]
    return normalized[:10]


def normalize_market_relationship_key(market: MarketInfo) -> str:
    normalized = relationship_tokens(market)
    return "-".join(normalized[:8])


def relationship_keys(market: MarketInfo) -> set[str]:
    tokens = relationship_tokens(market)
    keys: set[str] = set()
    unique_tokens = list(dict.fromkeys(tokens))
    if len(tokens) >= 2:
        keys.add("-".join(tokens[:2]))
    if len(tokens) >= 3:
        keys.add("-".join(tokens[:3]))
    if len(tokens) >= 4:
        keys.add("-".join(tokens[:4]))
    if len(unique_tokens) >= 2:
        keys.add("-".join(sorted(unique_tokens[:2])))
    if len(unique_tokens) >= 3:
        keys.add("-".join(sorted(unique_tokens[:3])))
    full_key = normalize_market_relationship_key(market)
    if full_key:
        keys.add(full_key)
    return {key for key in keys if key}


def markets_are_related(left: MarketInfo, right: MarketInfo) -> bool:
    left_tokens = set(relationship_tokens(left))
    right_tokens = set(relationship_tokens(right))
    if not left_tokens or not right_tokens:
        return False
    shared = left_tokens & right_tokens
    smaller = min(len(left_tokens), len(right_tokens))
    overlap_ratio = len(shared) / max(smaller, 1)
    union_ratio = len(shared) / max(len(left_tokens | right_tokens), 1)
    left_category = (left.category or "").lower()
    right_category = (right.category or "").lower()
    if "sports" in left_category or "sports" in right_category:
        return len(shared) >= 4 and overlap_ratio >= 0.8 and union_ratio >= 0.6
    return overlap_ratio >= 0.75 and union_ratio >= 0.5


def parse_end_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def build_relationship_groups(markets: list[MarketInfo], min_group_size: int) -> list[list[MarketInfo]]:
    groups: dict[tuple[str, str], list[MarketInfo]] = defaultdict(list)
    for market in markets:
        if not market.market_id or not market.token_id or market.closed or not market.active:
            continue
        category = (market.category or "").lower()
        # Sports futures create lots of noisy same-team cross-competition groupings that
        # are not useful for this dislocation heuristic.
        if "sports" in category:
            continue
        for key in relationship_keys(market):
            groups[(market.category, key)].append(market)
    candidates: list[list[MarketInfo]] = []
    seen_market_sets: set[tuple[str, ...]] = set()
    for group in groups.values():
        deduped = {market.market_id: market for market in group}
        candidate = list(deduped.values())
        if len(candidate) < min_group_size:
            continue
        unique_end_dates = {parse_end_date(item.end_date_iso) for item in candidate if parse_end_date(item.end_date_iso) is not None}
        if len(unique_end_dates) < 2:
            continue
        candidate.sort(key=lambda item: (parse_end_date(item.end_date_iso) or datetime.max, item.market_id))
        market_ids = tuple(item.market_id for item in candidate)
        if market_ids in seen_market_sets:
            continue
        seen_market_sets.add(market_ids)
        candidates.append(candidate)
    return candidates


def find_best_dislocation_pair(
    group: list[MarketInfo],
    mid_prices: dict[str, float],
    min_gap_pct: float,
) -> tuple[MarketInfo, MarketInfo, float] | None:
    best_pair: tuple[MarketInfo, MarketInfo, float] | None = None
    for earlier_index, earlier in enumerate(group):
        earlier_mid = mid_prices.get(earlier.market_id)
        earlier_end = parse_end_date(earlier.end_date_iso)
        if earlier_mid is None or earlier_end is None:
            continue
        for later in group[earlier_index + 1 :]:
            later_mid = mid_prices.get(later.market_id)
            later_end = parse_end_date(later.end_date_iso)
            if later_mid is None or later_end is None or later_end <= earlier_end:
                continue
            if not markets_are_related(earlier, later):
                continue
            gap = earlier_mid - later_mid
            if gap < min_gap_pct:
                continue
            confidence = gap / max(earlier_mid, 1e-6)
            if best_pair is None or confidence > best_pair[2]:
                best_pair = (earlier, later, round(confidence, 6))
    return best_pair
