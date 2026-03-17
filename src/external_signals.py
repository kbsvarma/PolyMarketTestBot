from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from src.models import MarketInfo
from src.utils import read_json


_STOPWORDS = {
    "a",
    "an",
    "and",
    "before",
    "by",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "will",
}


class OfficialSignalStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict[str, Any]]:
        payload = read_json(self.path, [])
        if not isinstance(payload, list):
            return []
        rows: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            rows.append(item)
        return rows


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))).strip()


def _tokenize(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if token and token not in _STOPWORDS and not token.isdigit()
    }


def _outcome_matches(market: MarketInfo, outcome: str) -> bool:
    if not outcome:
        return True
    normalized_outcome = _normalize_text(outcome)
    if not normalized_outcome:
        return True
    return normalized_outcome in _normalize_text(market.title)


def resolve_official_signal_market(row: dict[str, Any], markets: list[MarketInfo]) -> tuple[MarketInfo | None, str]:
    market_id = str(row.get("market_id") or "").strip()
    token_id = str(row.get("token_id") or "").strip()
    market_slug = str(row.get("market_slug") or row.get("slug") or "").strip()
    outcome = str(row.get("outcome") or "").strip()
    category = str(row.get("category") or "").strip().lower()
    title_query = str(
        row.get("market_title")
        or row.get("title")
        or row.get("market_title_contains")
        or row.get("market_query")
        or ""
    ).strip()
    keyword_tokens = _tokenize(row.get("market_keywords") or title_query)
    title_query_normalized = _normalize_text(title_query)

    candidates = [market for market in markets if market.active and not market.closed]
    if category:
        filtered = [market for market in candidates if market.category.lower() == category]
        if filtered:
            candidates = filtered

    if market_id and token_id:
        if not candidates:
            return (
                MarketInfo(
                    market_id=market_id,
                    token_id=token_id,
                    title=str(row.get("market_title") or row.get("title") or market_id),
                    slug=str(row.get("market_slug") or row.get("slug") or ""),
                    category=str(row.get("category") or "unknown"),
                ),
                "exact_market_and_token",
            )
        for market in candidates:
            if market.market_id == market_id and market.token_id == token_id:
                return market, "exact_market_and_token"
        # Market not in the local cache but has explicit IDs (e.g. momentum signal for
        # a short-term market not returned by the standard Gamma /markets endpoint).
        # Trust the explicit IDs and create a synthetic stub so the strategy can proceed.
        return (
            MarketInfo(
                market_id=market_id,
                token_id=token_id,
                title=str(row.get("market_title") or row.get("title") or market_id),
                slug=str(row.get("market_slug") or row.get("slug") or ""),
                category=str(row.get("category") or "unknown"),
            ),
            "exact_market_and_token",
        )

    if token_id:
        token_matches = [market for market in candidates if market.token_id == token_id and _outcome_matches(market, outcome)]
        if len(token_matches) == 1:
            return token_matches[0], "token_id"
        if len(token_matches) > 1:
            return None, "ambiguous_token_id"

    if market_id:
        market_matches = [market for market in candidates if market.market_id == market_id and _outcome_matches(market, outcome)]
        if len(market_matches) == 1:
            return market_matches[0], "market_id"
        if len(market_matches) > 1:
            return None, "ambiguous_market_id"

    if market_slug:
        slug_matches = [market for market in candidates if market.slug == market_slug and _outcome_matches(market, outcome)]
        if len(slug_matches) == 1:
            return slug_matches[0], "market_slug"
        if len(slug_matches) > 1:
            return None, "ambiguous_market_slug"

    if not title_query_normalized and not keyword_tokens:
        return None, "missing_mapping_hints"

    scored: list[tuple[float, MarketInfo]] = []
    for market in candidates:
        if not _outcome_matches(market, outcome):
            continue
        haystack = f"{market.title} {market.slug}"
        haystack_normalized = _normalize_text(haystack)
        haystack_tokens = _tokenize(haystack)
        overlap = len(keyword_tokens & haystack_tokens)
        coverage = overlap / max(len(keyword_tokens), 1)
        exact_phrase = 1.0 if title_query_normalized and title_query_normalized in haystack_normalized else 0.0
        if exact_phrase == 0.0 and coverage < 0.75:
            continue
        score = exact_phrase + coverage + min(market.liquidity / 100_000.0, 0.25)
        scored.append((score, market))

    if not scored:
        return None, "no_market_match"
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_market = scored[0]
    if len(scored) > 1 and best_score - scored[1][0] < 0.20:
        return None, "ambiguous_title_match"
    return best_market, "title_match"


def mapping_confidence(mapping_reason: str) -> float:
    return {
        "exact_market_and_token": 1.0,
        "token_id": 0.97,
        "market_id": 0.94,
        "market_slug": 0.9,
        "direct_market_lookup": 0.88,
        "title_match": 0.82,
    }.get(mapping_reason, 0.0)


def parse_signal_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
