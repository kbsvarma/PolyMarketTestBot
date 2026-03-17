"""
Bayesian completion probability tracker for legged pair positions.

When we open leg-1 of a paired trade (e.g. buy YES), we need to know:
  - What is the current probability that leg-2 will complete before expiry?
  - Should we abort (flatten leg-1) if the probability drops too low?

This module implements a lightweight Bayesian belief updater:
  - Prior: P(complete) based on current price distance, vol, and time.
  - Update: shift posterior toward 0 if price moves against us, toward 1 if favourable.
  - Abort threshold: if posterior < abort_threshold, emit ABORT signal.

The model is intentionally simple (no full particle filter):
  - Uses a Beta distribution as the conjugate prior for P(complete).
  - Observations are "favorable" (price moved toward target) or "adverse".
  - The posterior Beta(alpha, beta) is updated each tick.
  - Posterior mean = alpha / (alpha + beta) = our current belief.

This replaces the current "hope it bounces" logic in the bot.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Default thresholds (overridden in config)
# ---------------------------------------------------------------------------

DEFAULT_ABORT_THRESHOLD: float = 0.40     # abort if P(complete) falls below this
DEFAULT_ENTRY_THRESHOLD: float = 0.70     # only enter leg-1 if P(complete) > this
DEFAULT_DEADLINE_FRACTION: float = 0.60   # abort deadline at 60 % of time-to-expiry


# ---------------------------------------------------------------------------
# First-Passage-Time inspired prior
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun approximation)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    cdf = 1.0 - pdf * poly
    return cdf if x >= 0 else 1.0 - cdf


def fpt_completion_probability(
    *,
    current_price: float,
    target_price: float,
    time_remaining_seconds: float,
    realized_vol_per_second: float,
    drift_per_second: float = 0.0,
) -> float:
    """
    First-Passage-Time inspired P(price hits target before expiry).

    Uses the analytical formula for GBM hitting a constant barrier:
        P(hit) ≈ 2 * Phi(-|d| / sigma_total)
    where:
        d           = target_price - current_price   (distance)
        sigma_total = realized_vol_per_second * sqrt(T)

    When drift ≠ 0, we use the absorbing-barrier approximation.
    """
    if time_remaining_seconds <= 0 or realized_vol_per_second <= 0:
        return 0.0 if current_price < target_price else 1.0

    dist = target_price - current_price
    sigma_t = realized_vol_per_second * math.sqrt(time_remaining_seconds)
    drift_t = drift_per_second * time_remaining_seconds

    if sigma_t < 1e-9:
        return 1.0 if dist <= 0 else 0.0

    # Probability of hitting target: P(max_t X_t >= target) for up move
    # Simplified: use distance-to-vol ratio
    standardised = (dist - drift_t) / sigma_t
    # One-sided: probability that price reaches target at some point in [0, T]
    p_hit = max(0.0, min(1.0, 2.0 * _norm_cdf(-abs(standardised))))
    return round(p_hit, 6)


# ---------------------------------------------------------------------------
# Beta-distributed belief state
# ---------------------------------------------------------------------------

@dataclass
class BetaBelief:
    """
    Beta(alpha, beta) belief over P(completion).

    alpha   ~ pseudo-count of "favorable" observations
    beta    ~ pseudo-count of "adverse" observations
    """
    alpha: float = 2.0   # weakly optimistic prior
    beta: float = 1.0

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        a, b, n = self.alpha, self.beta, self.alpha + self.beta
        return (a * b) / (n * n * (n + 1))

    def update_favorable(self, strength: float = 1.0) -> None:
        """Shift belief toward P=1 by *strength* pseudo-observations."""
        self.alpha += max(strength, 0.0)

    def update_adverse(self, strength: float = 1.0) -> None:
        """Shift belief toward P=0 by *strength* pseudo-observations."""
        self.beta += max(strength, 0.0)

    def update_from_price_move(
        self,
        *,
        price_move_toward_target: float,
        vol_normaliser: float = 0.01,
    ) -> None:
        """
        Update belief based on a price move relative to normalised vol.

        Positive price_move_toward_target → favorable.
        Negative → adverse.
        """
        if vol_normaliser <= 0:
            return
        strength = abs(price_move_toward_target) / vol_normaliser
        strength = min(strength, 5.0)  # cap to avoid single-observation dominance
        if price_move_toward_target >= 0:
            self.update_favorable(strength)
        else:
            self.update_adverse(strength)

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "mean": round(self.mean(), 4),
            "variance": round(self.variance(), 6),
        }


# ---------------------------------------------------------------------------
# Per-position completion tracker
# ---------------------------------------------------------------------------

@dataclass
class CompletionState:
    """Tracks Bayesian completion belief for a single legged position."""
    position_id: str
    market_id: str
    token_id: str
    target_price: float              # leg-2 threshold price
    entry_price: float               # leg-1 execution price
    expiry_epoch: float              # unix epoch seconds when market resolves
    created_at: float = field(default_factory=time.time)

    belief: BetaBelief = field(default_factory=lambda: BetaBelief(alpha=2.0, beta=1.0))
    last_price: float = 0.0
    last_updated_at: float = field(default_factory=time.time)
    tick_count: int = 0
    abort_emitted: bool = False

    abort_threshold: float = DEFAULT_ABORT_THRESHOLD
    deadline_fraction: float = DEFAULT_DEADLINE_FRACTION

    def time_remaining_seconds(self, now: float | None = None) -> float:
        now = now or time.time()
        return max(self.expiry_epoch - now, 0.0)

    def total_window_seconds(self) -> float:
        return max(self.expiry_epoch - self.created_at, 1.0)

    def fraction_elapsed(self, now: float | None = None) -> float:
        now = now or time.time()
        elapsed = now - self.created_at
        return min(elapsed / self.total_window_seconds(), 1.0)

    def past_deadline(self, now: float | None = None) -> bool:
        """True if we've passed the configurable abort deadline fraction."""
        return self.fraction_elapsed(now) >= self.deadline_fraction

    def tick(
        self,
        current_price: float,
        *,
        realized_vol_per_second: float = 0.005,
        drift_per_second: float = 0.0,
        now: float | None = None,
    ) -> "CompletionResult":
        """
        Process a new price observation and return the updated belief.

        Parameters
        ----------
        current_price           : latest mark/executable price of the position
        realized_vol_per_second : short-window vol estimate
        drift_per_second        : optional drift estimate
        """
        now = now or time.time()
        t_remaining = self.time_remaining_seconds(now)
        self.tick_count += 1

        # Update belief from price movement
        if self.last_price > 0:
            price_move = current_price - self.last_price
            move_toward_target = price_move if self.target_price > self.entry_price else -price_move
            self.belief.update_from_price_move(
                price_move_toward_target=move_toward_target,
                vol_normaliser=max(realized_vol_per_second, 1e-6),
            )

        self.last_price = current_price
        self.last_updated_at = now

        # FPT prior for comparison / logging
        fpt_p = fpt_completion_probability(
            current_price=current_price,
            target_price=self.target_price,
            time_remaining_seconds=t_remaining,
            realized_vol_per_second=realized_vol_per_second,
            drift_per_second=drift_per_second,
        )

        posterior_mean = self.belief.mean()

        # Blend FPT prior with running Beta posterior (equal weight first tick)
        blend_weight = min(self.tick_count / max(10.0, float(self.tick_count)), 0.8)
        blended_p = round(blend_weight * posterior_mean + (1.0 - blend_weight) * fpt_p, 6)

        should_abort = (
            blended_p < self.abort_threshold
            or self.past_deadline(now)
            or t_remaining <= 0
        )

        if should_abort and not self.abort_emitted:
            self.abort_emitted = True

        return CompletionResult(
            position_id=self.position_id,
            current_price=current_price,
            target_price=self.target_price,
            time_remaining_seconds=round(t_remaining, 1),
            fpt_probability=fpt_p,
            posterior_mean=posterior_mean,
            blended_probability=blended_p,
            should_abort=should_abort,
            abort_threshold=self.abort_threshold,
            fraction_elapsed=round(self.fraction_elapsed(now), 4),
            past_deadline=self.past_deadline(now),
            belief=self.belief.to_dict(),
            tick_count=self.tick_count,
        )


@dataclass
class CompletionResult:
    position_id: str
    current_price: float
    target_price: float
    time_remaining_seconds: float
    fpt_probability: float
    posterior_mean: float
    blended_probability: float
    should_abort: bool
    abort_threshold: float
    fraction_elapsed: float
    past_deadline: bool
    belief: dict[str, float]
    tick_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "current_price": round(self.current_price, 6),
            "target_price": round(self.target_price, 6),
            "time_remaining_seconds": self.time_remaining_seconds,
            "fpt_probability": round(self.fpt_probability, 6),
            "posterior_mean": round(self.posterior_mean, 6),
            "blended_probability": round(self.blended_probability, 6),
            "should_abort": self.should_abort,
            "abort_threshold": self.abort_threshold,
            "fraction_elapsed": self.fraction_elapsed,
            "past_deadline": self.past_deadline,
            "belief": self.belief,
            "tick_count": self.tick_count,
        }


# ---------------------------------------------------------------------------
# Registry: one tracker per open legged position
# ---------------------------------------------------------------------------

class CompletionTrackerRegistry:
    """
    Registry of CompletionState objects indexed by position_id.
    Thread-safe for asyncio single-thread model (no locks needed).
    """

    def __init__(self) -> None:
        self._trackers: dict[str, CompletionState] = {}

    def register(
        self,
        *,
        position_id: str,
        market_id: str,
        token_id: str,
        target_price: float,
        entry_price: float,
        expiry_epoch: float,
        abort_threshold: float = DEFAULT_ABORT_THRESHOLD,
        deadline_fraction: float = DEFAULT_DEADLINE_FRACTION,
    ) -> CompletionState:
        state = CompletionState(
            position_id=position_id,
            market_id=market_id,
            token_id=token_id,
            target_price=target_price,
            entry_price=entry_price,
            expiry_epoch=expiry_epoch,
            abort_threshold=abort_threshold,
            deadline_fraction=deadline_fraction,
        )
        self._trackers[position_id] = state
        return state

    def tick_all(
        self,
        price_by_token: dict[str, float],
        *,
        realized_vol_per_second: float = 0.005,
    ) -> list[CompletionResult]:
        """Update all tracked positions and return their results."""
        results: list[CompletionResult] = []
        for state in list(self._trackers.values()):
            price = price_by_token.get(state.token_id, state.last_price)
            if price <= 0:
                continue
            result = state.tick(price, realized_vol_per_second=realized_vol_per_second)
            results.append(result)
        return results

    def abort_positions(self) -> list[str]:
        """Return position IDs that should be aborted."""
        return [
            state.position_id
            for state in self._trackers.values()
            if state.abort_emitted
        ]

    def remove(self, position_id: str) -> None:
        self._trackers.pop(position_id, None)

    def remove_closed(self, open_position_ids: set[str]) -> None:
        for pid in list(self._trackers):
            if pid not in open_position_ids:
                del self._trackers[pid]

    def __len__(self) -> int:
        return len(self._trackers)


# ---------------------------------------------------------------------------
# Module-level singleton registry
# ---------------------------------------------------------------------------

_registry: CompletionTrackerRegistry | None = None


def get_completion_registry() -> CompletionTrackerRegistry:
    global _registry
    if _registry is None:
        _registry = CompletionTrackerRegistry()
    return _registry
