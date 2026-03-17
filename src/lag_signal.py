"""
Oracle-aligned lag signal detector.

Strategy:
  1. Subscribe to RTDS (Binance + Chainlink prices with ms timestamps).
  2. Watch the CLOB orderbook for the same market.
  3. Fire a "lag signal" when the Polymarket CLOB price significantly
     lags a Chainlink-consistent move, after deducting fees and spread.

This is the correct framing for 5-min Up/Down markets:
  - Settlement is on Chainlink (not Binance spot).
  - We use Binance as a *leading indicator* of where Chainlink will move.
  - We only trade when the CLOB executable price is materially off vs.
    our Chainlink-derived P(Up) estimate, net of fees.

Signal logic:
  Given a 5-min Up/Down market:
    - We estimate P(Up) from the Chainlink stream drift + vol.
    - We compare to executable YES/NO ask prices.
    - If (P_estimate - ask_price) > min_net_edge, emit a BUY signal.
    - Staleness gates: skip if RTDS or CLOB data is too old.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from src.fees import taker_fee, meets_min_edge
from src.rtds_client import RTDSSnapshot


# ---------------------------------------------------------------------------
# Configurable defaults (overridden via LagSignalConfig)
# ---------------------------------------------------------------------------

# Min Binance vs Chainlink price divergence to consider a "lag" opportunity (%)
DEFAULT_MIN_PRICE_DIVERGENCE_PCT: float = 0.003   # 0.3 %

# Min Binance move over the short window to signal momentum (%)
DEFAULT_MIN_SPOT_MOVE_PCT: float = 0.003   # 0.3 %

# Max age of RTDS data before we refuse to trade (seconds)
DEFAULT_RTDS_STALENESS_MAX_SECONDS: float = 1.5

# Max age of CLOB orderbook before we refuse to trade (seconds)
DEFAULT_CLOB_STALENESS_MAX_SECONDS: float = 0.8

# Min net edge after fees to emit a signal
DEFAULT_MIN_NET_EDGE_TAKER: float = 0.020   # 2 ¢
DEFAULT_MIN_NET_EDGE_MAKER: float = 0.010   # 1 ¢

# Min Binance-Chainlink lag (ms) that we require before acting
# (Chainlink is typically a few ms behind Binance)
DEFAULT_MIN_LAG_MS: float = 50.0

# Time-remaining veto: don't lag-arb if less than N seconds remain
DEFAULT_MIN_TIME_REMAINING_SECONDS: float = 30.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LagSignalConfig:
    min_price_divergence_pct: float = DEFAULT_MIN_PRICE_DIVERGENCE_PCT
    min_spot_move_pct: float = DEFAULT_MIN_SPOT_MOVE_PCT
    rtds_staleness_max_seconds: float = DEFAULT_RTDS_STALENESS_MAX_SECONDS
    clob_staleness_max_seconds: float = DEFAULT_CLOB_STALENESS_MAX_SECONDS
    min_net_edge_taker: float = DEFAULT_MIN_NET_EDGE_TAKER
    min_net_edge_maker: float = DEFAULT_MIN_NET_EDGE_MAKER
    min_lag_ms: float = DEFAULT_MIN_LAG_MS
    min_time_remaining_seconds: float = DEFAULT_MIN_TIME_REMAINING_SECONDS


@dataclass
class LagSignalResult:
    fire: bool = False
    side: str = ""                   # "UP" or "DOWN"
    token_id: str = ""
    net_edge: float = 0.0
    gross_edge: float = 0.0
    taker_fee_paid: float = 0.0
    executable_price: float = 0.0
    estimated_fair_p: float = 0.0
    binance_price: float = 0.0
    chainlink_price: float = 0.0
    lag_ms: float = 0.0
    price_divergence_pct: float = 0.0
    rtds_staleness_seconds: float = 0.0
    clob_staleness_seconds: float = 0.0
    skip_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fire": self.fire,
            "side": self.side,
            "token_id": self.token_id,
            "net_edge": round(self.net_edge, 6),
            "gross_edge": round(self.gross_edge, 6),
            "taker_fee_paid": round(self.taker_fee_paid, 6),
            "executable_price": round(self.executable_price, 6),
            "estimated_fair_p": round(self.estimated_fair_p, 6),
            "binance_price": round(self.binance_price, 2),
            "chainlink_price": round(self.chainlink_price, 2),
            "lag_ms": round(self.lag_ms, 1),
            "price_divergence_pct": round(self.price_divergence_pct, 6),
            "rtds_staleness_seconds": round(self.rtds_staleness_seconds, 3),
            "clob_staleness_seconds": round(self.clob_staleness_seconds, 3),
            "skip_reason": self.skip_reason,
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Fair probability estimator
# ---------------------------------------------------------------------------

def _estimate_fair_p_up(
    *,
    chainlink_price: float,
    start_price: float,
    time_remaining_seconds: float,
    realized_vol_per_second: float,
) -> float:
    """
    Estimate P(Up) = P(Chainlink at T_end > Chainlink at T_start).

    Uses a simplified drift-and-vol model:
      - Assume Chainlink follows GBM with instantaneous vol = realized_vol
      - drift = 0 (neutral prior; we don't know direction)
      - P(Up) = N(d) where d = (log(S/K) / (vol * sqrt(T)))
      - S = current Chainlink price
      - K = start price (the bar we need to beat)
      - vol = realized_vol_per_second * sqrt(time_remaining_seconds)

    This gives a P(Up) that naturally converges to 0 or 1 as time runs out
    and price diverges from the starting bar, and sits near 0.50 when we're
    in the middle of the window with no movement.
    """
    if start_price <= 0 or chainlink_price <= 0 or time_remaining_seconds <= 0:
        return 0.5

    sigma = max(realized_vol_per_second * math.sqrt(time_remaining_seconds), 1e-9)
    d = math.log(chainlink_price / start_price) / sigma

    # Standard normal CDF approximation (Abramowitz & Stegun 26.2.17)
    def _norm_cdf(x: float) -> float:
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
        cdf = 1.0 - pdf * poly
        return cdf if x >= 0 else 1.0 - cdf

    return round(_norm_cdf(d), 6)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_lag_signal(
    *,
    rtds: RTDSSnapshot,
    start_price: float,
    time_remaining_seconds: float,
    realized_vol_per_second: float,
    yes_ask: float,
    no_ask: float,
    yes_token_id: str,
    no_token_id: str,
    clob_timestamp_epoch: float,
    category: str = "crypto price",
    is_taker: bool = True,
    config: LagSignalConfig | None = None,
) -> LagSignalResult:
    """
    Evaluate whether an oracle-aligned lag signal exists.

    Parameters
    ----------
    rtds                     : current RTDS snapshot (binance + chainlink)
    start_price              : Chainlink price at the start of the 5-min window
    time_remaining_seconds   : seconds until market resolves
    realized_vol_per_second  : short-window realized vol of Chainlink, per second
    yes_ask / no_ask         : best executable ask prices for YES and NO tokens
    yes_token_id / no_token_id : token identifiers
    clob_timestamp_epoch     : unix epoch seconds when the CLOB orderbook was fetched
    category                 : market category (used for fee calculation)
    is_taker                 : True = taker order (full fee), False = maker (rebate)
    config                   : optional config overrides
    """
    cfg = config or LagSignalConfig()

    # ------------------------------------------------------------------ #
    # 1. Staleness gates — hard skip if data is too stale
    # ------------------------------------------------------------------ #
    rtds_stale = rtds.staleness_seconds()
    clob_stale = max(time.time() - clob_timestamp_epoch, 0.0)

    if rtds_stale > cfg.rtds_staleness_max_seconds:
        return LagSignalResult(
            fire=False,
            skip_reason=f"RTDS_TOO_STALE rtds_staleness={rtds_stale:.2f}s limit={cfg.rtds_staleness_max_seconds}s",
            rtds_staleness_seconds=rtds_stale,
            clob_staleness_seconds=clob_stale,
            binance_price=rtds.binance_price,
            chainlink_price=rtds.chainlink_price,
        )

    if clob_stale > cfg.clob_staleness_max_seconds:
        return LagSignalResult(
            fire=False,
            skip_reason=f"CLOB_TOO_STALE clob_staleness={clob_stale:.2f}s limit={cfg.clob_staleness_max_seconds}s",
            rtds_staleness_seconds=rtds_stale,
            clob_staleness_seconds=clob_stale,
            binance_price=rtds.binance_price,
            chainlink_price=rtds.chainlink_price,
        )

    # ------------------------------------------------------------------ #
    # 2. Time-remaining veto
    # ------------------------------------------------------------------ #
    if time_remaining_seconds < cfg.min_time_remaining_seconds:
        return LagSignalResult(
            fire=False,
            skip_reason=f"TOO_CLOSE_TO_EXPIRY time_remaining={time_remaining_seconds:.1f}s",
            rtds_staleness_seconds=rtds_stale,
            clob_staleness_seconds=clob_stale,
        )

    # ------------------------------------------------------------------ #
    # 3. Price sanity checks
    # ------------------------------------------------------------------ #
    if rtds.chainlink_price <= 0 or rtds.binance_price <= 0:
        return LagSignalResult(
            fire=False,
            skip_reason="RTDS_PRICES_MISSING",
            rtds_staleness_seconds=rtds_stale,
            clob_staleness_seconds=clob_stale,
        )

    if yes_ask <= 0 or no_ask <= 0:
        return LagSignalResult(
            fire=False,
            skip_reason="CLOB_PRICES_MISSING",
            rtds_staleness_seconds=rtds_stale,
            clob_staleness_seconds=clob_stale,
        )

    # ------------------------------------------------------------------ #
    # 4. Lag gate — Binance must be sufficiently ahead of Chainlink in time
    # ------------------------------------------------------------------ #
    lag_ms = rtds.lag_ms()
    if abs(lag_ms) < cfg.min_lag_ms:
        return LagSignalResult(
            fire=False,
            skip_reason=f"LAG_TOO_SMALL lag_ms={lag_ms:.1f}",
            lag_ms=lag_ms,
            binance_price=rtds.binance_price,
            chainlink_price=rtds.chainlink_price,
            rtds_staleness_seconds=rtds_stale,
            clob_staleness_seconds=clob_stale,
        )

    # ------------------------------------------------------------------ #
    # 5. Price divergence gate — Binance must have moved away from Chainlink
    # ------------------------------------------------------------------ #
    div_pct = rtds.price_divergence_pct()
    if div_pct < cfg.min_price_divergence_pct:
        return LagSignalResult(
            fire=False,
            skip_reason=f"PRICE_DIVERGENCE_TOO_SMALL div_pct={div_pct:.4f}",
            price_divergence_pct=div_pct,
            lag_ms=lag_ms,
            binance_price=rtds.binance_price,
            chainlink_price=rtds.chainlink_price,
            rtds_staleness_seconds=rtds_stale,
            clob_staleness_seconds=clob_stale,
        )

    # ------------------------------------------------------------------ #
    # 6. Estimate fair P(Up) from Chainlink current + start price
    # ------------------------------------------------------------------ #
    fair_p_up = _estimate_fair_p_up(
        chainlink_price=rtds.chainlink_price,
        start_price=start_price,
        time_remaining_seconds=time_remaining_seconds,
        realized_vol_per_second=realized_vol_per_second,
    )
    fair_p_down = round(1.0 - fair_p_up, 6)

    # ------------------------------------------------------------------ #
    # 7. Edge calculation — compare fair_p vs executable ask, net of fees
    # ------------------------------------------------------------------ #
    # Check YES leg
    gross_edge_yes = round(fair_p_up - yes_ask, 6)
    fee_yes = taker_fee(yes_ask, category=category) if is_taker else 0.0
    net_edge_yes = round(gross_edge_yes - fee_yes, 6)

    # Check NO leg
    gross_edge_no = round(fair_p_down - no_ask, 6)
    fee_no = taker_fee(no_ask, category=category) if is_taker else 0.0
    net_edge_no = round(gross_edge_no - fee_no, 6)

    min_edge = cfg.min_net_edge_taker if is_taker else cfg.min_net_edge_maker

    # Pick the best leg (if any)
    best_side: str = ""
    best_net_edge: float = -999.0
    best_gross: float = 0.0
    best_fee: float = 0.0
    best_ask: float = 0.0
    best_token_id: str = ""
    best_fair_p: float = 0.5

    if net_edge_yes > net_edge_no and net_edge_yes >= min_edge:
        best_side = "UP"
        best_net_edge = net_edge_yes
        best_gross = gross_edge_yes
        best_fee = fee_yes
        best_ask = yes_ask
        best_token_id = yes_token_id
        best_fair_p = fair_p_up
    elif net_edge_no >= min_edge:
        best_side = "DOWN"
        best_net_edge = net_edge_no
        best_gross = gross_edge_no
        best_fee = fee_no
        best_ask = no_ask
        best_token_id = no_token_id
        best_fair_p = fair_p_down

    if not best_side:
        return LagSignalResult(
            fire=False,
            skip_reason=(
                f"NET_EDGE_INSUFFICIENT yes_net={net_edge_yes:.4f} "
                f"no_net={net_edge_no:.4f} min={min_edge:.4f}"
            ),
            estimated_fair_p=fair_p_up,
            binance_price=rtds.binance_price,
            chainlink_price=rtds.chainlink_price,
            lag_ms=lag_ms,
            price_divergence_pct=div_pct,
            rtds_staleness_seconds=rtds_stale,
            clob_staleness_seconds=clob_stale,
            diagnostics={
                "fair_p_up": fair_p_up,
                "fair_p_down": fair_p_down,
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "gross_edge_yes": gross_edge_yes,
                "gross_edge_no": gross_edge_no,
                "net_edge_yes": net_edge_yes,
                "net_edge_no": net_edge_no,
                "fee_yes": fee_yes,
                "fee_no": fee_no,
            },
        )

    return LagSignalResult(
        fire=True,
        side=best_side,
        token_id=best_token_id,
        net_edge=best_net_edge,
        gross_edge=best_gross,
        taker_fee_paid=best_fee,
        executable_price=best_ask,
        estimated_fair_p=best_fair_p,
        binance_price=rtds.binance_price,
        chainlink_price=rtds.chainlink_price,
        lag_ms=lag_ms,
        price_divergence_pct=div_pct,
        rtds_staleness_seconds=rtds_stale,
        clob_staleness_seconds=clob_stale,
        diagnostics={
            "fair_p_up": fair_p_up,
            "fair_p_down": fair_p_down,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "min_edge": min_edge,
        },
    )
