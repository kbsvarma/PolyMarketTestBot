"""
Window Report Writer — per-15-minute-cycle summary report.

Generates a human-readable Markdown report (logs/window_report.md) that
records every 15-minute window cycle: whether a signal fired, why it didn't
if not, and hypothetical PnL if bets had been placed.

One record per window, updated at every window transition. Designed to be
left open overnight and checked periodically.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Gate result display names
# ---------------------------------------------------------------------------

_GATE_LABELS: dict[str, str] = {
    # Keys must match constants in crypto_direction_signal.py exactly
    "WINDOW_SETTLING":          "Settling (AMM noise)",
    "TIME_GATE_FAIL":           "Too late in window",
    "ASSET_MOVE_INSUFFICIENT":  "BTC/ETH move too small",
    "PRICE_RANGE_FAIL":         "Price outside entry range",
    "PRICES_STALE":             "Orderbook prices stale",
    "CHOP_FILTER_FAIL":         "Choppy / no clean direction",
    "LAG_GAP_INSUFFICIENT":     "No GBM lag (market already priced in)",
    "ALREADY_FIRED_THIS_WINDOW": "Signal already fired this window",
    "SIGNAL_FIRED":             "✅ Signal fired",
    "NO_WINDOW_STATE":          "No window state",
}

_GATE_RANK: dict[str, int] = {
    # Higher rank = further through the gate sequence = better signal quality
    # Keys must match gate result strings from crypto_direction_signal.py
    "NO_WINDOW_STATE":          0,
    "WINDOW_SETTLING":          1,
    "TIME_GATE_FAIL":           2,
    "ASSET_MOVE_INSUFFICIENT":  3,
    "PRICE_RANGE_FAIL":         4,
    "PRICES_STALE":             5,
    "CHOP_FILTER_FAIL":         6,
    "LAG_GAP_INSUFFICIENT":     7,
    "ALREADY_FIRED_THIS_WINDOW": 8,
    "SIGNAL_FIRED":             9,
}


# ---------------------------------------------------------------------------
# Per-window summary record
# ---------------------------------------------------------------------------

@dataclass
class WindowSummary:
    window_ts: int              # Unix timestamp of window start
    window_close_ts: int        # window_ts + 900
    asset: str                  # "BTC" or "ETH"
    asset_open: float           # price at window open
    asset_close: float          # price at window close (last BTC/ETH price seen)
    yes_ask_final: float        # YES ask at window close (0.99 → YES likely won)
    no_ask_final: float         # NO ask at window close
    primary_gate_fail: str      # deepest gate reached (furthest in sequence)
    gate_counts: dict           # {gate: count} — how often each gate fired
    signal_fired: bool          # did a Phase 1 signal fire?
    signal_side: str            # "YES" or "NO" if fired, else ""
    signal_price: float         # momentum-side entry ask price
    phase2_triggered: bool      # did Phase 2 (bracket leg 2) trigger?
    phase2_price: float         # opposite-side entry price for leg 2
    resolved_yes: Optional[bool] = None   # True=YES won, False=NO won, None=unclear
    hyp_pnl_per_share: float = 0.0        # hypothetical PnL per $1 notional
    hyp_pnl_dollars: float = 0.0          # hypothetical PnL in dollars (bet_size scaled)

    @property
    def move_pct(self) -> float:
        if self.asset_open <= 0:
            return 0.0
        return (self.asset_close - self.asset_open) / self.asset_open

    @property
    def window_label(self) -> str:
        """Format: '9:15 PM – 9:30 PM ET'"""
        def _fmt(ts: int) -> str:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            # Convert UTC → ET (UTC-4 for EDT, UTC-5 for EST)
            # Use simple -4 offset (Eastern Daylight Time, March)
            et_hour = (dt.hour - 4) % 24
            suffix = "AM" if et_hour < 12 else "PM"
            h = et_hour % 12 or 12
            return f"{h}:{dt.minute:02d} {suffix}"
        return f"{_fmt(self.window_ts)} – {_fmt(self.window_close_ts)} ET"

    @property
    def date_label(self) -> str:
        dt = datetime.fromtimestamp(self.window_ts, tz=timezone.utc)
        et_hour = (dt.hour - 4) % 24
        et_date = dt.replace(hour=et_hour)
        return et_date.strftime("%b %d, %Y")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

class WindowReportWriter:
    """
    Maintains an in-memory list of per-window summaries and rewrites the
    Markdown report file on every window close.

    Usage (in the observer loop, on each window transition):
        writer.record_window_close(
            window_ts=..., asset=..., asset_open=..., asset_close=...,
            yes_ask_final=..., no_ask_final=...,
            last_signal_event=..., eval_log_path=...,
        )
    """

    def __init__(
        self,
        report_path: str,
        hypothetical_bet_size: float = 10.0,
        max_windows: int = 200,
        live_execution: bool = False,   # True when BracketExecutor is wired in
    ) -> None:
        self._report_path = Path(report_path)
        self._bet_size = hypothetical_bet_size
        self._max_windows = max_windows
        self._live_execution = live_execution
        self._windows: list[WindowSummary] = []
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        # Seed with an empty report on startup
        self._write_report()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record_window_close(
        self,
        window_ts: int,
        asset: str,
        asset_open: float,
        asset_close: float,
        yes_ask_final: float,
        no_ask_final: float,
        last_signal_event: dict | None,
        eval_log_path: str,
        yes_won: bool | None = None,
    ) -> None:
        """
        Called when a window transition is detected. Reads the evaluation log
        for the closing window, builds a summary, appends it, and rewrites
        the report.

        yes_won: directly passed from the evaluator (asset_close > asset_open).
          Preferred over inferring from ask prices, which are unreliable at
          settlement time (both tokens go near 0, making inference ambiguous).
        """
        gate_counts = self._scan_eval_log(eval_log_path, window_ts, asset)

        # Determine the deepest gate reached (furthest in sequence)
        primary_gate = max(
            gate_counts.keys(),
            key=lambda g: _GATE_RANK.get(g, 0),
            default="NO_STATE",
        )

        # Signal details from the last signal event (if any)
        signal_fired = False
        signal_side = ""
        signal_price = 0.0
        phase2_triggered = False
        phase2_price = 0.0

        if last_signal_event:
            signal_fired = True
            signal_side = last_signal_event.get("momentum_side", "")
            signal_price = float(last_signal_event.get("momentum_price", 0.0))
            # Check if phase2 was triggered (present in post-signal observations)
            obs = last_signal_event.get("observation") or {}
            if obs and obs.get("phase2_would_have_triggered"):
                phase2_triggered = True
                # phase2_trigger_price is the PostSignalObservation field for the
                # actual y price at which Phase 2 would have triggered.
                phase2_price = float(obs.get("phase2_trigger_price") or 0.0)

        # Resolve outcome — prefer the directly-passed yes_won flag (most reliable).
        # Post-settlement ask prices are unreliable: after market close both tokens
        # drop to near-zero (no one quotes a settled market), so price-based inference
        # frequently returns wrong results.  The yes_won flag is computed directly
        # from asset_close > asset_open which is always available and correct.
        resolved_yes: Optional[bool] = None
        if yes_won is not None:
            resolved_yes = yes_won   # direct — always use this when available
        elif yes_ask_final >= 0.90:
            resolved_yes = True      # YES near 1.0 = YES won (pre-settlement snap)
        elif no_ask_final >= 0.90:
            resolved_yes = False     # NO near 1.0 = NO won (pre-settlement snap)
        elif yes_ask_final <= 0.10 and no_ask_final > yes_ask_final:
            resolved_yes = False     # YES near 0 AND NO higher → NO won
        elif no_ask_final <= 0.10 and yes_ask_final > no_ask_final:
            resolved_yes = True      # NO near 0 AND YES higher → YES won
        # else: both near 0 and can't tell — leave None (ambiguous post-settlement)

        # Compute hypothetical PnL
        hyp_pnl_per_share = 0.0
        if signal_fired and signal_price > 0 and resolved_yes is not None:
            # Phase 1 entry: bet momentum_side at signal_price
            # Payout $1.00 if momentum direction wins, $0.00 if it loses
            fee = _taker_fee(signal_price)
            phase1_cost = signal_price + fee

            if phase2_triggered and phase2_price > 0:
                # Bracket complete: guaranteed profit regardless of outcome
                opp_fee = _taker_fee(phase2_price)
                total_cost = phase1_cost + phase2_price + opp_fee
                hyp_pnl_per_share = 1.0 - total_cost
            else:
                # Phase 1 only: depends on resolution
                momentum_won = (
                    (signal_side == "YES" and resolved_yes is True) or
                    (signal_side == "NO" and resolved_yes is False)
                )
                if momentum_won:
                    hyp_pnl_per_share = 1.0 - phase1_cost
                else:
                    hyp_pnl_per_share = -phase1_cost

        hyp_pnl_dollars = hyp_pnl_per_share * self._bet_size

        summary = WindowSummary(
            window_ts=window_ts,
            window_close_ts=window_ts + 900,
            asset=asset,
            asset_open=asset_open,
            asset_close=asset_close,
            yes_ask_final=yes_ask_final,
            no_ask_final=no_ask_final,
            primary_gate_fail=primary_gate,
            gate_counts=gate_counts,
            signal_fired=signal_fired,
            signal_side=signal_side,
            signal_price=signal_price,
            phase2_triggered=phase2_triggered,
            phase2_price=phase2_price,
            resolved_yes=resolved_yes,
            hyp_pnl_per_share=hyp_pnl_per_share,
            hyp_pnl_dollars=hyp_pnl_dollars,
        )

        self._windows.append(summary)
        # Keep cap to avoid unbounded memory
        if len(self._windows) > self._max_windows:
            self._windows = self._windows[-self._max_windows:]

        self._write_report()
        logger.info(
            "Window report updated window={} asset={} signal={} gate={} hyp_pnl=${:.2f}",
            summary.window_label, asset,
            "FIRED" if signal_fired else "none",
            primary_gate,
            hyp_pnl_dollars,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _scan_eval_log(
        self,
        eval_log_path: str,
        window_ts: int,
        asset: str,
    ) -> dict[str, int]:
        """Read the evaluation log and count gate results for this window."""
        counts: Counter = Counter()
        path = Path(eval_log_path)
        if not path.exists():
            return {}
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if (
                            rec.get("asset") == asset
                            and rec.get("window_open_ts") == window_ts
                        ):
                            result = rec.get("result", "UNKNOWN")
                            counts[result] += 1
                    except Exception:
                        continue
        except Exception as exc:
            logger.debug("Could not scan eval log: {}", exc)
        return dict(counts)

    def _write_report(self) -> None:
        """Overwrite the Markdown report file with current state."""
        now_et = _now_et()
        total_windows = len(self._windows)
        windows_with_signal = sum(1 for w in self._windows if w.signal_fired)
        brackets_complete = sum(1 for w in self._windows if w.phase2_triggered)
        total_hyp_pnl = sum(w.hyp_pnl_dollars for w in self._windows)
        wins = sum(
            1 for w in self._windows
            if w.signal_fired and w.hyp_pnl_dollars > 0
        )
        losses = sum(
            1 for w in self._windows
            if w.signal_fired and w.hyp_pnl_dollars < 0
        )

        lines: list[str] = []

        # ---- Header ----
        if self._live_execution:
            mode_line = "> **Mode:** 🔴 LIVE EXECUTION — real orders are being placed"
        else:
            mode_line = "> **Mode:** Observation Only — no real money is being spent"
        lines += [
            "# 📊 Bracket Observer — Live Window Report",
            "",
            f"{mode_line}  ",
            f"> **Last Updated:** {now_et}  ",
            f"> **Hypothetical Bet Size:** ${self._bet_size:.2f} per signal  ",
            "",
            "---",
            "",
        ]

        # ---- Running totals ----
        pnl_color = "🟢" if total_hyp_pnl > 0 else ("🔴" if total_hyp_pnl < 0 else "⚪")
        win_rate_str = f"{wins}/{windows_with_signal}" if windows_with_signal else "—"
        lines += [
            "## 💰 Running Totals",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Windows observed | **{total_windows}** |",
            f"| Signals fired | **{windows_with_signal}** ({_pct(windows_with_signal, total_windows)}) |",
            f"| Brackets completed | **{brackets_complete}** |",
            f"| Win / Loss (Phase 1) | **{win_rate_str}** |",
            f"| Hypothetical PnL | {pnl_color} **${total_hyp_pnl:+.2f}** |",
            "",
            "---",
            "",
        ]

        # ---- Gate funnel stats ----
        if self._windows:
            all_gate_counts: Counter = Counter()
            for w in self._windows:
                all_gate_counts.update(w.gate_counts)
            total_evals = sum(all_gate_counts.values()) or 1
            lines += [
                "## 🔬 Signal Gate Funnel (all windows)",
                "",
                "| Gate | Hits | % of evals |",
                "|------|------|-----------|",
            ]
            for gate in [
                "WINDOW_SETTLING", "TIME_GATE_FAIL", "ASSET_MOVE_INSUFFICIENT",
                "PRICE_RANGE_FAIL", "PRICES_STALE", "CHOP_FILTER_FAIL",
                "LAG_GAP_INSUFFICIENT", "ALREADY_FIRED_THIS_WINDOW", "SIGNAL_FIRED",
            ]:
                count = all_gate_counts.get(gate, 0)
                if count:
                    label = _GATE_LABELS.get(gate, gate)
                    lines.append(
                        f"| {label} | {count:,} | {count/total_evals*100:.1f}% |"
                    )
            lines += ["", "---", ""]

        # ---- Per-window log (newest first) ----
        lines += [
            "## 📋 Window Log (newest first)",
            "",
        ]

        if not self._windows:
            lines.append("*No windows completed yet. Waiting for first 15-minute cycle...*")
            lines.append("")
        else:
            for w in reversed(self._windows):
                lines += _format_window(w)

        content = "\n".join(lines)
        try:
            self._report_path.write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write window report: {}", exc)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_window(w: WindowSummary) -> list[str]:
    """Format a single window as a Markdown section."""
    lines: list[str] = []

    # --- Section heading ---
    signal_badge = "🔔 SIGNAL FIRED" if w.signal_fired else "— no signal"
    lines.append(f"### {w.date_label} · {w.window_label} · {w.asset} · {signal_badge}")
    lines.append("")

    # --- Price / move ---
    move_pct = w.move_pct * 100
    direction = "▲" if move_pct > 0 else ("▼" if move_pct < 0 else "—")
    move_str = f"{direction} {abs(move_pct):.3f}%"
    price_str = (
        f"${w.asset_open:,.2f} → ${w.asset_close:,.2f} ({move_str})"
        if w.asset_open > 0 else "*(mid-window start)*"
    )
    lines.append(f"- **{w.asset} price:** {price_str}")

    # --- Final market prices ---
    if w.yes_ask_final > 0 or w.no_ask_final > 0:
        resolution_hint = ""
        if w.resolved_yes is True:
            resolution_hint = "  → ✅ **YES won**"
        elif w.resolved_yes is False:
            resolution_hint = "  → ✅ **NO won**"
        lines.append(
            f"- **Final prices:** YES {w.yes_ask_final:.3f} / NO {w.no_ask_final:.3f}{resolution_hint}"
        )

    # --- Gate result ---
    if w.signal_fired:
        lines.append(f"- **Signal:** {w.signal_side} @ {w.signal_price:.3f}")
        if w.phase2_triggered:
            lines.append(
                f"- **Phase 2:** ✅ Bracket leg triggered — {_opp(w.signal_side)} @ {w.phase2_price:.3f}"
            )
        else:
            lines.append("- **Phase 2:** ⏳ Not triggered (opposite side never fell to target)")
    else:
        gate_label = _GATE_LABELS.get(w.primary_gate_fail, w.primary_gate_fail)
        lines.append(f"- **Could not place bet:** {gate_label}")

        # Show the gate count breakdown if interesting
        if w.gate_counts:
            dominated = max(w.gate_counts, key=lambda g: w.gate_counts[g])
            count = w.gate_counts[dominated]
            total = sum(w.gate_counts.values())
            lines.append(
                f"  - *(deepest gate: `{dominated}`, {count}/{total} eval cycles)*"
            )

    # --- PnL ---
    if w.signal_fired and w.resolved_yes is not None:
        pnl_icon = "🟢" if w.hyp_pnl_dollars > 0 else "🔴"
        lines.append(
            f"- **Hypothetical PnL:** {pnl_icon} ${w.hyp_pnl_dollars:+.2f} "
            f"({w.hyp_pnl_per_share:+.4f}/share × ${10:.0f})"
        )
    elif w.signal_fired:
        lines.append("- **Hypothetical PnL:** ⏳ Pending resolution")
    else:
        lines.append("- **Hypothetical PnL:** $0.00 (no bet placed)")

    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def _taker_fee(p: float) -> float:
    """Polymarket taker fee: p * 0.25 * (p*(1-p))^2"""
    if p <= 0 or p >= 1:
        return 0.0
    return p * 0.25 * (p * (1 - p)) ** 2


def _pct(n: int, d: int) -> str:
    if d == 0:
        return "—"
    return f"{n/d*100:.0f}%"


def _opp(side: str) -> str:
    return "NO" if side == "YES" else "YES"


def _now_et() -> str:
    """Current time formatted as ET (EDT = UTC-4)."""
    from datetime import timedelta
    now_utc = datetime.now(timezone.utc)
    et = now_utc - timedelta(hours=4)
    return et.strftime("%Y-%m-%d %I:%M %p ET")
