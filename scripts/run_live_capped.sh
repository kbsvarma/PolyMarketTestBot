#!/bin/zsh
set -euo pipefail

ROOT="/Users/varmakammili/Documents/GitHub/PolyMarketTestBot"
APP_STATE="$ROOT/data/app_state.json"
SYSTEM_LOG="$ROOT/logs/system.log"

cd "$ROOT"
source .venv/bin/activate

export POLYBOT_CONFIG_PATH="${POLYBOT_CONFIG_PATH:-config.yaml}"
export POLYBOT_LIVE_SESSION_MAX_USD="${POLYBOT_LIVE_SESSION_MAX_USD:-30}"
export POLYBOT_LIVE_MAX_TRADE_USD="${POLYBOT_LIVE_MAX_TRADE_USD:-5}"
export POLYBOT_LIVE_MAX_POSITIONS="${POLYBOT_LIVE_MAX_POSITIONS:-10}"
export POLYBOT_LIVE_WALLET_COUNT="${POLYBOT_LIVE_WALLET_COUNT:-5}"

echo "Starting capped live session"
echo "  config: $POLYBOT_CONFIG_PATH"
echo "  session max usd: $POLYBOT_LIVE_SESSION_MAX_USD"
echo "  max trade usd: $POLYBOT_LIVE_MAX_TRADE_USD"
echo "  max positions: $POLYBOT_LIVE_MAX_POSITIONS"
echo "  live wallet count: $POLYBOT_LIVE_WALLET_COUNT"
echo "  logs: $SYSTEM_LOG"
echo "  state: $APP_STATE"
echo ""
echo "Live heartbeat will print every 10s."

# Main wallet-following bot
python main.py &
BOT_PID=$!

# BTC 15-minute bracket observer (separate process — must be started alongside main bot)
python -u main.py --execute-crypto-instance polymarket_btc_15m &
CRYPTO_PID=$!

cleanup() {
  echo "[shutdown] stopping all bot processes..."
  if kill -0 "$BOT_PID" 2>/dev/null; then
    kill "$BOT_PID" 2>/dev/null || true
    wait "$BOT_PID" 2>/dev/null || true
  fi
  if kill -0 "$CRYPTO_PID" 2>/dev/null; then
    kill "$CRYPTO_PID" 2>/dev/null || true
    wait "$CRYPTO_PID" 2>/dev/null || true
  fi
}

trap cleanup INT TERM

while kill -0 "$BOT_PID" 2>/dev/null && kill -0 "$CRYPTO_PID" 2>/dev/null; do
  if [[ -f "$APP_STATE" ]]; then
    python -c '
import json
from pathlib import Path

path = Path("'"$APP_STATE"'")
try:
    data = json.loads(path.read_text())
except Exception as exc:
    print(f"[heartbeat] unable to read app_state: {exc}")
else:
    print(
        "[heartbeat] status={status} paused={paused} ready={ready} "
        "detections={detections} decisions={decisions} "
        "open_orders={open_orders} positions={positions}".format(
            status=data.get("system_status", "?"),
            paused=data.get("paused", "?"),
            ready=data.get("live_readiness_last_result", {}).get("ready", "?"),
            detections=data.get("last_cycle_detection_count", "?"),
            decisions=data.get("last_cycle_decision_count", "?"),
            open_orders=data.get("open_orders_detail", "?"),
            positions=data.get("positions_detail", "?"),
        )
    )
' || true
  else
    echo "[heartbeat] waiting for $APP_STATE"
  fi
  sleep 10
done

echo "[shutdown] a bot process exited — cleaning up remaining processes"
cleanup
wait "$BOT_PID" 2>/dev/null || true
wait "$CRYPTO_PID" 2>/dev/null || true
