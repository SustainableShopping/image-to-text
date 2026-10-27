#!/usr/bin/env bash
# Starts the API in the background when the workspace starts or is attached to,
# and waits until /health answers; skipped in CI. Settings come from the repo's
# .env (created from .env.example on first setup). Pass --restart to pick up
# .env changes.
set -euo pipefail

[ "${CI:-}" = "true" ] && exit 0

PATTERN="uvicorn image_to_text"
LOG=/tmp/itt-server.log
PORT=8100
START_TIMEOUT=${ITT_START_TIMEOUT:-300}  # first start may download ~100 MB of models
STOP_TIMEOUT=10

if pgrep -f "$PATTERN" > /dev/null; then
    [ "${1:-}" = "--restart" ] || { echo "image-to-text API already running"; exit 0; }
    pkill -f "$PATTERN"
    for _ in $(seq $((STOP_TIMEOUT * 5))); do
        pgrep -f "$PATTERN" > /dev/null || break
        sleep 0.2
    done
    pkill -9 -f "$PATTERN" 2> /dev/null || true
fi

cd "$(dirname "$0")/../backend"
env_args=()
[ -f ../.env ] && env_args=(--env-file ../.env)

setsid nohup uv run "${env_args[@]}" uvicorn image_to_text.api.app:app \
    --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 < /dev/null &
pid=$!

echo "image-to-text API starting (log: $LOG)"
for _ in $(seq "$START_TIMEOUT"); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "image-to-text API ready on port $PORT"
        exit 0
    fi
    kill -0 "$pid" 2> /dev/null || break
    sleep 1
done

echo "image-to-text API failed to start; last lines of $LOG:" >&2
tail -n 20 "$LOG" >&2
exit 1
