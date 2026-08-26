#!/usr/bin/env bash
# Serve one MLX model, run the bench against it, stop the server.
#
# Only one model process at a time — this machine has 8 GB, and two loaded
# models means swap, which would make the timings meaningless.
#
#   harness/bench.sh <model-dir> <run-label> [task-limit]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"; LABEL="$2"; LIMIT="${3:-20}"
PORT=8080

pkill -f "mlx_lm server" 2>/dev/null || true
sleep 2

.venv/bin/python -m mlx_lm server --model "$MODEL" --port "$PORT" \
    > "runs/server-$LABEL.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
    sleep 3
    curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
done

.venv/bin/python harness/run_model.py --name "$LABEL" \
    --base "http://127.0.0.1:$PORT/v1" --model "$MODEL" --limit "$LIMIT"
