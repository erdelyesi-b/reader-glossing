#!/usr/bin/env bash
# Same bench, against TurboFieldfareServer instead of mlx-lm.
#
# TurboFieldfare streams Gemma 4 26B-A4B's experts from SSD and keeps ~1.35 GB
# resident, which is the only way a 26B-class model runs on this 8 GB machine
# at all — the same weights as MLX are 15.6 GB and do not fit.
#
# Its README insists on one model-owning process at a time, so this kills any
# mlx-lm server first.
#
#   harness/bench_turbo.sh <path-to-TurboFieldfareServer> <gturbo-dir> <label> [limit]
set -euo pipefail
cd "$(dirname "$0")/.."

SERVER_BIN="$1"; MODEL="$2"; LABEL="$3"; LIMIT="${4:-20}"
PORT=8081

pkill -f "mlx_lm server" 2>/dev/null || true
pkill -f "TurboFieldfareServer" 2>/dev/null || true
sleep 3

"$SERVER_BIN" --model "$MODEL" --port "$PORT" --max-context 8192 \
    > "runs/server-$LABEL.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

# The server loads the whole shared core before opening the port, so this is a
# real wait, not a formality.
for _ in $(seq 1 120); do
    sleep 5
    curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
done

# The server registers the model under its own id and 404s on anything else,
# so ask it rather than guessing.
MODEL_ID=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
    | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "model id: $MODEL_ID"

.venv/bin/python harness/run_model.py --name "$LABEL" \
    --base "http://127.0.0.1:$PORT/v1" --model "$MODEL_ID" --limit "$LIMIT" \
    --timeout 3600
