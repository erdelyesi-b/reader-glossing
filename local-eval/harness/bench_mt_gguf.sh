#!/usr/bin/env bash
# Score a GGUF translator on the ranking stage, via llama.cpp's llama-server.
#
# The models worth testing here (EuroLLM-9B) have no MLX 4-bit port and are too
# big to convert on an 8 GB machine — converting loads the bf16 weights whole.
# llama.cpp mmaps the quantised file instead, which is the only way they run.
#
# Context is deliberately small: the ranking stage sends one sentence, so a
# large KV cache would only compete with the weights for the same 8 GB.
#
#   harness/bench_mt_gguf.sh <model.gguf> <label> [limit] [n-gpu-layers]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"; LABEL="$2"; LIMIT="${3:-60}"; NGL="${4:-99}"
PORT=8082

pkill -f "mlx_lm server" 2>/dev/null || true
pkill -f "llama-server" 2>/dev/null || true
sleep 2

# A 9B at IQ4_XS is 5.05 GB and Metal's working-set limit on this machine is
# about 5.4 GB, so the weights fit and the compute buffers do not — it dies
# with kIOGPUCommandBufferCallbackErrorOutOfMemory on the first decode. Pass
# n-gpu-layers 0 for those: llama.cpp mmaps the file and runs on the CPU.
llama-server --model "$MODEL" --port "$PORT" --ctx-size 2048 --batch-size 256 \
    --n-gpu-layers "$NGL" --no-warmup \
    > "runs/server-$LABEL.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 1 90); do
    sleep 3
    curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
done

.venv/bin/python harness/stage_mt_rank.py \
    --backend openai --base "http://127.0.0.1:$PORT/v1" \
    --model "$LABEL" --label "$LABEL" --limit "$LIMIT"
