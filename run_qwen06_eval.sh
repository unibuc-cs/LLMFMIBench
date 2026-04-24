#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
PORT="${PORT:-8011}"
THREADS="${THREADS:-16}"
CONTEXT="${CONTEXT:-4096}"
LIMIT="${LIMIT:-0}"   # 0 = all tasks

mkdir -p results

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing model: $MODEL_PATH"
  exit 1
fi

cleanup() {
  echo "Stopping server..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}

wait_for_server() {
python3 - <<PY
import sys, time, urllib.request
url = "http://127.0.0.1:${PORT}/v1/models"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if r.status == 200:
                print("Server ready.")
                sys.exit(0)
    except Exception:
        pass
    time.sleep(1)
print("Server did not start.")
sys.exit(1)
PY
}

echo "Starting Qwen3-0.6B server..."
"$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  -t "$THREADS" \
  -c "$CONTEXT" \
  > results/qwen3_0_6b_server.log 2>&1 &

SERVER_PID=$!
trap cleanup EXIT

wait_for_server

echo "Running benchmark..."
python3 main.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --api-model local \
  --model-label qwen3_0_6b_cpu \
  --out-jsonl results/qwen3_0_6b_cpu.jsonl \
  --out-csv results/qwen3_0_6b_cpu_summary.csv \
  --temperature 0.0 \
  --max-tokens 1024 \
  --request-timeout 900 \
  --test-timeout 8 \
  --code-mem-mb 512 \
  --repeats 1 \
  --limit "$LIMIT" \
  --no-think

echo "Done."
echo "Summary:"
cat results/qwen3_0_6b_cpu_summary.csv
