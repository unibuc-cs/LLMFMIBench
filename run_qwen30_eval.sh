#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
PORT="${PORT:-8012}"
THREADS="${THREADS:-32}"
CONTEXT="${CONTEXT:-4096}"
LIMIT="${LIMIT:-0}"   # 0 = all tasks

mkdir -p results

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing model: $MODEL_PATH"
  echo "Check with: find ~/models -type f -name '*.gguf' -ls"
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
for _ in range(300):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if r.status == 200:
                print("Server ready.")
                sys.exit(0)
    except Exception:
        pass
    time.sleep(2)
print("Server did not start.")
sys.exit(1)
PY
}

echo "Starting Qwen3-Coder-30B-A3B server..."
"$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  -t "$THREADS" \
  -c "$CONTEXT" \
  > results/qwen3_coder_30b_a3b_server.log 2>&1 &

SERVER_PID=$!
trap cleanup EXIT

wait_for_server

echo "Running benchmark..."
python3 main.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --api-model local \
  --model-label qwen3_coder_30b_a3b_cpu \
  --out-jsonl results/qwen3_coder_30b_a3b_cpu.jsonl \
  --out-csv results/qwen3_coder_30b_a3b_cpu_summary.csv \
  --temperature 0.0 \
  --max-tokens 1536 \
  --request-timeout 3600 \
  --test-timeout 8 \
  --code-mem-mb 1024 \
  --repeats 1 \
  --limit "$LIMIT" \
  --no-think

echo "Done."
echo "Summary:"
cat results/qwen3_coder_30b_a3b_cpu_summary.csv
