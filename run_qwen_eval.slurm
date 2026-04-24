#!/bin/bash
#SBATCH --job-name=qwen-code-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm-qwen-code-cpu-%j.out

set -euo pipefail

# Run from the folder where main.py, coding_tasks.py, and report_results.py exist.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results

# Optional: activate local virtual environment if present.
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# Paths. Override when submitting if needed, e.g.:
# sbatch --export=ALL,BENCH_LIMIT=3 run_qwen_eval.slurm
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
BENCH_SCRIPT="${BENCH_SCRIPT:-$PWD/main.py}"
REPORT_SCRIPT="${REPORT_SCRIPT:-$PWD/report_results.py}"

SMALL_MODEL="${SMALL_MODEL:-$HOME/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}"
LARGE_MODEL="${LARGE_MODEL:-$HOME/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"

THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
CTX_SIZE="${CTX_SIZE:-4096}"
BENCH_LIMIT="${BENCH_LIMIT:-0}"   # 0 = all tasks; use 3 for smoke test

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

if [ ! -x "$LLAMA_SERVER" ]; then
  echo "Missing or non-executable llama-server: $LLAMA_SERVER"
  exit 1
fi

if [ ! -f "$BENCH_SCRIPT" ]; then
  echo "Missing benchmark script: $BENCH_SCRIPT"
  exit 1
fi

if [ ! -f "coding_tasks.py" ]; then
  echo "Missing coding_tasks.py in current directory: $PWD"
  exit 1
fi

wait_for_server() {
  local base_url="$1"

  python3 - <<PY
import sys
import time
import urllib.request

url = "${base_url}/models"

for _ in range(180):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if r.status == 200:
                print("Server ready:", url)
                sys.exit(0)
    except Exception:
        pass
    time.sleep(2)

print("Server did not become ready:", url)
sys.exit(1)
PY
}

run_one_model() {
  local label="$1"
  local model_path="$2"
  local port="$3"
  local max_tokens="$4"
  local code_mem_mb="$5"

  echo "============================================================"
  echo "Starting model: $label"
  echo "Model path: $model_path"
  echo "Port: $port"
  echo "Threads: $THREADS"
  echo "Context size: $CTX_SIZE"
  echo "============================================================"

  if [ ! -f "$model_path" ]; then
    echo "Missing model file: $model_path"
    exit 1
  fi

  rm -f "results/${label}.jsonl" "results/${label}_summary.csv"

  "$LLAMA_SERVER" \
    -m "$model_path" \
    --host 127.0.0.1 \
    --port "$port" \
    -t "$THREADS" \
    -c "$CTX_SIZE" \
    > "results/server_${label}.log" 2>&1 &

  local server_pid=$!

  cleanup_server() {
    echo "Stopping server PID $server_pid"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }

  trap cleanup_server EXIT

  local base_url="http://127.0.0.1:${port}/v1"
  wait_for_server "$base_url"

  local limit_args=()
  if [ "$BENCH_LIMIT" != "0" ]; then
    limit_args=(--limit "$BENCH_LIMIT")
  fi

  python3 "$BENCH_SCRIPT" \
    --base-url "$base_url" \
    --api-model local \
    --model-label "$label" \
    --out-jsonl "results/${label}.jsonl" \
    --out-csv "results/${label}_summary.csv" \
    --temperature 0.0 \
    --max-tokens "$max_tokens" \
    --request-timeout 3600 \
    --test-timeout 8 \
    --code-mem-mb "$code_mem_mb" \
    --repeats 1 \
    --no-think \
    "${limit_args[@]}"

  cleanup_server
  trap - EXIT

  echo "Finished model: $label"
}

run_one_model "qwen3_0_6b_cpu" "$SMALL_MODEL" 8011 1024 512
run_one_model "qwen3_coder_30b_a3b_cpu" "$LARGE_MODEL" 8012 1536 1024

if [ -f "$REPORT_SCRIPT" ]; then
  echo "Generating comparison report..."
  python3 "$REPORT_SCRIPT" \
    --results-dir results \
    --out-prefix results/qwen_report
else
  echo "report_results.py not found; skipping report generation."
fi

echo "All done. Results:"
ls -lh results/
