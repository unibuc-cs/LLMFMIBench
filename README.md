# LLM FMI CPU Coding Benchmark 
### (qwen3 evaluated for now)

This repository benchmarks small and large Qwen models on deterministic Python programming tasks using a local OpenAI-compatible `llama.cpp` server. The benchmark compares functional correctness, generation latency, test execution time, and speed metrics such as tokens per second when returned by the inference server.


## Benchmark tasks

1. `two_sum` — Finds two array indices whose values sum to a target. The task tests basic hash-map reasoning and simple edge cases.
2. `longest_valid_parentheses` — Computes the longest valid parentheses substring. The task tests stack-based or dynamic-programming reasoning over strings.
3. `coin_change` — Finds the minimum number of coins required to form an amount. The task evaluates unbounded dynamic programming.
4. `num_islands` — Counts connected land components in a grid. The task tests graph traversal with DFS or BFS.
5. `edit_distance` — Computes the Levenshtein distance between two strings. The task tests two-dimensional dynamic programming.
6. `word_break` — Determines whether a string can be segmented into dictionary words. The task tests prefix-based dynamic programming.
7. `topological_order` — Returns a valid ordering under prerequisite constraints. The task tests graph construction, indegree tracking, and cycle detection.
8. `shortest_path_binary_matrix` — Finds the shortest clear path in a binary grid with 8-directional movement. The task tests BFS and boundary handling.
9. `merge_intervals` — Merges overlapping intervals and returns a sorted result. The task tests sorting and interval reasoning.
10. `min_cost_cut_stick` — Computes the minimum total cost of cutting a stick. The task tests interval dynamic programming.
11. `dijkstra` — Computes shortest paths in a directed weighted graph with non-negative weights. The task tests priority queues and graph algorithms.
12. `max_profit_k_transactions` — Maximizes stock profit with at most `k` transactions. The task tests dynamic programming with state constraints.
13. `longest_increasing_subsequence` — Computes the length of the longest strictly increasing subsequence. The task tests sequence DP or binary-search optimization.
14. `decode_ways` — Counts valid decodings of a digit string. The task tests careful dynamic programming with invalid zero cases.
15. `subarray_sum_equals_k` — Counts subarrays whose sum equals a target. The task tests prefix sums and hash-map counting.
16. `kth_largest` — Returns the kth largest element in a list. The task tests sorting, heap use, or selection logic.


## Files

- `main.py` — benchmark runner; sends tasks to the model, extracts Python code, runs tests, and stores JSONL results.
- `coding_tasks.py` — programming task definitions and hidden unit tests.
- `download_models.sh` — downloads the GGUF model files.
- `run_qwen06_eval.sh` — local CPU runner for Qwen3-0.6B.
- `run_qwen30_eval.sh` — local CPU runner for Qwen3-Coder-30B-A3B.
- `run_qwen_eval.slurm` — Slurm script that runs both models and generates reports.
- `report_results.py` — aggregates JSONL outputs into CSV and Excel reports.

## Requirements

Install Python dependencies:

```bash
uv pip install openpyxl huggingface_hub hf_xet
```

Build or install `llama.cpp`, then check that the server binary exists:

```bash
ls -lh ~/llama.cpp/build/bin/llama-server
```

## Download models

```bash
chmod +x download_models.sh
./download_models.sh
```

Expected model paths:

```text
~/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf
~/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
```

## Run locally

Run Qwen3-0.6B:

```bash
chmod +x run_qwen06_eval.sh
./run_qwen06_eval.sh
```

Run Qwen3-Coder-30B-A3B:

```bash
chmod +x run_qwen30_eval.sh
./run_qwen30_eval.sh
```

## Run with Slurm

Submit the full benchmark:

```bash
sbatch run_qwen_eval.slurm
```

Submit a quick test with only three tasks:

```bash
sbatch --export=ALL,BENCH_LIMIT=3 run_qwen_eval.slurm
```

Check job status:

```bash
squeue -u $USER
```

## Generate reports manually

After running one or both models:

```bash
python3 report_results.py \
  --results-dir results \
  --out-prefix results/qwen_report
```

Main outputs:

```text
results/qwen_report_model_summary.csv
results/qwen_report_task_details.csv
results/qwen_report_task_comparison.csv
results/qwen_report.xlsx
```

View the model-level summary:

```bash
cat results/qwen_report_model_summary.csv
```

## Copy results to a local machine

From Windows PowerShell:

```powershell
scp fizlabrl@hpc-ctrl01.acc-ub.local:~/LLMBench/results/qwen_report.xlsx .
```

To copy all results:

```powershell
scp -r fizlabrl@hpc-ctrl01.acc-ub.local:~/LLMBench/results .
```

## Metrics

The benchmark records correctness, generation latency, unit-test execution time, and generated output. If the inference server returns token usage, the report also includes prompt tokens, completion tokens, total tokens, and completion tokens per second. If token usage is unavailable, generated characters per second is used as a fallback speed metric.