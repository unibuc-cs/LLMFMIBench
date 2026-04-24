#!/usr/bin/env python3
"""
Benchmark local OpenAI-compatible LLM servers on Python coding tasks.

The tasks and hidden tests are in coding_tasks.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any

from coding_tasks import TASKS


SYSTEM_PROMPT = """
You are a careful coding assistant.
Return only valid Python code.
Do not include explanations.
Do not include Markdown unless the code is inside a single Python code block.
"""


def call_chat_completion(
    base_url: str,
    api_model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    api_key: str,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_text_from_response(response: dict[str, Any]) -> str:
    return response["choices"][0]["message"]["content"]


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        for block in blocks:
            if "def " in block or "class " in block:
                return block.strip()
        return blocks[0].strip()

    match = re.search(r"(^|\n)(from\s+\S+\s+import\s+|import\s+|def\s+|class\s+)", text)
    if match:
        return text[match.start(2):].strip()

    return text.strip()


def truncate(s: str, max_chars: int = 4000) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n... [truncated] ..."


def make_preexec_fn(mem_mb: int, cpu_seconds: int):
    def set_limits():
        try:
            import resource

            mem_bytes = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        except Exception:
            pass

    return set_limits


def run_tests(code: str, test_code: str, timeout_s: int, mem_mb: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        solution_path = tmp_path / "solution.py"
        test_path = tmp_path / "test_solution.py"

        solution_path.write_text(code, encoding="utf-8")
        test_path.write_text(textwrap.dedent(test_code).strip() + "\n", encoding="utf-8")

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-B", str(test_path)],
                cwd=str(tmp_path),
                text=True,
                capture_output=True,
                timeout=timeout_s,
                preexec_fn=make_preexec_fn(mem_mb, timeout_s),
            )
            elapsed = time.perf_counter() - start
            return {
                "passed": proc.returncode == 0,
                "returncode": proc.returncode,
                "test_time_s": elapsed,
                "stdout": truncate(proc.stdout),
                "stderr": truncate(proc.stderr),
            }
        except subprocess.TimeoutExpired as e:
            elapsed = time.perf_counter() - start
            return {
                "passed": False,
                "returncode": None,
                "test_time_s": elapsed,
                "stdout": truncate(e.stdout or ""),
                "stderr": "TIMEOUT",
            }


def make_user_prompt(task: dict[str, str], no_think: bool) -> str:
    prefix = "/no_think\n\n" if no_think else ""
    return prefix + f"""
Solve the following Python programming task.

Rules:
- Return only Python code.
- Implement exactly the requested function name and signature.
- Do not read from stdin.
- Do not print anything.
- You may use only the Python standard library.

Task:
{task["prompt"]}
""".strip()


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]], model_label: str) -> None:
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    avg_latency = sum(r["latency_s"] for r in rows) / total if total else 0.0
    avg_test_time = sum(r["test_time_s"] for r in rows) / total if total else 0.0

    summary = {
        "model": model_label,
        "total_attempts": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "avg_generation_latency_s": avg_latency,
        "avg_test_time_s": avg_test_time,
    }

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Example: http://127.0.0.1:8080/v1")
    parser.add_argument("--api-model", default="local", help="Model name sent to the API")
    parser.add_argument("--model-label", required=True, help="Label saved in results")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--test-timeout", type=int, default=8)
    parser.add_argument("--code-mem-mb", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Use first N tasks only; 0 means all")
    parser.add_argument("--no-think", action="store_true", help="Add /no_think to the prompt")
    args = parser.parse_args()

    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tasks = TASKS[: args.limit] if args.limit > 0 else TASKS
    rows: list[dict[str, Any]] = []

    for rep in range(args.repeats):
        for task in tasks:
            print(f"[{args.model_label}] rep={rep + 1}/{args.repeats} task={task['id']}", flush=True)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": make_user_prompt(task, no_think=args.no_think)},
            ]

            generation_error = ""
            raw_text = ""
            code = ""
            response : dict[str, Any] = {}

            start = time.perf_counter()

            try:
                response = call_chat_completion(
                    base_url=args.base_url,
                    api_model=args.api_model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout_s=args.request_timeout,
                    api_key=args.api_key,
                )
                raw_text = extract_text_from_response(response)
                code = extract_code(raw_text)
            except Exception as e:
                generation_error = repr(e)

            latency = time.perf_counter() - start

            if generation_error:
                test_result = {
                    "passed": False,
                    "returncode": None,
                    "test_time_s": 0.0,
                    "stdout": "",
                    "stderr": generation_error,
                }
            else:
                test_result = run_tests(
                    code=code,
                    test_code=task["tests"],
                    timeout_s=args.test_timeout,
                    mem_mb=args.code_mem_mb,
                )

            row = {
                "model": args.model_label,
                "task_id": task["id"],
                "repeat": rep,
                "passed": bool(test_result["passed"]),
                "latency_s": latency,
                "test_time_s": test_result["test_time_s"],
                "returncode": test_result["returncode"],
                "generation_error": generation_error,
                "stdout": test_result["stdout"],
                "stderr": test_result["stderr"],
                "raw_generation": truncate(raw_text, 12000),
                "extracted_code": truncate(code, 12000),

                "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                "total_tokens": response.get("usage", {}).get("total_tokens"),

                "completion_tokens_per_second": (
                    response.get("usage", {}).get("completion_tokens") / latency
                    if response.get("usage", {}).get("completion_tokens") is not None and latency > 0
                    else None
                ),

                "generated_chars_per_second": (
                    len(raw_text) / latency if latency > 0 else None
                ),
            }

            rows.append(row)
            write_jsonl(out_jsonl, row)
            status = "PASS" if row["passed"] else "FAIL"
            print(f"  -> {status} latency={latency:.2f}s", flush=True)

    write_summary_csv(out_csv, rows, args.model_label)
    passed = sum(1 for r in rows if r["passed"])
    total = len(rows)
    print(f"\nSummary for {args.model_label}: {passed}/{total} passed ({passed / total:.2%})")


if __name__ == "__main__":
    main()
