#!/usr/bin/env python3
"""Create LiteLLM virtual API keys from a CSV file."""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://llm.fmi.unibuc.ro"
SENSITIVE_KEY_RE = re.compile(r"sk-[A-Za-z0-9._-]+")


class ApiError(RuntimeError):
    """Wrap an HTTP or network error with a readable LiteLLM API message."""

    def __init__(self, status: int | None, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(self.__str__())

    def __str__(self) -> str:
        if self.status is None:
            return self.message
        return f"HTTP {self.status}: {self.message}"


@dataclass(frozen=True)
class KeyInput:
    """Normalized data from one CSV row used to create one LiteLLM key."""

    csv_line: int
    alias: str
    name: str
    group: str
    email: str
    user_id: str | None


def redact_secrets(text: str) -> str:
    """Hide API keys before errors or responses are shown to the user."""

    return SENSITIVE_KEY_RE.sub("sk-***", text)


def normalize_base_url(base_url: str) -> str:
    """Trim whitespace and trailing slashes from the LiteLLM base URL."""

    return base_url.strip().rstrip("/")


def split_csv_arg(value: str | None) -> list[str]:
    """Parse comma-separated command-line options such as models or tags."""

    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def request_json(
    method: str,
    base_url: str,
    path: str,
    api_key: str,
    *,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    changed_by: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Send an authenticated LiteLLM API request and return a JSON object."""

    url = f"{normalize_base_url(base_url)}{path}"
    if params:
        query = urllib.parse.urlencode(params, doseq=True)
        url = f"{url}?{query}"

    data = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "x-litellm-api-key": api_key,
    }
    if changed_by:
        headers["litellm-changed-by"] = changed_by
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return read_json_response(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ApiError(exc.code, extract_error_message(body)) from exc
    except urllib.error.URLError as exc:
        raise ApiError(None, f"request failed: {exc.reason}") from exc


def read_json_response(response: Any) -> dict[str, Any]:
    """Decode an HTTP response body into a dictionary."""

    body = response.read().decode("utf-8", errors="replace")
    if not body.strip():
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}
    if isinstance(parsed, dict):
        return parsed
    return {"data": parsed}


def extract_error_message(body: str) -> str:
    """Extract the useful LiteLLM error text from a failed HTTP response."""

    redacted = redact_secrets(body.strip())
    if not redacted:
        return "empty response body"
    try:
        parsed = json.loads(redacted)
    except json.JSONDecodeError:
        return redacted

    detail = parsed.get("detail") if isinstance(parsed, dict) else None
    if isinstance(detail, str):
        return detail
    if detail is not None:
        return json.dumps(detail, ensure_ascii=True)
    return json.dumps(parsed, ensure_ascii=True)


def cleaned_cell(row: dict[str, str], column: str, csv_line: int, *, required: bool) -> str:
    """Read and validate one CSV cell by column name."""

    if column not in row:
        if required:
            raise ValueError(f"line {csv_line}: missing required CSV column '{column}'")
        return ""
    value = row[column].strip()
    if required and not value:
        raise ValueError(f"line {csv_line}: empty required CSV column '{column}'")
    return value


def build_alias(group: str, name: str, *, normalize_name: bool) -> str:
    """Build the required key alias in the Gr{Grupa}_{Nume} format."""

    clean_group = re.sub(r"\s+", "", group)
    clean_name = re.sub(r"\s+", " ", name).strip()
    if normalize_name:
        clean_name = re.sub(r"\s+", "_", clean_name)
    return f"Gr{clean_group}_{clean_name}"


def load_csv_inputs(
    csv_path: Path,
    *,
    name_column: str,
    group_column: str,
    email_column: str,
    user_id_column: str | None,
    normalize_name: bool,
) -> list[KeyInput]:
    """Load the input CSV and convert each row to a KeyInput record."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row")
        reader.fieldnames = [field.strip() for field in reader.fieldnames]

        rows: list[KeyInput] = []
        for csv_line, raw_row in enumerate(reader, start=2):
            row = {
                key.strip(): (value or "").strip()
                for key, value in raw_row.items()
                if key is not None
            }
            name = cleaned_cell(row, name_column, csv_line, required=True)
            group = cleaned_cell(row, group_column, csv_line, required=True)
            email = cleaned_cell(row, email_column, csv_line, required=False)
            user_id = (
                cleaned_cell(row, user_id_column, csv_line, required=False)
                if user_id_column
                else ""
            )
            alias = build_alias(group, name, normalize_name=normalize_name)
            rows.append(
                KeyInput(
                    csv_line=csv_line,
                    alias=alias,
                    name=name,
                    group=re.sub(r"\s+", "", group),
                    email=email,
                    user_id=user_id or None,
                )
            )
    return rows


def make_key_payload(args: argparse.Namespace, csv_path: Path, row: KeyInput) -> dict[str, Any]:
    """Create the JSON body sent to LiteLLM /key/generate for one row."""

    payload: dict[str, Any] = {
        "key_alias": row.alias,
        "key_type": args.key_type,
        "metadata": {
            "source": "auto_key_register.py",
            "source_csv": csv_path.name,
            "csv_line": row.csv_line,
            "name": row.name,
            "group": row.group,
        },
    }

    if row.email:
        payload["metadata"]["email"] = row.email
    if row.user_id:
        payload["user_id"] = row.user_id
    if args.team_id:
        payload["team_id"] = args.team_id
    if args.duration:
        payload["duration"] = args.duration
    if args.max_budget is not None:
        payload["max_budget"] = args.max_budget
    if args.budget_duration:
        payload["budget_duration"] = args.budget_duration
    if args.rpm_limit is not None:
        payload["rpm_limit"] = args.rpm_limit
    if args.tpm_limit is not None:
        payload["tpm_limit"] = args.tpm_limit

    models = split_csv_arg(args.models)
    if models:
        payload["models"] = models

    tags = split_csv_arg(args.tags)
    if tags:
        payload["tags"] = tags

    return payload


def find_existing_key(
    base_url: str,
    api_key: str,
    alias: str,
    *,
    changed_by: str | None,
    timeout: int,
) -> dict[str, Any] | None:
    """Look up an existing LiteLLM key by exact alias to avoid duplicates."""

    page = 1
    while True:
        data = request_json(
            "GET",
            base_url,
            "/key/list",
            api_key,
            params={
                "page": page,
                "size": 100,
                "key_alias": alias,
                "return_full_object": "true",
            },
            changed_by=changed_by,
            timeout=timeout,
        )
        for item in data.get("keys", []):
            if isinstance(item, dict) and item.get("key_alias") == alias:
                return item

        total_pages = data.get("total_pages") or page
        if page >= total_pages:
            return None
        page += 1


def resolve_api_key(args: argparse.Namespace) -> str:
    """Get the admin API key from args, environment variables, or a prompt."""

    api_key = args.api_key
    if not api_key:
        api_key = getpass.getpass("LiteLLM API key/password: ")
    api_key = api_key.strip()
    if not api_key:
        raise SystemExit("No API key/password provided.")
    return api_key


def validate_auth(args: argparse.Namespace, api_key: str) -> None:
    """Check that the provided key can access key-management endpoints."""

    request_json(
        "GET",
        args.base_url,
        "/key/list",
        api_key,
        params={"page": 1, "size": 1},
        changed_by=args.username,
        timeout=args.timeout,
    )


def output_path_for(args: argparse.Namespace) -> Path:
    """Choose where generated keys and per-row results will be written."""

    if args.output:
        return args.output
    return args.csv_path.with_name(f"{args.csv_path.stem}_created_keys.csv")


def check_output_target(path: Path, *, append: bool, overwrite: bool) -> None:
    """Prevent accidental overwrite of an existing output CSV."""

    if path.exists() and not append and not overwrite:
        raise SystemExit(
            f"Output file already exists: {path}. Use --append-output or --overwrite-output."
        )


def prepare_output_file(path: Path, *, append: bool, overwrite: bool) -> tuple[Any, csv.DictWriter]:
    """Open the output CSV and write its header when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    check_output_target(path, append=append, overwrite=overwrite)

    mode = "a" if append else "w"
    handle = path.open(mode, encoding="utf-8", newline="")
    fieldnames = [
        "status",
        "message",
        "csv_line",
        "nume",
        "grupa",
        "email",
        "key_alias",
        "user_id",
        "key",
        "token_id",
        "key_name",
        "expires",
        "created_at",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not append or not file_exists or path.stat().st_size == 0:
        writer.writeheader()
        handle.flush()
    return handle, writer


def write_result(
    writer: csv.DictWriter,
    *,
    status: str,
    message: str,
    row: KeyInput,
    response: dict[str, Any] | None = None,
) -> None:
    """Write one created, skipped, or failed row to the output CSV."""

    response = response or {}
    writer.writerow(
        {
            "status": status,
            "message": message,
            "csv_line": row.csv_line,
            "nume": row.name,
            "grupa": row.group,
            "email": row.email,
            "key_alias": row.alias,
            "user_id": row.user_id or "",
            "key": response.get("key", ""),
            "token_id": response.get("token_id", ""),
            "key_name": response.get("key_name", ""),
            "expires": response.get("expires", ""),
            "created_at": response.get("created_at", ""),
        }
    )


def positive_int(value: str) -> int:
    """Validate a positive integer command-line argument."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_float(value: str) -> float:
    """Validate a non-negative float command-line argument."""

    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Define all command-line arguments supported by the script."""

    parser = argparse.ArgumentParser(
        description="Generate LiteLLM virtual API keys from a CSV file."
    )
    parser.add_argument("csv_path", type=Path, help="CSV file with Nume, Grupa, and email columns.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("LITELLM_BASE_URL", DEFAULT_BASE_URL),
        help=f"LiteLLM proxy base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.getenv("LITELLM_API_KEY")
            or os.getenv("LITELLM_MASTER_KEY")
            or os.getenv("LITELLM_PASSWORD")
        ),
        help="LiteLLM admin/master key. Prefer env vars or the prompt over shell history.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("LITELLM_USERNAME", "admin"),
        help="Value sent in litellm-changed-by for audit logs. Default: admin.",
    )
    parser.add_argument("--name-column", default="Nume", help="CSV column used for {Nume}.")
    parser.add_argument("--group-column", default="Grupa", help="CSV column used for {Grupa}.")
    parser.add_argument("--email-column", default="email", help="CSV email column.")
    parser.add_argument(
        "--user-id-column",
        default="email",
        help="CSV column to send as LiteLLM user_id. Use an empty value to disable.",
    )
    parser.add_argument(
        "--no-normalize-name",
        action="store_true",
        help="Keep spaces in {Nume}; by default whitespace is converted to underscores.",
    )
    parser.add_argument(
        "--key-type",
        default="llm_api",
        choices=["llm_api", "management", "read_only", "default"],
        help="LiteLLM key_type. Default: llm_api.",
    )
    parser.add_argument("--team-id", help="Optional LiteLLM team_id for every generated key.")
    parser.add_argument("--models", help="Comma-separated model allow-list. Empty means all models.")
    parser.add_argument("--tags", help="Comma-separated LiteLLM tags for every generated key.")
    parser.add_argument("--duration", help='Optional key duration, e.g. "30d", "12h".')
    parser.add_argument("--budget-duration", help='Optional budget reset duration, e.g. "30d".')
    parser.add_argument("--max-budget", type=non_negative_float, help="Optional max budget.")
    parser.add_argument("--rpm-limit", type=positive_int, help="Optional requests-per-minute limit.")
    parser.add_argument("--tpm-limit", type=positive_int, help="Optional tokens-per-minute limit.")
    parser.add_argument("--timeout", type=positive_int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between API writes.")
    parser.add_argument("--output", type=Path, help="CSV file where generated keys are saved.")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--append-output", action="store_true", help="Append to output CSV.")
    output_mode.add_argument("--overwrite-output", action="store_true", help="Overwrite output CSV.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create a new key even when the same key alias already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read the CSV and print planned aliases without calling the API.",
    )
    parser.add_argument(
        "--print-keys",
        action="store_true",
        help="Also print generated keys to stdout. By default keys are only written to the output CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CSV loading, duplicate checks, key creation, and reporting."""

    parser = build_parser()
    args = parser.parse_args(argv)
    args.base_url = normalize_base_url(args.base_url)
    args.user_id_column = args.user_id_column.strip() or None

    if not args.csv_path.exists():
        raise SystemExit(f"CSV file not found: {args.csv_path}")

    rows = load_csv_inputs(
        args.csv_path,
        name_column=args.name_column,
        group_column=args.group_column,
        email_column=args.email_column,
        user_id_column=args.user_id_column,
        normalize_name=not args.no_normalize_name,
    )
    if not rows:
        raise SystemExit("CSV has no data rows.")

    if args.dry_run:
        print(f"Dry run: {len(rows)} key(s) would be created at {args.base_url}")
        for row in rows:
            user_id = f", user_id={row.user_id}" if row.user_id else ""
            print(f"  line {row.csv_line}: {row.alias}{user_id}")
        return 0

    output_path = output_path_for(args)
    check_output_target(output_path, append=args.append_output, overwrite=args.overwrite_output)

    api_key = resolve_api_key(args)
    try:
        validate_auth(args, api_key)
    except ApiError as exc:
        raise SystemExit(f"Authentication check failed: {exc}") from exc

    output_handle, writer = prepare_output_file(
        output_path,
        append=args.append_output,
        overwrite=args.overwrite_output,
    )

    created = 0
    skipped = 0
    failed = 0
    try:
        # Take each row in the input
        for row in rows:
            try:
                existing = None
                if not args.force:
                    # If not forced try to find the key alias
                    existing = find_existing_key(
                        args.base_url,
                        api_key,
                        row.alias,
                        changed_by=args.username,
                        timeout=args.timeout,
                    )
                # Skip if exists
                if existing is not None:
                    skipped += 1
                    write_result(
                        writer,
                        status="skipped_existing",
                        message="key_alias already exists",
                        row=row,
                        response=existing,
                    )
                    print(f"skipped existing: {row.alias}")
                    continue

                # Make a call payload then call key/generate
                payload = make_key_payload(args, args.csv_path, row)
                response = request_json(
                    "POST",
                    args.base_url,
                    "/key/generate",
                    api_key,
                    payload=payload,
                    changed_by=args.username,
                    timeout=args.timeout,
                )
                created += 1
                write_result(writer, status="created", message="", row=row, response=response)
                if args.print_keys:
                    print(f"created: {row.alias} -> {response.get('key', '')}")
                else:
                    print(f"created: {row.alias}")
            except ApiError as exc:
                failed += 1
                message = str(exc)
                write_result(writer, status="failed", message=message, row=row)
                print(f"failed: {row.alias}: {message}", file=sys.stderr)
            finally:
                output_handle.flush()
                if args.sleep > 0:
                    time.sleep(args.sleep)
    finally:
        output_handle.close()

    print(
        f"Done. created={created}, skipped_existing={skipped}, failed={failed}. "
        f"Output: {output_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
