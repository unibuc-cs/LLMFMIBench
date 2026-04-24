#!/bin/bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-$HOME/models}"

SMALL_REPO="unsloth/Qwen3-0.6B-GGUF"
SMALL_FILE="Qwen3-0.6B-Q4_K_M.gguf"
SMALL_DIR="$BASE_DIR/qwen3-0.6b"

LARGE_REPO="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
LARGE_FILE="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
LARGE_DIR="$BASE_DIR/qwen3-coder-30b-a3b"

echo "Creating model directories..."
mkdir -p "$SMALL_DIR" "$LARGE_DIR"

echo "Checking Python environment..."

if command -v uv >/dev/null 2>&1; then
    echo "Using uv..."
    uv pip install -U huggingface_hub hf_xet
else
    echo "uv not found, using pip..."
    python3 -m pip install -U huggingface_hub hf_xet
fi

echo ""
echo "============================================================"
echo "Downloading small model"
echo "Repo: $SMALL_REPO"
echo "File: $SMALL_FILE"
echo "Destination: $SMALL_DIR"
echo "============================================================"

hf download "$SMALL_REPO" \
    "$SMALL_FILE" \
    --local-dir "$SMALL_DIR"

echo ""
echo "============================================================"
echo "Downloading large model"
echo "Repo: $LARGE_REPO"
echo "File: $LARGE_FILE"
echo "Destination: $LARGE_DIR"
echo "============================================================"

hf download "$LARGE_REPO" \
    "$LARGE_FILE" \
    --local-dir "$LARGE_DIR"

echo ""
echo "============================================================"
echo "Download finished."
echo "Downloaded GGUF files:"
echo "============================================================"

find "$BASE_DIR" -type f -name "*.gguf" -exec ls -lh {} \;

echo ""
echo "Small model path:"
echo "$SMALL_DIR/$SMALL_FILE"

echo ""
echo "Large model path:"
echo "$LARGE_DIR/$LARGE_FILE"