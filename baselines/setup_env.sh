#!/bin/bash
# Create a dedicated Python environment for baseline image generation.
# Uses uv for fast installation.
#
# Usage:
#   bash baselines/setup_env.sh [--venv-dir /path/to/venv]
#
# After setup, run with:
#   bash baselines/generate_mjhq.sh --gpus 0,1 \
#       --python /nfs/data/sohyun/venvs/baselines/bin/python

set -euo pipefail

VENV_DIR="/nfs/data/sohyun/venvs/baselines"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv-dir) VENV_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Creating venv at: $VENV_DIR"

# Create venv with uv if available, else python3 -m venv
if command -v uv &>/dev/null; then
    uv venv "$VENV_DIR" --python 3.10
    echo "Installing packages with uv ..."
    uv pip install --python "$VENV_DIR/bin/python" -r baselines/requirements.txt
else
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    echo "Installing packages with pip ..."
    "$VENV_DIR/bin/pip" install -r baselines/requirements.txt
fi

echo ""
echo "Done. Verify FLUX import:"
"$VENV_DIR/bin/python" -c "from diffusers import FluxPipeline; print('FluxPipeline OK')"
"$VENV_DIR/bin/python" -c "from transformers import AutoTokenizer; print('transformers OK')"

echo ""
echo "Run generation with:"
echo "  bash baselines/generate_mjhq.sh --gpus 0,1 --python $VENV_DIR/bin/python"
