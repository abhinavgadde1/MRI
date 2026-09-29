#!/usr/bin/env bash
# Lightweight environment setup for the MRI pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

if [[ ! -d "$VENV" ]]; then
  echo "Creating $VENV ..."
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Editable install for package imports; --no-deps keeps the freeze pins
python -m pip install -e ".[dev]" --no-deps

echo
echo "Setup complete."
echo "  source $VENV/bin/activate"
echo "  export PYTHONPATH=src"
echo "  make demo          # synthetic smoke"
echo "  make test"
echo
echo "Note: v3 weights (checkpoints/brats_scale_full/best_model.pt, ~54MB) are"
echo "gitignored. Place them locally for real inference; make demo works without them."
