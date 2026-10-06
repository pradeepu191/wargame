#!/bin/bash
# One-time environment setup on the SuperCloud LOGIN node (compute nodes have no internet).
set -euo pipefail
cd "$(dirname "$0")/.."
module load anaconda/2023b 2>/dev/null || module load anaconda 2>/dev/null || true
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
python -m pytest -q
echo "OK: venv at $(pwd)/.venv"
