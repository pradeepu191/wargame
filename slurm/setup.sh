#!/bin/bash
# One-time environment setup on the SuperCloud LOGIN node (compute nodes have no internet).
set -euo pipefail
cd "$(dirname "$0")/.."
# anaconda/2023b is Python 3.9 (too old); Python-ML-2025a is 3.10.14.  Check: module avail anaconda
module load anaconda/Python-ML-2025a
# The login node's shared /tmp is often full; keep pip's temp and cache in the repo.
export TMPDIR="$PWD/.tmp" PIP_CACHE_DIR="$PWD/.pipcache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
python -m pytest -q
rm -rf "$TMPDIR" "$PIP_CACHE_DIR"
echo "OK: venv at $(pwd)/.venv"
