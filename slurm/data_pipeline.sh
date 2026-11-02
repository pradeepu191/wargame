#!/bin/bash
#SBATCH --job-name=wargame-data
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
# The Hyperliquid data pipeline on the cluster: calibrate -> stress -> longwindow -> events, from
# data/raw/live (pushed from the Mac; the cluster has no route to the VPS) with the marked-fill cache
# under data/cache.  Outputs land in results/live_2026_10; commit them from the login node.
# Usage: sbatch slurm/data_pipeline.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RAW=data/raw/live; OUT=results/live_2026_10
python data/coverage.py --raw $RAW | tail -3
python analysis/calibrate.py --raw $RAW --out $OUT
python analysis/stress.py --raw $RAW --results $OUT
python analysis/longwindow.py --raw $RAW --results $OUT
python analysis/events.py --raw $RAW --results $OUT
