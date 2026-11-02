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
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# the module function is not inherited when the job is submitted from a non-interactive ssh command
if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh; do
    [ -f "$f" ] && source "$f" && break
  done
fi
set -euo pipefail
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
