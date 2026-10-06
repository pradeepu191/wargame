#!/bin/bash
#SBATCH --job-name=wargame-sweep
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
# Usage: sbatch slurm/sweep.sh <config.yaml> [sweep.py args...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONFIG="$1"; shift
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} config=${CONFIG} args=$*"
python experiments/sweep.py "$CONFIG" "$@" --jobs "${SLURM_CPUS_PER_TASK}"
