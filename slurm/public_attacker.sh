#!/bin/bash
#SBATCH --job-name=wargame-pa
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
# The public-information attacker (identify the reaction function from the tape, then solve) over
# every frozen cfl_exp2 cartel, N = 2 and 3, five probing budgets.  The N = 3 cartels have
# 24^4 states per DP solve, so they are the slow ones.  -> results/public_attacker.csv
# Usage: sbatch slurm/public_attacker.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python experiments/public_attacker.py --dirs "results/cfl_exp2_alpha*_seed*" --probes 0,200,1000,5000,20000 \
    --n-episodes 1000 --resolve-every 1000 --jobs "$SLURM_CPUS_PER_TASK" --out results/public_attacker.csv
