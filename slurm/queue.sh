#!/bin/bash
#SBATCH --job-name=wargame-queue
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
# The one-tick queue game (RQ3, tick lever) with LEARNED incumbents: two Q-learners + a fringe of
# two always-resting makers, 12k training episodes per cell, then the four entrant policies; and
# the like-for-like baseline (always-resting incumbents with the same fringe).  5 seeds x
# rho_z {0.9, 0.99} x L {1, 5, 20, 60} x spread {h^C, h^C+1}.  Each cell is ~15 min on one core.
# Usage: sbatch slurm/queue.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python experiments/queue_game.py --alphas 0.3 --seeds 0-4 --rhos-z 0.9,0.99 --mark-lags 1,5,20,60 --above 0,1 \
    --incumbents q --fringe 2 --discount 0.95 --n-train 12000 --n-episodes 300 \
    --jobs "$SLURM_CPUS_PER_TASK" --out results/queue_q_fringe2.csv
python experiments/queue_game.py --alphas 0.3 --seeds 0-4 --rhos-z 0.9,0.99 --mark-lags 1,5,20,60 --above 0,1 \
    --incumbents always --fringe 2 --n-episodes 300 \
    --jobs "$SLURM_CPUS_PER_TASK" --out results/queue_always_fringe2.csv
