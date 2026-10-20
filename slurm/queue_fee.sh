#!/bin/bash
#SBATCH --job-name=wargame-qfee
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
# The priority fee as a venue lever (RQ3).  No-regret incumbents over {always, never, <info>,
# <info>_pay} for info = anon and info = id, a fringe of two always-resting makers, fee sweep
# {0, 0.1, 0.25, 0.5, 0.75, 1.0} x rho_z {0.9, 0.99} x L {1, 5, 20} x spread {h^C, h^C+1}, 5 seeds.
# Per cell: profits net of fees (entrant / incumbents / fringe), venue revenue, who pays by regime,
# fill shares, informed shares, uninformed taker's fill probability by regime.
# Usage: sbatch slurm/queue_fee.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for info in anon id; do
    python experiments/queue_game.py --alphas 0.3 --seeds 0-4 --rhos-z 0.9,0.99 --mark-lags 1,5,20 --above 0,1 \
        --fees 0,0.1,0.25,0.5,0.75,1.0 --incumbents hedge --experts always,never,${info},${info}_pay --fringe 2 \
        --n-train 4000 --n-episodes 300 --jobs "$SLURM_CPUS_PER_TASK" --out "results/queue_fee_${info}.csv"
done
