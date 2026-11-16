#!/bin/bash
#SBATCH --job-name=wargame-ws2
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
# The Workshop 2 grids at full size (paper/WORKSHOP2_PLAN.md).  The laptop runs committed on
# Nov 15 are the same experiments on 30 two-maker cartels and 3 seeds; these overwrite them.
#   1. public-information attacker vs re-learning cartels   -> results/public_attacker_relearn.csv
#   2. double oracle / PSRO to convergence, two modes       -> results/psro.csv
#   3. queue game: steering no-regret incumbents            -> results/queue_steer.csv
#   4. queue game: the request allowance                    -> results/queue_budget.csv
#   5. quoter defences with the exact best response and the public attacker added
#                                                            -> results/defences.csv
# Prerequisite: the cfl_exp2 replication grid in results/ (slurm/sweep.sh).  N = 3 cartels have
# 24^4 states per DP solve; steps 1 and 2 use the two-maker cartels (add n_mm3 to the glob to
# include them, at roughly 20x the time).
# Usage: sbatch slurm/workshop2.sh            (or: sbatch slurm/workshop2.sh 5   to run one step)
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
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
J="${SLURM_CPUS_PER_TASK:-8}"
ONLY="${1:-all}"
run() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }

if run 1; then
python experiments/public_attacker_relearn.py --dirs "results/cfl_exp2_alpha*_n_mm2_seed*" \
    --n-episodes 3000 --n-frozen 500 --windows 2000,20000 --jobs "$J" --out results/public_attacker_relearn.csv
fi
if run 2; then
python experiments/psro.py --dirs "results/cfl_exp2_alpha*_n_mm2_seed*" --modes last,uniform --rounds 12 \
    --relearn-episodes 3000 --n-eval 500 --jobs "$J" --out results/psro.csv
fi
if run 3; then
python experiments/queue_attacks.py --part steer --seeds 0-9 --rhos-z 0.9,0.99 --mark-lags 5,20 --above 0,1 \
    --n-train 4000 --n-eval 300 --jobs "$J" --out results/queue_steer.csv
fi
if run 4; then
python experiments/queue_attacks.py --part budget --seeds 0-9 --rhos-z 0.9,0.99 --mark-lags 5,20 --above 0,1 \
    --budgets inf,8,4,2,1,0 --jobs "$J" --out results/queue_budget.csv
fi
if run 5; then
python experiments/defences.py --run cfl_exp2 --alphas 0.3,0.5 --seeds 0-9 --n-episodes 500 --n-train 12000 \
    --jobs "$J" --out results/defences.csv
fi
