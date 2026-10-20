#!/bin/bash
#SBATCH --job-name=wargame-qid
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
# Identity-aware defenders and liquidity in the toxic state (RQ3: anonymity as a venue lever, from
# the defenders' side).  No-regret incumbents over {always, never, anon} vs {always, never, anon, id},
# with and without a fringe of two always-resting makers; 5 seeds x rho_z {0.9, 0.99} x
# L {1, 5, 20, 60} x spread {h^C, h^C+1}.  Outputs per cell: entrant / incumbent / fringe profit by
# entrant policy, expert weights, share of toxic-state periods with nobody resting, and the fill
# probability of an uninformed taker by regime.
# Usage: sbatch slurm/queue_identity.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for ex in always,never,anon always,never,anon,id; do
  for f in 0 2; do
    tag=$(echo "$ex" | tr ',' '_')_fringe${f}
    python experiments/queue_game.py --alphas 0.3 --seeds 0-4 --rhos-z 0.9,0.99 --mark-lags 1,5,20,60 --above 0,1 \
        --incumbents hedge --experts "$ex" --fringe "$f" --n-train 4000 --n-episodes 300 \
        --jobs "$SLURM_CPUS_PER_TASK" --out "results/queue_hedge_${tag}.csv"
  done
done
