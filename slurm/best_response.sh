#!/bin/bash
#SBATCH --job-name=wargame-br
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
# The exact best response to every frozen cfl_exp2 cartel (backward induction over the cartel's
# reaction function) on the attacker's profit--damage frontier (lambda = 1 own profit ... 0 damage),
# then three double-oracle rounds: the cartel re-learns against the best response, the entrant
# re-solves.  -> results/best_response.csv (frontier) and results/best_response_relearn.csv
# Usage: sbatch slurm/best_response.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python experiments/best_response.py --dirs "results/cfl_exp2_alpha*_seed*" --lambdas 1,0.75,0.5,0.25,0 \
    --n-episodes 500 --jobs "$SLURM_CPUS_PER_TASK" --out results/best_response.csv
python experiments/best_response.py --dirs "results/cfl_exp2_alpha*_seed*" --lambdas 1 \
    --relearn-rounds 3 --relearn-episodes 3000 --n-episodes 500 --jobs "$SLURM_CPUS_PER_TASK" \
    --out results/best_response_relearn.csv
