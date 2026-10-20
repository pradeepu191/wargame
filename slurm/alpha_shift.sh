#!/bin/bash
#SBATCH --job-name=wargame-shift
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
# Train at alpha, test at alpha': every cfl_exp2 cartel (alpha 0.1 / 0.3 / 0.5, N 2 / 3, 10 seeds)
# evaluated on markets of toxicity 0.1, 0.3 and 0.5 -- rent, best-response loss and the fraction
# destroyed off the training distribution.  -> results/exploitability_alpha_shift.csv
# Usage: sbatch slurm/alpha_shift.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python experiments/exploitability.py --dirs "results/cfl_exp2_alpha*_seed*" --eval-alphas 0.1,0.3,0.5 \
    --n-episodes 500 --jobs "$SLURM_CPUS_PER_TASK" --out results/exploitability_alpha_shift.csv
