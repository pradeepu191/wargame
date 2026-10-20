#!/bin/bash
#SBATCH --job-name=wargame-def
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
# RQ3, the makers' own levers: quoter-side defences (random, hold, identity floor) on the frozen
# cfl_exp2 cartels (alpha 0.3, 0.5; seeds 0-9), evaluated on the heterogeneous market alone and
# against seven adversaries incl. a Q-learning entrant trained against each defended cartel (12k
# episodes); per-episode defender profit for the CVaR.  -> results/defences.csv
# Usage: sbatch slurm/defences.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python experiments/defences.py --run cfl_exp2 --alphas 0.3,0.5 --seeds 0-9 --n-episodes 500 --n-train 12000 \
    --jobs "$SLURM_CPUS_PER_TASK" --out results/defences.csv
