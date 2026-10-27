#!/bin/bash
#SBATCH --job-name=wargame-nr
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
# Strategizing against a live no-regret (EXP3) market maker, Deng-Schneider-Sivan setting:
# three toxicity levels, one and two incumbents, three learning rates, 3 selection + 7 evaluation
# seeds.  -> results/no_regret.csv.  Usage: sbatch slurm/no_regret.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python experiments/no_regret.py --alphas 0.1,0.3,0.5 --n-inc 1,2 --etas 0.02,0.05,0.2 --seeds 0-9 --n-select 3 \
    --train-episodes 3000 --attack-episodes 2000 --jobs "$SLURM_CPUS_PER_TASK" --out results/no_regret.csv
