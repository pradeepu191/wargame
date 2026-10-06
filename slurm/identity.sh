#!/bin/bash
#SBATCH --job-name=wargame-identity
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
# Identity experiment (RQ1), 10 seeds x 3 alphas x 3 kappas x 4 rhos x 5 policies, 1000 episodes.
# Usage: sbatch slurm/identity.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1
if ! ls results/cfl_exp2_alpha*_n_mm2_seed0/Q_agent0.npy >/dev/null 2>&1; then
    echo "ERROR: no trained incumbents in results/cfl_exp2_*. Run the replication grid first."; exit 1
fi
python experiments/identity.py --run cfl_exp2 --alphas 0.1,0.3,0.5 --seeds 0-9 \
    --kappas 1e6,5,1 --rhos 0,0.5,0.9,0.99 --n-wallets 50 --n-episodes 1000 \
    --jobs "$SLURM_CPUS_PER_TASK" --out results/cfl_exp2_identity_10seed.csv
