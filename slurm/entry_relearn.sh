#!/bin/bash
#SBATCH --job-name=wargame-relearn
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm-%A_%a.out
# Usage: sbatch slurm/entry_relearn.sh <policy>      (policy = markout | undercut | competitive)
# Array index -> (alpha, seed): 3 alphas x 5 seeds = 15 tasks.  Each task writes its own CSV;
# merge afterwards with:  python slurm/merge_relearn.py
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load anaconda/Python-ML-2025a
source .venv/bin/activate
export TMPDIR="$PWD/.tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1
POLICY="${1:-markout}"
# Prerequisite: trained incumbents from the replication grid.
if ! ls results/cfl_exp2_alpha*_n_mm2_seed0/Q_agent0.npy >/dev/null 2>&1; then
    echo "ERROR: no trained incumbents in results/cfl_exp2_*. Run first:"
    echo "  sbatch slurm/sweep.sh experiments/configs/replicate_cfl_exp.yaml --set market.alpha=0.1,0.3,0.5 --set market.n_mm=2,3 --seeds 0-9"
    exit 1
fi
ALPHAS=(0.1 0.3 0.5)
A=${ALPHAS[$((SLURM_ARRAY_TASK_ID / 5))]}
S=$((SLURM_ARRAY_TASK_ID % 5))
mkdir -p results/relearn_parts
# Each task writes directly to its own file: tasks of one policy finishing together must not
# share a path (an earlier version wrote a common file then mv'd it, and lost ~25% of tasks).
python experiments/entry_relearn.py --policy "$POLICY" --alphas "$A" --seeds "$S-$S" \
    --n-episodes 8000 --window 500 --eps0 0.3 --decay 0.00001 \
    --out "results/relearn_parts/${POLICY}_a${A}_s${S}.csv"
