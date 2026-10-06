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
# merge afterwards with:  python -c "import pandas as pd,glob; pd.concat(map(pd.read_csv, glob.glob('results/relearn_parts/*.csv'))).to_csv('results/cfl_exp2_entry_relearn_<policy>.csv', index=False)"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate
export OMP_NUM_THREADS=1
POLICY="${1:-markout}"
ALPHAS=(0.1 0.3 0.5)
A=${ALPHAS[$((SLURM_ARRAY_TASK_ID / 5))]}
S=$((SLURM_ARRAY_TASK_ID % 5))
mkdir -p results/relearn_parts
python experiments/entry_relearn.py --policy "$POLICY" --alphas "$A" --seeds "$S-$S" \
    --n-episodes 8000 --window 500 --eps0 0.3 --decay 0.00001
mv "results/cfl_exp2_entry_relearn_${POLICY}.csv" "results/relearn_parts/${POLICY}_a${A}_s${S}.csv"
