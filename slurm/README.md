# Running the grids on MIT SuperCloud

SuperCloud facts that matter here: SLURM; CPU nodes have 48 cores / 192 GB (`xeon-p8`);
**compute nodes have no internet**, so install everything from the login node first;
the scheduler is happiest with a few large allocations rather than many tiny ones.
Docs: https://supercloud.mit.edu/ (see "Submitting Jobs" and "Job Arrays").

## Known gotchas (both handled by the scripts)
* The default `anaconda/2023b` module is Python 3.9; the project needs >= 3.10. Scripts load
  `anaconda/Python-ML-2025a` (3.10.14). If that module disappears, `module avail anaconda`.
* The login node's shared `/tmp` (5 GB) is frequently full, which breaks pip with
  "No space left on device". Scripts set `TMPDIR` inside the repo. Your home quota is not the issue.

## One-time setup (login node)
```bash
git clone https://github.com/pradeepu191/wargame.git ~/wargame
cd ~/wargame
bash slurm/setup.sh          # creates a venv in ~/wargame/.venv with the project installed
```

## Run a grid
```bash
sbatch slurm/sweep.sh experiments/configs/replicate_cfl_exp.yaml \
    --set market.alpha=0.1,0.3,0.5 --set market.n_mm=2,3 --seeds 0-9
```
That requests one node, 48 cores, and runs the sweep with `--jobs 48`. The 60-run
replication grid finishes in roughly 3–4 minutes of wall time (vs ~1 h on a laptop).
Everything after the config path is passed through to `experiments/sweep.py`.

Other grids:
```bash
sbatch slurm/sweep.sh experiments/configs/robustness_exp.yaml \
    --set market.n_mm=2,3,5 --agent-set learning_rate=0.05,0.15 --agent-set init_q=0.0,2.0 --seeds 0-9
```

## Run the entry experiments (sequential scripts, parallel over seeds via a job array)
**Prerequisite:** the replication grid (`replicate_cfl_exp.yaml`) must have run on this machine;
the entry experiments load its trained Q-tables from `results/cfl_exp2_*/`.
```bash
sbatch slurm/entry_relearn.sh markout      # one array task per (alpha, seed)
# when the array is done:
python slurm/merge_relearn.py              # -> results/cfl_exp2_entry_relearn_markout.csv
```

## Identity experiment (RQ1: value of wallet identity), 10 seeds
```bash
sbatch slurm/identity.sh                   # -> results/cfl_exp2_identity_10seed.csv (~1 h on 12 cores)
python analysis/identity_fig.py            # figures (point --run / file name as needed)
```

## Watching
```bash
squeue -u $USER                  # queue state
tail -f slurm-<jobid>.out        # live log
```

## Getting results back
Results land in `~/wargame/results/<run>_*`. Commit only the small summary files
(`*_summary.csv`, `*_runs.csv`, `*_impulse.csv`, `*_mechanism.csv`, `fig_*`), never the
per-run directories (they are gitignored). From the login node:
```bash
python analysis/replication.py --run cfl_exp2
python analysis/mechanism.py --run cfl_exp2
git add results/cfl_exp2_summary.csv ...   && git commit && git push
```

## Notes
* SuperCloud sets `OMP_NUM_THREADS=1` by default in some configs; our code is pure Python +
  numpy scalars and never benefits from BLAS threads, so this is fine and avoids oversubscription.
* The `--jobs` count should equal `--cpus-per-task`. Do not exceed 48 on xeon-p8.
* Walltime: the sweep script asks for 2 h, which is far more than any current grid needs;
  raise it for N = 5 PPO runs later.

## Exploitability at ten seeds (RQ2)
```bash
sbatch slurm/exploitability.sh      # trains robust_exp2 seeds 0-9 if missing (~10 min), then attacks
                                    # every population with 500 episodes; -> results/exploitability_10seed.csv,
                                    # exploitability_summary_10seed.csv, fig_exploitability_10seed.*
git add results/exploitability_10seed.csv results/exploitability_summary_10seed.csv results/fig_exploitability_10seed.* && git commit -m "Exploitability, 10 seeds" && git push
```

## Queue game with learned incumbents (RQ3, tick lever)
```bash
sbatch slurm/queue.sh               # ~30 min on 48 cores -> results/queue_hedge_fringe2.csv (+ queue_always_fringe2.csv if missing)
git add results/queue_hedge_fringe2.csv results/queue_always_fringe2.csv && git commit -m "Queue game, no-regret incumbents + fringe" && git push
```

## Identity-aware defenders and liquidity in stress (RQ3, anonymity lever)
```bash
sbatch slurm/queue_identity.sh      # ~45 min on 48 cores -> results/queue_hedge_always_never_anon{,_id}_fringe{0,2}.csv
git add results/queue_hedge_always_never_anon*_fringe*.csv && git commit -m "Identity-aware defenders, liquidity in stress" && git pull --no-rebase && git push
```

## Priority fee (RQ3, venue lever)
```bash
sbatch slurm/queue_fee.sh           # ~1.5 h on 48 cores -> results/queue_fee_anon.csv, queue_fee_id.csv
git add results/queue_fee_*.csv && git commit -m "Priority fee sweep" && git pull --no-rebase && git push
```

## Quoter-side defences and the frontier (RQ3)
```bash
sbatch slurm/defences.sh            # ~1 h on 48 cores -> results/defences.csv
git add results/defences.csv && git commit -m "Quoter defences" && git pull --no-rebase && git push
```

## Train at alpha, test at alpha' (robustness)
```bash
sbatch slurm/alpha_shift.sh         # ~20 min on 48 cores -> results/exploitability_alpha_shift.csv
git add results/exploitability_alpha_shift.csv && git commit -m "Alpha shift" && git pull --no-rebase && git push
```
