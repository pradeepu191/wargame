"""Run a config over a grid of overrides and seeds.

Example (RQ1 baseline grid):
  python experiments/sweep.py experiments/configs/replicate_cfl_2mm.yaml \
      --set market.alpha=0.1,0.3,0.5 --set market.n_mm=2,3 --seeds 0-9
"""
import argparse, copy, itertools, sys
from multiprocessing import Pool
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `sim` imports without install
from sim.runner import run


def _run_one(args):
    path, out = args
    return str(run(path, results_root=out))

def set_path(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def parse_val(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], help="key=v1,v2,...  (dotted path into the config)")
    ap.add_argument("--agent-set", action="append", default=[],
                    help="param=v1,v2,...  applied to every agent's params (e.g. learning_rate=0.05,0.15)")
    ap.add_argument("--seeds", default="0", help="e.g. 0-9 or 0,3,7")
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes")
    args = ap.parse_args()
    base = yaml.safe_load(Path(args.config).read_text())
    grid = []
    for s in args.set:
        k, vs = s.split("=")
        grid.append([(k, parse_val(v)) for v in vs.split(",")])
    for s in args.agent_set:
        k, vs = s.split("=")
        grid.append([("agents.*." + k, parse_val(v)) for v in vs.split(",")])
    if "-" in args.seeds:
        a, b = map(int, args.seeds.split("-")); seeds = range(a, b + 1)
    else:
        seeds = [int(x) for x in args.seeds.split(",")]
    tmp = Path(args.out) / "_configs"; tmp.mkdir(parents=True, exist_ok=True)
    jobs = []
    for combo in itertools.product(*grid) if grid else [()]:
        for seed in seeds:
            cfg = copy.deepcopy(base)
            tag = "_".join(f"{k.split('.')[-1]}{v}" for k, v in combo)
            for k, v in combo:
                if k.startswith("agents.*."):
                    continue
                set_path(cfg, k, v)
            if "n_mm" in [k.split(".")[-1] for k, _ in combo]:
                # keep agents list in sync with n_mm: replicate the first agent spec
                cfg["agents"] = [copy.deepcopy(cfg["agents"][0]) for _ in range(cfg["market"]["n_mm"])]
            for k, v in combo:                       # agent-param overrides, after the list is sized
                if k.startswith("agents.*."):
                    for ag in cfg["agents"]:
                        ag.setdefault("params", {})[k[len("agents.*."):]] = v
            cfg["seed"] = seed
            cfg["run_name"] = f"{base['run_name']}_{tag}" if tag else base["run_name"]
            p = tmp / f"{cfg['run_name']}_seed{seed}.yaml"
            p.write_text(yaml.safe_dump(cfg))
            jobs.append((str(p), args.out))
    print(f"{len(jobs)} runs, {args.jobs} parallel")
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            for out in pool.imap_unordered(_run_one, jobs):
                print("done:", out, flush=True)
    else:
        for j in jobs:
            print("done:", _run_one(j), flush=True)

if __name__ == "__main__":
    main()
