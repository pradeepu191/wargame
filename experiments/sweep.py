"""Run a config over a grid of overrides and seeds.

Example (RQ1 baseline grid):
  python experiments/sweep.py experiments/configs/replicate_cfl_2mm.yaml \
      --set market.alpha=0.1,0.3,0.5 --set market.n_mm=2,3 --seeds 0-9
"""
import argparse, copy, itertools
from pathlib import Path
import yaml
from sim.runner import run

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
    ap.add_argument("--set", action="append", default=[], help="key=v1,v2,...")
    ap.add_argument("--seeds", default="0", help="e.g. 0-9 or 0,3,7")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    base = yaml.safe_load(Path(args.config).read_text())
    grid = []
    for s in args.set:
        k, vs = s.split("=")
        grid.append([(k, parse_val(v)) for v in vs.split(",")])
    if "-" in args.seeds:
        a, b = map(int, args.seeds.split("-")); seeds = range(a, b + 1)
    else:
        seeds = [int(x) for x in args.seeds.split(",")]
    tmp = Path(args.out) / "_configs"; tmp.mkdir(parents=True, exist_ok=True)
    for combo in itertools.product(*grid) if grid else [()]:
        for seed in seeds:
            cfg = copy.deepcopy(base)
            tag = "_".join(f"{k.split('.')[-1]}{v}" for k, v in combo)
            for k, v in combo:
                set_path(cfg, k, v)
            cfg["seed"] = seed
            cfg["run_name"] = f"{base['run_name']}_{tag}" if tag else base["run_name"]
            p = tmp / f"{cfg['run_name']}_seed{seed}.yaml"
            p.write_text(yaml.safe_dump(cfg))
            run(str(p), results_root=args.out)

if __name__ == "__main__":
    main()
