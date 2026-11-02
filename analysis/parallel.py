"""Run one coin per process.  Each analysis script defines coin_job(coin, args) -> result and
calls per_coin(coin_job, coins, args, jobs); the workers' printed output is captured and shown
in coin order once they finish, so the log reads the same as the sequential version."""
from __future__ import annotations

import io
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path


def _worker(fn, coin, args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            result = fn(coin, args)
        except Exception as e:                       # report, do not kill the other coins
            print(f"{coin}: FAILED: {type(e).__name__}: {e}")
            result = None
    return buf.getvalue(), result


def per_coin(fn, coins, args, jobs: int = 1):
    jobs = max(1, min(int(jobs), len(coins)))
    if jobs == 1:
        out = []
        for c in coins:
            out.append(fn(c, args))
        return out
    results = []
    with ProcessPoolExecutor(jobs) as ex:
        futures = [ex.submit(_worker, fn, c, args) for c in coins]
        for c, f in zip(coins, futures):
            text, result = f.result()
            print(text, end="", flush=True)
            results.append(result)
    return results
