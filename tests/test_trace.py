"""The demo's traces are the simulator's own runs: the recorded profits must equal what the
experiment code measures on the same seed, so the visual layer cannot drift from the paper."""
import json

import numpy as np

from sim import GlostenMilgromEnv, MarketConfig
from experiments.best_response import CartelMDP, DPEntrant
from experiments.entry import FrozenQ, play
from experiments import retail_trader as rt
from experiments.trace import DEMO, SCENARIOS, SPEC, load_cartel, make


def test_best_response_trace_matches_play():
    tr = make("best_response", DEMO, n_periods=300)
    mk, Q, seed = load_cartel(DEMO)
    N, K, T = int(mk["n_mm"]), int(mk["max_half_spread"]), int(mk["horizon"])
    mdp = CartelMDP(Q, MarketConfig(**{**mk, "n_mm": N + 1}), T)
    policy, _, _ = mdp.solve(1.0)
    env = GlostenMilgromEnv(MarketConfig(**{**mk, "n_mm": N + 1, "seed": 10_000 + seed}))
    ent = DPEntrant(mdp, policy)
    pnl, _, _, periods = play(env, [FrozenQ(Q[i], K, N + 1) for i in range(N)] + [ent], 3, entrant_idx=N, entrant=ent)
    assert periods == 300 == len(tr["p"]["t"])
    np.testing.assert_allclose(np.array(tr["p"]["r"]).mean(axis=0), pnl, atol=2e-3)   # trace rounds to 3 decimals
    rent = tr["meta"]["benchmarks"]["rent"]
    assert abs(tr["meta"]["summary"]["profit_per_period"]["entrant"] - pnl[N] / rent) < 2e-3


def test_every_scenario_is_well_formed():
    for s in SCENARIOS:
        tr = make(s, DEMO, n_periods=120)
        m, p = tr["meta"], tr["p"]
        json.dumps(tr)                                           # serialisable
        if m["mode"] == "speculator":
            assert len(p["t"]) == 120 and all(len(p[k]) == 120 for k in p)
            continue
        A = len(m["agents"])
        assert len(p["t"]) == 120 and all(len(p[k]) == 120 for k in p)
        assert all(len(a) == A for a in p["a"]) and all(len(r) == A for r in p["r"])
        assert all(-1 <= f < A for f in p["f"])
        if m["mode"] == "queue":
            assert all(set(a) <= {0, 1, 2} for a in p["a"])
            assert all(f < 0 or f == sv[0] for f, sv in zip(p["f"], p["sv"]))   # the front of the queue is served
        else:
            assert all(1 <= h <= m["K"] for a in p["a"] for h in a)


def test_savings_runs_follow_the_solved_policy():
    """Each recorded savings run bets exactly when the solved belief-MDP policy says so, its savings are
    the running sum of its results, and it stops only at the quitting line or when the savings are gone."""
    for s in SPEC:
        tr = make(s)
        m, p, P = tr["meta"], tr["p"], tr["meta"]["params"]
        trade = None if P["delta"] is None else rt.solve_speculator(*P["prior"], P["x"], P["mu"], P["delta"])[0]
        n = k = 0
        w = P["savings"]
        stopped = False
        for t in range(len(p["t"])):
            should = not stopped and w > 0 and (trade is None or bool(trade[n, k]))
            assert p["bet"][t] == int(should), (s, t)
            stopped = stopped or not should
            if should:
                n += 1; k += p["right"][t]
                w += p["r"][t]
            assert abs(p["w"][t] - w) < 5e-3 and p["n"][t] == n and p["k"][t] == k   # both rounded to 4 decimals
        if m["ended"] is not None and m["ended_why"] == "stopped for good":
            assert trade is not None and not trade[n, k]