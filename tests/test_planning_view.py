"""The entrant's MDP (experiments/planning_view.py): the ordering of the policies is what the
Bellman equation implies, on the shipped demo cartel."""
import numpy as np

from experiments.best_response import CartelMDP
from experiments.planning_view import discounted_policy, walk
from experiments.trace import DEMO, load_cartel
from sim import MarketConfig


def _mdp():
    mk, Q, _ = load_cartel(DEMO)
    N, T = int(mk["n_mm"]), int(mk["horizon"])
    return CartelMDP(Q, MarketConfig(**{**mk, "n_mm": N + 1}), T), T


def test_optimal_beats_myopic_and_myopic_wins_today():
    mdp, T = _mdp()
    pol_opt, own_dp, _ = mdp.solve(1.0)
    myopic = mdp.r_own.argmax(axis=1)
    tot_opt, _, path_opt = walk(mdp, lambda t, x: pol_opt[t, x] - 1, T)
    tot_my, _, path_my = walk(mdp, lambda t, x: myopic[x], T)
    assert np.isclose(tot_opt / T, own_dp)                    # walking the policy reproduces backward induction
    assert tot_opt >= tot_my - 1e-9                            # the optimal total is at least the myopic one
    assert path_my[0][3] >= path_opt[0][3] - 1e-12             # but the myopic quote earns at least as much today


def test_discounted_policy_is_greedy_on_its_own_values():
    mdp, T = _mdp()
    p0 = discounted_policy(mdp, 0.0)
    assert np.array_equal(p0, mdp.r_own.argmax(axis=1))        # delta = 0 is the myopic policy
    p = discounted_policy(mdp, 0.5)
    tot, _, _ = walk(mdp, lambda t, x: p[x], T)
    tot_my, _, _ = walk(mdp, lambda t, x: p0[x], T)
    assert tot >= tot_my - 1e-9
