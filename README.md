# Rent or Toxicity? Exploiting and Defending Learning Market Makers on Hyperliquid

MIT 6.S974 (Games, Learning, and Security), Fall 2026 — wargame project, Example 4.

**Question.** Learning market makers drift to spreads above the competitive level
(Colliard–Foucault–Lovo; Cont–Xiong). A wide spread is *rent* a sharper entrant can
capture, or the correct price of *toxic* flow, and the two look identical on the tape.
Can a trader tell them apart, how exploitable are learned quoting policies, and what
quoter/venue design resists exploitation? We study this in a Glosten–Milgrom
simulator with Hyperliquid's mechanics (public wallet identities, request budgets,
priced latency, tick discontinuities, liquidation flow) and on Hyperliquid L4 data.

Paper draft: `paper/deliverable1.tex`. From-scratch explainer: `paper/companion.tex`.

## Quick start

```bash
pip install -e ".[dev]"
make test          # 9 unit tests, ~6 s
make smoke         # 50-episode run, writes results/smoke_seed0/
make refs          # known-competitive (Delta ~ 0) and known-collusive (Delta ~ 1) references
make replicate     # Colliard et al. baseline grid: alpha x N x 10 seeds (hours; run on cluster)
```

## Layout

```
sim/            environment, exact benchmarks, agents, runner, impulse-response test
  env.py          Glosten–Milgrom dealer market on a tick grid, N quoters, one taker per period
  benchmarks.py   exact competitive & joint-monopoly quotes; collusion index Delta
  agents.py       Random / FixedSpread / CompetitiveGM / GrimTrigger / tabular Q-learning
  impulse.py      force a deviation, watch for punishment (Calvano et al. 2020)
  runner.py       YAML config -> episodes.csv + tapes + meta.json (git hash, seed)
experiments/    configs/*.yaml and sweep.py (grid over parameters and seeds)
analysis/       markouts.py (effective/realized spread, price impact), wallets.py (stub)
data/           download.py + README; raw data is gitignored
paper/          LaTeX
results/        run outputs (gitignored except small summary tables)
tests/          pytest
```

## The model in one screen

* $V \in \{V_L, V_H\}$, prior 1/2. `redraw_v_each_period: true` gives the stationary
  repeated game (replication baseline); `false` makes $V$ persist so the public
  belief $\mu_t$ evolves and price discovery matters.
* Each MM posts symmetric quotes at half-spread $h \in \{1,\dots,K\}$ ticks around
  the mid. One taker per period: informed w.p. $\alpha$ (toxicity), else uninformed
  with valuation $m_t + L$, $L \sim N(0,\sigma_L)$. Best quote wins, ties split.
* Payoff per fill $(a - V)$ on a sale, $(V - b)$ on a purchase; inventory penalty $\phi I^2$.
* Benchmarks are exact: $h^C(\mu)$ is the first grid point with non-negative expected
  profit (Bertrand stops there), $h^M(\mu)$ maximises it. With the defaults
  (`tick=1, K=15, sigma_L=6`) the competitive/monopoly half-spreads are
  (2, 5) at $\alpha=0.1$, (6, 10) at $\alpha=0.3$, (9, 10) at $\alpha=0.5$.
  Note $\pi^M - \pi^C$ shrinks with $\alpha$: less rent is available in toxic markets.
* $\Delta = (\bar\pi - \pi^C)/(\pi^M - \pi^C)$ is reported per episode in `episodes.csv`.

## Conventions

* Every run records config, seed and git hash. Report distributions over seeds, never a single run.
* Notebooks are committed with outputs stripped (`nbstripout --install`).
* This repo never holds exchange keys. Any live bot lives elsewhere.
* Branch per task, PR to `main`, CI must pass (`pytest` + smoke run).

## Roadmap (see paper, Section 4)

- [ ] Replicate Colliard et al.: markups fall with $N$ and with $\alpha$
- [ ] Impulse-response classification: punishment vs under-exploration
- [ ] Entrant with markout inference vs competitive benchmark (RQ1)
- [ ] Data pipeline: one week of BTC L4, MM wallet classification, calibrate $\alpha$
- [ ] Attackers: undercutter, steering MM, rate-limit drainer, predatory taker, PSRO (RQ2)
- [ ] Request budgets $B_t$ and priced latency in `env.py`
- [ ] Quoter/venue design grid + cascade replay (RQ3)
