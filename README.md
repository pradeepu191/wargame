# Rent or Toxicity? Exploiting and Defending Learning Market Makers on Hyperliquid

MIT 6.S974 (Games, Learning, and Security), Fall 2026 — wargame project, Example 4.

**Question.** Learning market makers drift to spreads above the competitive level
(Colliard–Foucault–Lovo; Cont–Xiong). A wide spread is *rent* a sharper entrant can
capture, or the correct price of *toxic* flow, and the two look identical on the tape.
Can a trader tell them apart, how much of the rent can it take, and what is Hyperliquid's
public wallet identity worth to it? We study this in a Glosten–Milgrom simulator with
heterogeneous, regime-switching takers and a public identity-tagged tape, and on
Hyperliquid data (the open L4 archive and our own live recorder).

**State of the project (Oct 9 2026).** All four Workshop-1 items have results; see
`results/REPLICATION_NOTES.md` for the running findings log and
`paper/deliverable1_revised.tex` for the revised proposal with results. In short:

* Two or three tabular Q-learners reproduce the supra-competitive spreads
  (collusion index 0.77–0.87, robust over N × learning rate × initialisation, 10 seeds);
  a Q-value mechanism test shows restraint in 96–100% of contested states.
* A markout-reading entrant captures 79–96% of the monopoly rent against frozen cartels
  and keeps it when incumbents re-learn (the cartel re-forms around the entrant). The
  competitive-GM entrant collapses the cartel and earns least.
* "More collusive ⇒ more exploitable" holds only trivially; the fraction of rent a
  profit-maximising entrant destroys is flat (~0.7) in the collusion index.
* The value of wallet identity is zero under the proposal's original mechanism (wallet
  persistence; the data say same-wallet repeats are ~0.03) and reappears under the
  mechanism the data support: informed flow clusters in time, a toxic wallet's print
  reveals the regime before the price does. On Hyperliquid, conditioning on the previous
  taker's class moves the next fill's markout by 0.5–0.9 bps, and 78–99% of that survives
  conditioning on what an anonymous quoter sees.
* The quoted spread on BTC/ETH/SOL is one tick essentially always; true-mid adverse
  selection is −0.33 bps at 10 s on all four coins. On those books rent cannot live in the
  spread, only in which fills a maker gets.

Paper: `paper/deliverable1.tex` (as submitted, Oct 6), `paper/deliverable1_revised.tex`
(revised, no page limit, Oct 9). From-scratch explainer: `paper/companion.tex`.

## Quick start

Use a dedicated environment. Installing into an existing Anaconda `base` upgrades
numpy/pandas/scipy and breaks packages compiled against NumPy 1.x (numba, numexpr, bottleneck).

```bash
conda create -n wargame python=3.11 -y && conda activate wargame   # or: python -m venv .venv
pip install -e ".[dev]"
python -m pytest -q                       # 38 tests, ~15 s
make smoke                                # 50-episode run, writes results/smoke_seed0/
python experiments/sweep.py experiments/configs/replicate_cfl_exp.yaml \
    --set market.alpha=0.1,0.3,0.5 --set market.n_mm=2,3 --seeds 0-9 --jobs 8   # trains the incumbents (hours)
python experiments/entry.py --run cfl_exp2                                        # frozen-incumbent entry
python experiments/exploitability.py --jobs 2                                     # RQ2 test over all populations
python experiments/identity.py --kappas 1 --rhos 0 --rhos-z 0.9,0.99 --mark-lags 1,5,20 --margin 1
```
Cluster scripts for all of the above are in `slurm/` (SuperCloud; see `slurm/README.md`).

## Data

```bash
python data/record.py --coins BTC,ETH,SOL,HYPE --hours 6         # live feed: trades + L2 + top of book
python data/zenodo_to_parquet.py --tar data/raw/zenodo/trades_2026_01.tar --coins BTC,ETH,SOL,HYPE --dates 20260126-20260128
python data/coverage.py --raw data/raw/live                       # gaps, rows per hour, missing ids
python analysis/calibrate.py --raw data/raw/zenodo_parquet --out results/zenodo_2026_01
```
Raw data is never committed; `results/<sample>/calibration.csv` and `wallets_<COIN>.csv` are.
`data/README.md` documents both sources and the archive's conventions
(`side_info` is `[buyer, seller]`; the L1 hash is zero for same-block fills; `start_pos` is
the pre-trade position).

## Layout

```
sim/            environment, exact benchmarks, agents, entrants, runner, impulse-response test
  env.py          Glosten–Milgrom on a tick grid; exponential informed edge; heterogeneous wallets,
                  toxicity regime, mark lag; public post-commitment print
  benchmarks.py   closed-form competitive / monopoly quotes, Delta, type-conditional and identity bounds
  agents.py       Random / FixedSpread / CompetitiveGM / GrimTrigger / tabular Q-learning
  entrants.py     competitive, undercut, own-fill markout, WalletEntrant (anon / id / oracle, regime filter)
experiments/    sweep.py (grids), entry.py, entry_relearn.py, exploitability.py, identity.py, configs/
analysis/       replication, mechanism (restraint test), robustness, exploitability regressions,
                entry / relearn / identity figures, wallets (classifier, persistence, conditional
                markouts), l2 (true-mid markouts), calibrate (one row per coin from either data source)
data/           record.py (live recorder), zenodo_to_parquet.py (archive converter), coverage.py, README
slurm/          SuperCloud scripts and README
paper/          LaTeX: deliverable1 (submitted), deliverable1_revised, companion, REVIEW_RESPONSE.md
results/        small summary CSVs and figures (run directories are gitignored); REPLICATION_NOTES.md
tests/          pytest (38)
```

## The model in one screen

* $N$ makers post symmetric quotes at half-spread $h\in\{1,\dots,K\}$ ticks around the mid.
  One taker per period: informed w.p. $\alpha$ (trades only if the edge $d\sim\mathrm{Exp}(\mu)$
  exceeds the best half-spread, so no quote is immune), else uninformed with valuation
  $m+L$, $L\sim N(0,\sigma_L)$. Best quote wins, ties split.
* Market-wide profit of quoting $h$ is closed-form,
  $\Pi(h;\alpha)=(1-\alpha)\,2x\bar\Phi(x/\sigma_L)-\alpha\mu e^{-x/\mu}$, $x=h\cdot$tick;
  $h^C$ is the first grid point with $\Pi\ge0$, $h^M$ its maximiser,
  $\Delta=(\bar\Pi-\Pi^C)/(\Pi^M-\Pi^C)$.
* `n_wallets > 0`: a population of wallets with persistent types $\alpha_j$ (dispersion
  `wallet_concentration`), optional same-wallet persistence `wallet_persistence`, a latent
  toxicity regime `regime_persistence` that tilts wallet activity, and a `mark_lag` between a
  print and its public mark. With `n_wallets = 0` the RNG stream is untouched and every earlier
  result reproduces bit-for-bit (pinned in tests).
* Information sets for the entrant: anon (prints and marks, no ids) ⊂ id (prints keyed by wallet)
  ⊂ oracle (true types). Nobody sees the identity of the order about to hit them.

## Conventions

* Every run records config, seed and git hash. Report distributions over seeds, never a single run.
* Common random numbers across policies within a cell; classification on the first half of a data
  sample, measurement on the second.
* This repo never holds exchange keys or raw market data.
* Findings go into `results/REPLICATION_NOTES.md` as they happen, including the ones that
  contradict the proposal.

## Roadmap

- [x] Replicate Colliard et al.; mechanism (restraint) test; robustness grid
- [x] Entrant with markout inference vs competitive GM; re-learning incumbents; three policies
- [x] Exploitability vs rent across 120 populations (RQ2 as measurement)
- [x] Heterogeneous takers, regime, mark lag; value of identity grids
- [x] Data: Zenodo archive converter, live recorder (trades, L2, bbo), calibration pipeline,
      archive-vs-feed agreement, true-mid markouts, identity split on real data
- [ ] 10-seed cluster runs of the identity grids at the break-even rule (`slurm/identity.sh`)
- [ ] October archive (the Oct 10 cascade); clean book run with `bbo`; top-maker realized spread
- [ ] Queue-position model for one-tick books (RQ3); priority fee as the price of position
- [ ] Re-learning / identity-aware incumbents; steering attacker; inventory from `start_pos`
- [ ] December L4 lifecycles: reaction functions, cancellation before toxic flow
- [ ] Defences and the profit–resilience frontier (RQ4); preprint by mid-December
