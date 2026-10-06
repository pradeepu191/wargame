# Workshop 1, item (i): Colliard–Foucault–Lovo replication — findings

Grid: alpha ∈ {0.1, 0.3, 0.5} × N ∈ {2, 3} × 10 seeds, tabular Q-learning (lr 0.15, γ 0.95,
ε = exp(−4e-6·t)), i.i.d. V, 20k episodes × 100 periods = 2M steps/agent. Converged statistics
are over the last 2,000 episodes (ε < 1e-3). Files: `cfl_2mm_summary.csv`, `cfl_2mm_runs.csv`,
`cfl_2mm_impulse.csv`, `fig_cfl_2mm.{png,pdf}`. Reproduce: `make replicate` then
`python analysis/replication.py --run cfl_2mm`.

| α | N | learned h (ticks) | h^C | h^M | markup over h^C | Δ | cliff share | change rate |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 2 | 9.1 ± 0.3 | 2 | 5 | 7.1 | 0.49 ± 0.05 | 0.49 | 1.00 |
| 0.1 | 3 | 8.6 ± 0.6 | 2 | 5 | 6.6 | 0.58 ± 0.08 | 0.36 | 1.00 |
| 0.3 | 2 | 10.7 ± 0.2 | 6 | 10 | 4.7 | 0.73 ± 0.04 | 0.90 | 1.00 |
| 0.3 | 3 | 10.3 ± 0.2 | 6 | 10 | 4.3 | 0.82 ± 0.02 | 0.89 | 1.00 |
| 0.5 | 2 | 10.9 ± 0.2 | 9 | 10 | 1.9 | 0.75 ± 0.06 | 0.99 | 1.00 |
| 0.5 | 3 | 10.6 ± 0.1 | 9 | 10 | 1.6 | 0.83 ± 0.03 | 1.00 | 1.00 |

## Reproduced (qualitatively matches CFL)
1. Learned quotes are far above the competitive quote in every cell (markup 1.6–7.1 ticks).
2. Markup over the competitive quote **falls with toxicity α** and (weakly) **falls with N**.
3. Quoted spreads widen with α (9.1 → 10.9 ticks): the learners respond to adverse selection.

## Not reproduced / new
4. The normalized collusion index Δ **rises** with α and with N. Reason: the learned quote is nearly
   invariant (≈ 9–11 ticks) while the benchmarks move toward it as α rises. Δ is the wrong summary
   statistic for this comparison; CFL's markup-over-competitive is the one that matches.
   TODO: confirm CFL's exact markup definition from the paper before writing this up.
5. **The learners sit on the adverse-selection cliff.** With binary V and |V − mid| = 10 ticks, any
   quote with h ≥ 10 is immune to informed traders. `cliff share` = fraction of periods where the best
   quote is immune: 0.36–0.49 at α = 0.1, ≈ 0.90 at α = 0.3, ≈ 0.99 at α = 0.5. The agents have learned
   to *exclude* informed flow, not to *price* it. Mechanism: tight quotes deliver −8 to −10 shocks from
   informed fills; with lr = 0.15 (effective memory ~7 samples) those shocks dominate the Q-values of
   tight actions and the learners retreat to the immune region. This is CFL's noisy-feedback mechanism
   in an extreme form, and it is an artifact of a fixed, finite informed edge.
6. Converged greedy joint policies are **limit cycles, not fixed points**: at least one MM changes its
   quote every period (change rate 1.00), with cycle amplitude 5–11 ticks. Consistent with Klein (2021)
   Edgeworth cycles.
7. Paired impulse response (force one MM to h = 1, difference shocked vs baseline trajectories): rivals
   tighten by ≈ 2.4 ticks for one period at α = 0.3, ≈ 1–1.7 ticks elsewhere, then return within 2–3
   periods. Compare the hand-coded grim trigger: −4 ticks for 5 periods. There is a reaction to an
   undercut but no sustained punishment phase.

## Implications for the model (do before RQ1)
* Replace the fixed informed edge with a continuous one (e.g. V = mid ± d, d ~ Exponential or
  Uniform(0, 2v)) so that no quote is immune. Benchmarks in `sim/benchmarks.py` must integrate over d.
  Without this, RQ1's markout signal is degenerate: informed trades barely occur at α ≥ 0.3.
* Robustness grid on learner hyperparameters: Q initialization (0 vs optimistic), learning rate
  (0.05 vs 0.15). Referees will ask.
* Report cycle statistics and the paired impulse response alongside Δ in every table.

---

# Continuous (exponential) informed edge — the cliff removed

Same grid and learner settings; market: V = mid ± d, d ~ Exp(mean 5), σ_L = 8, tick 0.5, K = 24.
An informed trader trades at half-spread x w.p. e^(−x/5); expected loss per informed fill is 5 at every
quote (memorylessness). Files: `cfl_exp_summary.csv`, `cfl_exp_runs.csv`, `cfl_exp_impulse.csv`,
`fig_cfl_exp.{png,pdf}`. Config: `experiments/configs/replicate_cfl_exp.yaml`.

| α | N | learned h (ticks) | h^C | h^M | Δ | best-quote mean ± std | near-monopoly share | leader-switch rate |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 2 | 12.3 ± 0.3 | 2 | 13 | 0.86 ± 0.01 | 12.8 ± 4.2 | 0.46 | 0.51 |
| 0.1 | 3 | 11.8 ± 0.2 | 2 | 13 | 0.87 ± 0.01 | 12.2 ± 3.9 | 0.46 | 0.75 |
| 0.3 | 2 | 16.0 ± 0.2 | 4 | 14 | 0.85 ± 0.01 | 16.2 ± 4.4 | 0.27 | 0.61 |
| 0.3 | 3 | 15.9 ± 0.1 | 4 | 14 | 0.85 ± 0.02 | 16.3 ± 4.6 | 0.30 | 0.80 |
| 0.5 | 2 | 17.7 ± 0.3 | 8 | 17 | 0.79 ± 0.02 | 17.9 ± 4.0 | 0.32 | 0.50 |
| 0.5 | 3 | 17.6 ± 0.2 | 8 | 17 | 0.77 ± 0.02 | 18.0 ± 3.9 | 0.37 | 0.67 |

(best-quote = min half-spread across MMs each period; leader-switch = fraction of periods in which the
identity of the best quoter changes; near-monopoly = share of periods with |best − h^M| ≤ 2.)

## Findings
1. **Strong, stable learned collusion.** Δ = 0.77–0.87 in every cell with seed std ≤ 0.02. The mean
   learned quote tracks the monopoly benchmark (12.3 / 16.0 / 17.7 vs 13 / 14 / 17). Compare the
   fixed-edge model: Δ = 0.49–0.83 with quotes pinned to the immune cliff rather than to h^M.
2. **Δ now falls with α** (0.86 → 0.85 → 0.79), the CFL direction. **N has no effect** (N = 3 ≈ N = 2 in
   every cell), which does *not* match CFL's "markups fall with N". Test N = 5 before concluding.
3. **The collusion is turn-taking, not symmetric wide quoting.** Converged play is a limit cycle in
   which one MM holds the best quote near h^M while the other(s) park far away (up to the grid edge),
   then they swap; the best quoter changes in 50–80% of periods. The market-relevant quote (the
   minimum) is far more stable than any individual quote. This is the alternating-monopoly /
   market-sharing equilibrium of repeated Bertrand competition, found by independent Q-learners.
4. **Impulse response is not interpretable as punishment under turn-taking.** Rivals move −4 to −5
   ticks one period after a forced undercut, then oscillate ±2–4 ticks: the shock perturbs the cycle
   phase. A different mechanism test is needed (next item).

## Implications
* For RQ1/RQ2 this is the best case for an entrant: the "off" incumbent is parked far away and the
  "on" incumbent quotes near monopoly, so a one-tick undercut wins every period. And the scheme is
  *visible on the tape* as alternation of the best-quoter identity — which on Hyperliquid is
  observable directly through wallet IDs. Turn-taking detection is a concrete, data-testable signal.
* Next mechanism test (replaces impulse response for cycling policies): from the learned Q-tables,
  compare Q(s, undercut) vs Q(s, cooperate) in the on-turn states. If the undercut has the higher
  immediate reward but the lower Q, the punishment is encoded in the value function; if Q also
  favours cooperating only because the immediate reward does, there is no punishment.
* Robustness grid still pending: Q initialization, learning rate, N = 5.
