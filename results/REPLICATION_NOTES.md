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


---

# NOTE: state-encoding bug in all `cfl_2mm_*` and `cfl_exp_*` results above

`QLearningAgent._state` computed the rival component as `min(own, rivals)` rather than
`min(rivals)`: the agent could not distinguish a rival quoting *wider* than itself from one
quoting at the same level. Found by the Q-value mechanism test (`analysis/mechanism.py`), which
showed half the tape-visited (own, rival) states had never been updated. Fixed in commit
"Fix rival-state encoding". The tables above are retained as the record of that encoding; the
corrected replication is `cfl_exp2_*` / `robust_exp2_*` below. A quick A/B (4k episodes, 3 seeds,
α = 0.3, N = 2) gave Δ ≈ 0.71 with the fix, so the qualitative finding (strong learned
collusion under the continuous edge) survives; magnitudes and cycle structure may change.

---

# Corrected replication (`cfl_exp2`): exponential edge, fixed state encoding

Same grid as `cfl_exp` (α ∈ {0.1, 0.3, 0.5} × N ∈ {2, 3} × 10 seeds, 2M steps/agent), with the
rival-state bug fixed. Files: `cfl_exp2_summary.csv`, `cfl_exp2_runs.csv`, `cfl_exp2_impulse.csv`,
`cfl_exp2_mechanism.csv`, `fig_cfl_exp2.{png,pdf}`.

| α | N | learned h | h^C | h^M | Δ | best-quote | leader-switch | contested | restraint│contested | forgone r | Q-gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 2 | 10.6 ± 0.1 | 2 | 13 | 0.83 ± 0.01 | 10.9 ± 4.3 | 0.53 | 0.53 | 0.99 | 1.20 | 5.1 |
| 0.1 | 3 | 10.3 ± 0.1 | 2 | 13 | 0.82 ± 0.01 | 10.7 ± 4.2 | 0.62 | 0.67 | 0.99 | 1.42 | 4.4 |
| 0.3 | 2 | 14.4 ± 0.2 | 4 | 14 | 0.86 ± 0.01 | 14.9 ± 4.5 | 0.45 | 0.55 | 0.99 | 0.84 | 3.7 |
| 0.3 | 3 | 14.9 ± 0.2 | 4 | 14 | 0.86 ± 0.01 | 14.9 ± 4.7 | 0.58 | 0.70 | 0.99 | 1.02 | 3.5 |
| 0.5 | 2 | 17.4 ± 0.3 | 8 | 17 | 0.77 ± 0.02 | 17.5 ± 4.2 | 0.50 | 0.58 | 0.96 | 0.44 | 3.5 |
| 0.5 | 3 | 17.5 ± 0.1 | 8 | 17 | 0.78 ± 0.02 | 17.3 ± 4.5 | 0.68 | 0.74 | 0.96 | 0.55 | 3.2 |

Mechanism-test columns (`analysis/mechanism.py`): *contested* = visit-weighted share of states where
the rival is at or below the agent's greedy quote (undercutting is on the table); *restraint│contested*
= share of those where the greedy action is NOT the myopic best response although the myopic one pays
more now; *forgone r* = immediate expected reward given up per period in restrained states; *Q-gap* =
Q(greedy) − Q(myopic).

## Findings
1. **Learned collusion is strong and the encoding fix did not remove it.** Δ = 0.77–0.86, seed std
   ≤ 0.02, learned quotes track h^M (10.6 / 14.4 / 17.4 vs 13 / 14 / 17). Below h^M at α = 0.1, at or
   slightly above it for α ≥ 0.3.
2. **Δ falls with α** (0.83 → 0.86 → 0.77: flat-then-down). **N = 3 vs 2 still makes no difference.**
3. **Restraint is encoded in the value function.** In roughly half to three-quarters of visited states
   the agent could profitably undercut now; in 96–99% of those it does not, giving up 0.4–1.4 per
   period, and its Q-table ranks the cooperative action 3–5 above the myopic one. This is the
   reward–punishment structure (Harrington 2018) read directly from the learned values, with no
   perturbation. It replaces the impulse-response test as our collusion certificate for cycling
   policies.
4. **Turn-taking persists.** Best-quoter identity changes in 45–68% of periods; per-quote std ≈ 4.3
   ticks while the market-relevant best quote tracks h^M. Under the corrected encoding the parked
   agent sees the rival's actual quote, so the alternation is a learned convention, not a blind spot.
5. **Impulse response under the fix** (α = 0.3, N = 2): rivals tighten ≈ 5 ticks at t+1 and stay
   1–2 ticks tighter for ~10 periods before cycle noise takes over. Weak but now visible.

## Open
* Why no N effect? Candidate: with ε-greedy exploration all agents visit the same joint cycle; the
  payoff per agent falls with N but the quote does not. Needs N = 5 (robustness grid) and possibly
  asymmetric learning rates.
* Robustness to Q-init and learning rate: `robust_exp2` (running).

---

# Robustness grid (`robust_exp2`): N × learning rate × Q-init at α = 0.3

12 cells × **10 seeds** (run on MIT SuperCloud, 48 cores, 4 min), corrected encoding, exponential
edge (h^C = 4, h^M = 14). Files: `robust_exp2_summary.csv`, `robust_exp2_runs.csv`. Config:
`experiments/configs/robustness_exp.yaml`. Reproduce: see config header; `python analysis/robustness.py`.
The 5-seed table below was computed first in the dev container; the 10-seed cluster run agrees on
every cell within 0.01 in Δ, and on the 60 shared (cell, seed) pairs the per-run Δ matches
**exactly** (max abs difference 0.0): the simulator is bit-reproducible across machines.

| N | lr | Q-init | learned h | Δ | leader-switch | contested | restraint│contested | forgone r | Q-gap |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.05 | 0 | 11.3 ± 0.8 | 0.85 ± 0.04 | 0.49 | 0.54 | 1.00 | 0.70 | 5.4 |
| 2 | 0.05 | 2 | 11.3 ± 0.3 | 0.85 ± 0.02 | 0.45 | 0.52 | 0.99 | 0.73 | 4.1 |
| 2 | 0.15 | 0 | 14.4 ± 0.2 | 0.85 ± 0.01 | 0.44 | 0.57 | 0.99 | 0.85 | 3.8 |
| 2 | 0.15 | 2 | 14.4 ± 0.1 | 0.85 ± 0.01 | 0.54 | 0.54 | 0.98 | 0.81 | 3.3 |
| 3 | 0.05 | 0 | 9.4 ± 0.1 | 0.71 ± 0.02 | 0.63 | 0.67 | 0.99 | 0.76 | 3.3 |
| 3 | 0.05 | 2 | 10.0 ± 0.2 | 0.76 ± 0.01 | 0.57 | 0.66 | 1.00 | 0.81 | 2.9 |
| 3 | 0.15 | 0 | 15.0 ± 0.2 | 0.87 ± 0.00 | 0.61 | 0.70 | 1.00 | 1.02 | 3.5 |
| 3 | 0.15 | 2 | 14.9 ± 0.2 | 0.86 ± 0.02 | 0.61 | 0.69 | 0.99 | 1.01 | 3.0 |
| 5 | 0.05 | 0 | 9.7 ± 0.1 | 0.71 ± 0.02 | 0.64 | 0.79 | 0.96 | 0.86 | 2.5 |
| 5 | 0.05 | 2 | 11.6 ± 0.1 | 0.85 ± 0.02 | 0.71 | 0.80 | 0.99 | 1.03 | 1.9 |
| 5 | 0.15 | 0 | 15.0 ± 0.2 | 0.87 ± 0.01 | 0.61 | 0.82 | 0.99 | 1.18 | 2.7 |
| 5 | 0.15 | 2 | 15.4 ± 0.2 | 0.85 ± 0.01 | 0.66 | 0.83 | 0.99 | 1.18 | 2.1 |

## Findings
1. **Collusion survives every setting.** Δ ∈ [0.71, 0.87] across all 12 cells; the worst cell
   (N = 5, lr = 0.05, pessimistic init) is still at 71% of the monopoly rent. Restraint is present
   in 96–100% of contested states everywhere. The value-function certificate does not depend on
   learner hyperparameters.
2. **Learning rate shifts the *arrangement*, not the rent.** At lr = 0.15 learned quotes sit at h^M
   (14–15); at lr = 0.05 they sit 3–5 ticks below (9–11) yet Δ is nearly the same at N = 2. Slow
   learners settle on a tighter-but-more-regular alternation. Reporting quoted spread alone would
   misread lr = 0.05 as "more competitive"; the profit-based index does not.
3. **N finally matters, but only for slow learners.** At lr = 0.15, N = 2 / 3 / 5 all give Δ ≈ 0.85–0.87.
   At lr = 0.05 with pessimistic init, Δ drops from 0.85 (N = 2) to 0.71 (N = 3, 5). The CFL
   "markups fall with N" result appears to be a slow-learning phenomenon; fast learners coordinate
   regardless of N.
4. **Optimistic init does not break collusion** (contrary to the under-exploration story in its
   simplest form): init_q = 2 gives Δ within 0.01 of init_q = 0 except at (N = 5, lr = 0.05), where
   it *raises* Δ from 0.71 to 0.85. More exploration found a better cartel.
5. **Contested share rises with N** (0.54 → 0.69 → 0.81): with more rivals, the agent is more often
   in a state where undercutting would pay, and still does not. Forgone reward per period rises with
   N too (0.7 → 1.0 → 1.2).

## Status of Workshop 1 item (i)
Replicated: learned quotes far above competitive; markup falls with α; markup falls with N for slow
learners. Beyond CFL: continuous-edge model with closed-form benchmarks; turn-taking structure;
value-function restraint test as a perturbation-free collusion certificate; robustness across
N × lr × init. Caveat: CFL's exact markup definition and learner settings still need to be checked
against the paper before the write-up claims agreement or disagreement on specifics.

---

# Workshop 1 item (iii) / RQ1 first result: the strategic entrant (`cfl_exp2_entry`)

A third market maker enters a market of two trained, FROZEN incumbents (greedy play from saved
Q-tables; they react through their learned policy but do not re-learn). 500 episodes per
(α, seed, policy), 10 seeds. Files: `cfl_exp2_entry.csv`, `fig_cfl_exp2_entry.{png,pdf}`.
Code: `sim/entrants.py`, `experiments/entry.py`, `analysis/entry_fig.py`.

Entrant policies (all see only public quotes and their own fills):
* **competitive-GM**: quote the exact zero-profit half-spread h^C. Textbook benchmark.
* **one-tick undercutter**: quote one tick inside the best incumbent, always.
* **markout-inference**: estimate toxicity from own-fill PnL, derive h^C(α̂), undercut by one tick
  only if the best incumbent quote exceeds h^C(α̂) by ≥ 2 ticks, else quote h^C(α̂).

| α | policy | entrant π/period | share of π^M | fill share | incumbents' π before → after |
|---|---|---|---|---|---|
| 0.1 | competitive | 0.37 ± 0.05 | 32% | 0.98 | 1.00 → 0.00 |
| 0.1 | markout | 0.95 ± 0.17 | 82% | 0.54 | 1.00 → 0.36 |
| 0.1 | undercut | 1.00 ± 0.16 | 87% | 0.60 | 1.00 → 0.35 |
| 0.3 | competitive | 0.12 ± 0.03 | 15% | 0.98 | 0.66 → 0.00 |
| 0.3 | markout | 0.72 ± 0.12 | 96% | 0.58 | 0.66 → 0.25 |
| 0.3 | undercut | 0.67 ± 0.19 | 89% | 0.58 | 0.66 → 0.25 |
| 0.5 | competitive | 0.10 ± 0.02 | 27% | 0.95 | 0.33 → −0.02 |
| 0.5 | markout | 0.36 ± 0.09 | 94% | 0.60 | 0.33 → 0.12 |
| 0.5 | undercut | 0.30 ± 0.18 | 79% | 0.80 | 0.33 → 0.09 |

## Findings
1. **The learned cartel is almost fully exploitable.** A one-line policy (undercut by one tick when
   the incumbents' quote is far above the competitive level) earns 82–96% of the *monopoly* profit
   per period, 3–6× what the competitive entrant earns. The incumbents' profit falls by roughly
   two-thirds. Exploitability of the learned policies at α = 0.3: ≈ 0.72 per period against a
   monopoly rent of 0.75 — the entrant captures essentially everything the cartel was earning.
2. **Winning every fill is the wrong objective.** The competitive entrant wins 95–98% of fills and
   earns the least: it is quoting the zero-profit spread, so each fill is worth ≈ 0 in expectation.
   The incumbents learn that a rival at h^C is unbeatable and park their quotes wide, collecting
   nothing. Nobody makes money. This is the "race to the bottom" outcome a textbook predicts, and
   it is the outcome a desk should avoid.
3. **Markout inference beats blind undercutting as toxicity rises** (0.95 vs 1.00 at α = 0.1;
   0.72 vs 0.67 at α = 0.3; 0.36 vs 0.30 at α = 0.5) and has roughly half the seed variance.
   The undercutter chases the incumbents' cycle and the incumbents chase back, so its PnL is
   erratic; the markout entrant undercuts into a stable configuration and otherwise sits at h^C.
   Its toxicity estimate is biased upward (α̂ ≈ 0.32 / 0.36 / 0.48 vs true 0.1 / 0.3 / 0.5) because
   it counts any losing fill as informed, including unlucky uninformed ones; a better estimator is
   an obvious improvement.
4. **The incumbents do not punish the entrant.** Against the competitive entrant they settle into a
   fixed 3-cycle wide of the entrant and never contest; against the undercutter their quotes
   bounce 4–24 but never pin the entrant at a loss. The restraint certified by the mechanism test
   (they do not undercut *each other*) does not generalise to a third party quoting inside them.
   This is the central vulnerability of tacit collusion learned at N: the policy is a response to
   the rivals it trained with.

## What this means for the paper
* RQ1 answered in the simulator for one scenario: yes, a simple public-information rule separates
  rent from toxicity well enough to capture most of the rent, and the naive alternative (quote
  competitively) is the worst option.
* The "rent is where toxicity is lowest" prediction (Colliard et al.) shows as entrant profit per
  period: 0.95 at α = 0.1 vs 0.36 at α = 0.5. The entrant should go where the flow is benign.
* Next: (a) let the incumbents re-learn during entry — does the cartel re-form around the entrant
  or collapse? (b) entrant with a less biased toxicity estimator (markout against a public mark
  rather than own-fill sign); (c) exploitability of the lr = 0.05 cartel, which quotes tighter.

---

# Entry with re-learning incumbents, markout entrant (`cfl_exp2_entry_relearn_markout`)

Same trained incumbents, but they resume Q-learning after entry with a fresh exploration schedule
(ε₀ = 0.3, decay 1e-5 → ε ≈ 0 by episode 8000). 5 seeds × 3 α, 500-episode windows.
All three entrant policies run (SuperCloud, `slurm/entry_relearn.sh`); the three-way comparison
is at the end of this file.

| α | window | entrant π | incumbents π | Δ (3-MM market) | best quote |
|---|---|---|---|---|---|
| 0.1 | first (ε=0.18) | 0.87 | 0.39 | 0.65 | 7.3 |
| 0.1 | last (ε=0) | **0.95** | 0.44 | 0.75 | 8.7 |
| 0.3 | first | 0.56 | 0.21 | 0.63 | 9.3 |
| 0.3 | last | **0.67** | 0.23 | 0.74 | 11.2 |
| 0.5 | first | 0.25 | 0.03 | 0.28 | 11.0 |
| 0.5 | last | **0.39** | 0.07 | 0.62 | 14.8 |
| | frozen incumbents (for comparison) | 0.95 / 0.72 / 0.36 | 0.36 / 0.25 / 0.12 | | |

## Finding: the cartel re-forms *around* the entrant, and the entrant keeps its rent
* As the incumbents' exploration decays, the market-wide Δ **rises** (0.65 → 0.75 at α = 0.1;
  0.28 → 0.62 at α = 0.5): the 3-MM market re-collusifies. The best quote drifts wider over the run.
* The entrant's profit at the end is the same as against frozen incumbents (0.95 / 0.67 / 0.39 vs
  0.95 / 0.72 / 0.36). Re-learning did not squeeze it out; the incumbents learned to live with a
  rival who undercuts whenever they quote wide, and the new equilibrium has the entrant on the
  inside of a wider cartel.
* The incumbents' profit roughly matches the frozen case (0.44 / 0.23 / 0.07 vs 0.36 / 0.25 / 0.12):
  at α = 0.1 they recovered a little by widening; at α = 0.5 they are close to zero either way.
* Fill share stays near 50–60%: the entrant does not win every fill, it wins the ones at
  profitable prices.

Interpretation for RQ2: the exploitability of this learned collusion is **persistent**, not a
transient that the incumbents learn away. An entrant with the markout rule is absorbed into a
three-way convention on terms it set. Whether a *second* entrant can do the same to the first
(entry cascades until rent is gone) is the natural next question, and it is the sim analogue of
"how many market makers does Hyperliquid's book support before rent disappears?"

### Undercut entrant under re-learning incumbents (`cfl_exp2_entry_relearn_undercut`)

| α | window | undercut entrant π | markout entrant π (above) |
|---|---|---|---|
| 0.1 | first / last | 0.75 / 0.91 | 0.87 / 0.95 |
| 0.3 | first / last | 0.45 / 0.58 | 0.56 / 0.67 |
| 0.5 | first / last | **0.03** / 0.30 | 0.25 / 0.39 |

Same qualitative story (cartel re-forms, entrant keeps rent), but the markout entrant beats the
blind undercutter at every α and every stage, and the gap is widest while the incumbents are still
exploring: at α = 0.5 in the first window the undercutter earns ≈ 0 (it chases exploring incumbents
into toxic territory) while the markout entrant earns 0.25. Inference matters most when the
competitors are unpredictable.

---

# Three-way comparison under re-learning incumbents (`fig_cfl_exp2_entry_relearn.png`)

`analysis/relearn_fig.py`. 5 seeds × 3 α × 3 policies, 8000 episodes, 500-episode windows.
Cluster runs reproduce the container's markout and undercut runs bit-for-bit on the shared seeds
(same `episodes.csv` rows), so the three CSVs are directly comparable.

Last window (ε = 0), mean over 5 seeds. π^C, π^M are per-incumbent benchmarks of the 2-MM
market (so 2π^C = total competitive profit Π(h^C), which is what a sole quoter at h^C earns).

| α | policy | entrant π | incumbents π (each) | Δ (3-MM market) | best quote | entrant fill share |
|---|---|---|---|---|---|---|
| 0.1 | competitive-GM | 0.39 | 0.00 | 0.00 | 2.0 (= h^C) | 1.00 |
| 0.1 | one-tick undercut | 0.91 | 0.42 | 0.71 | 8.4 | 0.55 |
| 0.1 | markout-inference | **0.95** | 0.44 | 0.75 | 8.7 | 0.53 |
| 0.3 | competitive-GM | 0.12 | 0.00 | 0.00 | 4.0 (= h^C) | 1.00 |
| 0.3 | one-tick undercut | 0.58 | 0.23 | 0.67 | 10.9 | 0.57 |
| 0.3 | markout-inference | **0.67** | 0.23 | 0.74 | 11.2 | 0.56 |
| 0.5 | competitive-GM | 0.11 | 0.00 | 0.00 | 8.0 (= h^C) | 1.00 |
| 0.5 | one-tick undercut | 0.30 | 0.06 | 0.47 | 14.1 | 0.63 |
| 0.5 | markout-inference | **0.39** | 0.07 | 0.62 | 14.8 | 0.61 |
| | π^C / π^M per incumbent | 0.20/1.15 · 0.06/0.75 · 0.06/0.38 | | | | |

First window (ε = 0.18), entrant π only: competitive 0.39 / 0.10 / 0.09; undercut 0.75 / 0.45 /
**0.03**; markout 0.87 / 0.56 / **0.25**.

## Findings
* **Three distinct outcomes, one per policy.** The competitive entrant *collapses* the cartel:
  Δ → 0 exactly, the best quote sits at h^C, the entrant wins every fill and earns Π(h^C) = 2π^C
  (0.39 / 0.12 / 0.11), and the re-learning incumbents converge to zero profit (they learn that
  every quote at or inside h^C loses money and quote wide, where they are never hit). The undercut
  and markout entrants both let the cartel *re-form around them* (Δ 0.47–0.75) and take the largest
  share of it.
* **Collapsing the cartel is the worst thing an entrant can do to itself.** Quoting at h^C leaves
  59% (α = 0.1), 82% (α = 0.3) and 72% (α = 0.5) of the markout entrant's profit on the table. The
  rent exists only while someone else is quoting wide; the entrant's job is to stay one tick inside
  the convention, not to end it. This is the "value of the rent to the entrant" number RQ1 asks for.
* **Inference beats blind undercutting at every α and every stage**, by 4% (α = 0.1) to 30%
  (α = 0.5) in the final window, and by far more while the incumbents are still exploring
  (α = 0.5 first window: 0.25 vs 0.03). The gap grows with toxicity because the undercutter follows
  exploring incumbents into loss-making quotes; the markout rule stops at its estimated h^C(α̂).
* **The competitive entrant is the only one that makes the incumbents lose money**, and only
  while they explore (incumbents' π = −0.065 at α = 0.5 in the first window; Δ = −0.23). If the
  objective were to damage incumbents rather than to earn, h^C is the tool; for a profit-maximizer
  it is the wrong one.
* The incumbents end up roughly indifferent between the undercut and markout entrants
  (0.42 vs 0.44 at α = 0.1); the entrant's inference pays the entrant, not the cartel.

## What this means for the paper
* The RQ1/RQ2 deliverable now has the full 3 × 3 table: three readings of the same public
  information (ignore it; mimic the best quote; infer toxicity from it) × three toxicity regimes,
  against incumbents that are allowed to respond. Exploitability is persistent across all of them.
* The ordering competitive < undercut < markout is the paper's central empirical claim and should be
  stated as a loss ranking: L_BR(competitive) ≫ L_BR(undercut) > L_BR(markout).
* Next: a *second* markout entrant (does the first entrant's rent survive a cascade?), and the
  Hyperliquid analogue — count the wallets that quote inside the resting spread on a given coin and
  check whether the inside quoter's markout is the least negative (the empirical signature of the
  markout rule).

---

# RQ2 hypothesis test: ∂L_BR/∂C > 0 (`exploitability.csv`, `fig_exploitability.png`)

`experiments/exploitability.py` → `analysis/exploitability.py`. 120 frozen populations: the
`robust_exp2` grid (α = 0.3; N ∈ {2,3,5} × lr ∈ {0.05, 0.15} × init_q ∈ {0, 2} × 5 seeds) plus
`cfl_exp2` (α ∈ {0.1, 0.3, 0.5} × N ∈ {2,3} × 10 seeds). Seed-level rent C(π) spans 0.43–0.97
(cell means 0.70–0.89; the lr = 0.05 cells are the low-C ones). Four attackers, 500 episodes each,
common random numbers: competitive-GM, one-tick undercut, markout-inference, and **oracle**
(the markout rule handed the true α — the F^oracle bound for RQ1).

Definitions (market-wide, per period): Π_nominal = frozen incumbents alone; J(π; A) = incumbents'
profit with attacker A present; L_BR^A = Π_nominal − J; L_BR = max_A L_BR^A (the proposal's
min_A J); everything divided by Π^M − Π^C. κ = L_BR / (Π_nominal − Π^C) = **fraction of the rent
lost**.

| slope on C (OLS, bootstrap 95% CI) | pooled | α = 0.3 only | within-cell (seeds) |
|---|---|---|---|
| L_BR^norm, max over attackers | 0.99 [0.85, 1.19] | 1.04 [1.00, 1.09] | 1.06 [0.93, 1.27] |
| L_BR^norm, markout attacker | 0.64 [0.47, 0.87] | 0.89 [0.68, 1.09] | 0.63 [0.48, 0.88] |
| L_BR^norm, undercut attacker | 0.60 [0.31, 0.95] | 0.91 [0.70, 1.16] | 0.56 [0.21, 1.02] |
| κ, markout attacker | −0.17 [−0.46, 0.23] | 0.27 [−0.01, 0.54] | −0.24 [−0.49, 0.17] |
| κ, undercut attacker | −0.37 [−0.97, 0.26] | 0.25 [−0.02, 0.54] | −0.53 [−1.21, 0.29] |
| κ, max over attackers | −0.32 [−0.59, 0.02] | −0.10 [−0.13, −0.03] | −0.30 [−0.54, 0.05] |
| entrant profit, markout | 0.27 [0.10, 0.50] | 0.61 [0.37, 0.94] | 0.10 [−0.10, 0.29] |
| entrant profit, undercut | 0.69 [0.31, 1.00] | 0.72 [0.38, 1.17] | 0.59 [0.03, 0.98] |

Means over populations (α = 0.3): κ = 0.70 markout, 0.71 oracle, 0.72 undercut, **1.11 competitive**;
entrant profit 0.51 / 0.50 / 0.49 / 0.08 rent units.

## Findings
* **The literal hypothesis holds, but mostly by construction.** L_BR rises with C for every
  attacker (slopes 0.6–1.0, every CI excludes 0). With the max-over-attackers definition the slope
  is 1.0: the maximizer of incumbent loss is the competitive-GM quoter in 119/120 populations, and
  it leaves the incumbents *below* Π^C (κ = 1.1; J/Π^C = −0.05 — frozen incumbents sometimes
  respond to a quote at h^C by undercutting it and lose money). So L_BR ≈ Π_nominal − 0, and
  ∂L_BR/∂C = 1 says only "more rent, more to lose".
* **The substantive version is not supported.** κ, the fraction of rent a profit-maximizing
  attacker destroys, is flat in C: ≈ 0.7–0.8 across the whole range for undercut, markout and
  oracle, with slopes whose CIs include 0 (pooled and within-cell; weakly positive at α = 0.3
  only). Higher-C populations are *not* more regular/exploitable per unit of rent in this
  attacker set; they are exploitable in proportion to the rent. The proposal's phrase "the
  regularity required to sustain rents makes the policy easier to manipulate" is **not** what the
  data say for one-shot public-information attackers; it may still hold for attackers that target
  the reaction function itself (steering a la Deng–Schneider–Sivan), which is untested.
* **Damage and profit are different objectives.** The attacker that hurts the incumbents most
  (competitive) earns the least (0.08 rent units); the attackers that earn the most (markout /
  oracle, 0.50) leave the incumbents 30% of their rent. The proposal's L_BR = Π_nominal − min_A J
  is the damage number; the trader-relevant number is the loss under the attacker's *own* best
  response, which is the markout rule in 68/120 populations, the oracle in 36, the undercutter
  in 16.
* **Oracle ≈ markout.** Knowing the true α adds nothing over estimating it from own fills
  (0.50 vs 0.51 rent units; the markout rule wins more populations than the oracle). In a
  stationary market the public tape already carries the full value of the toxicity *level*. For
  the identity arm F^id to have value, toxicity must be *heterogeneous and persistent across
  takers* (wallet-level α_j), which is exactly what Zhai measures and what this simulator does
  not yet model. That is the required next model change, not more entrants.
* **Entrant profit rises with C** (markout: 0.61 per unit C at α = 0.3; undercut 0.69 pooled):
  the trader does earn more from more collusive incumbents, even though the fraction destroyed
  is constant. Within-cell (seed-only variation) the markout slope is ≈ 0: the across-cell
  effect is partly hyper-parameters (lr = 0.05 cartels are both lower-C and leave the entrant
  less), so the causal reading is "populations that learned higher rents also leave more for an
  entrant", not that C alone is the driver.

## What this means for the paper
* RQ2's hypothesis should be restated. Candidate: "L_BR under a profit-maximizing attacker grows
  with C, and the fraction of rent destroyed, κ, is roughly constant (≈ 0.7); the attacker captures
  about two thirds of what it destroys, the rest is passed to takers as tighter quotes." That is
  a quantitative, falsifiable statement that the data support.
* Define the attacker set to exclude policies that would not be chosen by a profit-maximizer, or
  report both L_BR^damage and L_BR^BR explicitly. The competitive quoter is a useful *bound*
  (what a price war costs the cartel), not an attacker anyone would run.
* RQ1's nested information: π^anon ≈ π^oracle here, so the simulator must add persistent taker
  heterogeneity before the F^id arm is meaningful. Priority for the next model revision.

---

# Identity experiment: the executable value of wallet identity (`cfl_exp2_identity.csv`)

Model change (`sim/env.py`, `n_wallets > 0`): a population of 50 wallets with persistent types
α_j (stratified Beta quantiles around ᾱ; concentration κ sets the dispersion, κ → ∞ is the old
homogeneous model), bursty arrivals (the previous period's wallet returns w.p. ρ), and a public
post-commitment print (event, price, wallet id, ex-post mark). With `n_wallets = 0` the RNG
stream is untouched; `tests/test_wallets_env.py` pins the old traces. Benchmarks use the
activity-weighted mean type, so Π^C, Π^M and Δ are unchanged by heterogeneity.

Entrant (`sim/entrants.py: WalletEntrant`): exact grid-Bayes estimation of ᾱ from every period
(print or not) and of each wallet's α_j from its prints (print-conditional likelihood, so no
fill-selection bias; `abar_hat` lands within 0.005 of the truth, per-wallet mean |error| 0.03
after 300 episodes). Forecast of the next arrival α_next = ρ^k α̂_j + (1 − ρ^k) ᾱ̂; quote
`min(best − 1, h^M(α_next))` when `best − h^C(α_next) ≥ 2`, else `h^C(α_next)` (which, for a toxic
active wallet, is wider than the incumbents' quote: the entrant withdraws). Three information
sets differ only in what is linkable: **anon** (one print's markout, no ids), **id** (each
wallet's whole history), **oracle** (true types).

Frozen `cfl_exp2` incumbents (N = 2, trained on the homogeneous market, 5 seeds), ᾱ ∈ {0.3, 0.5},
κ ∈ {∞, 5, 1} (type sd 0, 0.19, 0.32 at ᾱ = 0.3), ρ ∈ {0, 0.5, 0.9, 0.99}, 500 episodes, common
random numbers within a cell. Entrant profit in rent units (Π^M − Π^C), mean over seeds:

| ᾱ | κ | ρ | competitive | own-fill markout | anon | **id** | oracle | id − anon (paired) |
|---|---|---|---|---|---|---|---|---|
| 0.3 | ∞ | any | 0.08 | 0.49 | 0.69 | 0.69 | 0.69 | 0.00 |
| 0.3 | 5 | 0 / 0.9 / 0.99 | 0.07 | 0.49 | 0.68 / 0.62 / 0.62 | 0.68 / 0.69 / **0.77** | 0.68 / 0.71 / 0.75 | 0.00 / 0.07 / 0.15 |
| 0.3 | 1 | 0 / 0.9 / 0.99 | 0.08 | 0.49 | 0.68 / 0.62 / 0.62 | 0.68 / 0.78 / **0.83** | 0.69 / 0.75 / 0.81 | 0.00 / 0.16 / 0.21 |
| 0.5 | ∞ | any | 0.16 | 0.60 | 0.73 | 0.73 | 0.72 | 0.00 |
| 0.5 | 5 | 0 / 0.9 / 0.99 | 0.15 | 0.58 | 0.75 / 0.66 / 0.68 | 0.75 / 0.80 / **0.98** | 0.74 / 0.80 / 0.91 | 0.00 / 0.14 / 0.30 |
| 0.5 | 1 | 0 / 0.9 / 0.99 | 0.16 | 0.54 | 0.77 / 0.72 / 0.94 | 0.77 / 0.97 / **1.20** | 0.71 / 0.94 / 1.14 | 0.00 / 0.25 / 0.26 |

Paired id − anon at ρ = 0.9: +0.16 ± 0.07 (ᾱ = 0.3, κ = 1), +0.25 ± 0.10 (ᾱ = 0.5, κ = 1);
every seed positive. Figures: `fig_cfl_exp2_identity.png` (levels), `_value.png` (paired
differences), `_as.png` (adverse-selection shift).

## Findings
* **Identity is worth nothing without persistence or dispersion, and a lot with both.** At
  ρ = 0 or κ = ∞ the three information sets coincide exactly (the sanity check passes). At
  ρ = 0.9 identity adds 26% (ᾱ = 0.3) to 35% (ᾱ = 0.5) to the entrant's profit with κ = 1, and
  the gain keeps growing to ρ = 0.99. Persistence matters more than dispersion: at ρ = 0.5 the
  gain is ≈ 0 even with κ = 1, because after one no-trade period the forecast has already
  collapsed to the mean. **ρ is the number to measure on Hyperliquid** (`analysis/wallets.py:
  arrival_persistence`, against the random-matching baseline `activity_herfindahl`).
* **id attains the oracle.** π^oracle − π^id is within ±0.03 at ᾱ = 0.3 and −0.03 to −0.08 at
  ᾱ = 0.5 (the estimate sometimes beats the truth: the discrete quote rule, not information, is
  the binding constraint there). The public tape plus wallet ids is *executably* as good as
  knowing every wallet's type; there is no further information premium to buy.
* **The mechanism is fill selection, not price setting.** The identity-aware sole quoter's
  bound (`identity_monopoly_profit`) exceeds Π^M by < 2% at κ = 5 and by 3–9% at κ = 1: a
  monopolist gains little from identity. The entrant gains because it can choose which flow to
  take against identity-blind rivals: at ᾱ = 0.5, κ = 1, ρ = 0.99 it withdraws in 31% of periods,
  its fills are 18% informed (vs 39% with no entrant) and the incumbents' are 53%. The cartel's
  residual book becomes a loss (incumbents at −0.13 rent units, i.e. below Π^C) while the
  market-wide Δ stays at 0.9: the rent did not disappear, it moved to the entrant.
* **A correct tape estimator beats the own-fill heuristic even without ids**: anon earns
  0.62–0.69 vs the own-fill markout rule's 0.48–0.49 (ᾱ = 0.3). Reading the whole public tape
  with an arrival-level (not print-level) likelihood is worth ≈ 0.15–0.2 rent units on its own.
* Incumbents trained on the homogeneous market; retraining on the heterogeneous one and
  re-learning after entry (identity-blind vs identity-aware incumbents) are the RQ3 follow-ups.

## What this means for the paper
* RQ1 now has its intended answer in the form the proposal states it: π^anon ≤ π^id ≈ π^oracle,
  with the gap a function of (ρ, κ), both measurable from the public feed. The headline for a
  trader: on a venue with persistent, bursty, heterogeneous flow, the public wallet tag is
  worth a quarter to a third of the entrant's profit, and you need no private data to realize it.
* The adverse-selection shift is the empirical signature to look for in the Hyperliquid data:
  wallets that quote inside the resting spread intermittently should show *less* negative
  markouts than the resting quoters, and the resting quoters' markouts should worsen when such a
  wallet is active.
* Calibration targets from `data/record.py` output: ρ̂ (same-taker repeat rate vs Herfindahl
  baseline), the cross-sectional dispersion of per-taker markouts (κ), and ᾱ from the markout
  level. 10-seed cluster run: `sbatch slurm/identity.sh`.

---

# First Hyperliquid data: Jan 26–28 2026, Zenodo trades archive (`results/zenodo_2026_01/`)

Source: Albers et al. record 18184441, `trades_2026_01.tar` → `data/zenodo_to_parquet.py` →
`analysis/calibrate.py`. January coverage is Jan 1–3 and Jan 19–31 (a 15-day hole in the
archive); we used three full consecutive days. 3.42M prints for BTC/ETH/SOL/HYPE, 1.44M taker
orders after collapsing sweeps. `side_info` is `[buyer, seller]`; 2.9% of aggressors have an
older oid than the resting order (trigger/liquidation flow), flagged `taker_older`. Markouts use
the trade-price mid proxy (bid–ask bounce at short horizons; the L2 mid is the fix).

| | BTC | ETH | HYPE | SOL |
|---|---|---|---|---|
| orders / min | 97 | 68 | 135 | 34 |
| maker markout 10 s (bps) | −0.69 | −0.79 | −0.12 | −0.41 |
| ρ_order (same taker next) / random baseline | 0.020 / 0.005 | 0.034 / 0.009 | 0.038 / 0.012 | 0.045 / 0.011 |
| ρ_fill (mechanical, sweeps) | 0.57 | 0.53 | 0.53 | 0.43 |
| type rank corr, split-half (p) | 0.17 (1e-6) | 0.39 (1e-17) | 0.11 (2e-3) | 0.42 (8e-11) |
| markout after toxic / after benign (bps) | −0.90 / −0.09 | −0.77 / −0.21 | −0.73 / +0.11 | −0.84 / +0.04 |
| MM wallets (classifier), volume share | 374, 56% | 250, 52% | 284, 52% | 165, 50% |
| MM–MM avoidance ratio | 1.00 | 1.05 | 0.93 | 0.99 |

## Findings
* **The identity signature is present and large.** Conditioning on whether the previous
  order's taker is in the toxic quartile (classified on the first 36 h, measured on the second)
  moves the maker's markout on the next fill by 0.55–0.88 bps, against a mean of −0.1 to −0.8.
  "After benign" is ≈ 0 on every coin; "after toxic" is ≈ −0.8. n ≈ 18k–56k orders per coin.
* **But not through the channel the simulator has.** Same-wallet persistence at the order level
  is 0.02–0.045 — three to four times random matching, and negligible in absolute terms. The
  simulator at ρ = 0.5 already gave zero value of identity. Informed flow clusters *in time*,
  and a toxic wallet's print is a signal that the market is in a toxic regime shared by the next
  arrival, whoever it is. Identity is a regime indicator, not a repeat-customer forecast. The
  model change this forces: a latent toxicity regime with persistence, wallet activity that
  depends on it, and an entrant that filters the regime from the identity-tagged tape.
* **Toxicity is a persistent wallet type** (split-half rank correlation 0.11–0.42, all
  significant), weaker than Zhai's ten-day 0.52, which is expected with 36-hour halves and a
  trade-price proxy.
* **HYPE is different:** mean 10-s markout −0.12 bps vs −0.4 to −0.8 for the cross-listed coins,
  and the top HYPE makers show *positive* 10-s markouts (+0.5 to +0.96 bps). HYPE has no Binance
  / OKX price to be arbitraged against. A clean cross-coin contrast for the "rent is where
  toxicity is lowest" prediction, pending the L2-mid check.
* **No MM–MM avoidance** (ratio ≈ 1 on all four coins; the 0.5–0.65 in the first run was an
  artefact of the 3% role swaps before the converter fix). The CCG anonymity-signalling result
  does not show in three days of Hyperliquid data at this classifier setting. The top-10 MM
  wallets carry 27–29% of all volume; turn-taking tests should use that set with the shuffled
  null now in `calibrate.py`, not the 374-wallet set.
* The HLP vault address does not appear with ≥ 30 fills on any of the four coins in this window.

## What this means for the paper
* Workshop 1 item (iv) is done; the data section can state coverage, the conversion rules, and
  this table.
* RQ1's framing survives, the mechanism in the simulator does not: replace wallet persistence
  with regime persistence. The empirical statistic that calibrates it is the conditional
  markout split above, and the entrant's rule becomes a regime filter on the public tape.
* Next data steps: October tar (the cascade; check contiguity), the Mac recorder with the L2
  book for a true-mid markout cross-check, and the Dec 2025 L4 files for reaction functions.

---

# Regime channel in the simulator (`cfl_exp2_identity_regime.csv`)

Model (`MarketConfig.regime_persistence` = ρ_z): latent z ∈ {calm, toxic}, kept w.p. ρ_z else
redrawn from a stationary law chosen so the mean toxicity stays at ᾱ; wallet activity tilts by
α_j in the toxic state and (1 − α_j) in the calm one. With κ = 1 the regime toxicities are
0.15 / 0.65, with κ = 5 they are 0.25 / 0.415. The entrant runs a two-state filter on the public
tape: with ids the evidence is *who* printed (likelihood ratio α_j(1−ᾱ)/((1−α_j)ᾱ)); without ids
it is whether the last print *lost money* for its maker. Quote rule now undercuts at break-even
(`--margin 1`); the first identity grid used margin 2, which can make a better-informed
entrant earn *less* (it declines positive-EV fills when its forecast is pessimistic), and did.

Entrant profit, rent units, 5 seeds, wallet persistence ρ = 0 (as the data say):

| ᾱ | κ | ρ_z | anon | id | oracle | id − anon (paired) | oracle − anon |
|---|---|---|---|---|---|---|---|
| 0.3 | 1 | 0 / 0.9 / 0.99 | 0.68 / 0.64 / 0.79 | 0.68 / 0.68 / 0.80 | 0.69 / 0.69 / 0.84 | 0 / +0.04 / +0.01 | 0 / +0.05 / +0.06 |
| 0.5 | 1 | 0 / 0.9 / 0.99 | 0.77 / 0.67 / 0.96 | 0.77 / 0.75 / 0.97 | 0.71 / 0.75 / 1.11 | 0 / +0.08 / +0.01 | 0 / +0.07 / +0.15 |
| 0.3 | 5 | 0.9 / 0.99 | 0.65 / 0.68 | 0.67 / 0.67 | 0.67 / 0.69 | +0.03 / −0.01 | +0.02 / +0.01 |
| 0.5 | 5 | 0.9 / 0.99 | 0.70 / 0.76 | 0.70 / 0.80 | 0.68 / 0.78 | −0.01 / +0.03 | −0.02 / +0.02 |

## Findings
* **Under the regime mechanism the anonymous tape already carries most of the value.** The
  last print's markout is itself a regime signal, and the regime — not the wallet — is what the
  next arrival shares. Identity sharpens the filter (regime hit rate 0.87 vs 0.76 against fixed
  quoters) but the profit increment is +0.01 to +0.08 rent units, versus +0.16 to +0.30 under
  the wallet-persistence channel that the data rule out.
* Identity only pays when the regime contrast crosses the entrant's decision threshold: at κ = 1
  the toxic state's 0.65 makes an undercut at the incumbents' quote unprofitable, and there the
  oracle gains +0.15 at ᾱ = 0.5, ρ_z = 0.99 (withdraws 15% of periods, informed share of its
  fills 0.28 vs incumbents' 0.50). At κ = 5 the toxic state (0.415) never crosses it and all three
  information sets coincide.
* So the empirical question is not "does identity predict the next fill's toxicity" (it does,
  −0.8 bps) but **"does identity predict it beyond what the anonymous tape already shows"**.
  `calibrate.py` now reports that directly: the 2 × 2 split of the next fill's markout by
  (previous taker toxic?) × (price moved against the previous maker by the time the next order
  arrived?). `two_diff_id_given_anon_bps` is identity's increment; if it is near zero, a quoter
  gets the regime from the tape and the wallet tag is redundant for *this* use — which would
  move the identity story to the belief-update / skew channel (post-fill), Zhai's own mechanism.
* The margin-2 identity grid (`cfl_exp2_identity.csv`) should be re-run at margin 1 before any
  of its numbers are quoted; `slurm/identity.sh` now does both channels at margin 1, 10 seeds.

---

# Identity beyond the anonymous tape (data), and the mark lag (model)

## Data: the 2 × 2 split (`results/zenodo_2026_01/calibration.csv`, `two_*` columns)
Next fill's maker markout (10 s, bps), second half of Jan 26–28, split by whether the previous
taker is a toxic wallet (classified on the first half) and by whether the price had already
moved against the previous maker when the next order arrived (what an anonymous quoter sees):

| | BTC | ETH | HYPE | SOL |
|---|---|---|---|---|
| identity alone (toxic − benign) | −0.81 | −0.55 | −0.84 | −0.88 |
| anonymous signal alone (adverse − not) | −0.74 | −0.55 | −0.31 | −0.46 |
| **identity given the anonymous signal** | **−0.63** | **−0.47** | **−0.74** | **−0.87** |
| P(price already moved against the last maker) | 0.18 | 0.21 | 0.21 | 0.20 |
| P(toxic | moved) / P(toxic | not moved) | 0.56 / 0.16 | 0.58 / 0.28 | 0.34 / 0.16 | 0.57 / 0.17 |

78–99% of identity's effect survives conditioning on the anonymous signal. **The wallet tag
carries regime information the tape has not yet revealed** — the opposite of the simulator's
regime result, where the anonymous entrant captured most of the value.

## Why the simulator got it wrong, and the fix
The simulator revealed each print's ex-post mark V one period later, so the anonymous entrant
knew with oracle precision whether the last fill was informed before the next arrival. On
Hyperliquid the next order arrives in ~0.6 s and the markout realises over seconds to minutes:
four times out of five the price has not moved yet (`P(adverse)` ≈ 0.2). Identity is public at
the print; the mark is late. **The lag between the print and its mark is the value of
identity.** `MarketConfig.mark_lag` now delays the mark by L periods; the entrant's evidence is
split into immediate (who printed / that a print happened) and delayed (the mark), with the
delayed evidence transferred to the current regime through ρ_z^L (the keep-or-redraw chain is
reversible, so this is exact). Against fixed quoters the anonymous regime hit rate falls from
0.77 (L = 1) to 0.71 (L = 20) while id / oracle stay at 0.87 / 0.89.

Also from the top-10 MM set: the leader-switch statistic is *below* its shuffled null on BTC
(0.49 vs 0.64) and ETH (0.64 vs 0.70) — a persistent leader, not turn-taking — and above it only
on HYPE (0.65 vs 0.56). MM–MM avoidance among the top-10 is strong on HYPE (0.33) and SOL (0.27),
weak on ETH (0.68), absent on BTC (0.88): the CCG signal appears on the less liquid books.

## Simulator with the mark lag (`cfl_exp2_identity_marklag.csv`; κ = 1, ρ = 0, margin 1, 5 seeds)

| ᾱ | ρ_z | mark lag | anon | id | oracle | id − anon (paired) | oracle − anon |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.9 | 1 / 5 / 20 | 0.65 / 0.65 / 0.65 | 0.68 / 0.67 / 0.68 | 0.69 | +0.03 / +0.03 / +0.03 | +0.04 / +0.04 / +0.03 |
| 0.3 | 0.99 | 1 / 5 / 20 | 0.78 / 0.80 / 0.79 | 0.80 / 0.81 / 0.80 | 0.84 | +0.02 / +0.02 / +0.01 | +0.07 / +0.05 / +0.05 |
| 0.5 | 0.9 | 1 / 5 / 20 | 0.68 / 0.68 / 0.65 | 0.76 / 0.75 / 0.79 | 0.75 | +0.08 / +0.07 / **+0.14** | +0.07 / +0.07 / +0.10 |
| 0.5 | 0.99 | 1 / 5 / 20 | 0.97 / 0.94 / 0.89 | 1.02 / 1.00 / 1.02 | 1.11 | +0.05 / +0.06 / **+0.13** | +0.14 / +0.17 / +0.22 |

* **The lag does what the data say it should, where the regime matters.** At ᾱ = 0.5 the
  anonymous entrant's profit falls as the mark arrives later (0.97 → 0.89 at ρ_z = 0.99) while
  the identity entrant's is flat, so identity's increment grows from +0.05 to +0.13 rent units
  (every seed positive at ρ_z = 0.9, lag 20). The anonymous entrant is still trading on a regime
  belief that is 20 arrivals stale; the identity entrant is not.
* At ᾱ = 0.3 the lag changes nothing (+0.02 throughout): with the incumbents at ~11 ticks, even the
  toxic state leaves an undercut profitable, so the regime is not decision-relevant and neither
  is knowing it sooner. Identity pays when the toxic state crosses the undercut threshold —
  which depends on how wide the cartel quotes relative to break-even. That is a sharp, testable
  statement: on Hyperliquid, the value of the wallet tag to an inside quoter should be larger on
  books where the resting spread is close to the toxic-regime break-even.
* Calibration targets for the sim, all from `calibrate.py`: regime contrast from the toxic /
  benign conditional markouts; mark lag from order inter-arrival (0.4–1.8 s) against the markout
  horizon (seconds to a minute) → L ≈ 5–60 arrivals; the cartel's quote relative to break-even
  from the L2 spread once the book is recorded.

## Live feed vs archive (`results/live_2026_10/`, Mac recorder, Oct 6–7 2026, 19 h, no book)

| archive Jan 26–28 / live Oct 6–7 | BTC | ETH | HYPE | SOL |
|---|---|---|---|---|
| prints / min | 262 / 248 | 157 / 135 | 313 / 225 | 61 / 65 |
| orders / min | 97 / 122 | 68 / 68 | 135 / 77 | 34 / 35 |
| ρ_order | 0.020 / 0.018 | 0.034 / 0.031 | 0.038 / 0.051 | 0.045 / 0.045 |
| ρ_fill | 0.57 / 0.49 | 0.53 / 0.47 | 0.53 / 0.65 | 0.43 / 0.47 |
| type rank corr | 0.17 / 0.15 | 0.39 / 0.41 | 0.11 / 0.30 | 0.42 / 0.40 |
| maker markout 10 s (bps) | −0.69 / −0.10 | −0.79 / −1.02 | −0.12 / −0.36 | −0.41 / −0.42 |
| identity given anon signal (bps) | −0.63 / −0.24 | −0.47 / −1.25 | −0.74 / −0.50 | −0.87 / −0.92 |
| P(adverse move before next order) | 0.18 / 0.11 | 0.21 / 0.13 | 0.21 / 0.25 | 0.20 / 0.14 |

Structural statistics (rates, sweep collapsing, wallet persistence, type persistence, identity's
increment) reproduce across the two sources; the feed and the archive are the same data. State
statistics moved: BTC was much less toxic in the live window and identity's increment on BTC
shrank with it — the simulator's prediction that identity pays when the toxic regime crosses the
undercut threshold. HYPE is at $90 vs $29 in January. The top-10 MM statistics (avoidance,
leader-switch vs null) flipped between samples and are not stable at 19 h / 3 days with a
volume-selected wallet set; do not report them until a fixed set over a longer window exists.
Coverage (`data/coverage.py`): 20 hours, 770k prints, no missing hours or wallet ids; one 42-min
gap (laptop lid) in hour 17 UTC on Oct 7.
