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

## True-mid markouts (`results/live_2026_10/`, six hours of L2 on Oct 8 2026 + 36 h of trades)

| | BTC | ETH | HYPE | SOL |
|---|---|---|---|---|
| quoted spread, median / p90 (bps) | 0.12 / 0.12 | 0.39 / 0.39 | 0.11 / 0.23 | 0.87 / 0.87 |
| = one tick at 5 significant figures | $1 on $83.7k | $0.10 | $0.001 | $0.01 |
| top-of-book depth, median (USD) | 508k | 314k | 10.6k | 100k |
| true-mid maker markout 1 s / 10 s / 60 s (bps) | +0.33 / −0.33 / −0.31 | +0.58 / −0.33 / −0.31 | +1.36 / −0.36 / −0.15 | +0.90 / −0.30 / −0.48 |
| proxy markout 10 s (bps), for comparison | −0.15 | −0.83 | −0.42 | −0.40 |
| identity split vs true mid: after toxic / after benign | −0.56 / −0.07 | −0.56 / −0.06 | −0.77 / −0.30 | −0.64 / +0.05 |

* **The inside is pinned at one tick.** p90 = median on BTC, ETH and SOL: the quoted spread is the
  5-sig-fig tick essentially all the time. A CFL-style rent cannot appear as a wider quoted spread
  on these books; it can only live in queue position at the one-tick spread and in how far sweeps
  walk the book. For the simulator this moves the relevant action from "how wide" to "where in the
  queue at one tick" — the priority-fee lever in the proposal — and makes the tick-size result of
  Cartea–Chang–Penalva the binding constraint, not a robustness check. HYPE (depth $10.6k) is the
  one book that opens to two ticks.
* **Adverse selection at 10 s is −0.33 bps on every coin** once bid–ask bounce is removed; the
  trade-price proxy had spread it from −0.15 to −0.83. A resting fill earns the half-spread at
  1 s and is under water by 10 s: on net, resting liquidity on Hyperliquid loses to the mid within
  ten seconds. The identity split survives the true mid unchanged.
  *Superseded in level by the bbo run below: the sign reproduces on every coin, the uniform
  −0.33 does not (−0.52 / −0.92 / −0.25 / −0.02), and the +0.3 to +1.4 at 1 s was the stale
  mid — with a fresh mid the maker is already under water at 1 s on three coins.*
* **Measurement caveat.** The `l2Book` feed delivered 4,020 snapshots in six hours (one per 5.4 s),
  not one per block. A stale mid inflates the effective spread (0.9–3.6 bps vs the 0.1–0.9 quoted)
  because trade direction is correlated with the move since the last snapshot; the 10-s markouts
  are unaffected. The recorder now also subscribes to `bbo` (top of book on every change);
  `analysis/l2.py` prefers it. Effective-spread numbers from this run should not be quoted.

## The book at top-of-book resolution (`results/live_2026_10/`, six hours of `bbo` on Oct 11–12 2026, 125.8 h of trades)

Recorder with the `bbo` subscription, 21:00 Oct 11 – 04:00 Oct 12 UTC (6.0 h in one session,
124k top-of-book changes on BTC, median gap 0.125–0.133 s; the 5.4-s `l2Book` throttle is gone).
Marked fills: 41.0k / 25.7k / 47.3k / 15.3k (BTC / ETH / HYPE / SOL). Two alignment rules were
needed and are now in `analysis/l2.py` (tests in `tests/test_l2.py`):

1. **The fill's reference mid is the last snapshot strictly before the trade time.** Hyperliquid
   stamps a trade and the top-of-book update it caused with the same block time, so "at or
   before" returns the post-trade book. On the first pass that gave an effective spread of
   −0.02 bps on BTC (a taker paying a negative spread). The check that the rule is right: a fill
   at the pre-trade touch now pays exactly the quoted spread on every coin (row 5 below).
2. **Marks exist only inside the recorder's coverage.** Trades run continuously (the long
   recorder) but the book stream exists only while a `bbo` recorder is alive; a lookup whose
   latest snapshot is older than 60 s returns NaN instead of a frozen mid. `calibrate` reports
   `l2_coverage_h` and `l2_sessions`. (On this run every marked fill was already inside the
   window; the guard changed nothing here and is there for the next run.)

| | BTC | ETH | HYPE | SOL |
|---|---|---|---|---|
| quoted spread, median / p90 (bps) | 0.12 / 0.12 | 0.39 / 0.39 | 0.12 / 0.58 | 0.90 / 0.90 |
| one tick (bps) | 0.12 | 0.39 | 0.11 | 0.86 |
| top-of-book depth, median (USD) | 611k | 283k | 6.2k | 78k |
| prints at the pre-trade touch | 73% | 79% | 54% | 80% |
| effective spread: fills at the touch (bps) | 0.12 | 0.42 | 0.17 | 0.92 |
| effective spread: fills the order walked to (bps) | 2.35 | 3.20 | 3.57 | 5.20 |
| effective spread: print-weighted / per order (VWAP) (bps) | 0.73 / 0.27 | 1.01 / 0.60 | 1.73 / 0.40 | 1.76 / 1.09 |
| orders that walk past the touch; fills per order | 10%; 2.36 | 10%; 1.88 | 19%; 2.57 | 8%; 2.05 |
| true-mid maker markout 1 / 10 / 60 s (bps) | −0.26 / −0.52 / −0.51 | −0.30 / −0.92 / −0.38 | −0.19 / −0.25 / −0.10 | +0.15 / −0.02 / +0.05 |
| 10-s markout, fills at the touch / walked fills (bps) | −0.41 / −0.80 | −0.56 / −2.26 | −0.45 / −0.01 | −0.27 / +0.98 |
| 10-s markout, Oct 8 window, for comparison (bps) | −0.33 | −0.33 | −0.36 | −0.30 |
| trade-price proxy, 10 s (125.8 h) (bps) | −0.19 | −0.85 | −0.40 | −0.36 |
| identity split vs true mid: after toxic / after benign (bps) | −0.97 / −0.27 | −0.73 / −0.79 | −0.65 / −0.17 | −0.48 / −0.01 |

### Findings
* **A fill at the touch pays one tick; the taker's cost is in depth.** 20–46% of prints are not at
  the pre-trade touch. One order in ten (one in five on HYPE) walks past it, and those prints
  pay 2.4–5.2 bps — twenty ticks on BTC. Print-weighted, the effective spread is 6× the quoted
  spread on BTC and 15× on HYPE; equal-weighted across orders (the order's VWAP against the
  pre-trade mid) it is 1.2–3.4× quoted. "The spread is one tick" is true of the quote and false
  of what a large taker pays. For the simulator: on a one-tick book the maker's price decision
  is not the inside quote but *how much to show at the inside and behind it* — depth is the
  action, and a sweep is the event that reveals it. (Caveat: the pre-trade book is the previous
  block's; a touch that moved within the block before the fill is counted as a walk, so the
  walk share is an upper bound. With BTC moving 0.3 bps per 0.13 s this cannot produce twenty
  ticks.)
* **The ordinary resting fill loses within ten seconds on every coin, and it is not a sweep
  artefact.** Fills at the touch — the fill a quiet maker gets — are at −0.41 / −0.56 / −0.45 /
  −0.27 bps at 10 s against a half-spread earned of 0.06 / 0.20 / 0.06 / 0.45. The adverse move
  at 10 s is 0.5–0.8 bps on all four books. With a fresh mid the maker is under water at *one*
  second on BTC, ETH and HYPE; the +0.3 to +1.4 bps "half-spread earned at 1 s" of the Oct 8 run
  was the stale mid. Sign and ordering (adverse move ≫ half-spread) reproduce across the two
  windows; the level does not (Oct 8 −0.33 on every coin was a six-hour coincidence). Report as:
  negative on all four coins in both windows, 0.0–0.9 bps, always larger than the half-spread.
* **Walked fills split by book.** On the deep BTC and ETH books the makers an order walks to lose
  *more* than the touch (−0.80, −2.26): an order that pays twenty ticks to walk BTC is informed,
  and the impact is permanent. On HYPE and SOL walked fills break even or earn (−0.01, +0.98):
  walking a $6k-deep or 0.9-bps-wide book is cheap and uninformative, the price reverts, and the
  maker who was resting deep keeps the walk. This is the simulator's regime contrast, measured by
  depth instead of by wallet: the toxic order is the one that walks a deep book.
* **Identity vs the true mid** holds on BTC, HYPE and SOL (after toxic − after benign −0.70 /
  −0.48 / −0.47 bps); on ETH it is +0.06 at 6 h of marked fills (the trades-only split over
  125.8 h is −0.98 on ETH). Six hours is not enough for the per-wallet classification; the
  125.8-h trades-only numbers are the ones to quote for identity.
* **Trades-only statistics at 125.8 h** (Oct 7–12; supersede the 19-h column of the live-vs-
  archive table): ρ_order 0.025 / 0.036 / 0.049 / 0.050 (baseline 0.004–0.015); type rank corr
  0.35 / 0.41 / 0.30 / 0.51 (all p < 1e-6); identity given the anonymous signal −0.66 / −1.02 /
  −0.52 / −0.63 bps, anonymous signal alone −0.31 / −0.29 / −0.10 / −0.47; P(adverse move
  before the next order) 0.11 / 0.13 / 0.22 / 0.16. All within the ranges of the January archive.

### What this changes in the paper and the model
* §5.6.1 of the revised deliverable rewritten around this table; the "−0.33 on every coin"
  sentence and the audit row are replaced; the synthesis's "average resting fill loses 0.33 bps"
  becomes "the fill at the touch loses 0.3–0.6 bps within ten seconds".
* The queue game's premise (fixed one-tick spread, value only in selection) is confirmed for the
  touch. The depth result adds the second action the queue model needs: size shown at and behind
  the inside. The priority-fee lever buys position *at the touch*; depth behind it is where the
  informed sweep lands on BTC/ETH.
* Next book run should be days, not hours, to put intervals on the touch-fill markout by coin
  and by hour of day, and to run the per-wallet classification on marked fills. The recorder is
  cheap; the laptop lid is the constraint. A small always-on machine (a $5/month VPS) would
  remove it.

---

# Identity grid re-run at the break-even rule (`cfl_exp2_identity_m1.csv`)

Same grid as `cfl_exp2_identity.csv` (wallet persistence ρ × dispersion κ × ᾱ, 5 seeds) with the
entrant undercutting at break-even (margin 1) instead of margin 2. The result stands:
id − anon = +0.17 / +0.17 (ᾱ = 0.3, κ = 1, ρ = 0.9 / 0.99), +0.23 / +0.17 (ᾱ = 0.5, κ = 1), +0.06 to
+0.24 at κ = 5, exactly 0.00 at ρ = 0 or κ = ∞; oracle − id within ±0.06. §5.4(a) of the revised
deliverable is no longer provisional.

---

# The one-tick queue game (`sim/queue_env.py`, `experiments/queue_game.py`, `queue_always.csv`)

Model: the spread is fixed (h = h^C(ᾱ) at break-even, or h^C + 1 when a coarse tick leaves the
inside slightly supra-competitive); makers choose REST or OUT each period; resting makers form a
time-priority queue; a fill consumes the front order, which rejoins at the back if it keeps
resting (round-robin); stepping out and returning means joining at the back; a period with
nobody resting has no trade and no print (`noliq`, uninformative about the regime). Takers, wallet
types, regime and mark lag are the spread game's. Per-fill value at h against type α is
(1 − q)·x − q·μ with q the fill-conditional informed probability: at h^C it is +0.15 on average,
+1.10 in the calm state and −2.26 in the toxic one (κ = 1); at h^C + 1, +0.58 / +1.57 / −1.99.

Incumbents here are two always-resting makers (the tick-constrained competitive book); one entrant
per policy; 300 episodes; 5 seeds; ᾱ = 0.3, κ = 1. Profit per period per maker:

| spread | ρ_z | L | always-rest | anon | id | oracle | id − anon (paired, min over seeds) |
|---|---|---|---|---|---|---|---|
| h^C | 0.9 | 1 / 5 / 20 | 0.04 | 0.10 / 0.09 / 0.05 | 0.14 / 0.14 / 0.15 | 0.15 | +0.04 / +0.05 / **+0.10** (min +0.05) |
| h^C | 0.99 | 1 / 5 / 20 | 0.03 | 0.16 / 0.15 / 0.12 | 0.16 / 0.16 / 0.16 | 0.18 | +0.01 / +0.01 / +0.04 |
| h^C + 1 | 0.9 | 1 / 5 / 20 | 0.15 | 0.16 / 0.14 / 0.14 | 0.21 / 0.21 / 0.21 | 0.21 | +0.06 / +0.07 / +0.07 (min +0.04) |
| h^C + 1 | 0.99 | 1 / 5 / 20 | 0.11 | 0.22 / 0.20 / 0.19 | 0.24 / 0.23 / 0.23 | 0.25 | +0.02 / +0.03 / +0.04 |

Informed share of fills (h^C, ρ_z = 0.9): always-rest 0.27; id entrant 0.16; the two incumbents
0.29 with their profit at 0.00 to −0.03. The id entrant rests in ~65% of periods and takes 20% of
fills instead of a round-robin third.

## Findings
* **On a one-tick book, identity is worth something in every cell and every seed but one**, and the
  increment grows with the mark lag: at break-even with ρ_z = 0.9 the anonymous selector's edge
  over always-resting collapses from +0.06 (L = 1) to +0.01 (L = 20) while the identity entrant's
  stays at +0.10. With a 20-arrival lag the anonymous tape is too stale to time the regime;
  identity is not. The identity entrant attains 90–100% of the oracle.
* **At break-even the whole profit is selection.** Always-resting earns ≈ 0, as the data say the
  average resting fill on BTC does; the identity entrant earns 0.14–0.16 per period by resting
  when the regime is calm and leaving the toxic flow to those who stay. The incumbents' fills get
  dirtier (0.27 → 0.29–0.31 informed) and their profit goes to zero or below.
* **Above break-even the tick rent is shared by rotation and identity adds ~45% to it** (0.15 →
  0.21), taken from the incumbents (0.14 → 0.11).
* The spread game and the queue game now give the same answer by different mechanisms: a quoter
  who knows who just printed captures the benign flow and leaves the toxic flow to identity-blind
  rivals. In the spread game that shows as undercutting selectively; here as resting selectively.
* Not yet done: learned incumbents in the queue game. Tabular Q-learners with (status, depth,
  anonymous signal) state and a persistent signal fall into a self-fulfilling lock (nobody rests →
  no prints → no signal → nobody rests) at break-even and learn to stay out; above break-even with
  optimistic initialisation is the next attempt. The priority fee (pay to jump the queue) is the
  lever to add after that.

### Learned incumbents in the queue game (`queue_q.csv`) — not yet credible
Two tabular Q-learners (state: own status, depth, anonymous signal; optimistic init 1.0, lr 0.05,
12k episodes) one tick above break-even end up resting rarely: their profit is 0.01–0.03 per
period against 0.146 for always resting, and an always-resting entrant takes 90–95% of fills.
The bootstrapped value of "out" (exactly 0 reward, value borrowed from states reached through the
other learner) beats the noisy value of "rest" whose rewards are ±2 and clustered by regime.
Entrant numbers against these incumbents measure a near-empty book and are not reported. Options:
Monte-Carlo returns instead of one-step bootstrapping; a longer, slower exploration schedule;
or treat always-rest as the incumbent benchmark (which is what the data say the competitive
one-tick book is) and reserve learning for the entrant side.

---

# Identity grids at ten seeds (`cfl_exp2_identity_10seed.csv`, `cfl_exp2_identity_regime_10seed.csv`; SuperCloud job 5897441, 70 min on 12 cores)

Same two grids as the 5-seed runs (margin 1, 50 wallets, 1000 evaluation episodes), seeds 0–9,
one trained two-maker cartel per seed. Everything below is in rent units, paired within seed
(id − anon on the same cartel with common random numbers), with 95% intervals over seeds and the
minimum over seeds. Figures: `fig_cfl_exp2_identity{,_value,_as}_10seed.*` (persistence
channel), `fig_cfl_exp2_marklag_10seed.*` (regime channel, paired). The 5-seed numbers reproduce
seed for seed (Monte-Carlo noise of the evaluation only); the intervals are what is new.

## Wallet-persistence channel (ρ)
| ᾱ | κ_α | ρ | anon | id | oracle | id − anon (±95%) | min | seeds > 0 |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 1 | 0.9 / 0.99 | 0.61 / 0.64 | 0.76 / 0.86 | 0.76 / 0.84 | +0.15 ± .05 / +0.22 ± .09 | +0.02 / +0.08 | 10 / 10 |
| 0.5 | 1 | 0.9 / 0.99 | 0.72 / 0.88 | 0.94 / 1.18 | 0.95 / 1.16 | +0.22 ± .08 / +0.30 ± .14 | +0.08 / −0.06 | 10 / 8 |
| 0.1 | 1 | 0.9 / 0.99 | 0.59 / 0.59 | 0.62 / 0.66 | 0.64 / 0.68 | +0.03 ± .09 / +0.07 ± .13 | −0.23 / −0.15 | 6 / 5 |
| 0.3 | 5 | 0.9 / 0.99 | 0.65 / 0.66 | 0.69 / 0.74 | 0.69 / 0.74 | +0.04 ± .04 / +0.07 ± .05 | −0.05 / −0.04 | 8 / 8 |
| 0.5 | 5 | 0.9 / 0.99 | 0.66 / 0.71 | 0.77 / 0.94 | 0.78 / 0.93 | +0.11 ± .04 / +0.23 ± .08 | −0.02 / −0.03 | 9 / 9 |
| any | any | 0, or κ_α = ∞ | — | — | — | 0.00 exactly | 0.00 | 0 |

Oracle − id within ±0.04 everywhere: the identity entrant attains the oracle on this channel.

## Regime channel (ρ_z, mark lag L), κ_α = 1
| ᾱ | ρ_z | anon (L = 1/5/20/60) | id | oracle | id − anon (±95%) | seeds > 0 |
|---|---|---|---|---|---|---|
| 0.3 | 0.9 | 0.65 / 0.65 / 0.64 / 0.66 | 0.68 / 0.68 / 0.67 / 0.67 | 0.69 | +0.03±.02 / +0.03±.03 / +0.03±.06 / +0.01±.06 | 8 / 9 / 6 / 5 |
| 0.3 | 0.99 | 0.76 / 0.77 / 0.75 / 0.72 | 0.78 / 0.78 / 0.78 / 0.76 | 0.80 | +0.02±.02 / +0.02±.02 / +0.03±.02 / +0.04±.02 | 8 / 7 / 9 / 10 |
| 0.5 | 0.9 | 0.71 / 0.69 / 0.66 / 0.68 | 0.76 / 0.76 / 0.77 / 0.76 | 0.76 | +0.06±.02 / +0.08±.03 / **+0.11±.05** / +0.08±.05 | 9 / 10 / 9 / 9 |
| 0.5 | 0.99 | 0.95 / 0.94 / 0.88 / 0.84 | 1.01 / 1.01 / 1.01 / 1.00 | 1.05 | +0.06±.04 / +0.07±.04 / +0.13±.07 / **+0.16±.07** | 7 / 9 / 9 / 9 |
| 0.1 | 0.9 / 0.99 | 0.53–0.63 | 0.54–0.60 | 0.59 / 0.66 | −0.03 to +0.01, none significant | 4–6 |

κ_α = 5 (types within 0.2 of each other): id − anon = −0.04 to +0.04 in every cell, none
significant; minima −0.14 to **−0.53**; 5–7 seeds positive.

## Findings
* **The persistence channel is robust and large** (+0.15 to +0.30, 10/10 seeds at ρ = 0.9 for
  ᾱ ≥ 0.3), zero exactly at ρ = 0 or κ_α = ∞, oracle ≈ id. It is the theory check: the data put
  ρ at 0.02–0.05.
* **The regime channel is real, smaller, and sharply conditional.** At ᾱ = 0.5 identity's value
  grows with the mark lag — +0.06 → +0.16 at ρ_z = 0.99 (still rising at L = 60) and
  +0.06 → +0.11 at ρ_z = 0.9 (peak at L = 20, back to +0.08 at L = 60). The peak is the regime's
  lifetime: at ρ_z = 0.9 a regime lasts ~10 arrivals, so a mark 60 late is no worse than one 20
  late — the tape was already useless at 20. *Identity's value grows with the mark lag as long as
  the lag is shorter than the regime's lifetime.* At ᾱ = 0.3 the increment is +0.02 to +0.04 and
  significant only at ρ_z = 0.99, L ≥ 20; at ᾱ = 0.1 it is zero or slightly negative. The 5-seed
  statement "+0.05 → +0.13 at ᾱ = 0.5" survives with intervals; the general "identity's value
  reappears" must be read as "where the toxic state crosses the undercut threshold".
* **Why the spread game understates identity.** Against a cartel at 11–17 ticks the entrant's
  main instrument is the undercut, which captures most of the rent whatever it knows (anon
  0.65–0.95 rent units). Identity is a few percent on top. On a one-tick book there is no
  undercut, and identity is the *whole* edge (queue game: 0.14 vs 0.08 vs 0.04 per period for
  id / anon / always-rest). The two games are consistent: information is worth most where the
  action space is smallest.
* **Where information hurts (the reaction function).** At κ_α = 5, ᾱ = 0.3, ρ_z = 0.9, L = 20,
  seeds 1, 2 and 8 have anon 1.07–1.48 > id 0.94–1.05 ≈ oracle. In those cartels the anonymous
  entrant undercuts 97–100% of periods with a steady quote, takes 86–99% of fills, and the
  incumbents earn 0.02–0.27; the id/oracle entrants' quotes move with their regime belief and the
  incumbents' learned reaction takes back 26–33% of fills (incumbent profit 0.19–0.33). The
  better-informed entrant provokes the defenders. In seed 0 it is the reverse (anon 0.41, id
  0.78: the anon entrant's filter believes toxic 29% of the time and gives up fills). Entrant
  profit against frozen Q-tables is the information effect plus an interaction with the
  incumbents' reaction function, and when the information effect is small (κ_α = 5) the
  interaction dominates and has either sign. This is the steering channel of RQ3 observed by
  accident, the reason the paper reports the minimum over seeds, and an argument for the
  queue game, where the incumbents have no quote to react with.
* Level intervals (±0.05–0.1) are dominated by which cartel the seed trained; only paired
  differences are informative. `analysis/marklag_fig.py` plots the paired differences.
* `κ` for the concentration is now `κ_α` in all documents (it collided with the fraction of rent
  destroyed `κ_A`); the CSV column is still `kappa`.

---

# Who earns the spread: per-maker realized spread on marked fills (`live_2026_10/makers_<COIN>.csv`, `fig_makers_live_2026_10.*`)

RQ3's empirical anchor. Same six hours of top-of-book data; every marked fill carries the
maker's wallet. `analysis/makers.py`: 10-s true-mid markout per maker with a 5-minute-cluster SE;
the top-10 MM wallets by **full-window** volume as a fixed set (chosen on activity over 125 h,
not on the six hours being scored); "at the touch" splits by where the print landed.

| | BTC | ETH | HYPE | SOL |
|---|---|---|---|---|
| all marked fills (bp) | −0.52 ± .06 | −0.92 ± .38 | −0.25 ± .09 | −0.02 ± .23 |
| top-10 MM wallets: markout; fill share; touch share | −0.14 ± .06; 9%; 0.79 | +0.09 ± .10; 23%; 0.88 | +0.04 ± .08; 19%; 0.70 | +0.58 ± .10; 21%; 0.94 |
| other MM wallets | −0.44 ± .05; 68%; 0.73 | −0.74 ± .20; 46%; 0.79 | −0.11 ± .10; 57%; 0.43 | −0.13 ± .31; 50%; 0.77 |
| not classified as MM | −0.90 ± .11; 23%; 0.68 | −1.97 ± .91; 30%; 0.73 | −0.81 ± .15; 24%; 0.66 | −0.26 ± .24; 29%; 0.77 |
| **fills at the touch only: top-10 / everyone else** | −0.04 ± .05 / −0.45 ± .04 | +0.24 ± .08 / −0.84 ± .26 | +0.01 ± .07 / −0.60 ± .08 | +0.60 ± .09 / −0.54 ± .08 |
| makers with ≥ 100 fills; share with mean > 0; significantly > 0 | 77; 21%; 5 | 52; 23%; 3 | 77; 38%; 5 | 40; 55%; 8 |
| fill share of the significantly-positive makers | 7% | 16% | 8% | 22% |
| split-half rank corr of per-maker markout (n makers) | 0.86 (61) | 0.64 (36) | 0.57 (63) | 0.76 (32) |

All split-half p < 1e-3.

## Findings
* **The hierarchy the simulator predicts is in the data.** Makers the classifier does not call
  MMs (retail limit orders) lose most (−0.26 to −1.97 bp per fill at 10 s); the other MMs lose
  less (−0.11 to −0.74); the top-10 MMs break even or earn (−0.14 to +0.58). The toxic flow lands
  on the slow and the identity-blind. This is the "selection cartel" structure: not a spread held
  above competitive, but a fill distribution in which the toxic fills go to those who do not
  select.
* **Same action, different outcome — so it is selection.** Restricting to fills *at the touch*
  (the same price, the same tick) the top-10 earn 0.4–1.1 bp per fill more than everyone else:
  −0.04 / +0.24 / +0.01 / +0.60 against −0.45 / −0.84 / −0.60 / −0.54. The figure makes it
  visible: at touch share ≈ 1 the per-maker markouts run from −1.8 to +0.9 bp, with the top-10 at
  the top of the column. Makers resting at the same price get different fills; the difference is
  *when* they are at the front — queue position and cancellation, i.e. selection.
* **Part of the edge is not being swept.** Top-10 touch share 0.70–0.94 vs 0.43–0.79 for other
  MMs: the top makers show less behind the inside. Two styles are visible on BTC/ETH/HYPE: the
  touch-only makers (touch share ≈ 1, effective ≈ one tick) and the depth makers (touch share
  0.05–0.5, effective 1.5–5 bp); on HYPE the depth makers earn (+0.03 to +0.81, the "walked fills
  earn on thin books" result from the maker's side), on ETH they lose (−0.44 to −1.07).
* **Skill is persistent, not luck.** Split-half (first 3 h vs last 3 h) Spearman of per-maker
  markout: 0.57–0.86, all p < 1e-3. The same makers are at the top in both halves. Caveat: six
  hours; part of this persistence is persistent *position* (the same wallets at the front
  throughout), which is the mechanism rather than a confound, but a multi-day window is needed
  to separate skill from a single calm spell.
* **Few makers select, and they take a fifth of the fills.** 3–8 makers per coin have a
  significantly positive 10-s markout; they take 7–22% of all fills. The rest of the book — most
  wallets, most fills — loses to the mid within ten seconds.
* **BTC is the hard book.** Even the top-10 are slightly under water at the touch (−0.04 ± .05)
  and hold only 9% of fills; the deepest book with the finest tick (0.12 bp) leaves the least
  room. SOL (0.9-bp tick) is the easy one: +0.60 at the touch for the top-10.
* For the paper: "makers who select earn a positive realized spread while the fill at the touch
  loses" holds on ETH, HYPE and SOL and nearly on BTC; identity-blind resting liquidity is the
  counterparty of the selection rent. The statistic to carry forward is the top-vs-rest gap at
  the touch (+0.4 to +1.1 bp) with its persistence.

---

# Exploitability at ten seeds (`exploitability_10seed.csv`, `exploitability_summary_10seed.csv`, `fig_exploitability_10seed.*`; SuperCloud job 5969375)

180 frozen populations (robust_exp2 at seeds 0–9: 120; cfl_exp2: 60), four attackers, 500 episodes,
common random numbers. C spans 0.42–0.97. Every entry of the 5-seed table moved by less than its
interval; what changed is the precision.

| | pooled | α = 0.3 | within cell |
|---|---|---|---|
| L_BR slope on C, max over attackers | 1.02 [0.90, 1.18] | 1.03 [1.00, 1.07] | 1.07 [0.96, 1.22] |
| L_BR slope, markout attacker | 0.71 [0.56, 0.90] | 0.86 [0.71, 1.02] | 0.74 [0.59, 0.98] |
| κ_A slope, markout | −0.06 [−0.31, 0.26] | 0.23 [0.03, 0.43] (ρ 0.08, p 0.33) | −0.07 [−0.33, 0.31] |
| κ_A slope, undercut | −0.17 [−0.71, 0.30] | 0.25 [0.04, 0.49] (ρ 0.11, p 0.18) | −0.30 [−0.94, 0.35] |
| κ_A slope, oracle | −0.11 [−0.41, 0.22] | 0.10 [−0.11, 0.34] | −0.07 [−0.46, 0.40] |
| κ_A slope, competitive (damage-maximiser) | −0.18 [−0.35, 0.02] | −0.10 [−0.14, −0.06] | −0.15 [−0.29, 0.04] |

Means: κ_A = 1.14 (competitive), 0.71 (markout), 0.71 (oracle), 0.73 (undercut). Attacker profit
0.10 / 0.51 / 0.49 / 0.48 rent units. Damage-maximiser is the competitive quoter in 179/180
populations; the attacker's own best response is the markout rule in 89, the undercutter in 58, the
oracle in 33.

* The literal hypothesis holds by construction (slope ≈ 1 for the damage-maximiser); the
  substantive one — more collusive ⇒ a *larger fraction* lost — is at most weakly supported: at
  α = 0.3 the OLS slope of κ_A on C for the profit-maximising attackers is +0.23 to +0.25 with
  intervals just excluding zero, the rank correlations are not significant, and over the observed
  range of C that is ≤ 0.13 in κ_A; pooled and within-cell slopes include zero. Report as "κ_A ≈ 0.7
  with at most a weak dependence on C". The damage-maximiser's κ_A *falls* slightly with C.
* Paper §5.2, Table tab:exploit and the audit row updated to the 180-population numbers.

---

# Queue game with learned incumbents (`queue_hedge_fringe2.csv`, `queue_always_fringe2.csv`, `queue_q_fringe2.csv`; `fig_queue_incumbents.*`; SuperCloud jobs 5969376 and the resubmit)

## Two failures of tabular Q-learning, and the fix
1. **The lock** (first attempt, no fringe): two learners that both step out in a toxic spell leave
   an empty book → no prints → frozen tape signal → nobody rests. Fixed by a *fringe* of two
   always-resting makers (retail orders, the vault), which is also the realistic population.
2. **The flicker** (`queue_q_fringe2.csv`, 12k episodes, fringe 2): the learners rest 40–60% of
   periods but hold 0–4% of fills and earn ≈ 0. With time priority the reward of resting arrives
   only after several consecutive periods in line; ε-greedy breaks the chain, the learner never
   reaches the front, sees zero reward whether it stays or leaves, and alternates. Tested on one
   cell at 12k episodes: three-status state, rank-in-state (out/front/2nd/3rd/deeper),
   optimistic init 5.0, and both together — fill share 0.00 in every variant. (The one-seed
   diagnostic at 4k episodes that looked healthy was ε = 0.45 random resting.) Conclusion:
   *identity-blind tabular Q-learners cannot learn to queue.* Kept as a result, not as an
   incumbent.
3. **The fix**: the proposal's no-regret incumbent — EXP3 over complete queue policies
   {always rest, never rest, anonymous selector}, per-episode bandit feedback, η = 0.1, 5%
   exploration, rewards scaled by 30 (`QueueHedgeAgent`). Converges in a few hundred episodes.

## No-regret incumbents (2 EXP3 + 2 fringe), 5 seeds, ᾱ = 0.3, κ_α = 1
| spread | ρ_z | L | weight on anon-selector / always | incumbent pnl | fringe pnl (always entrant) | always | anon | id | oracle | id − anon (paired, ±95%) |
|---|---|---|---|---|---|---|---|---|---|---|
| h^C | 0.9 | 1 / 5 / 20 / 60 | 1.0 / .9 / .66 / .59 vs 0 / .1 / .34 / .42 | .06 / .05 / .04 / .03 | .00 / .02 / .02 / .03 | .01 | .05 / .04 / .03 / .03 | .10 / .10 / .09 / .08 | .10 / .11 / .10 / .10 | +.053±.01 / +.057±.02 / +.053±.01 / +.056±.02 (5/5 each) |
| h^C | 0.99 | 1 / 5 / 20 / 60 | 1.0 at every L | .10 / .10 / .08 / .05 | −.04 / −.04 / −.02 / .00 | −.04 / −.02 / −.02 / .00 | .09 / .09 / .07 / .05 | .10 / .10 / .11 / .11 | .11 / .11 / .12 / .13 | +.003 / +.015 / +.039 / **+.063** (±.01) |
| h^C+1 | 0.9 | 1 / 5 / 20 / 60 | .9 / .47 / .46 / .58 vs .1 / .53 / .54 / .42 | .10 / .09 / .09 / .09 | .08 / .09 / .08 / .08 | .08 | .09 / .08 / .08 / .08 | .13 / .13 / .12 / .11 | .14 / .13 / .13 / .13 | +.041 / +.043 / +.036 / +.032 (±.01–.02) |
| h^C+1 | 0.99 | 1 / 5 / 20 / 60 | 1.0 / 1.0 / 1.0 / .9 | .15 / .14 / .12 / .10 | .04 / .04 / .05 / .07 | .03 / .04 / .05 / .06 | .13 / .13 / .12 / .09 | .15 / .15 / .16 / .15 | .15 / .16 / .16 / .17 | +.014 / +.019 / +.043 / +.052 (±.01) |

Same grid with always-resting incumbents + the same fringe (`queue_always_fringe2.csv`): id − anon
+.03 to +.04 at h^C, ρ_z = 0.9; 0.01 → 0.05 with L at ρ_z = 0.99.

## Findings
* **What incumbents learn.** Fresh tape (L ≤ 5, or ρ_z = 0.99): both become anonymous selectors.
  Stale tape (L ≥ 20 at ρ_z = 0.9): a third to 40% of weight moves to always-rest. One tick above
  at ρ_z = 0.9: half always-rest — the rotation rent of the tick beats selecting. Never-rest is
  never chosen. Learned incumbents earn 2–3× the fringe with 15–26% informed fills vs 26–31%.
* **Identity is worth more against incumbents that select.** +0.05 to +0.06 per period at
  break-even, ρ_z = 0.9, every L, 5/5 seeds — vs +0.03 to +0.04 against always-resters. The
  anonymous entrant's edge over always-rest shrinks with L (+0.04 → +0.01): once the incumbents
  read the tape, a tape-reading entrant is one more of them; the wallet tag is what is left. At
  ρ_z = 0.99 the increment grows with L from 0.00 to +0.06 — the mark-lag mechanism, again.
* **The fringe pays.** Fringe profit at break-even: +0.00 to +0.03 with an always-resting
  entrant, −0.02 to −0.04 with a selecting one, −0.10 at ρ_z = 0.99. Against no-regret
  incumbents at ρ_z = 0.99 even the always-resting *entrant* is under water (−0.04 at L = 1).
  Identity-blind resting liquidity is the counterparty of the selection rent — the structure in
  the per-maker Hyperliquid data (non-MM −0.3 to −2.0 bp; top-10 MM ≥ 0).
* Paper §5.7 rewritten around this; `fig_queue_incumbents` (break-even, both incumbent types);
  RQ3 restated. Open: priority fee; identity-aware incumbents (an "id" expert in the hedge set
  is a one-line change and is the identity-aware-control defence of RQ3).

---

# Identity-aware defenders and liquidity in stress (preliminary: 2 seeds, 3k training episodes, L = 5, break-even; `slurm/queue_identity.sh` is the full grid)

Add an identity expert to the no-regret incumbents' policy set, with and without the fringe.

| ρ_z | incumbents' set, fringe | they choose | incumbent pnl | entrant: always / anon / id | fringe pnl (id entrant) | nobody resting in toxic state (id entrant) | uninformed taker's fill prob, toxic: always / anon / id entrant |
|---|---|---|---|---|---|---|---|
| 0.9 | anon, fringe 2 | anon 1.0 | .03–.05 | .01 / .04 / .11 | −.02 | 0 | .81 / .81 / .81 |
| 0.9 | anon+id, fringe 2 | id .99 | .09–.11 | −.03 / .03 / .08 | −.06 | 0 | .81 / .81 / .81 |
| 0.9 | anon, fringe 0 | anon 1.0 | .04–.07 | .01 / .07 / .17 | — | .43 | .81 / .45 / .47 |
| 0.9 | anon+id, fringe 0 | id 1.0 | .12–.15 | **−.16** / −.06 / .10 | — | **.70** | .81 / .50 / **.25** |
| 0.99 | anon, fringe 0 | anon 1.0 | .14–.19 | −.10 / .14 / .12 | — | .77 | .82 / .15 / .18 |
| 0.99 | anon+id, fringe 0 | id 1.0 | .14–.22 | **−.16** / .11 / .07 | — | .84 (oracle entrant .97) | .82 / .17 / **.13** (oracle .02) |

* **Defenders that can see identities use them** (weight on the identity expert 0.75–1.0) and
  earn 2–3× what anonymous selectors earn.
* **Liquidity disappears in the toxic state when every maker can see it coming.** Without an
  uninformed fringe, the share of toxic-state periods with nobody at the inside is 0.70 (ρ_z = 0.9)
  to 0.84–0.97 (ρ_z = 0.99); an uninformed taker's fill probability in the toxic state falls from
  0.81 to 0.25 (ρ_z = 0.9) and 0.02–0.13 (ρ_z = 0.99). Anonymous selectors withdraw too (they read
  the tape) but later and less: 0.45–0.47 at ρ_z = 0.9. Identity makes the withdrawal faster and
  sharper where regimes are short relative to the mark lag — the same lag mechanism as the
  entrant's edge, seen from the venue's side. Uninformed takers in the toxic state (a third of its
  flow) are the collateral damage: *public identity reduces liquidity provision in stress.*
* **Whoever still rests is the sole counterparty.** An identity-blind always-resting entrant among
  identity-aware incumbents earns −0.16 per period at both persistences; with a fringe present the
  fringe's loss doubles (−0.02 → −0.06) when the incumbents become identity-aware.
* **The entrant's identity edge survives identity-aware incumbents** at ρ_z = 0.9 (+0.05 over the
  anonymous entrant with the fringe; the anonymous entrant is now the second-worst-informed agent
  on the book) and shrinks at ρ_z = 0.99 (+0.02).
* Caveat: 2 seeds, 200 evaluation episodes, one lag; the cluster grid (5 seeds, four lags, both
  spreads) is the reference. Metrics added to `experiments/queue_game.py`: `noliq_{calm,toxic}`,
  `uninformed_fill_{calm,toxic}`, `share_toxic`.

---

# The priority fee (preliminary: 2 seeds, L = 5, ρ_z = 0.9, break-even, fringe 2; `slurm/queue_fee.sh` is the full grid)

Mechanism (`sim/queue_env.py`): a third action REST-AND-PAY; payers are served ahead of non-payers,
time priority within each group; the fee f is charged per period of priority, filled or not; the
venue collects it. The selector's rule (`QueueWalletEntrant(pay=True)`): when it would rest and is
not already first in time priority, pay iff P(fill) · v(α_next) > f. Incumbents: EXP3 over
{always, never, anon, anon_pay}.

| f | incumbents' weight on paying | incumbent pnl (always / id_pay entrant) | fringe pnl | venue revenue / period | entrant: id (no pay) / id_pay / oracle_pay | id_pay entrant pays: calm / toxic | id_pay fill share | fringe informed share |
|---|---|---|---|---|---|---|---|---|
| 0 | — | .03 / .03 | .03 / −.01 | 0 | .10 / — / .11 | — | .13 | .29 |
| 0.25 | .70 | .00 / **−.05** | −.01 / **−.09** | .15 / .29 | .05 / **.13** / .18 | .73 / .11 | **.37** | **.43** |
| 0.5 | .48 | .01 / −.03 | .03 / −.06 | .08 / .23 | .09 / .09 / .11 | .43 / .04 | .33 | .35 |
| 0.75 | .39 | .03 / .03 | .03 / −.01 | 0 / 0 | .11 / .10 / .05 | 0 / 0 | .14 | .30 |
| 1.0 | — | .03 / .03 | .03 / −.01 | 0 | .10 / .10 / .11 | 0 / 0 | .13 | .29 |

* **The fee is an all-pay auction for position, and it dissipates the selectors' rent to the
  venue.** At f = 0.25 the incumbents pay in half of the calm periods and a quarter of the toxic
  ones, gain no position relative to each other, and their profit falls from +0.03 to 0 or below;
  the venue collects 0.15–0.31 per period, more than the whole maker rent at f = 0 (≈ 0.2). The
  learners notice: at f ≥ 0.5 their weight moves from `anon_pay` back to `anon`.
* **The best-informed agent gains from the fee.** The identity entrant pays selectively — 73% of
  calm periods, 11% of toxic ones at f = 0.25 — takes 37% of fills instead of 13%, and earns 0.13
  instead of 0.10; the oracle 0.18. The fee converts an information advantage into position,
  which a time-priority queue otherwise rations by patience.
* **The incidence falls on the identity-blind.** The fringe never pays, is pushed to the back of
  the serving order whenever anyone does, and keeps only the fills the selectors declined: its
  informed share rises from 0.29 to 0.43–0.49 and its profit from −0.01 to −0.09 / −0.10. Taker
  access is unchanged (the fringe is always there); the fee moves money from identity-blind makers
  to the venue and the informed, not from takers.
* The threshold at which paying stops is the calm-state fill value times the fill probability,
  ≈ 0.86 for an agent that knows the state (the oracle still pays at f = 0.75) and ≈ 0.5–0.6 for
  one estimating it.
* For the frontier: the fee is a venue lever that raises venue revenue and the informed entrant's
  profit while lowering identity-blind makers' profit and leaving taker welfare unchanged — i.e. it
  is not a defence of the incumbents; it is a tax on selection whose burden the selectors pass on.

## Full grid (`queue_hedge_always_never_anon{,_id}_fringe{0,2}.csv`; SuperCloud job 5971218; 5 seeds, 4k training episodes, 300 evaluation episodes; `fig_liquidity_identity.*`)

Supersedes the preliminary table above. Break-even, no fringe, identity entrant present — the
venue-design numbers (95% seed intervals in the figure):

| incumbents see | ρ_z | L = 1 / 5 / 20 / 60: nobody resting in toxic state | uninformed taker filled in toxic state | incumbent pnl |
|---|---|---|---|---|
| tape only | 0.9 | .54 / .09 / .06 / .12 | .36 / **.73** / **.75** / **.71** | .08 / .00 / .00 / .01 |
| wallet ids | 0.9 | .54 / .62 / .72 / .69 | .37 / .30 / **.23** / **.25** | .10 / .12 / .14 / .14 |
| tape only | 0.99 | .86 / .83 / .75 / .64 | .11 / .14 / .20 / .30 | .14 / .13 / .10 / .06 |
| wallet ids | 0.99 | .87 / .85 / .84 / .86 | .11 / .12 / .13 / .11 | .16 / .14 / .14 / .17 |

Always-resting (identity-blind) maker among those incumbents, no fringe: tape-only incumbents at
ρ_z = 0.9: −.07 / +.02 / +.03 / +.03; identity incumbents: **−.13 / −.16 / −.18 / −.18**; at ρ_z = 0.99:
−.24 → −.09 (tape) vs **−.27 to −.29** (ids). With the fringe present (fringe 2): the fringe earns
−.03 → +.02 (ρ_z = 0.9) and −.10 → −.05 (0.99) under tape-only incumbents, **−.06** and **−.12** under
identity incumbents, at every lag. Expert weights: identity incumbents choose the identity expert
0.70–1.0; tape-only incumbents at ρ_z = 0.9 move to always-rest as the tape goes stale (0.43–0.57
at L = 5–20 without a fringe, 0.77 at L = 60 with one).

### Findings
* **Under anonymity, the mark lag is what keeps liquidity in the toxic state.** With the tape
  only and a lag of five or more arrivals, the incumbents cannot tell a toxic spell from a calm
  one in time; they stay (many learn to always rest), an uninformed taker is filled 71–75% of the
  time in the toxic state, and the incumbents earn nothing — they are the counterparty. At L = 1
  they withdraw as sharply as identity-aware makers (0.36 vs 0.37).
* **Identity removes the lag, and the liquidity with it.** With wallet ids the incumbents withdraw
  in 62–72% of toxic-state periods at any lag; the uninformed taker's fill probability in the
  toxic state is 0.23–0.30 against 0.71–0.75 under anonymity (ρ_z = 0.9, L ≥ 5), and the
  incumbents earn 0.12–0.14 instead of ≈ 0. At ρ_z = 0.99 regimes last long enough for even a
  stale tape to catch them, and the two converge (0.11–0.13 either way at L ≤ 5; anonymity
  recovers to 0.30 only at L = 60). The venue's choice is therefore sharpest exactly where the
  entrant's identity edge is: short regimes relative to the lag.
* **Whoever is identity-blind on an identity book is the sole counterparty.** An always-resting
  maker earns −0.13 to −0.29 per period among identity-aware incumbents, against +0.02 to +0.03
  among anonymous incumbents with a stale tape. With a fringe present, the fringe's loss doubles
  (−0.06 vs −0.03; −0.12 vs −0.10) once the incumbents can see identities.
* **The entrant's own identity edge survives identity-aware incumbents** and is larger, not
  smaller: id − anon = +0.04 to +0.07 per period (fringe 2, break-even, 5/5 seeds, min +0.04),
  growing with L; against anonymous incumbents +0.05 (ρ_z = 0.9) and 0.00 → +0.06 (0.99). The
  anonymous entrant is the worst-informed selector on the book.
* One tick above break-even the same pattern holds with more liquidity overall (tape-only,
  ρ_z = 0.9, L ≥ 20: 0.74 filled; ids: 0.38–0.48).
* **For the frontier.** Anonymity is a transfer from makers to uninformed takers in stress:
  takers keep access (0.7 vs 0.25) and the makers who provide it earn nothing. Public identity is
  the reverse. Neither is a defence of the incumbents' rent; the lever decides who holds the toxic
  flow when nobody wants it. This is the venue-side statement of the mark-lag mechanism.

## Full grid (`queue_fee_anon.csv`, `queue_fee_id.csv`; SuperCloud job 5971619; 5 seeds × ρ_z {0.9, 0.99} × L {1, 5, 20} × spread {h^C, h^C+1} × fee {0, 0.1, 0.25, 0.5, 0.75, 1}; `fig_priority_fee.*`)

Supersedes the preliminary table. Break-even, fringe 2, means over ρ_z and L (the pattern is the
same in every cell; the by-cell numbers are in the CSVs):

| incumbents see | f | incumbents' weight on paying | pay in calm / toxic (incumbents) | id entrant (paying) | incumbents | fringe | venue |
|---|---|---|---|---|---|---|---|
| tape | 0 | — | — | .10 | .06 | −.05 | 0 |
| tape | 0.1 | .96 | .77 / .27 | **.20** | .03 | **−.15** | .16 |
| tape | 0.25 | .87 | .66 / .18 | .14 | −.02 | −.16 | .33 |
| tape | 0.5 | .54 | .44 / .06 | .05 | −.04 | −.14 | **.41** |
| tape | 0.75 | .26 | .02 / .00 | .10 | .06 | −.06 | .02 |
| ids | 0 | — | — | .09 | .11 | −.09 | 0 |
| ids | 0.1 | 1.00 | .94 / .09 | .14 | .11 | **−.21** | .17 |
| ids | 0.25 | 1.00 | .93 / .08 | .04 | .04 | −.21 | .41 |
| ids | 0.5 | .78 | .86 / .06 | −.04 | −.03 | −.20 | **.62** |
| ids | 0.75 | .23 | .31 / .01 | .06 | .07 | −.13 | .17 |

A non-paying identity entrant among paying identity-aware incumbents at f ≤ 0.25: fill share
**0.000–0.001**, profit 0.00 — if you do not pay you do not trade in calm periods.

### Findings
* **The fee is an all-pay auction for position.** Every selector pays, in the calm state: tape-only
  incumbents 66–77% of calm periods and 18–27% of toxic ones (they cannot tell), identity-aware
  incumbents 93–94% of calm and 8–9% of toxic periods (they can). The fee becomes the selection
  instrument; time priority stops mattering ("if you don't pay you don't trade").
* **The venue's revenue-maximising fee leaves every maker at or below zero.** Revenue peaks at
  f = 0.5: 0.41 per period (tape-only incumbents) and 0.62 (identity-aware) — two to four times
  the makers' total rent at f = 0. At that fee the entrant and the incumbents are at −0.04 to
  +0.05 and the fringe at −0.14 to −0.20. The excess over the dissipated rent is the fringe's
  loss: the fee is funded by the identity-blind.
* **A cheap fee is a gift to the best-informed.** At f = 0.1 the identity entrant among tape-only
  incumbents doubles its profit (0.10 → 0.20) by paying in 88% of calm and 8% of toxic periods and
  taking 34% of fills; identity-aware incumbents facing a non-paying entrant double theirs
  (0.11 → 0.21). Information converts into position at the price of the fee, and the fee is cheap
  relative to the calm-state fill value (~0.86).
* **Incidence.** The fringe's loss triples (−0.05 → −0.15 under tape-only incumbents; −0.09 →
  −0.21 under identity-aware) at any fee that anyone pays, because it is pushed behind every payer
  and keeps the fills the selectors declined (its informed share 0.29 → 0.40–0.54). Taker access
  is unchanged (the fringe is always there). The learners notice the price: weight on the paying
  expert falls from ~1.0 at f ≤ 0.25 to 0.23–0.26 at f = 0.75.
* **For the frontier.** The priority fee is not a defence of the incumbents' rent: at the venue's
  optimum it dissipates all of it. It is a tax on selection whose burden is passed to whoever
  cannot select, and whose only beneficiaries besides the venue are the best-informed at fees
  below the venue's optimum. Mechanism-design reading: a pay-for-priority queue converts a
  time-priority rationing of the selection rent into a price, and the price is set by the
  uninformed makers' losses.

---

# Two book sessions (`results/live_2026_10/`, Oct 11–12 and Oct 19–20 2026: 15.0 h of top-of-book in two sessions eight days apart, 327 h of trades)

Every number from the six-hour run holds, and the maker persistence is now across nights.

| | BTC | ETH | HYPE | SOL |
|---|---|---|---|---|
| quoted spread, median (bps) | 0.12 | 0.42 | 0.13 | 0.96 |
| effective spread, fills at the touch | 0.13 | 0.43 | 0.18 | 0.95 |
| 10-s markout: all / at the touch / walked (bps) | −0.45 / −0.37 / −0.64 | −0.61 / −0.46 / −1.26 | −0.20 / −0.41 / +0.04 | −0.08 / −0.26 / +0.70 |
| identity given the anonymous signal, next fill (bps, 327 h) | −0.64 | −1.01 | −0.59 | −0.59 |
| makers with ≥ 100 marked fills | 180 | 107 | 135 | 67 |
| top-10 MM / other MM / non-MM makers, 10 s (bps) | −0.10 / −0.36 / −0.88 | +0.19 / −0.53 / −1.25 | +0.11 / −0.12 / −0.71 | +0.52 / −0.17 / −0.41 |
| at the touch: top-10 / everyone else | −0.02 / −0.41 | +0.24 / −0.70 | +0.09 / −0.56 | +0.52 / −0.56 |
| significantly positive makers; their fill share | 8; 8% | 9; 20% | 9; 13% | 15; 30% |
| rank corr of per-maker markout, median split (n) | 0.76 (128) | 0.58 (66) | 0.42 (77) | 0.59 (45) |
| **rank corr, night vs night, eight days apart (n)** | **0.75 (108)** | **0.50 (65)** | **0.43 (68)** | **0.56 (45)** |

All p < 1e-3. Marked fills 112k / 55k / 106k / 29k.

* **Persistence is across days, not a calm spell.** Makers with ≥ 50 marked fills on each of two
  nights eight days apart keep their ranking: Spearman 0.43–0.75. The top-vs-rest gap at the
  touch is unchanged (0.4–1.1 bp), the number of significantly positive makers roughly doubled
  with the sample (8–15 per coin, 8–30% of fills), and the touch-fill loss is unchanged
  (−0.26 to −0.46 bp). This is the version to quote.
* The identity split at 327 h: −0.59 to −1.01 bp given the anonymous signal — stable since 125 h.
* `mk_session_*` in `calibrate.py` cuts at the largest gap > 1 h between marked fills.

---

# Quoter-side defences (preliminary: 6 cfl_exp2 cartels at ᾱ ∈ {0.3, 0.5}, 150 episodes, hand-crafted adversaries; `slurm/defences.sh` is the full grid with the learned adversary)

`experiments/defences.py`. Frozen cartels wrapped in a defence, evaluated on the heterogeneous
market (κ_α = 1, ρ_z = 0.9, L = 5) alone and against competitive / undercut / markout / anon / id /
oracle entrants. Rent units; "worst" = minimum over the profit-maximising adversaries.

| defence | Π_nominal | worst J under attack | worst 5% CVaR | ΔJ vs none: anon / id / oracle / markout / undercut |
|---|---|---|---|---|
| none | 0.98 | 0.13 | −0.59 | — |
| random (ε = 0.1 / 0.3, ±2 ticks) | 0.96 / 0.97 | 0.13 / 0.14 | −0.59 / −0.72 | ≈0 / +.02 / ≈0 / **−.06 to −.09** / **−.09 to −.10** |
| hold (requote every 2 / 5 periods) | 1.00 / 1.00 | **0.03 / −0.04** | −0.60 / −0.73 | −.12 / −.09 / −.10 / **−.33** / **−.33** (m = 2); worse at m = 5 |
| floor on the tape's forecast (anon) | 0.99 | 0.16 | −0.51 | +.06 / +.04 / +.05 / ≈0 / −.01 |
| floor on the identity forecast (id) | **1.02** | **0.18** | **−0.51** | +.06 / **+.07** / +.06 / +.01 / −.01 |

Quoted spread and fill rate are unchanged by every defence (no withdrawal).

* **Speed is necessary.** A cartel that requotes every other period loses essentially all its rent
  to any entrant (κ_A ≈ 1.05–1.15): the only response to an undercut is to undercut back, and a
  slow responder sits with the residual flow. The opposite of the Brown–MacKay coercion logic,
  because the entrant here takes flow rather than disciplining a rival.
* **Noise is a cost without a benefit** against these adversaries: randomising the quote lowers
  nominal profit and loses 0.06–0.10 rent units to the undercutter and the markout entrant (a
  randomly wide quote hands them the flow; a randomly tight one undercuts the cartel partner).
  Whether it blunts an adversary that *learns* the reaction function is the point of the learned
  adversary in the full grid.
* **Identity-aware control is the one lever that improves both axes**, modestly: nominal profit
  0.98 → 1.02 (the floor avoids toxic fills when alone) and the worst-case profit under attack
  0.13 → 0.18, i.e. +0.05 rent units, with the worst-case CVaR −0.59 → −0.51. Most of it is the
  floor itself (the tape's forecast gives 0.16); identity adds 0.02–0.03. It does nothing against
  the undercutter: a floor cannot stop someone quoting inside you.
* **The makers' levers barely move the frontier.** The best of them leaves the entrant with
  ~80% of the rent it took before. The defenders' problem is not that the adversary reads their
  reaction function; it is that the adversary undercuts, and the only answer to an undercut
  destroys the rent. The rent is not defensible by the makers; the venue's levers (tick,
  anonymity, fee) decide its incidence. (Hand-crafted adversaries; the learned one is pending.)
* Learned adversary, one cartel, 600 training episodes (smoke test): J = 0.20 vs 0.48 under the
  undercutter — a Q-learning entrant already hurts the cartel more than the best hand-crafted rule.

## Full grid (`defences.csv`; SuperCloud job 5978770; 20 cfl_exp2 cartels, 500 episodes, learned adversary at 12k training episodes; `fig_defence_frontier{,_cvar}.*`)

Supersedes the preliminary table. Worst case = minimum over the six profit-maximising adversaries
per cartel, then mean over ten cartels (95% CI).

| ᾱ | defence | Π_nominal | worst-case J | worst-case 5% CVaR | ΔJ vs none: anon / id / oracle / markout / undercut / learned |
|---|---|---|---|---|---|
| 0.3 | none | .96 ± .03 | .18 ± .04 | −.31 | — |
| 0.3 | random ε = .1 / .3 | .94 / .94 | .22 / .23 | −.31 / −.31 | ≈0 everywhere; learned +.04 / +.03 |
| 0.3 | hold 2 / 5 | .96 / .95 | **.07 / .02** | −.34 / −.35 | −.14 to −.22 / −.20 to −.29; learned +.03 / +.05 |
| 0.3 | floor anon / id | .97 / .97 | .22 / .22 | −.27 / −.26 | +.01 to +.02 / +.01 to +.03; learned +.01 / −.02 |
| 0.5 | none | 1.01 ± .06 | .08 ± .03 | −1.07 | — |
| 0.5 | random ε = .1 / .3 | .97 / .95 | .07 / .08 | −.98 / −.99 | ≈0; markout −.12 / −.14; learned +.01 |
| 0.5 | hold 2 / 5 | 1.01 / .99 | **−.01 / −.09** | −1.00 / −1.11 | −.08 to −.17 / −.14 to −.37; learned +.01 / −.13 |
| 0.5 | floor anon / id | 1.02 / **1.02** | .16 / **.19** | −.83 / **−.75** | +.03 to +.10 / **+.09 to +.12**; learned +.06 / +.09 |

Learned adversary (tabular Q, 12k episodes, the incumbents' own algorithm and budget): earns
0.28–0.42 rent units vs 0.67–1.02 for the best hand-crafted rule, and leaves the cartel J = 0.47–0.59
(κ_A ≈ 0.5–0.6) vs 0.02–0.26 under the rules. Quoted spread and fill rate are unchanged by every
defence; every entrant tightens the book (best quote 11–15 ticks under attack vs 17 alone).

### Findings
* **Speed is necessary.** Requoting every other period costs 0.11–0.17 rent units of worst-case
  profit (0.18 → 0.07 at ᾱ = 0.3; 0.08 → −0.01 at 0.5); every fifth period, essentially everything
  (κ_A = 1.0–1.3). The only answer to an undercut is to undercut back; a slow responder keeps the
  residual flow. The reverse of Brown–MacKay's coercion result, because the entrant here takes
  flow rather than disciplining a rival.
* **Noise is a cost.** Randomising the quote lowers nominal profit by 0.02–0.06 and buys nothing
  in the worst case at ᾱ = 0.5; at ᾱ = 0.3 it narrows the cartel-to-cartel dispersion of the worst
  case (+0.04 in the mean, tighter CI) at a nominal cost of 0.02. Against the learned adversary
  — the one it is meant for — it adds 0.03–0.04 at ᾱ = 0.3 and nothing at 0.5.
* **Identity-aware control is the one lever that improves both axes, and only where the toxic
  regime is decision-relevant.** At ᾱ = 0.5 the identity floor raises nominal profit 1.01 → 1.02
  and the worst-case profit under attack 0.08 → 0.19 (CIs ±0.03 / ±0.02), with the worst-case
  CVaR −1.07 → −0.75; the tape's forecast gets about 70% of it (0.16), identity the rest. At
  ᾱ = 0.3 the gain is +0.04 and not distinguishable from zero — against ~11-tick cartels the
  toxic state rarely bites, as in the identity grids. The floor is the same object as the
  identity entrant's rule, used by the defenders.
* **A learned adversary is a weak one.** A Q-learning entrant with the incumbents' own algorithm
  and training budget finds a policy worth 0.3–0.4 rent units where the hand-crafted undercut-
  with-inference rules take 0.7–1.0; it leaves the cartel half its rent. κ_A ≈ 0.7 is the
  exploitability by structured best responses, not an artefact of weak adversaries; generic RL
  with the incumbents' budget does *worse* than the rules, which is a remark about what an entrant
  needs (the undercut and a toxicity estimate, not a value function).
* **The makers' levers barely move the frontier.** The best of them leaves the entrant 75–80% of
  what it took before. The defenders' problem is not that the adversary learns their reaction
  function; it is that the adversary undercuts, and the only response to an undercut destroys the
  rent. The rent is not defensible by the makers. The venue's levers (tick, anonymity, fee) decide
  who holds the toxic flow and who pays for position; they do not defend the rent either. Nothing
  in the game defends it — which is the answer to RQ3.

---

# Held-out adversaries (re-analysis of `defences.csv`)

The defence conclusions were drawn with all adversaries in view. Check: choose each cartel's
defence on {undercut, markout, anon} (maximise the worst-case profit under attack), score it on
{id, oracle, learned}.

| ᾱ | ordering of defences, selection vs held-out set (Spearman) | defence picked (10 cartels) | held-out worst-case J: picked / none / best in hindsight | gain of the pick over none (paired mean, min) |
|---|---|---|---|---|
| 0.3 | 0.75 | random .3 ×4, random .1 ×2, floor anon ×2, floor id ×2 | .23 / .20 / .25 | +.04 (min −.01) |
| 0.5 | 0.96 | floor id ×5, floor anon ×5 | .18 / .09 / .20 | **+.10 (min +.02)** |

The ordering is the same on adversaries the choice never saw; at ᾱ = 0.5 the floor is chosen in
every cartel and keeps 93% of its hindsight gain out of sample, positive in every cartel. At
ᾱ = 0.3 the choice is split among defences that are all within noise of each other, which is the
"nothing helps here" conclusion restated out of sample.

---

# Train at ᾱ, test at ᾱ′ (`exploitability_alpha_shift.csv`; SuperCloud; 60 cfl_exp2 cartels × 3 evaluation toxicities × 4 attackers, 500 episodes)

Frozen cartels trained at ᾱ ∈ {0.1, 0.3, 0.5} (N = 2, 3; ten seeds) evaluated on markets of
toxicity ᾱ′ ∈ {0.1, 0.3, 0.5}; benchmarks and rent units are the evaluation market's.

| trained at ᾱ | C on ᾱ′ = 0.1 / 0.3 / 0.5 | κ_A, profit-maximisers, on 0.1 / 0.3 / 0.5 | attacker profit (rent units) on 0.1 / 0.3 / 0.5 |
|---|---|---|---|
| 0.1 | .82 / .72 / **.23** | .75–.79 / .71–.76 / **2.0–2.5** | .52 / .33–.37 / −.12 to .20 |
| 0.3 | .84 / .86 / .69 | .73–.75 / .68–.69 / .88–.95 | .65 / .55–.57 / .39–.51 |
| 0.5 | .80 / .84 / .80 | .72–.73 / .65–.69 / .78–.84 | .62 / .54–.57 / .45–.57 |

No cartel earns below the competitive benchmark on any market on its own (0 of 180 cells).

* **Robust in one direction.** A cartel trained on more toxic flow quotes wide, and a wide quote is
  still a collusive quote on a calmer market: C stays 0.80–0.86 and κ_A 0.65–0.75 when ᾱ′ ≤ ᾱ.
  Training toxicity does not matter for the rent or its exploitability as long as the market is
  no more toxic than the training one.
* **Degrades gracefully, then becomes over-exploitable, in the other.** As toxicity rises past the
  training level the fixed quote approaches the new break-even: the 0.3-cartel keeps C = 0.69 on
  the 0.5 market, the 0.1-cartel only 0.23 — but never negative. The entrant's undercut then
  pushes a thin-rent cartel *below* the competitive benchmark: κ_A = 2.0–2.5 for the 0.1-cartel on
  the 0.5 market (the attacker destroys more than the rent that was there), 0.9 for the
  0.3-cartel. The attacker earns less too (the undercutter loses on the most mismatched cell): a
  wide quote on a toxic market leaves little for anyone.
* For the paper: κ_A ≈ 0.7 is a property of cartels whose rent is intact, not of the training
  toxicity; the vulnerable case is a cartel caught by a toxicity regime it was not trained for —
  which is the regime-shift of §takers at the scale of the whole training distribution, and one
  more reason the regime filter (identity-aware control) is the lever that matters.

---

# Who is at the touch when the flow turns toxic, and what the earning makers condition on (`stress_*.csv`, `stress_scale.csv`, `stress_fill.csv`; two nights, Oct 8 and Oct 16; Mac)

`analysis/stress.py`.  Top-10 MM wallets fixed by full-window volume as in the maker table.
Fills at the pre-trade touch, marked against the true mid.

## 1. No withdrawal after stress, at any timescale

The anonymity lever (§anonymity) predicts that incumbents who read the regime step out of the
inside after toxic flow and the fringe absorbs it.  Tested as: top-10 share of touch fills in a
block vs the mid's absolute 10 s move per fill in the PREVIOUS block (30 s, 60 s, 300 s blocks;
lag only across contiguous blocks), and at fill level vs the mid's move over the 10 s / 60 s
before each fill.

| | BTC | ETH | SOL | HYPE |
|---|---|---|---|---|
| Spearman, top share vs lagged vol, 30 s / 60 s / 300 s blocks | +.13 / +.13 / +.12 | −.02 / −.01 / −.03 | +.03 / −.00 / −.07 | +.01 / −.02 / +.03 |
| fill-level, vs move over previous 10 s / 60 s | +.04 / +.04 (p<.001) | −.01 / −.05 | −.00 / −.01 | +.03 / −.00 |
| top share, calm / mid / stress tercile of previous 10 s move | .097 / .095 / .123 | .248 / .272 / .247 | .282 / .290 / .275 | .212 / .246 / .240 |

Flat on ETH/SOL/HYPE; on BTC the top makers' share RISES after a move.  The earning makers do
not leave the touch when the tape turns.

## 2. But their selection is real, contemporaneous, and not the tape's

* Within the same block the top share falls as that block's realized toxicity (adverse selection
  of everyone else's fills, excluding the top makers' own) rises: ETH −.09 / −.13 / −.28 at
  30 / 60 / 300 s (p<.001), SOL −.12 / −.15 / −.20, HYPE −.05 / −.06 / −.01, BTC ≈ 0.  They are
  absent from the fills that turn out toxic without having left the touch.
* Their markout edge over the rest holds or widens in stress (previous-10 s move terciles, top
  minus rest, calm → stress): ETH 0.94 → 0.99 bp, SOL 0.95 → 1.31, HYPE 0.55 → 0.73; BTC narrows
  0.56 → 0.26 (BTC is where they lean in).
* Nothing in the public tape's recent move predicts when.  So the `anon` policy of the queue game
  (read the stale tape, step out) is not what the earning makers do.

## 3. Identity selection: typed on night 1, measured on night 2

Every taker with ≥ 30 touch fills on the first night is typed by the mean markout it inflicted
(rank terciles: benign / middle / toxic); every touch fill of the second night is tagged with its
taker's type, or "unseen".  The typing holds up: on night 2 the toxic takers cost the maker
1.0 / 1.7 / 2.3 / 1.2 bp more than the benign ones (BTC/ETH/SOL/HYPE).  Toxic typed takers are
20–27% of all touch fills; benign typed takers 1–4%; unseen 67–77%.

Top-10 share of touch fills from:

| | benign | unseen | toxic | benign / toxic | Spearman(top, past toxicity), p |
|---|---|---|---|---|---|
| BTC | .121 | .133 | .027 | 4.5× | −.21, <.001 |
| ETH | .383 | .308 | .059 | 6.5× | −.33, <.001 |
| SOL | .422 | .388 | .121 | 3.5× | −.17, <.001 |
| HYPE | .301 | .253 | .118 | 2.6× | −.26, <.001 |

Within calm tape (previous-10 s move, bottom tercile) benign / toxic: .158/.027, .350/.049,
.471/.116, .263/.115; within stressed tape: .115/.036, .400/.072, .387/.132, .302/.112.  The gap
is the same inside calm and inside stressed tape — it is the wallet, not the moment.

**Speed or identity.**  A stale quote hit by a fast taker moves the mid within a second.  Among
fills after which the mid did NOT move within 1 s (37–61% of fills), the top makers' share of
toxic takers' fills is still below their share of benign takers' fills: BTC .087 vs .141 (1.6×),
ETH .151 vs .424 (2.8×), SOL .180 vs .478 (2.7×), HYPE .238 vs .300 (1.3×).  On pick-offs the
gap is larger (BTC .006 vs .067, ETH .038 vs .259, SOL .106 vs .273, HYPE .094 vs .302): never
being stale is most of the avoidance, and avoiding the wallet is the rest.  Those quiet toxic
fills still lose 0.3–0.8 bp at 10 s, so the wallet's information plays out over seconds and the
wallet tag is worth having before the tape shows it — the mark-lag mechanism of §identity.

Even when they do fill a toxic taker the top makers lose less than the rest (BTC −0.44 vs −1.35,
ETH −0.28 vs −1.58, SOL −1.36 vs −1.83, HYPE −0.03 vs −1.35).

## For the paper
* RQ1's empirical anchor: the makers who earn the spread take 3–12% of last night's toxic
  wallets' fills against 12–42% of the benign wallets' and 13–39% of unseen wallets', on every
  coin, at every level of the tape, and on fills that were not pick-offs.  Identity-conditioned
  fill selection is operating on the real book.
* The anonymity lever's "withdraw in stress" is NOT the mechanism: the earning makers do not
  leave the touch after a move; they are absent from the wallet.  §anonymity's result stands as
  a statement about incumbents whose only regime signal is the tape (and about what anonymity
  would do to a book of such makers); the data say the top makers have a better one.
* Figures: `fig_identity_selection.png`, `fig_stress_withdrawal.png`.
* Caveats: two nights; types from ≥ 30 fills so the typed set is the high-volume takers;
  "quiet" = mid exactly unmoved at 1 s on the bbo feed.  A cross-venue price lead (Binance) is
  not excluded as the top makers' signal on pick-offs; it cannot produce the quiet-fill gap.

---

# The exact best response, and the double oracle (`best_response.csv`, `best_response_relearn.csv`; SuperCloud; 60 cfl_exp2 cartels)

`experiments/best_response.py`.  A frozen cartel is a deterministic reaction function (greedy Q
over (own last h, min rival last h); no inventory state; i.i.d. periods), so the entrant's
problem is a finite-horizon MDP with known deterministic transitions and closed-form expected
rewards (`profit_curve`); backward induction over the 100-period episode gives the exact best
response for any objective λ·own − (1−λ)·incumbents.  DP-predicted values match the simulated
ones (mean |ΔL| 0.00, |Δe| 0.02; no ties in any cartel's argmax).

## Frontier (500 episodes; mean over 10 seeds; L = cartel's loss, e = attacker's profit, rent units)

| ᾱ, N | C | undercut L / e | markout L / e | **exact BR (λ=1)** L / e / κ_A | damage-max (λ=0) L / e | incumbents' quote alone → under BR → under damage-max |
|---|---|---|---|---|---|---|
| 0.1, 2 | .84 | .69 / .52 | .67 / .50 | **1.06 / 1.20 / 1.14** | 1.06 / .26 | 10.8 → 15.9 → 4.8 |
| 0.1, 3 | .80 | .61 / .52 | .59 / .55 | **1.02 / 1.19 / 1.18** | 1.03 / .27 | 10.3 → 15.6 → 6.9 |
| 0.3, 2 | .87 | .59 / .49 | .59 / .52 | **.95 / 1.08 / 1.14** | 1.33 / .15 | 15.5 → 18.0 → 5.4 |
| 0.3, 3 | .84 | .59 / .62 | .59 / .62 | **.93 / 1.08 / 1.16** | 1.36 / .02 | 14.6 → 18.8 → 4.8 |
| 0.5, 2 | .83 | .71 / .46 | .64 / .55 | **1.00 / 1.17 / 1.18** | 2.82 / −.04 | 17.8 → 21.9 → 4.3 |
| 0.5, 3 | .77 | .59 / .43 | .58 / .54 | **.93 / 1.17 / 1.29** | 2.68 / .02 | 17.0 → 20.5 → 4.9 |

* **The exact best response takes the whole rent and more.**  L ≥ 0.9 C in 60 of 60 cartels,
  L > C (the incumbents end below the competitive benchmark) in 60 of 60; the entrant earns
  1.08–1.20 rent units — i.e. ≈ Π^M, the monopoly profit — and κ_A = 1.14–1.29.  The
  hand-crafted rules reach 55–70% of it.  κ_A ≈ 0.7 was a property of the undercut rules, not
  of the cartels.
* **How: it steers.**  The best response quotes one tick inside what the incumbents are *about
  to* quote (under_share 1.00 — it undercuts the reaction, not the last print) and holds the
  cartel at Q-cells it never visits on its own path (off-path share 0.86–0.99), where the learned
  "reaction" sends them wider: 10.8 → 15.9 at ᾱ=0.1, 17.8 → 21.9 at ᾱ=0.5.  The cartel is made
  to quote near the monopoly width while the entrant sits inside it.
* **No profit–damage trade-off until the attacker gives up most of its profit.**  λ = 1, 0.75,
  0.5 give the same policy at ᾱ ≤ 0.3 (the profit-maximiser already destroys everything).  The
  damage-maximiser is a different attack: it steers the cartel *tight* — to 4–7 ticks, inside
  break-even (h^C = 8–13) — by quoting *above* it (above_share 0.6–0.86), takes no fills, earns
  ≈ 0, and lets the incumbents absorb the toxic flow at losing quotes: L = 1.3 at ᾱ=0.3 and
  2.7–2.8 at ᾱ=0.5, i.e. the incumbents lose two to three times their rent per period.  A frozen
  learned reaction function can be driven anywhere its off-path rows point.

## Double oracle (λ=1; the cartel re-learns 3000 episodes against the fixed best response, eps0 0.3, decay 1e-5; the entrant re-solves; 3 rounds)

| round | cartel's rent alone C (mean; round 0 = .83) | stale BR still takes L / C | stale BR earns | fresh BR takes ≥ 0.9 C | cartels with C < 0.5 |
|---|---|---|---|---|---|
| 1 | .79 | .84 | .49 | 60 / 60 | 1 |
| 2 | .71 | .91 | .55 | 60 / 60 | 2 |
| 3 | .69 | .93 | .58 | 60 / 60 | 8 |

* Re-learning buys back about half of what the stale attacker took (its profit falls from ≈1.2
  to ≈0.5 — the level of the hand-crafted undercutter), and no more: the re-solved best
  response takes ≥ 90% of the remaining rent in every cartel in every round.
* While re-learning, the incumbents earn 0.08–0.27 rent units at ᾱ ≤ 0.3 and ≈ −0.05 (below
  competitive) at ᾱ = 0.5 (last fifth of the re-learning run).
* The cartel's own rent erodes round by round (0.83 → 0.79 → 0.71 → 0.69; at ᾱ = 0.5, N = 3,
  0.77 → 0.60 by round 3 with 13% of cartels below C = 0.5).  The trajectory is the collapse
  branch, slowly: the cartel cannot re-form around an attacker that re-solves.

## For the paper
* RQ2's headline changes: the rent is fully exploitable, not two-thirds exploitable; the open
  question about attackers that target the reaction function is answered — they take
  everything, and the damage-maximiser can impose losses of 2–3× the rent at no cost.
* The exact best response needs the cartel's reaction function (the F^oracle of the policy); it
  is the upper bound L_BR by definition.  The reaction function is identifiable from public
  quotes (every incumbent's response to (own, riv) is on the tape, and the entrant sets riv with
  its own quote), so a model-based learner with F^anon can approach it — the realistic attacker
  to build next.  The learned tabular Q entrant of `defences.py` was weaker than the rules
  because it was solving this MDP by sampling; the DP solves it exactly.
* Figure `fig_best_response.png`.

---

# Strategizing against a no-regret market maker (`no_regret.csv`, `fig_no_regret.png`; this container, 2 cores; ᾱ = 0.3, η = 0.05, 3 selection + 5 evaluation seeds)

Workshop 1 feedback: present the background as Glosten–Milgrom + Q-learners (Colliard, Foucault,
Lovo) and Deng, Schneider and Sivan (2019), "Strategizing against no-regret learners", and
nothing else.  The theorem: an optimizer playing against a no-regret learner can always guarantee
its Stackelberg value; against a no-swap-regret learner it can get no more; against a
mean-based learner (MWU, EXP3, follow-the-leader-like) with more than two actions it can get
strictly more, and the optimal play is a control problem.  Two additions make the mapping exact.

## The Stackelberg value is on the attackers figure (`sim.benchmarks.stackelberg_value`)

The entrant commits to a quote; the incumbents play a symmetric pure Nash equilibrium of the
resulting stage game (ties split).  On the tick grid the equilibrium set can hold more than one
quote, so both tie-breaks are computed; "strong" (best for the entrant) is drawn.  The entrant
can only share a quote the incumbents have no incentive to undercut, so the value sits at
h^C to h^C + 1: 0.10 / 0.15 / 0.21 of the rent against a single incumbent and 0.07 / 0.10 / 0.06
against two, at ᾱ = 0.1 / 0.3 / 0.5.  Everything an attacker earns above that line is
exploitation of the learner's dynamics, not of the game.  On `fig_attackers` the line sits at
0.06 against the 60 frozen Q-cartels: the undercut rule is at 0.51, the exact best response at 1.15.

## A no-regret cartel without memory earns no rent (`sim.agents.Exp3QuoteAgent`, `experiments/no_regret.py`)

Stateless EXP3 over h ∈ 1..24, per-period bandit feedback, never frozen.
* One learner alone finds the monopoly quote: C = 0.96–0.97 (mean best quote 12–19 ticks, h^M = 14).
* Two learners compete: C = 0.01–0.03, mean best quote 4.5–5 ticks (h^C = 4), for η = 0.02–0.2.

So the Colliard–Foucault–Lovo rent needs the learner's **memory** (state = last quotes): without
it there is no threat, and no-regret dynamics land at the coarse correlated equilibrium of the
stage game, which here is Bertrand.  That memory is also what makes the cartel a reaction
function, which is what the exact best response solves.  Rent and exploitability have one cause.

## A single live no-regret learner is exploited far beyond its Stackelberg value

Against one EXP3 monopolist (C = 0.97 alone), still learning throughout 2000 attack episodes,
evaluation seeds, rent units (whole window; second half in brackets):

| attacker | entrant earns | learner loses |
|---|---|---|
| undercut rule (one tick inside its last quote) | 0.72 (0.60) | 1.03 (1.03) |
| same, floored at h^C | 0.72 (0.61) | 1.03 (1.03) |
| lure-then-undercut cycle (K for 20 periods, then 10 ticks for 400; chosen on other seeds) | 0.29 (0.06) | 0.52 (0.31) |
| Q-learning entrant, online | 0.32 (0.28) | 0.58 (0.50) |
| Stackelberg value | 0.15 | |

* The trivial undercut rule takes five times the Stackelberg value, and the learner gives up its
  whole rent and more: a mean-based learner answers an undercut by slowly moving mass toward
  tighter quotes, the rule follows one tick inside, and the learner is always a step behind.
  This is the "strictly more than Stackelberg against mean-based learners" branch of the
  theorem, observed.
* The explicit lure/undercut cycle, our hand approximation to the control problem, is worse
  than the plain rule here: against this learner there is no need to lure, it never tightens
  for good.  The control problem is worth solving properly (the learner's state is its weight
  vector, which an attacker can estimate from its play) but the headline does not need it.
* Against two EXP3 learners there is nothing to take (C ≈ 0): the naive undercut rule goes
  inside h^C and loses 0.23; floored at h^C it earns 0.04.

## For the paper
* Background = two things: GM + Q-learners (the rent exists and is a reaction function) and
  Deng–Schneider–Sivan (what an optimizer can take from a learner, with Stackelberg as the
  benchmark).  Every result is placed against the Stackelberg line.
* The two findings here: (i) the rent needs memory, and memory is the attack surface;
  (ii) a live mean-based learner is exploited far beyond Stackelberg by a one-line rule.
* Not yet run: ᾱ = 0.1, 0.5 and η = 0.02, 0.2 (`slurm/no_regret.sh`, ~1 h on 48 cores); an
  attacker that solves the control problem against the estimated weight vector.
