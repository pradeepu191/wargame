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

12 cells × 5 seeds, corrected encoding, exponential edge (h^C = 4, h^M = 14). Files:
`robust_exp2_summary.csv`, `robust_exp2_runs.csv`. Config: `experiments/configs/robustness_exp.yaml`.
Reproduce: see config header; `python analysis/robustness.py`.

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
Undercut and competitive passes pending.

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
