# Workshop 2 and the preprint: feedback, status, next steps (Nov 15 2026)

## 0. Status on Nov 16 (what was done after this plan was written)

| Step of §3 | Status | Where |
|---|---|---|
| 1 Visual demonstration | **Done** | `experiments/trace.py`, `demo/index.html`, `analysis/trace_video.py`, `tests/test_trace.py` |
| 2 Public attacker vs re-learning cartels | **Interim** (6 of 15 two-maker cartels; laptop run in progress) | `experiments/public_attacker_relearn.py`, `fig_relearn_attack` |
| 3 Scripted grids on the cluster | **Not submitted** (no cluster login on the laptop). The queue identity and fee grids were already in the repo at 5 seeds. | `sbatch slurm/workshop2.sh` |
| 4 Double oracle to convergence | **Interim** (10 of 24 runs, six rounds) | `experiments/psro.py`, `fig_psro` |
| 5 Queue-mode attacks | **Preliminary** (3 seeds): steering is a null; the request allowance removes the selection rent | `experiments/queue_attacks.py`, `fig_queue_steer`, `fig_queue_budget` |
| 6 Data: full recording, December archive | Full-recording analysis was already in the repo (Nov 2). **December archive not started**: raw data is not on the laptop. | `slurm/data_pipeline.sh` |
| 7 One frontier figure | **Draft** from existing results; strong attackers (exact best response, public attacker) coded into `experiments/defences.py`, run pending | `analysis/frontier_fig.py`, `fig_frontier` |
| 8 Workshop 2 report and talk | **Drafts** with interim numbers | `paper/deliverable3.tex`, `paper/slides_workshop2.tex` |
| 9 Reproducibility pass, arXiv draft | **Draft** (17 pages); `make figures` added. Licence, citation file and Zenodo archive need the authors' decision. | `paper/arxiv_draft.tex` |

Findings are logged in `results/REPLICATION_NOTES.md` (two sections dated Nov 15). Two things in
this plan turned out differently: "steering memory learners in the queue" (step 5a) is not
well-posed, because tabular Q-learners cannot learn to queue, so the steering experiment targets
the no-regret incumbents; and "identification with a forgetting window" (step 2) was tested and
makes the attacker worse.

Not yet covered, and promised on the Workshop 1 slides: the customer-outcome comparison (taker
surplus and fringe profit under each venue rule, with loss-making participants allowed to leave)
and delayed wallet disclosure as a rule. The taker-surplus measure exists
(`sim.benchmarks.taker_surplus_curve`) and is recorded by the defence grid.

## 1. Feedback on Deliverable 2 (Workshop 1 report and prototype)

> 60/60. Very good report. I appreciate the definition of the technical jargon. You have done a
> good amount of work already. The preliminary results are interesting! I don't have any
> specific comments: Your plan for Workshop 2 makes sense!

There are no requested changes, so the feedback becomes three rules for the next documents:

| Feedback | How we act on it |
|---|---|
| The jargon definitions worked | Keep the `glossary.tex` discipline in the Workshop 2 report, the slides, the demo and the preprint: every finance or game-theory word is defined before its first use. New terms (e.g. *probing budget*, *coverage*, *Stackelberg value*, *request budget*) go into `glossary.tex` before they appear anywhere else. The demo's on-screen labels use the same words (§4). |
| The Workshop 2 plan makes sense | Keep the plan in `deliverable2.tex` §f without changing its scope. §2 below tracks it item by item. No new research questions before the preprint. |
| The preliminary results are interesting | Turn them into the paper's results: seed intervals, the N = 3 cartels, held-out attackers. Hold the same evidence bar the outline sets for reviewers (`OUTLINE.md`, "Reviewer expectations"). |

## 2. Status of the Workshop 2 plan (`deliverable2.tex` §f)

| # | Plan item | Status | What is left |
|---|---|---|---|
| 1 | Attacker with public information only; gap to the exact best response against the probing budget | **Done** (60 cartels, N = 2 and 3, five budgets; `public_attacker.csv`). It reaches the oracle at 1000 probing periods and the gap is closed for frozen cartels. | Extend it to cartels that keep learning: identification with a forgetting window against the re-learning cartel. Also stochastic cartels (ties, ε > 0). This is the one open caveat on RQ2. |
| 2 | Both sides learning (PSRO): equilibrium rent and κ_A; are adversarially trained cartels less exploitable or just less collusive? | **Partly done.** Double oracle, 3 rounds, with the exact BR as the attacker: re-learning buys back half against a stale attacker and nothing against one that re-solves. | (a) More rounds until rent and κ_A converge, 10 seeds. (b) A *learned* entrant in the PSRO population (the public-information attacker from 1 is the natural one). (c) The test the plan promised: plot rent C against exploitability across PSRO rounds to see whether the cartel ends less exploitable or only less collusive. |
| 3 | Queue-mode attacks: steering no-regret incumbents; draining the request budget | **Steering half done** (no-regret grid: a single EXP3 learner gives up its whole rent to the one-line undercut rule; two stateless learners never hold a rent). | (a) Steering against learners *with memory* in the queue game (the Q-learners of `queue_q*.csv`), since memory is what holds the rent. (b) Induced self-exhaustion of the request budget, plus the defences named in `REVIEW_RESPONSE.md` (hysteresis, minimum quote lifetime, reserve). Not started. |
| 4 | Resilient designs: venue and maker levers on the frontier (E[profit], −CVaR₀.₀₅ under attack) against all attackers, plus uninformed fill probability in stress | **Pieces exist separately**: quoter defences (20 cartels, 7 adversaries), anonymity and stress liquidity (preliminary, 2 seeds), priority fee (preliminary, 2 seeds). | Run the full grids that are already scripted (`slurm/queue_identity.sh`, `slurm/queue_fee.sh`). Add the exact BR and the public-information attacker to the defence grid, because the frontier must hold against the strongest attacker, not the rule attackers. Add the tick and request-allowance levers. Merge everything into **one** frontier figure with paired intervals. |
| D | Data: always-on recorder; stress episode; December L4 archive | Recorder running since Oct 25. Stress case done: the Oct 29 Binance-hack event study replaces the Oct 2025 cascade. | (a) Re-run the identity-selection gap on the full recording (by coin, hour, regime) with weekly intervals. (b) December L4 order lifecycles: which action carries the selection (cancel before toxic flow, or not posting). This is the empirical counterpart of the queue game. (c) Add events to `data/events.csv` as they happen. |

## 3. Next steps, in priority order

The paper's spine is RQ1 + RQ2 with RQ3 as the required third section (`OUTLINE.md`), so the
order below follows what the preprint needs and finishes the Workshop 2 deliverables on the way.

**Week of Nov 16 (simulator)**
1. **Visual demonstration** (§4). Build it first, because every later experiment can then be
   shown in it, and it is also the figure-generating tool for the slides.
2. **Re-learning cartel vs the public-information attacker** (item 1, the forgetting window).
   This closes RQ2's last caveat.
3. **Launch the scripted full grids** on SuperCloud: `queue_identity.sh`, `queue_fee.sh`, and
   `defences.sh` with the exact BR and the public attacker added. These are mostly cluster time
   and should run in the background while we do 2.

**Week of Nov 23 (simulator + data)**
4. **PSRO to convergence** with a learned entrant (item 2), 10 seeds; the rent vs
   exploitability plot across rounds.
5. **Queue-mode steering of memory learners and self-exhaustion** (item 3), at least as a
   preliminary 2-seed result with the full grid scripted.
6. **Data:** the identity-selection gap on the full recording; first pass over the December L4
   archive.

**Week of Nov 30 (the frontier and Workshop 2)**
7. **One frontier figure** (item 4) against all attackers, including the uninformed fill
   probability in stress.
8. **Workshop 2 report and talk.** Use the Workshop 1 form, which the feedback endorsed:
   definitions first, one figure per result, intervals on every number. The talk includes a
   live run of the demo (§4).

**Dec 1–15 (preprint)**
9. **Reproducibility pass** (§5) and the arXiv draft from `OUTLINE.md` §"The paper". Most
   sections already exist and need to be reorganised.
10. **Internal review** before submission: one read for definitions (every term in §3 of the
    paper), one for claims (every number traced to a CSV and a command).

## 4. The visual demonstration

**Goal.** Show the model running, period by period, at a speed the audience can follow. The
same tool serves three uses: a live demo in the Workshop 2 talk, a recorded clip as a backup and
for the arXiv supplementary material, and an interactive page linked from the paper for
reviewers.

**A note on words.** The paper already uses *tick* for the price grid (the smallest price
increment). To keep the glossary consistent, the demo calls its speed control the **playback
rate (periods per second)**, not "tick rate". A *period* is one taker arrival, the simulator's
unit of time. On Hyperliquid one period corresponds to about 0.6 s of real time (the median gap
between orders), and the demo can show that conversion in a corner.

**Design.** The demo has two parts, so the visual layer never contains model logic and cannot
drift from the paper's numbers.

1. `experiments/trace.py`: runs any existing scenario through the real `GlostenMilgromEnv` /
   `QueueEnv` and agents, with a fixed seed, and writes one JSON trace. Each period records:
   mid, true value V, regime z, every maker's half-spread (or queue position), the arriving
   taker's wallet id and type, who was filled, the fill's mark, and the running P&L per agent.
   The header records config, seed and git hash, like every other run. Scenarios (most reuse
   `experiments/demo.py`):
   * the cartel alone (coordination above Nash: quotes settle near h^M, not h^C);
   * the one-tick undercutter, the markout entrant, and the exact best response, with the
     steering visible by period 2;
   * the public-information attacker: the probing phase, then exploitation, with coverage of
     the reaction table rising;
   * the anonymous vs identity entrant under the regime and mark lag (RQ1): the regime strip
     turns toxic and the identity entrant steps back before the mark arrives;
   * queue mode: the one-tick book, queue positions, and the identity-aware maker avoiding
     toxic wallets.
2. `demo/index.html`: a single static page with no build step and no server. It loads a trace
   and plays it on a canvas:
   * a **quote ladder**: the price grid around the mid, each maker's bid and ask, h^C and h^M
     drawn as reference lines, and the taker's order flashing in on each arrival (coloured by
     informed or uninformed, shown only after the fill, because nobody sees the identity of the
     incoming order);
   * a **regime strip** (calm / toxic) and a **public tape** of recent prints with wallet ids,
     with marks appearing L periods later;
   * **running panels**: cumulative P&L per agent in rent units, the rent index C, and the
     attacker's take;
   * **controls**: play/pause, single step, a playback-rate slider (1 to 200 periods/s),
     scenario selector, and a side-by-side mode that runs two traces under common random
     numbers (for example anonymous vs identity entrant).
   For the paper it can be hosted on GitHub Pages from the public repository.
3. `analysis/trace_video.py`: renders the same trace to MP4/GIF with matplotlib, for the slides
   (in case there is no network or browser in the room) and for the supplementary material.

**Done when:** `python experiments/trace.py --scenario best_response --seed 0` followed by
opening `demo/index.html` plays the run. The demo's end-of-run numbers match `demo.py`'s
printed numbers for the same seed, and a test pins this. Each scenario has a 30–60 s clip in
`results/demo/`.

## 5. Publication checklist (to run alongside the work, not at the end)

* **One command per figure.** Every figure in the paper is produced by a Makefile target from
  committed CSVs. Add `make figures` and `make demo`.
* **Seeds and intervals everywhere.** Use 10 seeds as standard; any 2-seed "preliminary" result
  that appears in the paper is re-run first (queue identity, priority fee).
* **Archive.** Tag a release when the preprint is submitted and archive the code and the summary
  CSVs on Zenodo with a DOI. Raw market data stays out, as now. Cite the Albers et al. archive for
  the data and document the recorder so others can reproduce our own recording.
* **Licence and citation.** Add `LICENSE` and `CITATION.cff` to the repository before it is made
  public.
* **Limitations section** names the stylisations: tabular learners, stylised takers, one venue,
  and the one-tick finding (`OUTLINE.md` §9).
* **Venue.** arXiv by mid-December. Pick the conference target after the November meeting, as
  `OUTLINE.md` already says; the demo page doubles as the artifact for venues with an artifact
  track.
