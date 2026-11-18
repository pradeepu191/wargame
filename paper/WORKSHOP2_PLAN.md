# Workshop 2 and the preprint: feedback, status, next steps (Nov 15 2026)

## READ THIS FIRST (Nov 18): what this branch is, and what you need to do

### Which branch is this?

You are reading `workshop2-lecture15` (pull request #2). There are two branches:

| | `workshop2-next-steps` (PR #1) | `workshop2-lecture15` (PR #2, this one) |
|---|---|---|
| History | the base | every commit of PR #1 **plus four more**; no file deleted |
| Code, experiments, results, figures, tests | everything up to Nov 16 | the same, plus the planning view, the ordinary trader, 7 more tests (79) |
| arXiv draft | 17 pages | the same text plus a planning paragraph, "The ordinary trader" section and an ethics paragraph (20 pages) |
| Workshop 2 report and slides | the full version: seven results (public attacker, re-learning cartel, double oracle, queue steering, request allowance, frontier, data) | **rewritten shorter** in the frame of Lecture 15: myopic vs optimal attacker, public attacker, re-learning cartel, identity as state, the ordinary trader; the queue game, allowance and frontier are on one backup slide |
| Demo | nine market scenarios | the same, plus the myopic entrant, five "betting savings" runs, and the fix for screens with display scaling |

So PR #2 = PR #1 + more, except that the report and slides of PR #1 were replaced by the shorter
version on purpose (the supervisor's Lecture 15 showed the level he wants). The fuller wording of
those two files is still in PR #1 and in git history; all of the underlying work is still here.

**Suggested:** present and submit Workshop 2 from this branch; merge PR #2 into `main` and close PR #1
(merging #1 first and then #2 also works, with no conflicts).

### Your to-do list, in order

1. **Read the drafts** (all in `paper/`): `deliverable3.pdf` (report, 3 pages), `slides_workshop2.pdf`
   (13 slides), `arxiv_draft.pdf` (20 pages). Check that you can explain every slide; every term is
   defined in `paper/glossary.tex` (section G is the Lecture 15 vocabulary). Tell Claude what to change.
2. **Rehearse the live demo.** Double-click `demo/index.html` (no server or internet needed).
   Space = play/pause, arrow keys = step, the slider sets the speed. Order used on the slides:
   1. "The cartel alone" at about 5 periods/s;
   2. Scenario "Myopic entrant", compare with "Exact best response" (step through ~10 periods: the
      myopic one drives the cartel down, the optimal one keeps it wide);
   3. "Attacker with public information only";
   4. "Betting savings: never updates" next to "myopic Bayesian" (choosing the first fills in the
      second), then "Real edge: myopic Bayesian" next to "patient Bayesian" (the trap).
   Backup without a browser: `python analysis/trace_video.py --all` writes GIFs to `results/demo/`.
3. **Run the full-size experiments on SuperCloud** (Claude cannot log in to the cluster). From a terminal:
   ```bash
   ssh <your-username>@txe1-login.mit.edu
   cd ~/wargame        # first time only: git clone https://github.com/pradeepu191/wargame.git ~/wargame && cd ~/wargame && bash slurm/setup.sh
   git fetch && git checkout workshop2-lecture15 && git pull
   ls results | grep -c cfl_exp2_    # the trained cartels; if 0, first: sbatch slurm/sweep.sh experiments/configs/replicate_cfl_exp.yaml --set market.alpha=0.1,0.3,0.5 --set market.n_mm=2,3 --seeds 0-9
   sbatch slurm/workshop2.sh         # five experiments, up to 12 hours on one 48-core node
   squeue -u $USER                   # is it still running?
   ```
   When it has finished:
   ```bash
   python analysis/workshop2_figs.py
   git add results/*.csv results/fig_*.png results/fig_*.pdf && git commit -m "Workshop 2 grids at full size" && git push
   ```
   Then tell Claude "refresh the numbers": it updates the report, slides and paper from the new CSVs.
   Step 5 of that script also replaces the strong-attacker defence run that died on the laptop.
4. **December order-level archive** (whoever has the raw data; it is not on the laptop). Download
   instructions are in `data/README.md` section B. No analysis script exists yet for the order-status
   stream; the question for it is in the report (plan item 2): do makers that avoid a wallet return to it
   when it turns benign? Claude can write the script, but it can only be tested where the data is.
5. **Decide, with Deepu, before the repository goes public:** a licence (`LICENSE`), a citation file
   (`CITATION.cff`) and a Zenodo archive with a DOI at submission.
6. **Send Claude the Workshop 2 report template** if the course published one; the draft copies the
   Workshop 1 layout.
7. Optional: install the `microtype` package in MiKTeX (MiKTeX Console > Packages), so the older
   Workshop 1 report builds on this laptop.

### Running things on the laptop

Nothing is running now. Claude Code sessions and anything they start stop when the laptop sleeps or
shuts down: keep it on, plugged in and awake, or use the cluster. Useful commands (PowerShell, in the
project folder, with the project's Python `.\.venv\Scripts\python`):

| What | Command |
|---|---|
| all tests (about 2 min) | `.\.venv\Scripts\python -m pytest -q` |
| every figure from the committed CSVs | `make figures` (or run the scripts in `analysis/` one by one) |
| re-record the demo | `.\.venv\Scripts\python experiments\trace.py --all` |
| myopic vs optimal attacker table (needs the locally trained cartels) | `.\.venv\Scripts\python experiments\planning_view.py --simulate 100 --jobs 6` |
| the ordinary trader (part A needs the local cartels) | `.\.venv\Scripts\python experiments\retail_trader.py` then `analysis\retail_fig.py` |
| a PDF (run twice, inside `paper\`) | `pdflatex deliverable3.tex` (slow while experiments use the CPU) |

### Where to read more

* What each experiment found, with every number: `results/REPLICATION_NOTES.md` (newest sections at the end).
* How Lecture 15 maps onto our code: section 00 below.
* The status of each plan item: section 0 below.

## 00. Realignment with Lecture 15 (Nov 17, branch `workshop2-lecture15`)

The supervisor's Lecture 15 ("Planning with a Model", Nov 5) shows the form he wants: one small
model stated as (states, actions, transitions, rewards, horizon), solved by the Bellman equation,
read through a worked table, and a myopic policy against an optimal one. The Workshop 2 report and
talk on this branch are narrowed to that frame. Nothing is deleted: the wider results stay on
`workshop2-next-steps` and in `results/REPLICATION_NOTES.md`.

| Lecture 15 | Our model | Where |
|---|---|---|
| Finite MDP (X, A, P, c, δ); turn limits | The entrant against a frozen cartel: state = last quotes, action = own quote, P = the cartel's reaction (deterministic), 100-period episode | `experiments/best_response.py` (`CartelMDP`) |
| An opponent that does not respond is part of P (missile defence) | A frozen cartel is part of P | report §a |
| What must the state contain? | The learners condition on own last quote and best rival quote, so the last quote vector is valid | report §a |
| Myopic vs optimal ("a cheaper route can give a costlier policy") | Myopic entrant 0.91–0.95 rent units, optimal 1.08–1.21, undercut rule 0.46–0.53; δ = 0.5 is already enough | `experiments/planning_view.py`, report b1 |
| Value iteration table | First four periods, myopic vs optimal, on one cartel | slides, "A quote that earns more today…" |
| Model-based vs model-free | Public attacker = model-based with a model learned from the book; the cartel = model-free Q-learners | report b2 |
| An opponent who responds is a player: a game | A cartel that keeps learning; re-learning and alternating best responses | report b3 |
| Belief state (n, h); selective feedback; trap (Rothschild) | Per-wallet record of fills and toxic fills; avoiding a wallet stops the feedback; is avoidance a trap? | report b4 (open question) |
| More information can hurt in a game (Akerlof) | Public wallet identity | report b4 (open question) |

**The ordinary trader (Nov 17).** The paper now also takes the side of the participant with the least
information (`experiments/retail_trader.py`, `analysis/retail_fig.py`, report b5, arXiv §"The ordinary
trader", an ethics paragraph in both). Three parts, each built on what the model already had: (A) the
price an ordinary customer pays under each market structure, from the planning paths and the
closed-form taker surplus; (B) a speculator betting savings without information as the lecture's belief
MDP (Beta prior on her hit rate, quitting line, ruin), solved by backward induction and checked against
the simulator; (C) a retail limit order as the identity-blind fringe of the queue game. The attack that
is best for the attacker leaves the customer 36–39% of her Nash-quote gains, no more than the cartel;
the speculator who never updates is ruined in 74% of years, the Bayesian quitting rule in 0.3–3.8%.

Moved out of the talk and report (kept in the repository, marked preliminary): the queue-game
steering and request-allowance experiments, the one frontier for all levers, and the double oracle
as a headline (it is one sentence in b3). Claims are worded more carefully ("in our runs", "we do not
claim an equilibrium"), and new planning words are in `paper/glossary.tex` §G.

## 0. Status on Nov 18 (what was done after this plan was written)

| Step of §3 | Status | Where |
|---|---|---|
| 1 Visual demonstration | **Done** | `experiments/trace.py`, `demo/index.html`, `analysis/trace_video.py`, `tests/test_trace.py` |
| 2 Public attacker vs re-learning cartels | **Done at laptop size** (15 two-maker cartels; full size is on the cluster script) | `experiments/public_attacker_relearn.py`, `fig_relearn_attack` |
| 3 Scripted grids on the cluster | **Not submitted** (no cluster login on the laptop). The queue identity and fee grids were already in the repo at 5 seeds. | `sbatch slurm/workshop2.sh` |
| 4 Double oracle to convergence | **Done at laptop size** (12 two-maker cartels, two modes, six rounds; no convergence to a defence) | `experiments/psro.py`, `fig_psro` |
| 5 Queue-mode attacks | **Preliminary** (3 seeds): steering is a null; the request allowance removes the selection rent | `experiments/queue_attacks.py`, `fig_queue_steer`, `fig_queue_budget` |
| 6 Data: full recording, December archive | Full-recording analysis was already in the repo (Nov 2). **December archive not started**: raw data is not on the laptop. | `slurm/data_pipeline.sh` |
| 7 One frontier figure | **Draft** from existing results; strong attackers (exact best response, public attacker) coded into `experiments/defences.py`. The laptop run died; it is step 5 of the cluster script | `analysis/frontier_fig.py`, `fig_frontier` |
| 8 Workshop 2 report and talk | **Drafts**, final laptop numbers, rewritten in the frame of Lecture 15 (section 00) | `paper/deliverable3.tex`, `paper/slides_workshop2.tex` |
| 9 Reproducibility pass, arXiv draft | **Draft** (20 pages, with the ordinary trader and an ethics paragraph); `make figures` added. Licence, citation file and Zenodo archive need the authors' decision. | `paper/arxiv_draft.tex` |

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
