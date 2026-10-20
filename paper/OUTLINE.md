# Reorganised outline — Workshop 1 (Oct 27) and the paper

Written after Prof. Dahan's replies of Oct 12–13: *define the jargon; stop introducing things here
and there; one model; results first.* On the research questions his position is: RQ1 and RQ2 are
the natural paper (one model tackles both); RQ3 (defences) must be tackled for the course project
regardless, and whether it becomes the paper's primary contribution depends on the results. So the
security-game framing below organises all three, but the paper's spine is RQ1 + RQ2; RQ3 is built
because the project requires it and promoted only if its results earn it. This file is the plan
for the form. `glossary.tex` is the definitions section it refers to.

## The spine (used by every document from here on)

One sentence: **learning market makers settle on payoffs above the competitive level; we ask
what an adversary can take from them, what information the venue gives the adversary, and which
defences — the makers' own and the venue's — hold up, on a market where every print is signed.**

One model, read as a security game:

| role | in the simulator | on Hyperliquid |
|---|---|---|
| designer | tick grid θ, anonymity (which 𝓕 the entrant gets), priority rule, priority fee | five-significant-figure tick, public wallets, FIFO + priority fee |
| defenders | N tabular-Q makers, frozen after training | the classified MM wallets |
| adversary | entrant with 𝓕^anon / 𝓕^id / 𝓕^oracle; undercutter; damage-maximiser; (steering) | — (the thing a trading firm would build) |
| nature | takers with wallet types, hidden regime z, mark lag L | the tape |

Two modes of the same environment: spread free (the Glosten–Milgrom tick grid) and spread
pinned at one tick (the queue game). Never present these as two models.

Numbering, fixed from now on and matching the Oct 6 proposal the professor read:
- **RQ1** value of information to the adversary (anonymity as the venue lever) — paper core
- **RQ2** rent and exploitability (as a measurement: L_BR and κ_A vs C(π)) — paper core
- **RQ3** defences: quoter levers and venue levers, reported as a profit–resilience frontier —
  required for the project; a paper section, the headline only if the results say so. The
  tick-constrained queue game is *inside* RQ3 (the tick lever), not a separate RQ.

Everything else is an appendix.

---

## Workshop 1 (Oct 27): talk + written report

Adapt to the format once announced; the spine does not change. Assume ~12 minutes + a report of
at most 6 pages (the proposal was 3; the professor said less, better organised).

### Talk (10 slides)
1. **Problem in one sentence** + the cast table. (No finance words before this slide.)
2. **Definitions** — the one-slide version of `glossary.tex`: action (half-spread on a tick
   grid), payoff function Π(h;α), Nash point h^C, monopoly point h^M, rent index C(π),
   adversary and L_BR, information sets, regime + mark lag, markout. Say "we will use these
   words from now on".
3. **The model** — one figure: takers (hidden regime → informed type → fill), makers (quotes on
   the grid), entrant (information set), the two modes. Parameters in a corner.
4. **Result 1 — learners coordinate above Nash.** Δ = 0.77–0.86 across α, seeds and learning
   rates; the mechanism test (restraint in 96–100% of contested states); turn-taking, not a
   leader. One figure (the replication panel). *What the proposal promised as (i)–(ii).*
5. **Result 2 — rent is exploitable in proportion, not more.** The RQ2 hypothesis holds only
   trivially: slope of L_BR on C is ≈1; κ_A ≈ 0.7 flat for every attacker. Entrant keeps
   79–96% of monopoly rent vs frozen incumbents; re-learning incumbents re-form the cartel around
   it. One figure (fig_exploitability). *Proposal item (iii).*
6. **Result 3 — what identity is worth, and why.** Not wallet persistence (data: 0.02–0.05).
   Informed flow clusters in time and a toxic wallet's print reveals the regime before the price
   does: id − anon grows with the mark lag L, is zero when the toxic state leaves an undercut
   profitable. One figure (identity levels vs L). *RQ1.*
7. **Result 4 — Hyperliquid.** 125 h of signed trades + 6 h of top-of-book: quoted spread is
   one tick (p90 = median); a fill at the touch pays one tick and loses 0.3–0.6 bp within 10 s
   on all four coins; identity given the anonymous tape −0.5 to −1.0 bp; one order in ten walks
   the book and pays twenty ticks, and on the deep books those are the informed ones. One table.
   *Proposal item (iv).*
8. **Result 5 — the tick lever (queue game).** When the spread cannot move, selection is the
   whole game: identity worth +0.04–0.10 per period at break-even, growing with L; identity-blind
   makers' fills get dirtier. One figure (fig_queue_always). *The first RQ3 result.*
9. **What we build next (to the Nov meeting):** learned defenders in the queue game; the
   priority fee; quoter defences (randomisation, hysteresis, identity-aware control); the frontier
   (E[Π_nominal], −CVaR_0.05(Π_attack)) against three attackers; the top-maker realized-spread
   anchor from data. One schedule line.
10. **Two questions for the room.** (a) Stylised tick-grid model with closed-form benchmarks vs a
    queue-based LOB — we run both as modes of one environment; which should carry the paper?
    (b) Of the venue levers (anonymity, tick, priority fee), which is the security-relevant one?

Rules: every finance word on slides 4–10 was defined on slide 2; one figure per result; numbers
with seed intervals (the 10-seed grids are running); no audit table, no code paths, no
"we fixed a bug" stories.

### Report (≤ 6 pages), same order
1. Problem and contributions (½ page) — four bullets, one per result, in the spine's words.
2. Definitions and notation (1 page) — `glossary.tex`, compressed: the cast table, the fifteen
   terms that appear in §3–§6, the notation table.
3. Model (1 page) — one environment, two modes, parameters, closed-form benchmarks.
4. Results (2½ pages) — R1 coordination, R2 exploitability, R3 information, R4 data, R5 queue.
   One figure or table each, intervals, nothing else.
5. Plan to the November meeting (½ page) — the RQ3 programme and the schedule.
Appendix (not counted): the regime filter; the exploitability regressions; data conventions
(archive vs feed, alignment rules); the audit of what changed since Oct 6; replication notes.

### What moves out of the main text (from `deliverable1_revised.tex`)
- The §2 audit table → appendix. Useful record, wrong opening.
- The derivation of the regime filter and the grid-Bayes estimator → appendix.
- Zenodo conventions, `taker_older`, hash-zero, l2Book throttle, the alignment rules → data
  appendix / `data/README.md`.
- RQ1 "first form" (frozen incumbents, three entrant policies) → two sentences inside R2.
- The mark-lag and regime grids → one figure in R3; tables to appendix.
- Anything that starts "we found that the earlier version…" → replication notes.

---

## The paper (target: arXiv mid-December; ML venue after the November meeting)

Standard structure; everything below the abstract is already drafted somewhere in the repo and
needs reorganising, not rewriting.

0. **Abstract** (150 words): the sentence; four findings; the venue-design conclusion.
1. **Introduction** (1½ pp): algorithmic coordination without communication is documented;
   whether it is *attackable*, by whom, with what information, and what a venue can do about it
   is not. The security-game framing. Hyperliquid as the laboratory: a market where the
   information set is a design choice because prints are signed. Contributions as four bullets.
2. **Related work** (¾ p): algorithmic collusion (Calvano et al.; Colliard, Foucault & Lovo —
   the replication target); adversarial / exploitability in multi-agent RL (PSRO, best-response
   loss, Brown–MacKay, Dou–Goldstein–Ji); market-making RL; microstructure benchmarks
   (Glosten–Milgrom; Cartea–Chang–Penalva on tick size); venue design (anonymity, tick, priority).
3. **Definitions and notation** (1 p): `glossary.tex`, compressed further.
4. **Model** (1½ pp): the environment; the two modes; closed-form benchmarks; the taker model
   (types, regime, mark lag) with the data that motivate each piece in one sentence each.
5. **Rent and its exploitability — RQ2** (1½ pp): replication; mechanism test; L_BR and κ_A
   across 120 populations and three attackers; re-learning incumbents.
6. **Information — RQ1** (1½ pp): nested information sets; the identity grids; why persistence
   fails and regime + lag succeeds; the empirical counterpart (identity given the anonymous
   signal, by coin) and the empirical anchor: the makers who earn the spread take 2.6–6.5× fewer
   of last night's toxic wallets' fills, at every level of the tape and on non-pick-off fills,
   while never withdrawing from the touch after a move (fig_identity_selection). This is the
   figure that closes RQ1 on data; it goes in the RQ1 section, not RQ3.
7. **Defences — RQ3** (1½–2½ pp): the tick lever (one-tick books on data; the queue game, learned
   defenders, identity entrant); the priority fee; quoter defences; the frontier figure. Required
   for the project; its length in the paper follows its results. Write it last.
8. **Data** (1 p): sources (archive, feed), conventions, the calibration table (ρ_z, regime
   contrast, L, depth, touch-fill markout), the maker realized-spread anchor.
9. **Discussion and limitations** (¾ p): stylised takers; tabular learners; one venue; what
   generalises (the mechanism) and what may not (the one-tick finding).
Appendices: derivations; full grids; seed tables; data conventions and alignment rules;
replication notes pointer.

Reviewer expectations a course deliverable does not have (so they are built in now): seed
intervals everywhere; common random numbers; held-out attackers; train-at-α / test-at-α′;
chronological splits on data; a public repository with one command per figure; a limitations
section that names the stylisations.

---

## Fifteen days to Oct 27

| days | simulator (Claude + whoever takes it) | data (Deepu) | writing |
|---|---|---|---|
| Oct 12–14 | fringe of always-resting makers → learned defenders in the queue game | VPS recorder up; maker test run; identity 10-seed collected | glossary into the deliverable; outline agreed with Miguel |
| Oct 15–18 | exploitability grid at 10 seeds; identity-entrant vs learned defenders | October archive on SuperCloud (cascade window) | report §1–§3 drafted from the spine |
| Oct 19–22 | priority fee, first version; figures regenerated with intervals | calibration table assembled; coverage check on the VPS data | report §4 (five results, one figure each) |
| Oct 23–25 | — (freeze numbers) | — | slides; report §5; dry run with Miguel |
| Oct 26 | — | — | final read: every term used is in §2; cut anything not on the spine |

Owner suggestion for Miguel (to agree, not assign): the quoter defences and the frontier figure
(RQ3's second half) — self-contained, uses the agents that exist, and is the part of the project
that must be delivered whatever the paper ends up centred on.
