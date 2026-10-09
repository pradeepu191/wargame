> **Superseded (Oct 9 2026).** The revised deliverable `paper/deliverable1_revised.tex` incorporates these review responses and the Workshop-1 findings; its Section 2 is the audit of what changed since the submitted version.

# Response to the external assessment of Deliverable 1 (6 Oct 2026)

A deep-research review of the proposal was obtained and acted on. Every new reference it
introduced was verified against the primary source before being cited. Summary of what changed.

## Accepted and applied

| Review point | Change |
|---|---|
| Centre the paper on one falsifiable claim: learned rents → predictable reaction functions → strategic vulnerability | New "Central claim" paragraph in Motivation; RQ2 reformulated as $C(\pi)$ (rent) vs $L_{\mathrm{BR}}(\pi)$ (best-response loss), hypothesis $\partial L_{\mathrm{BR}}/\partial C>0$. Our experiments already support it: Δ 0.77–0.87, entrant captures 82–96% of monopoly rent, incumbents do not punish. |
| Zhai (2026) has opened the "wallet identity predicts toxicity" question; position our work after it | Cited; our contribution stated as the dynamic-execution question Zhai leaves open. |
| "Wallet identity," not "trader identity" | Terminology changed throughout both documents. |
| Information timing: identity is public only after commitment | Taker item now says makers see wallet histories through $t-1$, never the identity of the incoming order. Nested information sets $\mathcal F^{\mathrm{anon}}\subset\mathcal F^{\mathrm{id}}\subset\mathcal F^{\mathrm{oracle}}$ added; RQ1 is now a value-of-information experiment. |
| "Discrete time is exact" is overstated | Replaced with "tractable event-time approximation" with the irregular-block-clock fact. |
| Rate-limit "drain" is impossible (limits are per user) | Reframed as *induced self-exhaustion*; defences (hysteresis, min quote lifetime, reserve) named; Brogaard et al. cited for flicker responses. |
| Priority fee: 45 ms/bp is IOC only; ALO buys queue position | Corrected; priority modelled as an action component $\rho^i_t$ with mode-dependent effect. |
| HLP is not a known fixed policy | Now "HLP-inspired fixed benchmark calibrated to observable HLP behaviour." |
| Tick jump should be computed from the venue rule | Stated; simulator to compute the grid per asset/price region. |
| Oct 10 numbers conflate Hyperliquid liquidations with multi-venue spread stats | Separated and attributed; we will compute Hyperliquid's own path from the L4 data. |
| CFL is published (RFS 2026); CCP title changed | Bibliography updated. |
| Add Barone–Lillo, Brown–MacKay, Brogaard et al., Chilenje et al. | Added, all verified. |
| Workshop 1 too ambitious | Text now lists exactly four items and states what is *not* a Workshop 1 requirement. (Already how we were working.) |
| Drop multi-asset extension | Feedback question 3 now argues against it. |
| Report CVaR, common random numbers, held-out attackers, chronological splits | Added under "Rigor." |

## Considered and not adopted

* **Replace the fixed-$V$ Glosten–Milgrom fundamental with a random-walk $X_t$ for the empirical extension.** The specific defect the review is reacting to (a fixed, finite informed edge that makes wide quotes immune to informed flow) is already fixed by our continuous exponential-edge model, which keeps closed-form competitive and monopoly benchmarks. A random walk loses those benchmarks, and the benchmarks are what make $C(\pi)$ a number rather than a judgment. We keep our model for the simulator and adopt the review's markout-based toxicity definition for the empirical side, where it is the right object.
* **Retitle without Hyperliquid.** Kept Hyperliquid in the title (second of the review's two suggestions); the empirical section is a major contribution.

## Review claims we checked and found correct
Zhai arXiv:2608.04373 (persistence 0.52; $R^2$ 10.88% → 12.31%; execution model left open). CFL RFS advance article hhag010. Barone–Lillo arXiv:2606.15715. Brown–MacKay NBER 34070. Brogaard et al. SSRN 5963035. Chilenje et al. SSRN 5505480.
