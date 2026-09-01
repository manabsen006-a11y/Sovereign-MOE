# Negative results

Things that were built, measured, and removed. Recorded because the measurements
cost more than the code did, and because each of these looks obviously correct
until you run it.

---

## Cost-aware cut rollback (attempted, reverted)

**Idea.** Cuts sometimes slow a model down more than the bound they buy is
worth — on p0201 they took the solve from 8.16 s to 39.65 s. So measure what a
cut round costs and roll it back when it does not pay:

```
payoff = bound_gain / (fill_ratio - 1)      keep the round if payoff >= threshold
```

with cost measured as **basis-factorisation fill** rather than wall-clock, so
the decision stays deterministic. Fill is not a proxy for the slowdown — it is
the mechanism: dense cut rows make the LU dense, which is what stops FTRAN
being hypersparse.

**Calibration.** One cut round from the root, measured:

| instance | m | cuts | LU fill | bound gain | payoff |
|---|---|---|---|---|---|
| gt2 | 29 | 11 | 77 → 130 (×1.69) | 43.93% | **0.638** |
| khb05250 | 101 | 19 | 307 → 652 (×2.12) | 8.61% | **0.077** |
| p0201 | 133 | 4 | 1054 → 1684 (×1.60) | 1.46% | 0.024 |
| mod010 | 146 | 5 | 2356 → 2932 (×1.24) | 0.24% | 0.010 |
| flugpl | 18 | 5 | 62 → 122 (×1.97) | 0.33% | 0.004 |

A threshold of 0.05 separates the two groups exactly: keep cuts on gt2 and
khb05250 (both *need* them — gt2 closes at the root with them, khb05250 is
unsolvable without them), drop them on p0201, mod010 and flugpl (all slowed by
them). The calibration looked clean.

**Why it failed anyway.**

*Attempt 1, per-round.* Roll back the round that fails the test and stop.
This abandons models whose cuts only pay after several rounds. gt2 needs 64
cuts to close the root; the first flat round killed the loop and gt2 went from
OPTIMAL in 1.71 s to a timeout with an incumbent of 38256 against a true
optimum of 21166.

*Attempt 2, cumulative.* Judge the whole cut set against the uncut base
instead. This fails for a more fundamental reason: **`fill_ratio - 1` grows
roughly linearly in the number of rounds while the bound gain saturates**, so
the payoff decays monotonically and eventually rejects everything. gt2
accumulates a ~13× fill increase over 64 cuts and is discarded despite those
cuts closing the model. Measured outcome: gt2 TIME_LIMIT with no incumbent at
all (64.9 s), khb05250 TIME_LIMIT at 4.7% off, against OPTIMAL at 1.71 s and
7.70 s respectively without the rollback.

Net effect of the rollback: it fixed the two instances cuts were hurting and
broke the two cuts were saving. A straight trade, not progress — so it was
reverted.

**The actual flaw in the metric.** Fill only costs you if you are going to
explore nodes. A cut set that closes the root is worth *any* amount of fill,
because there are no nodes to pay it back over. The metric prices cost without
knowing the tree size it will be amortised over, and that quantity is precisely
what is unknown at the moment the decision has to be made.

**What a working version would need.** Either

* an estimate of the remaining tree size (so fill can be priced against the
  number of node LPs it will actually slow down) — e.g. from the gap and the
  observed node throughput, decided *after* some tree search rather than at the
  root; or
* deferring the decision: keep the cuts, start the tree, and drop the cut rows
  if the observed node rate collapses without the gap closing; or
* a gain-only rule, which is the stable quantity. Cumulative gains measured:
  p0201 3.94% over 19 cuts (fill ×3.24). The single-round gains above suggest a
  threshold near 5% might separate the groups, but the cumulative numbers for
  gt2, khb05250, mod010 and flugpl were never collected, so this is a
  hypothesis and not a result.

**What is in the code instead.** The density filter in
`CutPool.support_limit`, keyed to the basis size `m`. It catches the
pathological case (near-dense cuts on a model large enough for hypersparsity to
matter) without needing to price anything, and it is what keeps p0201 solvable
at all. It does not help p0201's 39.65 s, because p0201 has `m = 133` and falls
under the small-basis exemption.

---

## Density cap keyed to the column count (attempted, reverted)

The first version of the density filter capped cut support against `n`. It
fixed p0201 and broke gt2, which closes at the root on dense cuts. The damage
from a dense row scales with the **basis size `m`**, not the column count:
below a few hundred rows there is no hypersparsity to lose, so a cap keyed to
`n` only discards good cuts. See the docstring on `CutPool.support_limit`.
