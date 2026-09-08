# Negative results

Things that were built, measured, and removed — and, at the end, a published
measurement that was withdrawn. Recorded because the measurements cost more than
the code did, and because each of these looks obviously correct until you run it.

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

**Outcome: the third one worked, in a form that needs no threshold.** The
threshold was never found and is not needed. `MIPParams.mir_cuts=None`, now the
default, runs the root cut loop with MIR off and with MIR on and keeps whichever
reached the stronger bound -- a race rather than a cut-off, so nothing has to be
calibrated and nothing has to be priced. It is the gain-only rule with the one
degree of freedom removed. Root bounds: p0201 7054.62 without MIR against
7185.00 with, gt2 21166.00 without against 21104.83 with. Over the set at 60 s
it proves 6/11 at a 20.19 s shifted geomean, against 6/11 at 22.39 s with MIR
forced off and 5/11 at 23.90 s forced on -- the first rule here that wins both
gt2 and p0201 instead of trading one for the other. It costs a second root cut
loop on every model. What it still does not do is price fill against tree size;
it sidesteps that question rather than answering it, and a model whose root MIR
helps but whose tree MIR slows would be chosen wrongly.

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

---

## Objective-parallelism cut scoring (attempted, reverted)

**Idea.** Cuts are ranked by efficacy, the Euclidean distance from the
relaxation optimum to the cut:

```
efficacy = violation / ||a||
```

That is a *normalised* violation, so it mechanically rewards a sparse cut -- a
small `||a||` inflates the ratio however little the cut moves the bound. On gt2
this looked like exactly the problem: MIR cuts have median density 0.085
against GMI's 0.32, win the ranking, and leave a root bound 70 units *worse*
than GMI alone.

The textbook remedy is to score on efficacy and objective parallelism together
(Wesselmann & Suhl, *Implementing cutting plane management and selection
techniques*, Paderborn, 2012), since a cut only lifts the bound insofar as it
opposes the objective:

```
score = efficacy * (1 + w * |c.a| / (||c|| ||a||))
```

**The hypothesis measured true.** Objective parallelism of selected cuts on
gt2 separates the families cleanly -- GMI median 0.181, MIR median 0.084, and
MIR's single best cut (0.186) barely reaches GMI's median.

**Why it was reverted anyway.** The response to `w` is not monotonic, and not
close to it. Root gap on gt2, optimum 21166:

| w | 0.0 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|
| root gap | 84.4 | 17.8 | 127.3 | 82.8 | 112.9 |

These runs are deterministic, so that is not noise -- it is the greedy
selection trajectory swinging on small perturbations. Any `w` chosen from this
table is fitted to one instance's chaos. Over the full set `w = 1` measured as
a wash against `w = 0`: 5/11 proved either way, 405.8 s against 404.9 s, one
instance better (misc07's incumbent reaching the true optimum 2810) and one
worse (khb05250 7.4 s -> 8.2 s).

A change that does not fix the instance it was written for, and that cannot be
told apart from its own baseline on the rest of the set, has not earned the
knob it adds.

**What was wrong instead.** The efficacy ranking was never the problem. gt2 was
losing its root closure to the *orthogonality floor* -- 0.05 rejects any cut
within 3 degrees of one already chosen, and gt2's root is closed by a fan of
near-parallel cuts of very different depths. Lowering the floor to 0.001 closes
gt2 at the root exactly (21166.00, 86 cuts, 5.6 s). The efficacy floor was
checked in the same sweep and is not implicated: 1e-4 through 1e-8 all close it
once the orthogonality floor is right.

Worth recording as a habit: the measurement that confirmed the hypothesis
(parallelism really does separate the families) and the measurement that
decided the change (does it help?) were different measurements, and only the
second one mattered.

---

## The 22-item knapsack table (published, withdrawn)

Not a failed optimisation — a failed *measurement*. It is the first published
figure here to be re-checked against the commit that produced it, and it did
not survive; the other performance claims in the README have not had the same
treatment yet.

**What it said.** Under "The revised simplex", as the demonstration that bound
quality dominates tree size:

| node bound | nodes | time |
|---|---|---|
| batched first-order (BNR) | 40,211 | 22.6 s |
| exact LP, dual simplex warm start | 59 | 1.5 s |

**What re-measurement found.** The instance was not stored, so it was
reconstructed from the recipe both knapsack generators in the test suite share
(integer weights in [1,40), values in [1,60), capacity a fraction of total
weight, `default_rng(0)`, n = 22). On that instance the simplex row reproduces
*exactly* — 59 nodes, both at `a1c9df7` where the table was written and at
HEAD — which is strong evidence the instance is the right one.

BNR on the same instance gives **65 nodes**, not 40,211. At `a1c9df7` itself it
returns **INFEASIBLE**, on that instance and on all eight knapsack variants
tried, always in three nodes. At `901d738`, the commit before the simplex
existed and therefore BNR by default, the same instance gives **59 nodes**. The
figure 40,211 matches neither engine at any commit tested.

It reproduces only on a much harder *near-correlated* knapsack, and there it is
a stopwatch reading rather than a search:

| time limit | BNR nodes | BNR status | simplex |
|---|---|---|---|
| 45 s | 30,207 | TIME_LIMIT | 2,567 nodes, 1.83 s, OPTIMAL |
| 60 s | 41,535 | TIME_LIMIT | 2,375 nodes, 2.24 s, OPTIMAL |

The node count tracks the clock, not the problem. No instance was found on
which the simplex needs 59 nodes and BNR needs ~40,000: on easy knapsacks the
two are within a few nodes of each other (85 vs 83, 75 vs 79), and on hard ones
BNR times out while the simplex finishes in around two thousand.

**The conclusion.** The two rows were measured on different instances, and the
BNR row reported an unfinished search as a completed one. The table has been
replaced with a same-instance comparison at a stated time limit, which makes
the original point more strongly: on the hard knapsack BNR does not merely fail
to prove optimality, it never reaches it, finishing on −23003 against a true
−23007.

**Why it matters beyond one table.** The underlying claim was true, which is
exactly why the numbers went unchecked: nobody re-derives a figure that agrees
with what they already believe. A correct conclusion resting on numbers nobody
can reproduce is indistinguishable, from the outside, from a wrong one.

The replacement table states its instance completely enough to rebuild — n,
distribution, capacity fraction, seed, time limit, settings — which the
original did not. **The rest of the README's performance figures have not all
been audited to that standard**, and until they are, the honest status of any
one of them is "measured once, on an instance that was not kept". The GPU
kernel timings have since been checked (below), and so have the MIPLIB tables
-- which turned up a live regression rather than a bad number: gt2 was
published as OPTIMAL in 1.71 s and had come to return no incumbent at all,
bisected to the commit that added MIR cuts. The root cut loop has been repaired
and gt2 is now proved again, in 6.40 s. The diagnosis in this paragraph was
itself wrong for a while: the relaxation was never cycling, it was dual simplex
pricing, and once that was fixed what remained was cut selection -- an
orthogonality floor rejecting the fan of near-parallel cuts that closes the
root, and MIR crowding out the GMI and cover cuts that do the closing. Both are
fixed, the second by the rule recorded above. The lesson is the same one twice
over: a table nobody regenerates stops being a measurement and becomes a memory.

---

## The K = 256 batching row (published, corrected)

Checked because the entry above named it as the next thing to verify. Unlike
the knapsack, this one is stored: `python -m bench.gpu_bench --skip-pdlp`
rebuilds the instance deterministically, so it can simply be re-run.

**Four rows of five held.** K = 8, 32, 64 and 128 reproduced within a few
percent on the same RTX 3050 the README names, across four runs. The headline
claims held with them: 42 GFLOP/s sustained in fp64, an 8.8× peak speedup
(measured 8.85–9.21×), and 11 µs per node at K = 64 (measured 11.1 µs). The
published table is also internally consistent — both derived columns recompute
correctly from the times and the nonzero count.

**The K = 256 row did not.** Published 4.61 ms / 4.3× / 26.6 GFLOP/s; measured
2.88–2.92 ms / 6.7–6.8× / 42.1–42.6 GFLOP/s, stable across every run. The
throughput collapse it depicts is not there. Extending the sweep settles it:

| K | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| GPU GFLOP/s | 44.7 | 43.5 | 42.7 | 42.5 | 42.6 | 42.4 |
| CPU GFLOP/s | 13.1 | 7.4 | 5.0 | 4.6 | 4.6 | 4.6 |

GPU throughput saturates by K = 32 and is flat to K = 1024. **The fall-off is a
CPU effect that had been attributed to the GPU** — which also made the
`MIPParams.batch` docstring wrong, since it cited that row to justify a ceiling
that only exists on the CPU. The default of 64 is still right, for a different
reason: it has to be safe on a machine with no GPU.

Also withdrawn: "the 3.4× raw SpMV advantage over the CPU". Re-measured, a
single SpMV on this matrix ranges from 0.76× to 2.56× depending on size and
run — two runs at the same 240k size gave 0.76× and 1.17×. The variance is the
finding; a headline ratio should not rest on a quantity that swings that far.
Replaced with a per-node, same-operation comparison at K = 64: 11.1 µs on the
GPU against 65 µs on the CPU.

**The lesson is different from the knapsack's.** Nothing here was measured on
the wrong instance, and the error was in the *pessimistic* direction, which is
why it survived. But the one wrong row was the load-bearing one — the sole
evidence for a batch-size ceiling, cited by a default in the code. A table can
be 80% right and still have its conclusion resting entirely on the 20%.
