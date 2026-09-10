# SOVOPT

**A sovereign GPU-accelerated optimization engine — LP / MILP, built from mathematical foundations.**

SIH 2026 · Problem Statement 26119 · Mangalore Refinery and Petrochemicals Ltd

No solver library is used, linked, or vendored. Every algorithm here is
implemented from the published mathematics, with a citation header on each
module naming its source. See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) and the
clean-room policy in [`CLAUDE.md`](CLAUDE.md).

---

## Status

Working end to end, with **three independent LP engines** -- an exact revised
simplex, a primal-dual interior-point method and a GPU first-order method --
validated against **published MIPLIB reference values** and an **independent
verifier** that recomputes feasibility from the original model.

Reproduce with `python -m bench.harness --mode lp --method pdlp --device cpu
--time-limit 90 --tol 1e-8`. **Every timing in this README was measured on the
machine described under [Measurement conditions](#measurement-conditions)**;
quote the ratios rather than the seconds if you are comparing against your own
hardware.

```
SOVOPT benchmark  mode=lp  method=pdlp  device=cpu  time-limit=90s  tol=1e-8

instance         rows   cols      nnz status            objective        reference    relerr     time  chk
10teams           230   2025    12150 OPTIMAL           917.00001              nan       nan    0.61s   ok
dcmulti           290    548     1315 OPTIMAL           183975.54        183975.54  5.61e-11    2.67s   ok
flugpl             18     18       46 OPTIMAL           1167185.7        1167185.7  4.29e-09    0.34s   ok
gr4x6              34     48       96 OPTIMAL              185.55              nan       nan    0.29s   ok
gt2                29    188      376 OPTIMAL           13460.233        13460.233  3.07e-11    0.04s   ok
khb05250          101   1350     2700 OPTIMAL            95919464         95919464  7.60e-13    6.90s   ok
mas76              12    151     1640 OPTIMAL           38893.904              nan       nan   16.28s   ok
misc07            212    260     8619 OPTIMAL                1415              nan       nan    0.45s   ok
mod010            146   2655    11203 OPTIMAL           6532.0833          6532.08  5.08e-07    1.52s   ok
p0201             133    201     1923 OPTIMAL                6875             6875  1.11e-11    0.16s   ok
qnet1             503   1541     4622 OPTIMAL           14274.103        14274.103  1.12e-10    2.83s   ok

  status OPTIMAL       11/11
  verifier accepted    11/11
  within 1e-4 of ref   7/7
  shifted geomean time 1.496s (shift 1s)
  total time           32.1s
  worst relative error 5.08e-07 (mod010)
```

The other two engines solve the same set to the same published values, and
considerably faster:

| engine | optimal | verified | shifted geomean | total | worst rel. err |
|---|---|---|---|---|---|
| first-order (PDLP) | 11/11 | 11/11 | 1.496 s | 32.1 s | 5.08e-07 |
| revised simplex | 11/11 | 11/11 | 0.258 s | 3.9 s | 5.10e-07 |
| **interior point** | 11/11 | 11/11 | **0.140 s** | **1.6 s** | 5.10e-07 |

All four rows above -- the PDLP listing and these three -- were re-measured
back to back at this commit, so they are comparable with each other and *not*
with figures recorded earlier in this README: the same commands on this machine
now run about twice the seconds they did in the sitting those were taken in,
while every objective matches to the digit and the HiGHS ratio below reproduces
exactly. **Read the ratios, not the seconds.** The one ratio that genuinely
moved is the interior point's: it was at parity with the simplex until it was
given a fill-reducing ordering, and is now about 1.8x faster on this set.

Three engines agreeing to the published value on every instance is the point of
having three. They fail differently: the simplex is exact but walks vertices and
stalls on degeneracy, the interior-point method takes a fixed handful of
iterations but needs a factorisation in each one and returns no basis of its
own, and PDLP needs neither but converges slowly in the tail. `--method` picks
one; `--method auto` still chooses between the simplex and PDLP by size,
because the rule for when the interior-point method should be preferred has not
been measured on anything larger than this set.

**`--crossover` composes them instead of choosing.** An interior point is not a
vertex, so neither PDLP nor the interior-point method produced a basis -- and
without a basis there is no cost or RHS ranging, and no warm start for a
branch-and-bound node. Crossover identifies a basis from the interior point and
lets the simplex finish from it. Over the LP set that is **61.9% fewer simplex
pivots than starting cold** (6,747 → 2,573), every objective identical:

| | 10teams | dcmulti | gt2 | khb05250 | mas76 | misc07 | mod010 | p0201 | qnet1 |
|---|---|---|---|---|---|---|---|---|---|
| cold pivots | 3,698 | 360 | 65 | 105 | 43 | 269 | 1,244 | 84 | 833 |
| after crossover | 1,311 | 144 | 9 | **2** | **0** | 340 | 303 | **2** | 424 |

The case it was built for is the one the simplex cannot reach alone. On the
15,360 x 15,360 planning model of the [Scale](#scale) ladder -- where a cold
simplex hits the time limit having never reached feasibility -- PDLP solves in
148 s and crossover turns that into an **optimal basis in zero pivots**,
verifier-accepted, agreeing with PDLP's objective. Ranging is available on a
model the simplex on its own cannot solve at all.

misc07 (+26% pivots) and flugpl (14 → 21) go the other way, which is what a
heuristic identification does on some instances; the aggregate is the claim.

Per-instance the gap is much wider than the aggregate suggests -- on mas76 the
simplex finishes below the timer's resolution against PDLP's 5.08 s, and it is
114x faster on khb05250 and 10.6x on qnet1 -- while PDLP still takes 10teams
(0.18 s against 1.35 s), which is highly degenerate. They are genuinely
complementary, and `method="auto"` picks by size.

MILP is exact where it closes. With exact node LPs the tree is sharp: a hard
22-item knapsack closes in **2,375 nodes and 2.24 s**, where the batched
first-order bound is still running at the 60 s limit having never reached the
optimum (see [The revised simplex](#the-revised-simplex) for the full table).
On the full MIPLIB set at a 60 s limit, before and after adding cutting planes
and primal heuristics:

```
                                              measured on this tree
instance   before cuts+heuristics       status        objective     time
flugpl     OPTIMAL     1201500          OPTIMAL         1201500     4.50s
gr4x6      OPTIMAL      202.35          OPTIMAL          202.35     0.65s
gt2        TIME_LIMIT   (none)          OPTIMAL           21166     6.40s   <- see below
khb05250   TIME_LIMIT   (none)          OPTIMAL   1.0694023e+08     4.34s
mod010     OPTIMAL        6548          OPTIMAL            6548     3.60s
p0201      OPTIMAL        7615          OPTIMAL            7615    11.93s
mas76      TIME_LIMIT 40589.44          TIME_LIMIT   40589.436     60.27s   <- see below
misc07     TIME_LIMIT     2810          TIME_LIMIT        2810     60.30s
dcmulti    TIME_LIMIT   (none)          TIME_LIMIT     189263.2    60.98s  (0.58% off)
qnet1      TIME_LIMIT   (none)          TIME_LIMIT    20627.76     60.52s (28.7% off)
10teams    TIME_LIMIT   (none)          TIME_LIMIT      (none)     60.76s

                        before     now
  status OPTIMAL         4/11      6/11
  verifier accepted      6/11     10/11
  no incumbent at all    5/11      1/11
  shifted geomean       31.4s     20.2s
```

Every proved optimum matches the published value exactly, and the verifier
rejects nothing.

**Two rows moved for reasons that are not cuts. Both have been bisected rather
than left as drift.**

*dcmulti* returns an incumbent again -- 189263.2, 0.58% off, verifier-accepted
-- where the previous table had none. Each end isolates to one commit: it was
still 202598.49 at `50bf2f0`, gone by `ac66b9c`, and back at `6d6e588`.

*mas76 was never a regression.* This table used to report 40408.477, and that
number does not reproduce at the commit that published it -- `50bf2f0`, whose
own message is "re-measure the MIPLIB tables", returns **40662.142** under the
harness default. 40662.142 then holds unchanged across every commit through
`a5582b2`, and `d7fe05c` improves it to the 40589.436 above. Measured against
the figure that reproduces, mas76 got 72.7 *better*. Measured against the one
that was published, it looked worse. The published one was the memory -- the
same failure this README names below, caught this time in a table that had
been labelled a re-measurement.

**gt2 is a regression this table found, now repaired.** It was published
as `OPTIMAL 21166 in 1.71s`. Re-running the table showed it returning *nothing*
— timing out with no incumbent — and bisection put the break at `caa5fa3`, the
commit that added MIR cuts. It reproduced on demand, with `nodes=0` either way:
gt2 closes at the root, so the whole failure lived in the root cut loop.

Two things were wrong there, both now fixed. The loop built its LP solver with
the **full** time limit, so one stalling relaxation consumed everything the tree
was going to need -- `cut_time_frac` documented that cap and the loop never
applied it. And when that LP failed, the cut set that broke it was handed to the
tree anyway; every node inherits the root relaxation, so an unsolvable one
yields a search that explores no nodes at all. Cut rounds are now rolled back to
the last set whose own LP solved.

gt2 returns `21166` again -- the published optimum, verifier-accepted -- and the
set as a whole went from 9/11 to 10/11 verified and 2 instances with no
incumbent down to 1. flugpl fell from 9.28 s to 3.81 s on the same change.

**It is back to `OPTIMAL`, in 6.40 s.** The relaxation that used to hang was
never cycling -- it was the dual simplex pricing described below, and it now
solves in 439 iterations instead of running past 100,000. What remained after
that was cut *selection*, and it took two fixes. MIR cuts crowd out the GMI and
cover cuts that close gt2's root; and the orthogonality floor was rejecting the
fan of near-parallel cuts the closure actually needs, stalling the root at
21155.00 against an optimum of 21166. The floor is fixed, and MIR is no longer
one setting for the whole set -- it is decided per model by running the root
cut loop both ways, which is described under [Known limits](#known-limits).
gt2 takes the branch that leaves MIR out, closes its root at 21166.00 exactly,
and proves the model.

The regression went unnoticed for four commits because this table was not re-run
after the code beneath it changed — the same failure recorded in
[`docs/NEGATIVE-RESULTS.md`](docs/NEGATIVE-RESULTS.md): a table nobody
regenerates stops being a measurement and becomes a memory.

The motivating failure -- five instances returning *no answer at all* -- is
down to one, 10teams. gt2 and dcmulti were the last two to recover: gt2 is
proved outright now, and dcmulti returns a feasible point again, which the
entry under [Known limits](#known-limits) records without claiming credit for.

**The cost, stated plainly:** cuts make some node LPs more expensive than the
bound they buy. p0201 is the clearest case, and it used to be **marginal at a
60 s limit**: 11.93 s in the run above, against 20.68 s, 39.65 s and a 60.5 s
timeout on earlier runs of the same instance, finding the optimum 7615 every
time but not always proving it. Most of that spread was the MIR question below
-- forcing MIR off in the same sweep costs it 52.17 s -- but it is one draw
either way, and should be read as one.

A cost-aware cut rollback was built to fix exactly this and **removed again**:
it fixed p0201 and flugpl but broke gt2 and khb05250, which cuts were saving.
The measurements and the reason the metric cannot work as posed are recorded in
[`docs/NEGATIVE-RESULTS.md`](docs/NEGATIVE-RESULTS.md).

---

## Quick start

```bash
pip install -e .            # numpy + numba
pip install cupy-cuda12x[ctk]   # optional, for the GPU path
```

The `[ctk]` extra pulls the CUDA libraries in as wheels, so the GPU path needs
only an NVIDIA driver -- no system CUDA Toolkit, and no `CUDA_PATH`. Run
`python -m sovopt.cli devices` to confirm; it reports the driver and runtime
versions and whether the CUDA kernels actually compiled, which is the fact a
solve depends on.

```bash
python -m sovopt.cli demo                         # the whole story, end to end
python -m sovopt.cli devices                      # what hardware is usable
python -m sovopt.cli info   model.lp              # stats + numerical health
python -m sovopt.cli solve  model.mps --device gpu --out sol.json
python -m sovopt.cli solve  model.mps --sensitivity     # shadow prices + ranging
python -m sovopt.cli verify model.mps sol.json    # independent check
python -m ui.server                              # http://127.0.0.1:8000
```

Models are read from MPS or CPLEX LP, plain or `.gz`/`.bz2`/`.xz`; the reader is
chosen by extension. A `QUADOBJ` section is parsed and, if the Hessian is
positive semi-definite, solved as a convex QP.

Presenting this? [`docs/DEMO.md`](docs/DEMO.md) is the runbook.

```bash
python -m bench.fetch --set small     # download MIPLIB instances
python -m bench.harness --mode lp                  # validate against published values
python -m bench.harness --mode lp --method simplex  # force one engine
python -m bench.harness --mode lp --method ipm      # the interior-point engine
python -m bench.scale --mode lp       # how far the engines actually go
python -m bench.gpu_bench             # CPU vs GPU
python -m bench.comparator            # head-to-head against HiGHS
python -m pytest tests/               # 277 tests; the 15 GPU ones skip without a device
```

---

## What is actually built

| Layer | Module | Status |
|---|---|---|
| Sparse core, dual CSR/CSC | `core/sparse.py` | done |
| MPS reader / writer | `io/mps.py` | done (free + fixed, RANGES, MARKER, QUADOBJ) |
| CPLEX LP reader | `io/lp_format.py` | done (ranged rows, both-side terms, bounds, General/Binary) |
| Scaling: Ruiz, Curtis–Reid, Pock–Chambolle | `numerics/scaling.py` | done |
| Sparse LU, threshold Markowitz + Gilbert–Peierls | `numerics/lu.py` | done |
| Fill-reducing ordering (reverse Cuthill–McKee) | `numerics/ordering.py` | done (AMD not built) |
| Hypersparse FTRAN / BTRAN | `numerics/lu.py` | done |
| Iterative refinement, compensated residual | `numerics/refine.py` | done |
| Condition estimation (Hager) | `numerics/refine.py` | done |
| **Revised simplex**, primal + dual, bounded | `lp/simplex.py` | done |
| Basis, product-form update, singularity repair | `lp/basis.py` | done |
| First-order LP (PDLP-class), CPU + CUDA | `lp/pdlp.py` | done |
| Hand-written CUDA kernels | `core/backend.py` | done |
| **Crossover**, interior point → optimal basis | `lp/crossover.py` | done |
| Presolve + postsolve (primal, duals, reduced costs) | `presolve.py` | done (opt-in; see below) |
| Safe dual bounds (Neumaier–Shcherbina) | `mip/safebound.py` | done |
| **Batched Node Relaxation** | `mip/bnr.py` | done |
| Domain propagation | `mip/propagate.py` | done |
| Branch and bound, exact or batched node LPs | `mip/tree.py` | done |
| Gomory mixed-integer, knapsack cover, MIR cuts | `mip/cuts.py` | done |
| Symmetry detection + static breaking | `mip/symmetry.py` | done |
| Conflict analysis (LP-infeasibility clauses) | `mip/conflict.py` | done |
| Sensitivity: shadow prices, cost and RHS ranging | `lp/sensitivity.py` | done |
| **Global bilinear pooling** (McCormick + spatial B&B + OBBT) | `globalopt/` | done |
| Feasibility Jump, fix-and-propagate, feasibility pump | `mip/heuristics.py` | done |
| Refinery model templates + Haverly pooling | `models/` | done |
| CLI, web UI, verifier, harness | `cli.py`, `ui/`, `bench/` | done |
| **Convex QP** (proximal PDHG, Condat–Vũ) | `qp/proximal.py` | done (convex only) |
| **Interior point** (Mehrotra predictor-corrector) | `lp/ipm.py` | done |
| Crossover, presolve, MIQP, non-convex QP | — | **not built** (roadmap) |

---

## The revised simplex

The exact engine. It is what a first-order method cannot be: it terminates at a
**vertex**, with a basis, and therefore with duals, reduced costs and the
identity of every binding constraint. A refinery planner reads marginal prices
off that basis; an approximate optimal *value* is of no use for the question
"what is one more tonne of this crude worth".

Bounded-variable primal and dual over the augmented system `M z = 0` with
`M = [A | -I]`, so `<=`, `>=`, `=` and ranged rows are all one code path -- a
row's type lives entirely in its logical variable's bounds.

- **Dual simplex** is the branch-and-bound workhorse. Reduced costs are
  `d = c - Aᵀ B^-T c_B`, which depends on the basis and objective but **not on
  the bounds**, so a branching decision cannot disturb dual feasibility. The
  parent's basis is therefore still dual feasible at the child and the dual
  simplex resumes from it. Measured: **1.7 pivots per node**, 100% warm-start
  hit rate.
- **Harris two-pass ratio test** in both directions. Pass one finds the largest
  step with bounds relaxed by the feasibility tolerance; pass two takes the
  largest *pivot magnitude* among rows blocking within it. The textbook
  minimum-ratio row routinely means pivoting on 1e-9 because it tied, which
  wrecks the factorisation a few iterations later.
- **Devex pricing** rather than Dantzig's rule, which is scale-dependent and
  takes far more iterations.
- **Product-form basis update** with periodic refactorisation, and repair of
  singular bases by swapping in logicals. Forrest–Tomlin would keep the factors
  sparser for longer and is the natural next step; it is not built.
- **Anti-degeneracy**: random cost perturbation after a run of zero-length
  pivots, removed and re-optimised before the answer is reported.

What it unlocks, measured on a **near-correlated 22-item knapsack** — values
within ±5 of weights drawn from [1000, 4000), capacity 40% of total weight,
`default_rng(0)`. Correlation is what makes a knapsack hard: the LP relaxation
is nearly flat, so the tree is decided almost entirely by bound quality. Both
engines, same instance, same 60 s limit, default settings:

| node bound | nodes | time | outcome |
|---|---|---|---|
| batched first-order (BNR) | 41,535 | 60 s (limit) | **TIME_LIMIT**, best −23003, bound −23014.92 |
| **exact LP, dual simplex warm start** | **2,375** | **2.24 s** | **OPTIMAL, −23007** |

BNR does not merely fail to *prove* optimality here — it never finds the
optimum, finishing on −23003 against a true −23007. Its bound stays valid, as
it must; it is simply too loose to close anything. Reruns land within 0.5% on
nodes (41,535 / 41,663) and the simplex figure is stable to the node.

That is the honest verdict on the batched idea: valid-but-loose bounds are cheap
and parallel, but bound *quality* dominates tree size. BNR remains the right
tool when nodes are large enough that an exact solve is unaffordable; the tree
takes `node_solver="simplex"` or `"bnr"`.

> **This table replaces an earlier one** claiming 40,211 nodes / 22.6 s for BNR
> against 59 nodes / 1.5 s for the simplex, which did not survive re-measurement.
> The 59 is real and reproducible — on an *easy* uncorrelated knapsack, where
> BNR needs 65 and the contrast evaporates. The 40,211 only ever appears on a
> hard instance, and there it was a stopwatch reading rather than a search: the
> node count tracks the time limit (30,207 at 45 s, 41,535 at 60 s) because BNR
> never finished. The two rows had been measured on different instances and an
> unfinished run reported as a completed one. See `docs/NEGATIVE-RESULTS.md`.

---

## Verified against outside authorities

Every correctness bug in this project was caught by an oracle *outside* the
solver. None was caught by the solver agreeing with itself, which is why the
checks below all reach for something external.

| check | result |
|---|---|
| **HiGHS head-to-head** (`bench/comparator.py`) | 11/11 agree, max relative difference **7.0e-16** |
| Independent verifier | every reported optimum accepted, recomputed in compensated arithmetic |
| Exact rational arithmetic | forward error **0.0** at cond₁ 1.7e12 |
| Published MIPLIB optima | every proved optimum matches exactly |
| Brute-force global scan | Haverly 400 / 600 / 750 confirmed |
| Cut validity (brute force) | 58 cuts vs every feasible point, worst slack −1.4e-14 |
| Conflict clause validity | 40 clauses, worst slack 0.0 |
| Shadow-price prediction | 239 rows, worst error 3.9e-15 |
| Whole-solve vs brute force | 450 models × both node solvers, **0 disagreements** |
| Batched safe bound vs exact simplex | 40 root relaxations, **0 invalid bounds** |

The last two are recent, and they are there because everything above them
passed while `node_solver="bnr"` was returning wrong answers. Enumerating every
integer point of a five-variable model is a weak-looking check that no amount
of agreement between components can substitute for.

**Speed, stated plainly:** HiGHS solves the same eleven LP relaxations in 0.4 s
against our 2.7 s — roughly **6.7× slower on our side**, or 4.8× excluding the
degenerate 10teams. (That ratio is the durable part: re-measured at this commit
it came back 6.75×, with both sides scaled by the same machine-state factor
discussed under [Status](#status).) That gap was 27× until the dual simplex got the pricing it
had been missing (see below). Matching its answers exactly is the achievement
here; matching its clock is closer than it was, and presolve plus a
Forrest–Tomlin update are what is left.

A full requirement-by-requirement conformance audit against the problem
statement — including what is *not* built — is in the artifact linked from the
project notes.

---

## Symmetry

Refineries have identical parallel units; schedules have interchangeable slots.
A tree without symmetry handling re-derives the same plan under every
relabelling, wasting up to `k!` for `k` identical objects.

Detection is two-stage and **only the second stage is trusted**. Colour
refinement (1-WL) on the bipartite variable-constraint graph proposes
candidates; individualisation-refinement turns a candidate pair into a full
permutation; and that permutation is then checked against the model in its
entirety -- objective, bounds, kinds, row bounds, every matrix entry. The search
may be incomplete without ever being unsound: it can miss a symmetry, it cannot
invent one.

Measured on identical-unit refinery scheduling, with the optimum preserved in
every case:

| identical units | nodes without | nodes with |
|---|---|---|
| 4 | 8378 | **6633** |
| 5 | 21813 | **17636** |
| 6 | 20338 | **16516** |

On the MIPLIB subset the effect is **within measurement noise** (shifted geomean
22.7 s against 24.1 s, on a machine that has shown 2.3x swings from background
load alone). p0201 and misc07 do have detectable symmetry -- 40 and 13 verified
generators -- but breaking it neither helps nor hurts them measurably. 10teams
has none at all: refinement drives it to 2025 singleton colours, and refinement
provably cannot separate two variables in the same orbit, so that is a proof of
absence rather than a failure to look.

Two bugs found building this, both of the *silently does nothing* kind, and both
now pinned by tests: detection that only tried single transpositions (a real
symmetry swaps every variable of a unit across every period at once, so it could
never fire), and colour numbering by order of first appearance, which made two
independently-refined colourings incomparable and returned zero generators on
every input including obvious Sym(8).

A third, found later and far worse, is bug 4 below: an overflowing colour key
made the *exact verification step* accept a permutation that was not an
automorphism — the one outcome the paragraph above is supposed to rule out.

## Global bilinear pooling

The highest-value piece here for a refinery, because **crude blending is a
pooling problem**. A pool's sulfur content multiplied by the flow leaving it is
a product of two decisions, so the feasible set is non-convex and a local
optimum is not a global one. The industry's standard workaround --
*distributive recursion* inside PIMS and GRTMPS: guess the pool qualities, solve
the LP, re-read the qualities, repeat -- is a fixed-point iteration with no
global guarantee.

`globalopt/` solves these to **proven** global optimality: McCormick envelopes
give a convex relaxation whose optimum is a rigorous bound, optimality-based
bound tightening shrinks the box first, and spatial branch-and-bound splits
*continuous* ranges until bound and incumbent meet. Envelope error is quadratic
in box width, so halving a range roughly quarters the gap that term contributes.

On Haverly's benchmarks, checked three ways -- against the published values,
against an independent brute-force scan, and against the solver's own bound:

| instance | global optimum | dual bound | brute force | nodes |
|---|---|---|---|---|
| haverly1 | **400** | 400 | 400 | 3 |
| haverly2 | **600** | 600 | 600 | 9 |
| haverly3 | **750** | 750 | 750 | 15 |

And the reason it matters, measured against the method it replaces:

```
                 global   recursion, 21 starting points
  haverly1          400   reaches 400 from 1 start; stuck below it from 20
  haverly2          600   stuck below global from 20 of 21
  haverly3          750   stuck below global from 20 of 21
```

Recursion returns a blend and no way to know whether a better one exists. This
returns a blend **and a proof of how far from optimal it can be**. The gap is
quality give-away, and it has a price.

One structural nicety falls out of the formulation: when a factor's range
collapses to a point the McCormick envelope becomes *exact*, so distributive
recursion is simply the relaxation on a degenerate box. It is kept as the primal
heuristic inside the tree -- supplying incumbents while the search proves
optimality, instead of being the whole method with no guarantee.

### Multi-pool: the pq-formulation

Haverly has one pool. A refinery has many -- tanks, headers, intermediate
streams -- and the hard non-convexity is several pools feeding one product.

The **q-formulation** replaces pool qualities with the *proportions* `q_ij` of
each source in each pool, leaving products `v_ijk = q_ij * y_jk`. The
**pq-formulation** adds the reformulation-linearisation rows obtained by
multiplying the proportion identity through by each outgoing flow:

```
(sum_i q_ij) * y_jk = y_jk        becomes        sum_i v_ijk = y_jk
```

Linear, redundant in the exact model, and decisive in the relaxation: without it
each auxiliary floats independently inside its own McCormick box, and the
relaxation can buy cheap quality from one while selling volume from another.

Measured on random multi-pool networks, 60 s limit (maximisation, so a *lower*
root bound is tighter):

| network | terms | root q | root pq | tighter | q nodes/time | pq nodes/time |
|---|---|---|---|---|---|---|
| 4x2x2 s0 | 16 | 4391 | **1476** | 66.4% | 2528 / 60.0s | **2 / 0.4s** |
| 4x2x2 s1 | 16 | 2503 | **1293** | 48.4% | 1362 / 60.0s | **0 / 0.3s** |
| 5x2x3 s0 | 30 | 4539 | **3965** | 12.6% | 1581 / 60.0s | **23 / 1.1s** |
| 5x2x3 s1 | 30 | 6384 | **3704** | 42.0% | 1600 / 60.0s | **10 / 0.5s** |
| 6x3x3 s0 | 54 | 5725 | **3976** | 30.6% | 1645 / 60.0s | **4 / 0.6s** |
| 6x3x3 s1 | 54 | 5033 | **3119** | 38.0% | 1774 / 60.0s | **5 / 0.4s** |

**Every q run hits the time limit; every pq run closes in under 1.2 seconds.**
On the two rows where the reported objectives differ (5x2x3 s0, 6x3x3 s1) it is
because q never finished and returned a worse incumbent -- pq's value is the
higher one in both cases, and pq proved it.

On the single-pool Haverly instances pq matches the p-formulation rather than
beating it (root 500 against 500). The RLT rows are what rescue the *q* encoding
there, from 1950 to 500; the advantage over p is a multi-pool phenomenon, which
is the point of the formulation.

---

## Sensitivity analysis

The industrial payoff of having a basis, and the reason the simplex was worth
building. "The optimum is 4.2 crore" is the least interesting number a refinery
LP produces; the questions that change decisions are *what is one more tonne of
this crude worth*, *how far can its price move before I should buy something
else*, and *over what range is that price still valid*. An interior point cannot
answer any of them.

```
  shadow prices (binding rows, by value)
    row                            dual       activity           rhs range
    c3                          2.33333              3              [0, 3]
    c2                         0.666667              6              [3, 6]

  cost ranging (variables at non-zero value)
    column          value      cost   reduced        cost range
    x                   3         3        -0    [0.666667, +inf]
    y                   1         2        -0             [-0, 9]
```

Validated against reality rather than against a formula: perturb each row inside
its reported range and check the objective moves by exactly `dual x delta`.
Across 12 models and **239 binding rows**, the worst relative prediction error
is **3.9e-15**. A shadow price that does not predict the objective is worse than
none, because a planner acts on it.

Degeneracy is reported rather than hidden. At a degenerate optimum several bases
describe the same point and the ranges are those of the basis the solver stopped
at; a zero-width range means "this price is one of several valid ones here", not
"infinitely sensitive". Refinery LPs are massively degenerate, so the report
counts and states it.

---

## Conflict analysis

When a node is infeasible, an ordinary tree throws the fact away and
rediscovers it in every sibling repeating the same decisions. Conflict analysis
asks which decisions were to blame and forbids that combination globally.

The certificate is the Neumaier-Shcherbina bound with a **zero objective**: for
any `y`, every feasible point of the node satisfies `0 >= L(y)`, so `L(y) > 0`
proves the node empty. Because `L(y)` is a sum of independent per-variable
terms, testing whether a branching decision was *needed* costs one subtraction
rather than another LP solve:

```
L_without_j  =  L  -  term_j(node bounds)  +  term_j(root bounds)
```

That is what turns "these thirty decisions are jointly infeasible" -- useless,
it excludes one node -- into a short clause that prunes broadly.

Validated by brute force: 40 clauses over 15 small models, each checked against
**every** integer-feasible point, worst slack exactly `0.000e+00`. Clauses come
out at median length 2 on those models.

It fires rarely in practice, for a structural reason: propagation runs before
the node LP, so nodes it can refute never produce the dual ray this needs. See
`mip/conflict.py` for the measured numbers and what would fix it.

---

## The idea that is new here

### Batched Node Relaxation

Every node in a branch-and-bound tree solves an LP over the **same** matrix `A`,
the **same** objective, and the **same** row bounds. Branching changes nothing
but the variable bounds. Conventional solvers still process nodes one at a time,
which is right for a CPU and wrong for a GPU: one node's sparse product cannot
fill the device.

SOVOPT pops a *slab* of nodes and bounds them all in one sparse-times-dense
product. Reproduce with `python -m bench.gpu_bench --skip-pdlp`, which builds
the instance deterministically — 20,000 × 30,000, 239,952 nonzeros, `seed=1234`
— on an RTX 3050 Laptop:

| K | K × SpMV | SpMM | speedup | GFLOP/s |
|---|---|---|---|---|
| 8 | 0.81 ms | 0.20 ms | 4.1× | 19.7 |
| 32 | 3.17 ms | 0.34 ms | **9.2×** | 44.7 |
| 64 | 6.35 ms | 0.71 ms | **9.0×** | 43.5 |
| 128 | 12.13 ms | 1.44 ms | 8.4× | 42.7 |
| 256 | 19.56 ms | 2.89 ms | 6.8× | 42.5 |
| 512 | 39.17 ms | 5.77 ms | 6.8× | 42.6 |
| 1024 | 78.19 ms | 11.59 ms | 6.8× | 42.4 |

**42 GFLOP/s sustained in fp64** on a card whose fp64 *peak* is ~86 — about half
of peak, from a sparse kernel — and it stays there: throughput saturates by
K = 32 and is flat to K = 1024. The SpMM column repeats to about 1%; the
`K × SpMV` column is a Python loop of K launches and is much noisier (K = 128
ranged over 9.8–12.7 ms across runs), so the speedup column inherits that
noise. Read the SpMM and GFLOP/s columns as the measurement and the speedup as
an approximation.

The batch size ceiling is a **CPU** phenomenon, not a GPU one. The same sweep
on the CPU kernels:

| K | 8 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| CPU GFLOP/s | 12.1 | 13.1 | 7.4 | 5.0 | 4.6 | 4.6 | 4.6 |

That is the dense operands falling out of cache, and it is why `MIPParams.batch`
defaults to 64: the default has to be safe on the machine that has no GPU.

Per node, for the same operation at K = 64, the SpMM costs **11.1 µs on the GPU
against 65 µs on the CPU** — 5.9×. An earlier version of this section claimed a
"3.4× raw SpMV advantage"; re-measured, a single SpMV on this matrix ranges from
0.76× to 2.56× depending on size and run, with too much variance to quote a
figure, so the claim is withdrawn rather than restated.

### Why unconverged GPU bounds are still rigorous

A first-order method's dual iterate is not dual feasible, so its dual objective
is not a valid bound — using it would prune away the optimum. The
Neumaier–Shcherbina correction fixes this: for **any** vector `y`, feasible or
not,

```
c'x  ≥  Σ_i (y_i > 0 ? y_i·rl_i : y_i·ru_i)  +  Σ_j (d_j > 0 ? d_j·l_j : d_j·u_j)
```

with `d = c − A'y`, is a valid lower bound over the node. A poor `y` gives a weak
bound; it never gives a wrong one. That is what lets an unconverged GPU iterate
drive an exact search. `tests/test_solvers.py` asserts the property over 300
random dual vectors, and asserts that the failure mode on an unbounded column is
`-inf` (vacuous) rather than a finite invalid number.

Every scale factor is rounded to a **power of two**, so converting a bound out of
the scaled space is exact in binary floating point and contributes no error to a
number that must stay valid.

### A valid bound is not a valid search

The argument above is sound, and it was verified rather than assumed: measured
against the exact simplex, the batched safe bound gave **0 invalid bounds in 40
root relaxations**. It is also narrower than it looks, and the gap is where this
solver's worst bug lived.

A rigorous bound makes *pruning* safe. It says nothing about the other decision
a tree makes on the same iterate — **when a node is finished**. That one needs
the point to be the node's optimum, which is precisely what an unconverged
first-order iterate is not. `node_solver="bnr"` closed nodes on integral-looking
iterates and reported `OPTIMAL` with a wrong objective and a dual bound
agreeing with it. Four separate routes into that mistake are recorded under
"Bugs worth recording" below.

So the correct reading of Neumaier–Shcherbina here is narrow: an approximate
`y` can be trusted to *discard* a subtree, and never to *conclude* one.

---

## Numerical robustness

The problem statement asks specifically for ill-conditioned and degenerate
models. Measured against an **exact rational solve** on a matrix with
`cond₁ = 1.7e12`:

| | forward error | componentwise backward error |
|---|---|---|
| plain LU | 4.27e-13 | 4.46e-12 |
| **+ iterative refinement** | **0.0** (bit-exact) | **1.16e-16** |

The reason refinement works here and fails in naive implementations: on that
matrix the residual `b − A@x` computed in plain float64 is **42% wrong** from
cancellation. The compensated residual (Dekker/Knuth error-free transformations)
is accurate to 3.9e-19 — **10¹⁸× better**. Refining against a garbage residual
converges confidently to the wrong answer.

Curtis–Reid scaling on the refinery blending template: coefficient ratio
**6.9e+08 → 2.4e+02**.

---

## Where the GPU wins, and where it does not

Measured, not assumed:

| nonzeros | GPU vs CPU |
|---|---|
| 12k | **0.14×** (GPU loses) |
| 240k | 1.5× |
| 960k | 3.1× |

Below roughly 10⁵ nonzeros, kernel launch and transfer cost dominate and the CPU
path is faster — the solver dispatches on size rather than pretending otherwise.

The inner loop was latency-bound before optimisation: at 516 it/s the actual
arithmetic accounted for under a tenth of the time, the rest being device
synchronisation from `float(dot(...))` in the step-size search. Fusing the
elementwise work into three custom kernels and re-validating the step size only
periodically removed those stalls.

**Not** ported to the GPU, deliberately: `FTRAN`/`BTRAN` through a sparse LU
(hypersparse, sequential, latency-bound), Markowitz pivoting (sequential pivot
search), and tree node management (irregular, pointer-chasing).

---

## Bugs worth recording

Not one of these was found by a unit test passing. Each was surfaced by
something outside the solver's agreement with itself — the independent
verifier, a published value, the runtime objecting to an undefined cast, or
brute-force enumeration over a model small enough to check exhaustively. All
five now have regression tests.

1. **Dual unscaling multiplied by the objective scale where it must divide.**
   Every dual was wrong by `obj_scale²`. Residuals still looked converged and
   the sign conditions still held, so the only symptom was a duality gap pinned
   at a constant: flugpl burned 30,000 iterations instead of 1,920. Marginal
   prices given to a planner would have been silently wrong.

2. **Termination used a purely relative feasibility test.** On mas76
   (`‖rhs‖∞ = 1.6e5`) it permitted an absolute row violation of 1.6e-03 — a
   thousand times looser than the verifier's 1e-6 — so the solver reported
   OPTIMAL for points its own checker rejected. Feasibility is now judged
   absolutely, optimality relatively.

3. **The phase-1 ratio test computed a breakpoint for a basic variable already
   outside its box and moving further out.** That ratio is negative, clamping
   the step to zero and stalling phase 1 at a non-optimal point — so the
   feasible instance misc07 was reported **INFEASIBLE**. Such a variable has no
   breakpoint at all: its infeasibility grows at a constant rate the phase-1
   objective already accounts for.

4. **A symmetry colour key overflowed `int64`, and the exact verification step
   compares those keys.** They were built as
   `round(value / 1e-9).astype(int64)`, which needs `|value| / 1e-9` to fit in
   an `int64` — a ceiling of about 9.2e9 on the value itself. Everything past
   it cast to `INT64_MIN`: `±1e30`, and equally an ordinary big-M of 1e10 or
   1e12. All of them became **one key**. Inside colour refinement that would
   only cost wasted candidates, which the exact check then rejects; but
   `verify_permutation` compares the same keys, so it accepted a permutation
   swapping a lower bound of `-inf` onto a finite one. An unsound generator is
   the worst failure this module has, because its breaking constraint deletes
   real solutions and the solver then reports a worse optimum with complete
   confidence. It contradicted the only property the design promises — the
   search may be incomplete, it may never be unsound. The cast is gone; the key
   stays a float, and an infinity now compares equal to nothing but itself.

   Found by the `RuntimeWarning: invalid value encountered in cast` that NumPy
   had been emitting on every symmetry test run, which is worth its own note:
   the suite was green, and the warning was the only thing in the room saying
   otherwise.

5. **A node closed by a point that was never its optimum — the same mistake,
   four times.** This one is recorded as a family rather than an incident,
   because that is what it turned out to be.

   A branch-and-bound node is finished when its relaxation **optimum** is
   integral: an integral optimum is feasible, so nothing below it can be
   better. That reasoning is exact for an exact node LP, and it is worthless
   for a batched first-order iterate, which is not an optimum and not even
   necessarily feasible. `node_solver="bnr"` closed nodes on it anyway. Four
   distinct routes to the same wrong conclusion turned up, each hidden behind
   the last:

   * The iterate started **outside the node's box** — zero is outside it as
     soon as a column has a non-zero lower bound. Branching on `x_j = 0` where
     `lo_j = 57` gives a down-child with `ub = 0 < lb`, empty by construction.
     Both children die, the node dies, and flugpl was reported **INFEASIBLE**
     against a published optimum of 1201500.
   * The iterate looked integral but was LP-infeasible, so `_accept` rejected
     it and there was no fractional variable left to branch on. The node fell
     through and was dropped.
   * `_accept` **rounds before it validates**, so on a fractional vertex it
     often succeeds and returns a perfectly good incumbent — which says
     nothing about whether the node is done. Reading that success as "node
     finished" dropped precisely the nodes whose rounding worked.
   * The node's exact re-solve returned neither `OPTIMAL` nor `INFEASIBLE`,
     which is not knowledge about the node, and it was dropped regardless.

   Every route ends the same way. A dropped node takes its subtree with it,
   the frontier empties, and an empty frontier is read as proof that the search
   is exhausted — so the run reports `OPTIMAL` with `dual_bound == incumbent`.
   The failure is not a crash and not an obviously wrong `INFEASIBLE`: it is a
   plausible near-optimal number with a dual bound agreeing with it, because
   the node that would have refuted the bound was never opened. A
   five-variable model came back `OPTIMAL` at −134 against a true −136, in one
   node.

   A fifth, unrelated in mechanism but found in the same hunt: the **MIR
   un-shift constant had both signs inverted**, so a cut removed the integer
   optimum outright. Multiplying by `lo_j = 0` hides it completely, which is
   why 133 brute-forced cuts over 55 generated instances passed while the code
   was wrong.

   **What the bound had to do with it: nothing.** The natural suspicion is the
   Neumaier–Shcherbina correction above, and it is wrong. Tested in isolation
   against the exact simplex, the batched safe bound produced **0 invalid
   bounds in 40 root relaxations**. The bound was valid every time; the tree
   consuming it was not. Separately, the gap rule stops as soon as every
   remaining node is prunable — which is exactly when the frontier's best bound
   has risen *past* the incumbent — and the reported `dual_bound` was read off
   that non-empty frontier, so a run could report a bound above the optimum it
   had just proved. A dual bound is now clamped to the incumbent, which it can
   never legitimately exceed.

   Caught by brute force, and by nothing else: the suite was green through all
   of it. The models are small enough to enumerate every integer point, which
   is what makes the oracle possible. Measured after: **0 failures over 450
   models × 2 node solvers**, where one seed's 150 previously gave 41. The
   throughput cost of re-solving those nodes exactly was nil on the three
   instances tried — identical node counts on a 22-item knapsack, flugpl and
   gr4x6 — because the re-solve only fires when an iterate looks integral,
   which is rare.

---

## Known limits

- **Crossover is identification plus clean-up, not Megiddo's push.** It is
  built and it pays -- 61.9% fewer pivots over the LP set, and a basis in zero
  pivots on a model the simplex cannot solve at all (see [Status](#status)) --
  but the rigorous construction walks the optimal face to a vertex with a
  bounded number of steps, and this ranks candidates and lets the simplex
  finish. So there is no pivot bound, and two instances of eleven cost *more*
  pivots than a cold start. The clean-up is what makes the answer exact, so a
  poor identification is a cost and never an error.
- **Product-form update, not Forrest–Tomlin.** Fill grows linearly in the number
  of etas, forcing a refactorisation every 60 pivots. This is the main reason
  10teams takes 18k iterations and 12 s.
- **Conflict analysis only sees LP infeasibilities.** The tree propagates before
  solving a node, so nodes propagation can refute never reach the LP that would
  produce a dual ray. Measured: misc07 learns 42 clauses (3168 nodes -> 2848),
  while p0201, gr4x6 and gt2 analyse zero nodes. Propagation-based conflict
  analysis is the larger prize and needs the propagator to carry reasons.
- **Symmetry breaking is weak.** One inequality per generator rather than full
  lexicographic ordering, so it captures a fraction of what orbitopal fixing
  would. Sound, but nowhere near the `k!` the theory allows.
- **dcmulti's incumbent was lost and has come back; both ends are bisected,
  neither is explained.** It returned 202598.49 (7.7% off, unproven) at
  `50bf2f0`, nothing at all from `ac66b9c`, and 189263.2 (0.58% off,
  verifier-accepted) from `6d6e588` onward. The loss used to be attributed
  here to the dual simplex Devex pricing; **that attribution was wrong** -- the
  incumbent was already gone one commit earlier, at the root cut loop's time
  cap and rollback, and the pricing fix landed after it. The pricing fix is
  still worth what it was worth: a stalling relaxation from 106,400 iterations
  to 439, and the LP set from 5.6 s to 2.0 s. What neither end explains is why
  a cut-loop cap and a tree update should move a primal heuristic's luck at
  all, and that is the honest state of it -- the commits are known, the
  mechanism is not. dcmulti is still not proved inside the limit.
- **MIR cuts are decided per model, not set once for the set.** They are worth
  a factor of four on p0201 and cost gt2 its root closure outright, and no
  structural property separates the two: both sit under the small-basis
  exemption, both see MIR displace GMI cuts, and MIR is the *sparser* family in
  both. What does separate them is the root bound itself, and it is available
  before a single node is explored -- p0201 closes to 7185.00 with MIR against
  7054.62 without, gt2 to 21166.00 without against 21104.83 with. So the
  default (`mir_cuts=None`) runs the root cut loop both ways and keeps the
  stronger bound, splitting the same cut-time budget between the two attempts
  rather than doubling it; `True` or `False` forces one loop, and every
  solution reports which way it went in `info["root_mir_cuts"]`. Measured over
  the set at a 60 s limit:

  |                  | MIR off | MIR on  | decided per model |
  |---|---|---|---|
  | status OPTIMAL   | 6/11    | 5/11    | **6/11**   |
  | gt2              | 5.62s   | timeout | 6.40s      |
  | p0201            | 52.17s  | 11.82s  | 11.93s     |
  | shifted geomean  | 22.39s  | 23.90s  | **20.19s** |

  **What it costs:** a second root cut loop on every model, which is why gt2
  takes 6.40 s here against 5.62 s with MIR simply forced off. It buys p0201's
  40 s, and a geomean better than either fixed setting -- the point being that
  neither fixed setting can win both rows, and this is the first rule in this
  repository that does. It is still only a root-bound race: it says nothing
  about which family will pay off deeper in the tree, and a model whose root
  MIR helps but whose tree it slows would be chosen wrongly. No instance in
  this set behaves that way; that is evidence, not a proof.
- **`node_solver="bnr"` is the less-trusted path.** It is not the default —
  `"simplex"` is — and it is the only path that ever returned a wrong answer
  (bug 5 above). Those are fixed and pinned by a brute-force sweep, but the
  asymmetry is the honest summary: the exact node LP came through every sweep
  clean, and the batched path took four fixes to get there. It also now falls
  back to an exact re-solve whenever a node's iterate looks integral — free on
  the instances measured, but a crutch the original design did not have.
- **Cuts are separated at the root only** and are never rolled back when they
  fail to pay for themselves (see p0201 above). Local cuts in the tree and a
  cost-aware rollback are the next steps.
- **The two instances that fall furthest short get essentially no cuts, and the
  density filter is why.** Both sit above the small-basis threshold, and both
  generate GMI cuts that are near-dense, so the filter that exists to preserve
  hypersparsity rejects nearly all of them:

  | | rows | cols | support cap | candidate support min/median | admitted |
  |---|---|---|---|---|---|
  | qnet1   | 503 | 1541 | 45  | 712 / 946  | **0 of 40** |
  | 10teams | 230 | 2025 | 264 | 120 / 1750 | **1 of 40** |

  qnet1 therefore runs its entire tree on the uncut relaxation -- root gain
  0.0 -- and finishes 28.7% off. 10teams never finds a feasible point at all,
  with every heuristic failing (`fix_and_propagate` 0/2, `feasibility_jump`
  0/2, `feasibility_pump` 0/1) across 703 nodes.

  **This is not a mis-tuned constant.** qnet1's *sparsest* candidate touches
  712 of 1541 columns, so it fails even the looser `0.4n` branch of the cap at
  616. Admitting these cuts means admitting near-dense rows on a 503-row basis
  -- precisely the case the filter exists to prevent, and the one measured
  above as taking p0201 from 1619 nodes in 9 s to 63 nodes in 60 s. What is
  missing is a cut family that separates *sparsely* on these models, not a
  larger cap.
- **QP is convex-only, and first-order.** `sovopt.qp` solves a convex quadratic
  by proximal PDHG (Condat–Vũ), validated against six hand-derived optima
  including one whose answer is *not* a vertex — the case a simplex provably
  cannot reach. What it does not do: a non-convex `Q` (needs spatial
  branch-and-bound) and any MIQP (needs a QP at every node) are **refused**, not
  approximated. It returns no basis, so no ranging on a QP, and it reaches
  ~1e-8, not the simplex's 1e-12.
- **The fill-reducing ordering helps the interior point and not the simplex.**
  Reverse Cuthill-McKee is worth 20-45x on a banded KKT, but measured over the
  factorisations of four real simplex solves it produced **4-45% more** fill
  than the LU's own singleton-peeling order every time -- qnet1 +33%, mod010
  +25%, 10teams +45%. That order was designed for a basis that is already
  80-95% triangular, and it wins there. A simplex basis also changes at every
  refactorisation, so an ordering would have to be recomputed each time rather
  than once per solve, which is the opposite of the economics that make it pay
  in the interior point. Recorded in
  [`docs/NEGATIVE-RESULTS.md`](docs/NEGATIVE-RESULTS.md).
- **The interior-point method returns no basis and no certificate.** It solves
  all 11 instances to the published value at a 0.140 s shifted geomean, but it
  detects infeasibility by *stagnation* rather than by a Farkas certificate --
  so it reports `INFEASIBLE_OR_UNBOUNDED` where the simplex reports
  `INFEASIBLE`. That needs a homogeneous self-dual formulation, which this is
  not. The missing basis, and with it the ranging and the warm start, is now
  supplied by `--crossover`.
- **Presolve exists, is correct, and does not pay.** `--presolve` reduces
  fixed columns, singleton rows and redundant rows to a fixpoint and postsolves
  the primal, the duals and the reduced costs exactly -- objective identical to
  a direct solve on all eleven models, duality gap below 1e-10. It is **off by
  default because the aggregate is 0.96x**: 10teams goes from 3,698 pivots to
  3,698 after losing 225 columns, because a revised simplex already handles a
  fixed column almost for free. The reductions that would pay shrink the
  *basis*, and there are 36 row reductions across the whole set against 275
  column ones. Kept opt-in for the two cases it does help. Forcing rows are
  detected and not applied -- their dual needs an argument the other three do
  not. Full measurement in
  [`docs/NEGATIVE-RESULTS.md`](docs/NEGATIVE-RESULTS.md).
- **Netlib, Mittelmann and QPLIB are untouched.** Only 11 MIPLIB instances are
  held; Netlib needs an `emps` decompressor that is not written.
- **Scale is measured now, and bounded by three different things.** See
  [Scale](#scale) for the ladder. LP reaches 1.02M nonzeros and 102,400 columns
  on the first-order path and 15,360 square rows solved and verified; the
  interior point is bounded by LU fill for want of a fill-reducing ordering, and
  stops well short of the other two; MILP proves optimality to about 100
  binaries and finds no feasible point at all past about 400. Against the stated
  benchmark of "thousands to millions" of variables, the thousands are reached
  and the millions are not.
- **The branch-and-bound tree is single-threaded.** Nine kernels run in
  parallel; the search does not.
- GPU fp64 on a laptop RTX 3050 runs at 1/32 rate; a datacentre card changes the
  crossover point substantially.

## Scale

The problem statement names a benchmark of "thousands to millions" of variables,
and until this study every published figure here came from MIPLIB instances that
top out at 230x2025. `python -m bench.scale` replaces the claim with numbers.

The ladder is built from the refinery templates in `sovopt.models.refinery`
rather than from random sparse matrices, because *shape* decides an LP's
difficulty far more than dimension does: `blend` is wide and shallow,
quality-constrained; `plan` is square, multi-period and **deliberately
degenerate**, which is the case that punishes a simplex. Every point is checked
by the independent verifier, so a size that "solves" to an infeasible point is a
failure and not a time.

### LP, 120 s limit

| model | rows | cols | nnz | simplex | interior point | PDLP |
|---|---|---|---|---|---|---|
| blend k=1 | 130 | 400 | 4.0k | 0.48 s | **0.12 s** | 1.06 s |
| blend k=2 | 260 | 1,600 | 16k | **0.51 s** | 0.70 s | 4.28 s |
| blend k=4 | 520 | 6,400 | 64k | 2.19 s | 2.61 s | **1.67 s** |
| blend k=8 | 1,040 | 25,600 | 256k | 7.04 s | 26.4 s | **2.16 s** |
| blend k=16 | 2,080 | 102,400 | 1.02M | 102.5 s | timeout | **3.65 s** |
| plan k=1 | 240 | 240 | 1.4k | 0.14 s | **0.01 s** | 1.86 s |
| plan k=2 | 960 | 960 | 9.9k | 1.76 s | **0.07 s** | 4.11 s |
| plan k=4 | 3,840 | 3,840 | 73k | 53.2 s | **1.59 s** | 7.49 s |
| plan k=8 | 15,360 | 15,360 | 564k | timeout | 81.5 s | **45.6 s** |
| plan k=16 | 61,440 | 61,440 | 4.42M | timeout | *did not finish* | timeout |

The interior-point column is measured **with the fill-reducing ordering
described below**; before it, `plan k=4` took 14.9 s, `plan k=8` failed
outright, and `blend k=8` took 36.0 s.

**The largest LP solved and verified is 2,080 x 102,400 with 1.02M nonzeros, in
3.65 s; the largest square one is 15,360 x 15,360 with 564k nonzeros in 45.6 s.
Both by PDLP.** That reaches the "thousands" the benchmark names and passes a
million nonzeros. It is not millions of variables.

Four things the table says that an aggregate would hide:

* **The engines swap places, and the crossover is a shape rather than a size.**
  The interior point wins the degenerate `plan` model by 9x at k=2 and 3.6x at
  k=4 -- the degeneracy advantage the theory predicts, showing up on the model
  built to be degenerate -- and loses `blend` badly at every size above 1,600
  columns. PDLP wins everything past about 25k columns. This is why `--method
  auto` still chooses between the simplex and PDLP by size only: the rule that
  would pick the interior point needs a degeneracy estimate, and there isn't
  one.
* **A fill-reducing ordering moved the interior point's ceiling by 4x, and
  `plan k=16` still defeats all three.** The KKT pattern is identical at every
  iteration, so an ordering is chosen once per solve and reused. Reverse
  Cuthill-McKee on the `plan` KKT cuts fill from 10-15x to 2.3-3.4x and the
  factorisation from 1.5-3.4 s to 0.04-0.08 s -- a 20-45x speedup, and 97% of
  an interior-point solve is that one factorisation repeated. `plan k=8` went
  from failing to `OPTIMAL` in 81.5 s, and the largest model the interior point
  solves went from 3,840 rows to 15,360. At 61,440 x 61,440 with 4.42M
  nonzeros it still does not finish, so approximate minimum degree remains the
  next step -- see [`numerics/ordering.py`](src/sovopt/numerics/ordering.py)
  for why RCM was measured first and AMD deliberately not written yet.
* **A time limit does not bound the interior point.** The clock is checked
  between iterations and, now, before each factorisation -- but a factorisation
  already running cannot be interrupted, and a 120 s limit was measured
  overrunning to 210 s on `plan k=8`.
* **A timed-out solve no longer hands back a point the verifier rejects.**
  This table is what found it: at `plan k=8` and `k=16` the simplex returned
  `TIME_LIMIT` with an objective of 0 and a phase-1 iterate that had never
  reached feasibility, and `Status.TIME_LIMIT` reports `has_solution`, so a
  caller was invited to use it. Both engines now withhold `x` when the point
  misses the feasibility cap **on the unscaled model** -- the yardstick the
  independent verifier uses -- so those rows report no solution instead of a
  wrong one. A phase-2 timeout is feasible and keeps its point, which is the
  case worth reporting; `INFEASIBLE` and `UNBOUNDED` keep their last iterate
  too, since they already report `has_solution` false and the iterate is worth
  having for diagnostics.

### MILP, 120 s limit

| model | rows | cols | binaries | status | gap | nodes | time |
|---|---|---|---|---|---|---|---|
| sched k=1 | 70 | 48 | 24 | OPTIMAL | 0 | 20 | 1.32 s |
| sched k=2 | 284 | 192 | 96 | OPTIMAL | 5.5e-05 | 1,565 | 45.6 s |
| sched k=4 | 1,144 | 768 | 384 | TIME_LIMIT | 1.2% | 1,087 | 131 s |
| sched k=8 | 4,592 | 3,072 | 1,536 | TIME_LIMIT | no incumbent | 447 | 122 s |

**MILP proves optimality to about 100 binaries, returns a plan within 1.2% to
about 400, and finds nothing at all at 1,536.** `unit_scheduling` is big-M with
minimum up-time, which is the weak-relaxation formulation refinery scheduling
actually uses, so this is a fair test rather than a flattering one -- but it is
one model family. The honest reading is that MILP scale here is bounded by the
**primal heuristics and not by node throughput**: at k=8 the tree explored 447
nodes and every heuristic failed, which is the same failure mode as 10teams on
the MIPLIB set.

## Measurement conditions

Every wall-clock figure in this README came from one machine:

| | |
|---|---|
| CPU | Intel Core i5-12450H, 8 physical / 12 logical cores, 2.0 GHz base |
| RAM | 15.7 GB |
| GPU | NVIDIA GeForce RTX 3050 Laptop, compute 8.6, 4 GB (fp64 at 1/32 rate) |
| OS | Windows 11 |
| CUDA | driver 12.4 (552.27), runtime 12.9 from the `[ctk]` wheels; no system toolkit |
| Python | 3.12.5, numpy 2.2.6, numba 0.67.0 (12 threads), cupy 14.2.0 |

A laptop, thermally throttled, with no attempt to pin clocks or quiet the
machine — background load alone has produced 2.3x swings on repeated runs of
the same instance. **Treat every second in this README as one draw on that
hardware, and prefer the ratios.** The correctness columns are a different
matter: objectives and relative errors reproduce bit-for-bit across runs and
across machines, because the algorithms are deterministic.

Everything is re-runnable rather than recorded: `bench.harness` regenerates the
LP and MILP tables, `bench.gpu_bench` the kernel tables, `bench.comparator` the
HiGHS comparison. Where a table names a synthetic instance, the README states
its generator, size, seed and settings, so the instance can be rebuilt. That
convention exists because two published tables were found not to reproduce; see
[`docs/NEGATIVE-RESULTS.md`](docs/NEGATIVE-RESULTS.md).

## Layout

```
src/sovopt/
  core/       sparse structures, JIT shim, backend + CUDA kernels, problem types
  io/         MPS reader and writer
  numerics/   scaling, LU, fill-reducing ordering, hypersparse solves, refinement
  lp/         revised simplex, basis, interior point, crossover, first-order LP
  presolve.py reductions and the postsolve stack
  mip/        safe bounds, batched node relaxation, propagation, tree
  models/     refinery templates
bench/        fetch, harness, verifier, GPU benchmark
tests/        277 tests including regressions for every bug above
ui/           local single-page interface
```
