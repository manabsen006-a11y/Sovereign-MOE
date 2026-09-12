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

## A parallel branch-and-bound tree (built, measured, reverted -- then superseded)

Known limits said the tree is single-threaded while nine kernels run in
parallel. The profile says that is where the time is: **node LP solves are
86-97% of MIP time** (misc07 91%, qnet1 97%, 10teams 93%, p0201 88%, gt2 86%),
and every JIT kernel in the project is compiled `nogil=True`, so on paper
threads should run them concurrently.

**The tree's shape makes it look easy.** It already pops a *slab* of up to 64
nodes and decides what to explore before solving any of them, so the slab is
fixed, each node's LP is independent, and results can be applied in slab order.
Determinism is preserved by construction rather than by luck -- and that was
confirmed: at 1, 4 and 8 threads gt2 and p0201 returned identical status,
objective **and node count**.

**It is slower at every thread count.**

| | 1 thread | 4 | 8 |
|---|---|---|---|
| gt2 | 14.45 s | 17.04 s | 18.14 s |
| p0201 | 24.48 s | 28.24 s | 32.91 s |
| misc07 (nodes in 60 s) | 3,046 | 2,601 | 2,475 |

misc07 explores 20% *fewer* nodes in the same wall clock at 8 threads.

**Why, isolated from the tree entirely.** Sixteen independent node LPs, each
with its own `NodeSolver` so nothing is shared, solved sequentially against
solved on eight threads:

    sequential 6.440s     8 threads 10.076s     0.64x

Not "no gain" -- a loss, on the cleanest possible version of the experiment.
The reason is that `nogil` kernels are not the same thing as a `nogil`
*algorithm*: the revised simplex's iteration loop is Python, calling short
kernels for pricing, the ratio test and the solves. The GIL is held for the
Python between them, which is most of the wall clock, so the threads serialise
and pay coordination costs on top.

**Reverted rather than shipped dormant.** The implementation defaulted to one
thread and so changed nothing, but it put a thread pool, a work queue and a
per-thread solver pool into the hottest and most correctness-critical loop in
the project, in exchange for a path that provably cannot pay. The finding is
worth more than the code.

**What would have to change first.** The simplex iteration loop itself would
have to move inside a `nogil` kernel, so that a node solve is one long
GIL-free call rather than a Python loop around many short ones. That is a
rewrite of `lp/simplex.py`, not a threading change, and until it happens
parallelising the tree is measuring the GIL.

**Superseded.** That rewrite is `lp/nodelp.py`: the node-LP dual loop as one
compiled call, pinned pivot for pivot to the Python loop. With it the same
slab-parallel tree pays -- p0201 20.9 s to 4.7 s and dcmulti 55 s to 25 s on
four threads, node counts identical -- and the single-threaded kernel alone
moved the MIPLIB set from 6/11 to 7/11. The entry stays as the record of why
the first attempt could not have worked.

---

## Forrest-Tomlin basis update (built, measured, kept opt-in)

**Idea.** The product-form update appends an eta column per pivot and every
FTRAN and BTRAN pays for the whole eta file, so its cost grows with the pivots
since the last refactorisation and the refactorisation budget has to stay
short. Forrest-Tomlin updates `U` itself: the entering column becomes a spike
in `U`, the one row that breaks triangularity is eliminated against the rows
beneath it, and the multipliers are a *row* eta as short as that row. A solve
after a thousand pivots then costs what it cost after none, and the budget
can be an order of magnitude longer. `numerics/ft.py` implements it with `U`
row-wise in a row file with logical permutations (Suhl & Suhl), a growth cap
on the multipliers, and a rule that a fresh factor never refuses -- without
which woodw livelocked at 27,167 refusals in 27,360 pivots, one
refactorisation each.

**It works, and it is verified.** The dense identity `L R⁻¹ Ũ = P B Q̃` is
checked outright after every update on small cases; hundreds of random
column replacements -- an adversarial sequence, the trailing matrix goes
ill-conditioned fast -- keep the backward error below 1e-9 with the cap
refactorising a handful of times; and the simplex reaches the same optimum
under both updates on every instance tried.

**It does not pay here.** Solve cost per pivot, product form (PFI) against
Forrest-Tomlin (FT), across refactorisation budgets:

| set | PFI @150 | FT @150 | PFI @1000 | FT @1000 |
|---|---|---|---|---|
| Netlib woodw, 25fv47, pilot87, d2q06c | **2.44 ms**, 63.6 s | 2.44 ms, 79.0 s | 5.2 ms, 159.5 s | **2.13 ms**, 66.7 s |
| MIPLIB LP relaxations (11) | **0.87 s** | 1.34 s | 2.86 s | 2.92 s |
| dfl001 (6,071 rows, 300 s limit) | 11.7 ms | — | — | 11.3 ms (10.8 ms @400) |
| plan k=4 (3,840 rows) | **11.6 s** | — | — | 15.5 s (12.5 s @400) |

At a long budget FT is exactly what it promises -- 2.4× cheaper per pivot
than PFI at the same budget on the big four, and the factorisation count
falls from 177 to 82. But PFI at *its* best budget is already at 2.44 ms,
which is where the refactorisation-budget commit had put it, and an FT path
takes more pivots on most instances: +25% on the big four, +28% on the MIPLIB
set at 150, 2.2× on 10teams and woodw. The solves are accurate -- 2e-12
backward error against the product form's 1e-13 -- but a degenerate dual
simplex takes a different path on the last two digits, and the different
path is longer more often than it is shorter. Net: a wash on the large
instances and a loss on the small ones, where FT's per-pivot overhead (one
extra `L` solve for the spike and an `O(m)` position shift) is what shows.

**Why the layout mattered, and still does.** The first version kept `U`'s
rows as linked lists and the solves ran 2.9× slower than the column LU's on
the same matrix (88 µs vs 31 µs on woodw's basis); the row file brought that
to 1.7× (54 µs) and no further, because each entry still pays an `alive`
check and two indirections through the position arrays. The column-oriented
LU solve the product form uses is the fastest thing in the code, and FT
cannot use it.

**Kept, not reverted.** `SimplexParams(basis_update="ft")`, off by default.
What would flip it: a simplex whose per-pivot cost is dominated by the basis
solves rather than by pricing -- dfl001 spends 11 ms per pivot with the
solves a fraction of it -- together with hypersparse FTRAN/BTRAN, which the
product form cannot offer and FT's row file could.

---

## αBB for non-convex QP (built, measured, kept opt-in)

**Idea.** The direct way to make a non-convex quadratic solvable is to shift
its diagonal until it is convex and branch on the box: with
`d = u − l`, `α_i = max(0, −½(Q_ii − Σ_j |Q_ij| d_j/d_i))` makes
`Q + 2 diag α` positive semidefinite by the scaled Gershgorin theorem, and
`f + Σ α_j (x_j−l_j)(x_j−u_j)` is a convex underestimator on the box whose
gap is `≤ Σ α_j (u_j−l_j)²/4`. Solve the convex QP at each node, prune on the
certified bound, split the widest term. It was built first because it needs
nothing but the existing convex QP solver and no product variables, and its
bound is unconditional -- the convexity of the relaxation is a proof.

**It loses to the McCormick reformulation by a wide margin on dense Q.**
Same instances, same 120 s limit, gap 1e-4:

| n | αBB | McCormick |
|---|---|---|
| 10 | 39 / 103 nodes, 8.0 / 11.7 s | 7 / 15 nodes, 0.1 / 0.2 s |
| 15 | **time limit**, gap 10.7% / 12.0% | 81 / 205 nodes, 1.1 / 9.5 s |
| 20 | time limit, gap 17.4% / 25.8% | 187 / 891 nodes, 22.9 / 93.5 s |
| 25 | time limit, gap 79% / 57% | time limit, gap 150% / 59% |

**Why.** A diagonal shift pays for every cross term `Q_ij x_i x_j` through
the diagonal of both variables, and the gap it introduces is quadratic in the
box width on every axis at once. An envelope treats each product exactly --
it is the convex hull of that one term -- so its error is confined to the
products that are actually loose at the node. On a dense `Q` every term is a
cross term, and that is the whole difference. The node solver compounds it:
a proximal QP takes ~150 iterations to converge on even a two-variable node,
where a warm-started simplex takes a handful of pivots.

**What was worth keeping from it.** The marginals-based box reduction
derived from the certified bound's own pieces took the two-variable smoke
test from 319 nodes to 7; the certified `safe_qp_bound` it prunes on is what
the convex MIQP tree now uses too; and the polish heuristic was carried over
to the spatial tree, where it turned out to matter more than it did here.

**Kept, not reverted.** `relaxation="alphabb"` on
`sovopt.globalopt.nonconvex_qp`, off by default. It needs no product
variables -- `n²` of them for a dense `Q` -- so there is a regime of large
sparse `Q` where it may yet win, and the table above cannot be regenerated
without it (`python -m bench.nonconvex_qp --relaxation alphabb`).

---

## Proximal-point regularisation of the interior point (built, measured, kept opt-in)

**Idea.** The interior point's KKT matrix is regularised -- `δp` on the
(1,1) block, `δd` on the (2,2) -- so that a free column or an equality row
cannot make it singular, and the perturbation is then refined away against
the unregularised matrix. On `QPLIB_8559` that refinement was recovering 15%
per round where it usually recovers everything, and the solve did 87
iterations in 600 s without converging. The known answer (Friedlander &
Orban; Pougkakiotis & Gondzio) is to make the regularisation *part of the
method*: each iteration solves the Newton system of the proximal problem --
the objective plus `ρ/2‖x − x_k‖²`, the constraints relaxed by `δ(y − y_k)`
-- with the centres at the current iterate, so the proximal terms vanish at
the linearisation point, the system is exactly the regularised one, and it
is solved as-is. The method converges to the original problem's solution as
`ρ, δ → 0`, and the accuracy of a step is no longer limited by what
refinement can recover. Built as `IPMParams(regularisation="pmm")`, with
`ρ = δ = clamp(μ, 1e-10, cap)`.

**It is worse at every strength.** Netlib, 89 problems, 60 s limit, the
interior point alone:

| regularisation | optimal | iterations | LDLᵀ fallbacks to LU |
|---|---|---|---|
| static, refined away (default) | **78/89** | 3,269 | 20 |
| proximal, cap 1e-8 | 77/89 | 3,437 | 29 |
| proximal, cap 1e-6 | 70/89 | 4,358 | 39 |
| proximal, cap 1e-4 | 53/89 | 5,626 | 31 |

The MIPLIB LP set tells the same story in miniature: at a cap of 1e-4
khb05250 comes back `INFEASIBLE_OR_UNBOUNDED`; at 1e-8 the iteration counts
are the static method's to the iterate. The mechanism is visible in the
failures, which are overwhelmingly `INFEASIBLE_OR_UNBOUNDED`: after a full
proximal step the primal residual is `δ·dy`, not zero, so on an instance
whose duals are large the residual plateaus while the dual side converges,
and a stall test that expects the primal residual to fall reads the plateau
as infeasibility. The variant built here re-centres every iteration, which
is Friedlander & Orban's exact-regularisation form; the Pougkakiotis-Gondzio
method lags the centres and moves them only when the residuals of the
original problem have fallen below a multiple of `μ`, with its own
termination and infeasibility tests built around that. Making it work would
mean adopting those tests too, not just the direction.

**The instance it was built for did not need it.** `QPLIB_8559` has `c = 0`
and a Hessian whose diagonal runs from 4 to 95,000. The objective scale was
taken from `c` alone -- the geometric mean of its nonzero magnitudes,
rounded to a power of two -- and the scale of nothing is 1, so the interior
point started with a dual residual of 1.7e5, drove the iterate onto its
bounds before that residual was gone, and then crawled with the primal step
length pinned near zero: the residual bouncing, the objective creeping, 87
iterations without converging. A quadratic objective's scale is the scale of
its gradient `c + Qx`, and the diagonal of the column-scaled `Q` is the
curvature along each scaled unit direction, so it now joins the geometric
mean. The same instance then converges in **15 iterations, 93 s**, to the
published value at 1.6e-9; `QPLIB_8567`, the same family with 7,500 rows,
in 11 iterations and 104 s at 4e-12. Neither the AMD ordering nor the LDLᵀ
built for this instance was wasted -- each factorisation is still 7 s
rather than hundreds -- but the wall they were built against was a scaling
error. What found it was the pair of trajectories: with either
regularisation the primal step was pinned from the first iteration, `μ`
reached 1e-9 with the primal residual still at 1e-4, which is an iterate
on the boundary and not a matrix solved badly, and the only thing wrong
with the iterate was the size of its gradient.

And with the scale fixed, the proximal form still loses on the instance it
was built for: 19 iterations and 162 s at a cap of 1e-8 against the static
method's 15 and 93 s, and at 1e-4 a time limit at 72 iterations with the gap
at 1e-2.

**Kept, not reverted.** `IPMParams(regularisation="pmm", pmm_cap=...)`,
off by default, so the table above can be regenerated.

---

## Around the LDLᵀ refusal: five rules that lost before one won

**The question.** When the symmetric LDLᵀ refuses an iterate (a pivot on
the wrong side, or too small), what should take that iterate? The rule
that shipped with the LDLᵀ -- the pivoting LU takes the rest of the solve
-- was measured to be the wall on dfl001: 2 s per LDLᵀ, 147 s per LU on
the same ordering, the time limit at 31 iterations. Every rule below was
run over Netlib plus the MIPLIB LP relaxations (100 instances, 60 s), and
the count of `OPTIMAL` is the score. The rule that shipped scores 88.

| rule | score | what it lost |
|---|---|---|
| shift both blocks (1e-6, 1e-4) and retry; refine 2 passes toward `K` | 88 | pilot: 159 shifted iterates, dual residual stuck at 2e-8 |
| same, refine 8 passes unconditionally | 89 | boeing2: eight passes refine toward a direction of norm 1e9, 37 iterations become 200 |
| same, refine while a pass halves the residual | 88 | pilot: the max-norm residual sits on a few huge right-hand-side entries and stalls while the direction still improves |
| shifted factor's solve checked by its last correction, LU if inaccurate | (not scored) | on the instances it was built for it lost both ways -- pilot: every shifted solve passes the check and the iteration still stalls; dfl001: every solve fails it, the LU is taken, the wall is back |
| the LU first whenever its fill is within 8x the LDLᵀ's | 89 | dfl001: the LU fits the cap at 10x the LDLᵀ's fill and 70x its time, and the wall is back |
| **the LU first if it fits 4x the fill or 2M nonzeros, else shift; refine while a pass gains 10x** | **92** | forplan, pilot4 -- and those to the unit below, not to this rule |

The last line is what shipped, and with it dfl001 solves (49-53 iterations,
107 s). What the table says about pilot: the instance is ill-conditioned
enough that it wants the pivoting LU's accuracy at the refused iterates and
eight refinement passes at the others, and a shifted LDLᵀ never recovers
once its iterates drift. What it says about boeing2: refining hard toward
the *unregularised* matrix is not always better, because that matrix is the
near-singular one, and a two-pass refinement was acting as a regulariser.
Both are recorded in the parameter docstrings of `lp/ipm.py`.

**A per-column unit was wrong; a uniform one was the biggest win.** The
unit (below) was first built per column -- each column divided by the
power of two nearest its own bound magnitude, then the equilibration run
on `A D`. On a column bounded by 1e12 among columns bounded by 1 (the
mas76 shape, a regression test) the equilibration was pulled to a fixed
point the solve could not use and the model came back
`INFEASIBLE_OR_UNBOUNDED`. Feeding the equilibration a diagonally
prescaled matrix changes which fixed point it converges to, and not
predictably. A uniform unit, applied to every column and every row, leaves
`A` untouched and the equilibration exactly what it was; it took the
interior point from 77 to 81 of 89 on Netlib and fixed the 9002 wall, at
the price of forplan and pilot4. See the README's Known limits.

---

## Presolve (built, correct, does not pay -- kept opt-in)

The README named presolve, with a Forrest-Tomlin update, as what was left to
close the 6.7x speed gap to HiGHS. It is now built, and on everything measured
here **it does not close any of it.**

**Counted before building.** Over the eleven benchmark models: 275 fixed
columns (10teams 225, khb05250 50), 20 forcing rows, 19 singleton rows
(dcmulti 18), 17 redundant rows, **1** free column singleton, and **0** empty
rows or columns. The textbook headline reduction -- substituting out a free
column singleton, which removes a row *and* a column -- was worth one column on
one model, so it was not written. What was written is the three that are
actually present and whose duals can be recovered exactly.

**Measured, simplex, whole LP set:**

| | before | after | direct | presolve+solve | speedup |
|---|---|---|---|---|---|
| 10teams  | 230x2025 | 215x1800 | 2.013 s | 2.043 s | 0.99x |
| dcmulti  | 290x548  | 272x548  | 0.213 s | 0.224 s | 0.95x |
| gt2      | 29x188   | 28x188   | 0.023 s | 0.019 s | **1.22x** |
| khb05250 | 101x1350 | 100x1299 | 0.051 s | 0.066 s | 0.76x |
| qnet1    | 503x1541 | 502x1541 | 0.541 s | 0.611 s | 0.89x |
| **total** | | | **3.599 s** | **3.738 s** | **0.96x** |

mod010, p0201, mas76, misc07 and gr4x6 reduce by nothing at all. The interior
point fares no better in aggregate (1.04x; khb05250 a real 2.12x, dcmulti a
real 0.70x loss), and neither do the refinery models (plan k=2 1.38x, plan k=4
0.77x on the *same* 20% row reduction).

**Why, and it is not that the implementation is slow.** Presolve costs 1-25% of
the total and never more than 0.03 s absolute; postsolve is 0.0004 s. The
reduced model simply is not easier: **10teams goes from 3,698 simplex pivots to
3,698** after losing 225 columns and 15 rows. A revised simplex already handles
a fixed column almost for free -- it never prices, never enters the basis, and
costs one column of the matrix -- so deleting 275 of them deletes work the
solver was not doing. The reductions that would pay are the ones that shrink
the *basis*, and those are the row reductions, of which there are 36 across the
whole set.

**Kept anyway, opt-in behind `--presolve`,** on the same grounds as
`mir_cuts`: it is correct, it is cheap, and the two instances it does help
(gt2 1.22x on the simplex, khb05250 2.12x on the interior point) are real. What
it must not be is on by default while the aggregate says 0.96x.

**What is correct about it, since that is the part that was hard.** Postsolve
recovers the primal, the duals *and* the reduced costs for the original model:
objective identical to a direct solve on all eleven, primal violation below
1e-13, and duality gap below 1e-10 -- with the reduced costs rebuilt from
`d = c - Aᵀy` at the end rather than tracked through the stack, which is a
definition and so cannot drift. Forcing rows are detected and deliberately not
applied: their dual needs an argument the other three do not, and getting it
wrong would corrupt shadow prices on the one model they fire on.

**One real bug, found by the tests rather than the benchmarks.** The
empty-column rule parks an unconstrained column at whichever bound the
objective wants, and took the minimisation branch regardless of sense -- on a
maximisation that parks it at the *worst* end and returns a feasible,
suboptimal plan that no feasibility check flags. Fixed, and pinned by a test.

---

## Fill-reducing ordering for the simplex basis (attempted, not adopted)

**Idea.** A fill-reducing ordering cut interior-point factorisation time by
20-45x (see `numerics/ordering.py`). The simplex factorises too, and far more
often -- once per refactorisation, every ~60 pivots -- so the same ordering
should pay there as well.

**It does not.** Measured over every basis factorisation of four real solves,
reverse Cuthill-McKee produced *more* fill than the LU's own singleton-peeling
order in every case:

| instance | basis | factorisations | default nnz(L+U) | RCM nnz(L+U) |
|---|---|---|---|---|
| qnet1    | 503 | 15 | 45,469  | 60,296  (+33%) |
| mod010   | 146 | 22 | 47,647  | 59,519  (+25%) |
| khb05250 | 101 |  3 | 783     | 816     (+4%)  |
| 10teams  | 230 | 40 | 302,698 | 439,755 (+45%) |

**Why, and why it is not a surprise in hindsight.** A simplex basis is 80-95%
triangular already, and Suhl-Suhl singleton peeling eliminates that part with no
fill at all before any ordering heuristic gets a say. RCM knows nothing about
triangularity; it optimises bandwidth, and reordering a nearly-triangular matrix
for bandwidth destroys the very structure the peeling was going to exploit.

**The economics differ too, independently of the fill.** The interior point
factorises the *same pattern* every iteration, so an ordering is computed once
and amortised over the whole solve. A simplex basis changes at every
refactorisation, so the ordering would be recomputed each time and charged to
each one. Even at break-even fill it would lose.

Recorded because the generalisation is the tempting part: "a fill-reducing
ordering helps sparse factorisation" is true in general and false here, and the
reason is that one of the two callers already has a better one.

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
