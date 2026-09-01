# VYUHA

**A sovereign GPU-accelerated optimization engine — LP / MILP, built from mathematical foundations.**

SIH 2026 · Problem Statement 26119 · Mangalore Refinery and Petrochemicals Ltd

No solver library is used, linked, or vendored. Every algorithm here is
implemented from the published mathematics, with a citation header on each
module naming its source. See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) and the
clean-room policy in [`CLAUDE.md`](CLAUDE.md).

---

## Status

Working end to end, with **two independent LP engines** -- an exact revised
simplex and a GPU first-order method -- validated against **published MIPLIB
reference values** and an **independent verifier** that recomputes feasibility
from the original model.

```
VYUHA benchmark  mode=lp  device=cpu  time-limit=90s  tol=1e-8

instance         rows   cols      nnz status            objective        reference    relerr     time  chk
10teams           230   2025    12150 OPTIMAL           917.00001              nan       nan    0.29s   ok
dcmulti           290    548     1315 OPTIMAL           183975.54        183975.54  5.61e-11    2.61s   ok
flugpl             18     18       46 OPTIMAL           1167185.7        1167185.7  4.29e-09    0.32s   ok
gr4x6              34     48       96 OPTIMAL              185.55              nan       nan    0.27s   ok
gt2                29    188      376 OPTIMAL           13460.233        13460.233  3.07e-11    0.04s   ok
khb05250          101   1350     2700 OPTIMAL            95919464         95919464  7.60e-13    6.62s   ok
mas76              12    151     1640 OPTIMAL           38893.904              nan       nan   15.87s   ok
misc07            212    260     8619 OPTIMAL                1415              nan       nan    0.41s   ok
mod010            146   2655    11203 OPTIMAL           6532.0833          6532.08  5.08e-07    1.50s   ok
p0201             133    201     1923 OPTIMAL                6875             6875  1.11e-11    0.21s   ok
qnet1             503   1541     4622 OPTIMAL           14274.103        14274.103  1.12e-10    6.88s   ok

  status OPTIMAL       11/11
  verifier accepted    11/11
  within 1e-4 of ref   7/7
  shifted geomean time 1.592s (shift 1s)
  worst relative error 5.08e-07 (mod010)
```

The revised simplex solves the same set to the same published values, exactly,
and considerably faster:

| engine | optimal | verified | shifted geomean | total | worst rel. err |
|---|---|---|---|---|---|
| first-order (PDLP) | 11/11 | 11/11 | 1.592 s | 35.0 s | 5.08e-07 |
| **revised simplex** | 11/11 | 11/11 | **0.513 s** | **14.3 s** | 5.10e-07 |

Per-instance the gap is much wider than the aggregate suggests -- simplex is
794x faster on mas76, 132x on khb05250, 14x on qnet1 -- while PDLP wins
decisively on 10teams (0.29 s against 11.8 s), which is highly degenerate. They
are genuinely complementary, and `method="auto"` picks by size.

MILP is exact where it closes. With exact node LPs the tree is sharp: a 22-item
knapsack closes in **59 nodes** where the batched first-order bound needed
**40,211**. On the full MIPLIB set at a 60 s limit, before and after adding
cutting planes and primal heuristics:

```
instance   before                     after                        after time
flugpl     OPTIMAL     1201500        OPTIMAL     1201500              9.04s
gr4x6      OPTIMAL      202.35        OPTIMAL      202.35              0.55s
gt2        TIME_LIMIT   (none)        OPTIMAL       21166              1.71s   <-
khb05250   TIME_LIMIT   (none)        OPTIMAL  1.0694023e+08           7.70s   <-
mod010     OPTIMAL        6548        OPTIMAL        6548              2.52s
p0201      OPTIMAL        7615        OPTIMAL        7615             39.65s
mas76      TIME_LIMIT 40589.44        TIME_LIMIT 40408.48             60.33s
misc07     TIME_LIMIT     2810        TIME_LIMIT     2810             60.05s
dcmulti    TIME_LIMIT   (none)        TIME_LIMIT 202598.49  (7.7% off)   60.41s
qnet1      TIME_LIMIT   (none)        TIME_LIMIT  20627.76 (28.7% off)   60.86s
10teams    TIME_LIMIT   (none)        TIME_LIMIT   (none)             62.36s

                        before    after
  status OPTIMAL         4/11      6/11
  verifier accepted      6/11     10/11
  no incumbent at all    5/11      1/11     <- the motivating number
  shifted geomean       31.4s     22.7s
```

Every proved optimum matches the published value exactly, and the verifier
rejects nothing. The motivating failure -- five instances returning *no answer
at all* -- is down to one.

**The cost, stated plainly:** p0201 went from 8.16 s to 39.65 s and flugpl from
4.32 s to 9.04 s, because cuts made their node LPs more expensive than the
bound they bought. flugpl is still comfortably solved. **p0201 is now marginal
at a 60 s limit** -- it came out OPTIMAL at 39.65 s in the run above and
TIME_LIMIT at 60.5 s on a repeat, finding the optimum 7615 both times but only
proving it in one. Treat the 39.65 s as one draw from a borderline instance,
not a reliable figure.

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

```bash
python -m vyuha.cli devices                      # what hardware is usable
python -m vyuha.cli info   model.mps             # stats + numerical health
python -m vyuha.cli solve  model.mps --device gpu --out sol.json
python -m vyuha.cli solve  model.mps --sensitivity     # shadow prices + ranging
python -m vyuha.cli verify model.mps sol.json    # independent check
python -m ui.server                              # http://127.0.0.1:8000
```

```bash
python -m bench.fetch --set small     # download MIPLIB instances
python -m bench.harness --mode lp                  # validate against published values
python -m bench.harness --mode lp --method simplex  # force one engine
python -m bench.gpu_bench             # CPU vs GPU
python -m pytest tests/               # 105 tests
```

---

## What is actually built

| Layer | Module | Status |
|---|---|---|
| Sparse core, dual CSR/CSC | `core/sparse.py` | done |
| MPS reader / writer | `io/mps.py` | done (free + fixed, RANGES, MARKER, QUADOBJ) |
| Scaling: Ruiz, Curtis–Reid, Pock–Chambolle | `numerics/scaling.py` | done |
| Sparse LU, threshold Markowitz + Gilbert–Peierls | `numerics/lu.py` | done |
| Hypersparse FTRAN / BTRAN | `numerics/lu.py` | done |
| Iterative refinement, compensated residual | `numerics/refine.py` | done |
| Condition estimation (Hager) | `numerics/refine.py` | done |
| **Revised simplex**, primal + dual, bounded | `lp/simplex.py` | done |
| Basis, product-form update, singularity repair | `lp/basis.py` | done |
| First-order LP (PDLP-class), CPU + CUDA | `lp/pdlp.py` | done |
| Hand-written CUDA kernels | `core/backend.py` | done |
| Safe dual bounds (Neumaier–Shcherbina) | `mip/safebound.py` | done |
| **Batched Node Relaxation** | `mip/bnr.py` | done |
| Domain propagation | `mip/propagate.py` | done |
| Branch and bound, exact or batched node LPs | `mip/tree.py` | done |
| Gomory mixed-integer + knapsack cover cuts | `mip/cuts.py` | done |
| Symmetry detection + static breaking | `mip/symmetry.py` | done |
| Conflict analysis (LP-infeasibility clauses) | `mip/conflict.py` | done |
| Sensitivity: shadow prices, cost and RHS ranging | `lp/sensitivity.py` | done |
| Feasibility Jump, fix-and-propagate, feasibility pump | `mip/heuristics.py` | done |
| Refinery model templates | `models/refinery.py` | done |
| CLI, web UI, verifier, harness | `cli.py`, `ui/`, `bench/` | done |
| Crossover, sensitivity ranging, IPM, QP | — | **not built** (roadmap) |

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

What it unlocks, measured on a 22-item knapsack:

| node bound | nodes | time |
|---|---|---|
| batched first-order (BNR) | 40,211 | 22.6 s |
| **exact LP, dual simplex warm start** | **59** | **1.5 s** |

That is the honest verdict on the batched idea: valid-but-loose bounds are cheap
and parallel, but bound *quality* dominates tree size. BNR remains the right
tool when nodes are large enough that an exact solve is unaffordable; the tree
takes `node_solver="simplex"` or `"bnr"`.

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

VYUHA pops a *slab* of nodes and bounds them all in one sparse-times-dense
product. Measured on a 20,000 × 30,000 matrix with 240k nonzeros:

| K | K × SpMV | SpMM | speedup | GFLOP/s |
|---|---|---|---|---|
| 8 | 0.81 ms | 0.21 ms | 3.9× | 18.2 |
| 32 | 3.19 ms | 0.36 ms | **8.8×** | 42.4 |
| 64 | 6.37 ms | 0.72 ms | **8.8×** | 42.4 |
| 128 | 10.73 ms | 1.45 ms | 7.4× | 42.3 |
| 256 | 19.59 ms | 4.61 ms | 4.3× | 26.6 |

42 GFLOP/s sustained in fp64 on a card whose fp64 *peak* is ~86 — about half of
peak, from a sparse kernel. Combined with the 3.4× raw SpMV advantage over the
CPU, batched bounding costs roughly **11 µs per node** against 410 µs for
one-at-a-time CPU bounding.

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

## Two bugs worth recording

Both were caught by the independent verifier and by comparison against published
values — not by unit tests. Both now have regression tests.

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

---

## Known limits

- **No sensitivity ranging.** The basis is there and the duals are exact, so
  objective and RHS ranging is now a short step — but it is not written.
- **No crossover** from a first-order point to a basis, so the PDLP path still
  cannot hand over to the simplex on large models. The two engines are chosen
  between, not composed.
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
- **Cuts are separated at the root only** and are never rolled back when they
  fail to pay for themselves (see p0201 above). Local cuts in the tree and a
  cost-aware rollback are the next steps.
- **No IPM and no QP solve path** (QP models parse, but there is no quadratic
  solver behind them).
- **No global/bilinear (pooling) solver.** The blending template uses the linear
  blending assumption; genuine crude-blending non-convexity is not addressed.
- GPU fp64 on a laptop RTX 3050 runs at 1/32 rate; a datacentre card changes the
  crossover point substantially.

## Layout

```
src/vyuha/
  core/       sparse structures, JIT shim, backend + CUDA kernels, problem types
  io/         MPS reader and writer
  numerics/   scaling, LU, hypersparse solves, refinement
  lp/         revised simplex, basis, first-order LP
  mip/        safe bounds, batched node relaxation, propagation, tree
  models/     refinery templates
bench/        fetch, harness, verifier, GPU benchmark
tests/        105 tests including regressions for every bug above
ui/           local single-page interface
```
