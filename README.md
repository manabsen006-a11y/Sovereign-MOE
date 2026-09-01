# VYUHA

**A sovereign GPU-accelerated optimization engine — LP / MILP, built from mathematical foundations.**

SIH 2026 · Problem Statement 26119 · Mangalore Refinery and Petrochemicals Ltd

No solver library is used, linked, or vendored. Every algorithm here is
implemented from the published mathematics, with a citation header on each
module naming its source. See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) and the
clean-room policy in [`CLAUDE.md`](CLAUDE.md).

---

## Status

Working end to end. Validated against **published MIPLIB reference values** and
an **independent verifier** that recomputes feasibility from the original model.

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

MILP is exact on tested instances (matches a brute-force DP optimum on knapsacks)
but the tree is young: the bound comes from a fixed iteration budget of a
first-order method, so node counts are high on models with weak relaxations.
That is the honest state and the next thing to improve.

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
python -m vyuha.cli verify model.mps sol.json    # independent check
python -m ui.server                              # http://127.0.0.1:8000
```

```bash
python -m bench.fetch --set small     # download MIPLIB instances
python -m bench.harness --mode lp     # validate against published values
python -m bench.gpu_bench             # CPU vs GPU
python -m pytest tests/               # 23 tests
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
| First-order LP (PDLP-class), CPU + CUDA | `lp/pdlp.py` | done |
| Hand-written CUDA kernels | `core/backend.py` | done |
| Safe dual bounds (Neumaier–Shcherbina) | `mip/safebound.py` | done |
| **Batched Node Relaxation** | `mip/bnr.py` | done |
| Domain propagation | `mip/propagate.py` | done |
| Branch and bound | `mip/tree.py` | working, early |
| Refinery model templates | `models/refinery.py` | done |
| CLI, web UI, verifier, harness | `cli.py`, `ui/`, `bench/` | done |
| Revised simplex, crossover, cuts, IPM | — | **not built** (roadmap) |

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

---

## Known limits

- **No revised simplex yet.** LP accuracy is capped at what a first-order method
  reaches; there is no basis, so no sensitivity ranging and no exact crossover.
  This is the largest gap and the next major piece.
- **MILP node counts are high** on weak-relaxation models. The bound comes from
  a fixed 80-iteration budget; tightening it costs time per node, loosening it
  costs nodes. Cuts and better branching are unbuilt.
- **No cuts, no conflict analysis, no symmetry handling, no IPM, no QP solve
  path** (QP models parse, but there is no quadratic solver behind them).
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
  lp/         first-order LP
  mip/        safe bounds, batched node relaxation, propagation, tree
  models/     refinery templates
bench/        fetch, harness, verifier, GPU benchmark
tests/        23 tests including regressions for both bugs above
ui/           local single-page interface
```
