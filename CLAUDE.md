# SOVOPT — Sovereign Optimization Engine

SIH 2026 · PS 26119 · Mangalore Refinery and Petrochemicals Ltd (MRPL)

An LP / MILP / QP solver core built from mathematical foundations, with GPU
acceleration where it measurably pays.

---

## THE HARD RULE — clean room

The problem statement is explicit:

> *"It shall not be built upon any existing open source solver library but shall
> be built from scratch from mathematical foundation."*

Therefore, in this repository:

**Forbidden — never import, link, vendor, or read the source of:**
- Any solver: CBC, Clp, HiGHS, SCIP, GLPK, SoPlex, OR-Tools, cuOpt, lp_solve,
  Cbc/Cgl/Osi, BARON, Couenne, Ipopt, OSQP, ECOS, Clarabel.
- `scipy.optimize.*` — in particular `linprog`, `milp`, `minimize`. Not in the
  solve path, not in tests, not "just to check".
- Any LU / Cholesky *factorization* library: `scipy.sparse.linalg.splu`,
  SuperLU, UMFPACK, CHOLMOD, MUMPS, PARDISO, `scipy.linalg.lu`.

**Permitted:**
- `numpy` — dense array storage and elementwise/BLAS-level arithmetic only.
- `numba` — JIT compilation. A compiler, not an algorithm.
- `cupy` — GPU arrays and elementwise kernels. Vendor math, not a solver.
- `scipy.sparse` — **only** inside `bench/` and `tests/` for constructing or
  cross-checking test data. Never inside `src/sovopt/`.
- `scipy.optimize.linprog` — **only** inside `bench/comparator.py`, and only to
  run HiGHS as an external *comparator*. The problem statement requires that
  results be "compared against at least one established commercial or
  open-source solver", so a comparator is mandatory, not optional. It is
  quarantined to that one file, it never touches `src/sovopt/`, and
  `tools/check_provenance.py` still bans it everywhere the engine lives. Using a
  solver to *check* our numbers is the opposite of building on one.

**Every algorithm module carries a citation header** naming the paper or
textbook it implements. `tools/check_provenance.py` fails CI if one is missing.
See `docs/PROVENANCE.md`.

If you need to understand an algorithm, read the paper. Do not read another
solver's source. You cannot un-read it, and the claim is then gone.

---

## Layout

```
src/sovopt/
  core/       sparse structures (CSR+CSC), JIT shim, CPU/GPU backend, problem
              and solution types, named tolerances
  io/         MPS (read/write, QUADOBJ), CPLEX-LP, Netlib expander, QPLIB
  numerics/   scaling, Markowitz LU, Forrest-Tomlin update (opt-in), RCM
              ordering, hypersparse FTRAN/BTRAN, iterative refinement
  lp/         dual & primal simplex, node-LP kernel, basis, interior point
              (LP and convex QP), crossover, sensitivity, PDLP (CPU+GPU)
  qp/         proximal PDHG for convex QP (the GPU path) and the dispatcher
  mip/        propagation, branching, cuts, heuristics, tree (threaded),
              BNR, safe bounds (LP and QP), conflict analysis, symmetry, MIQP
  globalopt/  McCormick / OBBT / spatial branch-and-bound for pooling;
              non-convex QP (McCormick reformulation, aBB opt-in)
  models/     refinery templates, Haverly pooling (p and pq)
  presolve.py fixed/singleton/redundant reductions and postsolve (opt-in)
  cli.py      routing by model class; demo.py
bench/        harness, independent verifier, comparator (HiGHS, quarantined),
              fetch, Netlib, QPLIB, scale study, non-convex QP ladder,
              basis-update comparison
tests/        unit + regression, one file per module, plus fixtures/
tools/        check_provenance.py
ui/           minimal local interface (server.py)
```

## Conventions

- Indices are `int32`, values are `float64`. Always. Mixed dtypes silently
  destroy Numba performance.
- Sparse matrices are stored **both** CSR and CSC. `A @ x` walks CSR parallel
  over rows; `Aᵀ @ y` walks CSC parallel over columns. Neither needs atomics.
- No `-ffast-math` equivalent: `fastmath=True` is allowed only in kernels marked
  `# SAFE-FASTMATH` where reassociation cannot change the sign of a residual.
- Determinism: search effort is counted in **work units**, never wall-clock.
  Parallel reductions use fixed-order trees.
- Every numerical tolerance is a named constant in `core/tolerances.py`. No
  magic `1e-9` scattered through the code.

## Commands

```bash
python -m sovopt.cli solve model.mps          # solve an instance
python -m bench.harness --mode lp            # the MIPLIB LP/MIP set
python -m bench.netlib --fetch && python -m bench.netlib    # Netlib
python -m bench.qplib --fetch --run          # QPLIB
python -m bench.verify model.mps sol.json    # independent feasibility check
python -m pytest tests/                      # unit tests
python tools/check_provenance.py             # citation-header lint
```
