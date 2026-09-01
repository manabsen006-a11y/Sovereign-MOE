# VYUHA — Sovereign Optimization Engine

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
  cross-checking test data. Never inside `src/vyuha/`.
- `scipy.optimize.linprog` — **only** inside `bench/comparator.py`, and only to
  run HiGHS as an external *comparator*. The problem statement requires that
  results be "compared against at least one established commercial or
  open-source solver", so a comparator is mandatory, not optional. It is
  quarantined to that one file, it never touches `src/vyuha/`, and
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
src/vyuha/
  core/      sparse structures, JIT shim, problem/solution types
  io/        MPS / LP / QPS readers and writers
  numerics/  scaling, Markowitz LU, hypersparse FTRAN/BTRAN, refinement
  lp/        dual & primal simplex, PDLP (CPU+GPU), crossover, presolve
  mip/       propagation, branching, cuts, heuristics, tree, BNR, safe bounds
  globalopt/ McCormick / RLT / OBBT / spatial branch-and-bound for pooling
bench/       harness, independent verifier, instance manifests
tests/       unit + regression
ui/          thin CLI and a minimal local interface
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
python -m vyuha.cli solve model.mps          # solve an instance
python -m bench.harness --set netlib         # run a benchmark set
python -m bench.verify model.mps sol.json    # independent feasibility check
python -m pytest tests/                      # unit tests
python tools/check_provenance.py             # citation-header lint
```
