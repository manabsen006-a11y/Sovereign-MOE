# Demo runbook

One command carries the whole story. Everything else here is for answering
follow-up questions without improvising.

```bash
python -m sovopt.cli warmup     # the night before: compiles and caches every kernel
python -m sovopt.cli demo
```

The first solve of each kind on a fresh machine pays Numba's compile --
about 50 s for everything, most of it the node-LP kernel -- and `warmup`
pays it once, ahead of time. Cached, the same command takes about 5 s.

Eight stages, ~90 seconds, each guarded so a stage that cannot run on the day
says so and the demo keeps going. Nothing in it can be the reason the demo
stops.

---

## What each stage is for

| # | Stage | The claim it backs |
|---|-------|--------------------|
| 1 | Refinery blending model | Real industrial structure, not a toy. Coefficient ratio 6.9e8 → 2.4e2 after Curtis–Reid |
| 2 | Revised simplex **and interior point** | Built from mathematical foundations — the PS's hard constraint. Two engines from different mathematics agreeing to 2.3e-10 is a check neither can perform on itself |
| 3 | The arithmetic, shown | The blend, term by term, summed back to the objective the solver reported — not asserted, displayed |
| 4 | Shadow prices + ranging | What a planner actually reads. Worst error 3.9e-15 across 239 binding rows |
| 5 | Independent verifier | Correct by a check that shares no code with the solver |
| 6 | HiGHS head-to-head | Correct by an outside standard — the PS requires a comparator |
| 7 | CPU vs GPU | The GPU claim, measured both ways, crossover shown |
| 8 | Convex QP | Beyond LP — an optimum no vertex method can reach |

---

## If you are asked to show something specific

```bash
python -m sovopt.cli solve model.mps --sensitivity
```
Shadow prices and cost/RHS ranging. Reads `.mps` and `.lp`, plain or
`.gz`/`.bz2`/`.xz`.

```bash
python -m bench.harness --mode lp
```
11 MIPLIB instances against published reference values. 11/11 optimal, 11/11
verifier-accepted on all three engines; shifted geomean 0.140 s on the interior
point, 0.258 s on the simplex, 1.496 s on PDLP, measured together. `--method ipm` or
`--method simplex` forces one. Add `--mode mip` for the MILP set: 6/11 proved
at a 60 s limit, 10/11 verifier-accepted, and every proved optimum matches the
published value exactly.

```bash
python -m bench.scale --mode lp
```
How far the engines actually go, on refinery models rather than random
matrices. Largest solved and verified: 2,080 x 102,400 with 1.02M nonzeros in
3.65 s. See README, Scale.

```bash
python -m bench.comparator
```
Head-to-head with HiGHS. Agrees 11/11 to 7.0e-16.

```bash
python -m pytest tests/ -q
```
Full regression suite.

```bash
python -m ui.server
```
Browser interface — upload a model, solve, read the solution.

---

## Questions you should expect, and the honest answer

**"Is this really from scratch?"**
`python tools/check_provenance.py` — it fails the build on any import of a
solver library, and on any LU/Cholesky library, across all of `src/sovopt/`. The
one exception is `bench/comparator.py`, which runs HiGHS as the comparator the
PS requires; it is quarantined to that file and never touches the engine.

**"How does it compare to CPLEX/Xpress?"**
Not tested against either — no licence. Against HiGHS it is correct on 11/11 and
roughly 6.7× slower on the LP set (4.8× excluding the degenerate 10teams). The gap is presolve and a Forrest–Tomlin basis
update, both named in the roadmap. Say the number; it is more credible than
dodging it.

**"What's actually new?"**
Batched Node Relaxation. Every branch-and-bound node shares `A`, `c` and the row
bounds — only the column bounds differ — so a whole frontier of nodes is bounded
in one sparse matrix-matrix product instead of one LP solve each. The safe dual
bound (Neumaier–Shcherbina) makes an *unconverged* first-order solve still a
rigorous bound, which is what makes batching sound rather than just fast.

**"What doesn't work?"**
README § Known limits, kept current deliberately. The short version: no
presolve, no crossover, no interior-point, MILP has not been tested past
230×2025, QP is convex-only, cuts are root-only, the tree is single-threaded.

**"Show me a bug you found."**
`docs/NEGATIVE-RESULTS.md` — two optimisations measured, found to be a net loss,
and reverted with the calibration data kept. Also worth saying out loud: every
correctness bug in this project was caught by an oracle *outside* the solver —
exact rational arithmetic, published values, the independent verifier,
brute-force enumeration, HiGHS. None was caught by the solver agreeing with
itself.

---

## Before you present

```bash
python -m sovopt.cli demo
```

Run it once on the actual machine. Stage 6 needs CuPy and a GPU; without one it
prints a line saying so and continues, which is fine, but you want to know which
version you are showing before you are standing in front of it.
