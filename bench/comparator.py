"""Head-to-head against an established solver.

The problem statement requires that results be *"compared against at least one
established commercial or open-source solver"*. This file is the only place in
the repository permitted to invoke one, and it invokes it purely as a
**comparator**: HiGHS, reached through ``scipy.optimize.linprog``, solves the
same instances independently and its answers are checked against ours.

Nothing here is imported by ``src/sovopt``. ``tools/check_provenance.py`` still
bans ``scipy.optimize`` everywhere the engine lives, and the ban is what makes
this file meaningful -- using a solver to *check* our numbers is the opposite of
building on one.

Why HiGHS: it is the strongest open-source LP/MIP solver, it is what SciPy ships
as its LP backend, and it is open-licensed. Commercial solver licences generally
forbid publishing benchmark comparisons without written consent, so CPLEX,
Gurobi and Xpress are deliberately not run here; the roadmap's advice is to cite
Mittelmann's published tables for commercial reference points instead.

    python -m bench.comparator
    python -m bench.comparator --tol 1e-6
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sovopt.core.problem import Status, VarKind          # noqa: E402
from sovopt.core.tolerances import INF                    # noqa: E402
from sovopt.io.mps import read_mps                        # noqa: E402
from sovopt.lp.simplex import SimplexParams, solve_simplex  # noqa: E402


def solve_with_highs(prob, time_limit=60.0):
    """Run HiGHS on the same LP, via SciPy. Comparator only."""
    from scipy.optimize import linprog          # quarantined: see module docstring
    from scipy.sparse import csr_matrix

    m, n = prob.m, prob.n
    A = csr_matrix((prob.A.rx, prob.A.ri, prob.A.rp), shape=(m, n))

    # linprog wants  A_ub x <= b_ub  and  A_eq x == b_eq
    lo, hi = prob.row_lb, prob.row_ub
    eq = (lo > -INF) & (hi < INF) & (hi - lo <= 1e-12)
    ub_rows, ub_vals = [], []
    if np.any(hi < INF):
        idx = np.flatnonzero((hi < INF) & ~eq)
        if idx.size:
            ub_rows.append(A[idx])
            ub_vals.append(hi[idx])
    if np.any(lo > -INF):
        idx = np.flatnonzero((lo > -INF) & ~eq)
        if idx.size:
            ub_rows.append(-A[idx])
            ub_vals.append(-lo[idx])

    from scipy.sparse import vstack
    A_ub = vstack(ub_rows) if ub_rows else None
    b_ub = np.concatenate(ub_vals) if ub_vals else None
    eq_idx = np.flatnonzero(eq)
    A_eq = A[eq_idx] if eq_idx.size else None
    b_eq = lo[eq_idx] if eq_idx.size else None

    bounds = [(None if l <= -INF else l, None if u >= INF else u)
              for l, u in zip(prob.col_lb, prob.col_ub)]
    sense = 1.0 if prob.sense.value == 1 else -1.0

    t = time.perf_counter()
    r = linprog(sense * prob.c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
                options={"time_limit": time_limit})
    dt = time.perf_counter() - t
    if not r.success:
        return None, dt, r.message
    obj = sense * r.fun + prob.obj_offset
    return (obj, r.x), dt, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/instances")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--time-limit", type=float, default=60.0)
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(a.dir, "*.mps")))
    if not paths:
        print("no instances; run: python -m bench.fetch --set small")
        return 1

    print("SOVOPT vs HiGHS (via scipy.optimize.linprog) -- LP relaxations")
    print("HiGHS is run as an independent comparator; nothing in src/sovopt uses it.")
    print()
    print(f"{'instance':<12} {'sovopt obj':>16} {'HiGHS obj':>16} {'rel diff':>10} "
          f"{'sovopt':>8} {'HiGHS':>8} {'verdict':>9}")
    print("-" * 86)

    agree = disagree = failed = 0
    tv = th = 0.0
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        prob = read_mps(path)
        prob.kind[:] = VarKind.CONTINUOUS

        t = time.perf_counter()
        mine = solve_simplex(prob, SimplexParams(time_limit=a.time_limit))
        dt_v = time.perf_counter() - t

        got, dt_h, msg = solve_with_highs(prob, a.time_limit)
        tv += dt_v
        th += dt_h

        if mine.status != Status.OPTIMAL or got is None:
            print(f"{name:<12} {'-':>16} {'-':>16} {'-':>10} "
                  f"{dt_v:>7.2f}s {dt_h:>7.2f}s {'SKIP':>9}")
            failed += 1
            continue

        h_obj = got[0]
        rel = abs(mine.objective - h_obj) / max(1.0, abs(h_obj))
        ok = rel <= a.tol
        agree += ok
        disagree += (not ok)
        print(f"{name:<12} {mine.objective:>16.10g} {h_obj:>16.10g} {rel:>10.2e} "
              f"{dt_v:>7.2f}s {dt_h:>7.2f}s {'agree' if ok else 'DIFFER':>9}")

    print("-" * 86)
    print(f"  agree within {a.tol:g}   {agree}/{agree + disagree}")
    if disagree:
        print(f"  !! DISAGREEMENTS: {disagree}")
    if failed:
        print(f"  skipped (one side did not solve): {failed}")
    print(f"  total time      sovopt {tv:.1f}s   HiGHS {th:.1f}s")
    return 0 if disagree == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
