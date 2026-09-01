"""Global pooling vs distributive recursion -- the industry workaround.

Recursion guesses the pool qualities, solves the resulting LP, re-reads the
qualities off the answer, and repeats. It is what PIMS and GRTMPS do. It has no
global guarantee, and Haverly's instances are the classic demonstration.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from vyuha.core.problem import Status
from vyuha.globalopt.bilinear import build_relaxation
from vyuha.globalopt.spatial import SpatialParams, solve_global
from vyuha.lp.simplex import SimplexParams, solve_simplex
from vyuha.models import haverly

P = 6            # pool quality column


def recursion(bp, p0, max_iter=200, tol=1e-10):
    """Distributive recursion from a starting guess for the pool quality."""
    lo, hi = bp.linear.col_lb.copy(), bp.linear.col_ub.copy()
    p = float(np.clip(p0, lo[P], hi[P]))
    last = None
    for _ in range(max_iter):
        l, h = lo.copy(), hi.copy()
        l[P] = h[P] = p
        s = solve_simplex(build_relaxation(bp, l, h), SimplexParams())
        if s.status != Status.OPTIMAL or s.x is None:
            return None, p
        x = s.x
        flow = x[2] + x[3]                      # yX + yY out of the pool
        if flow <= 1e-9:
            return s.objective, p
        # re-read the pool quality implied by the feeds
        new_p = (3.0 * x[0] + 1.0 * x[1]) / flow
        new_p = float(np.clip(new_p, lo[P], hi[P]))
        if last is not None and abs(new_p - p) < tol:
            return s.objective, p
        last, p = p, new_p
    return s.objective, p


def main():
    print("Haverly pooling: global optimisation vs distributive recursion")
    print()
    print(f"  {'instance':<10} {'global':>9} {'bound':>9} {'nodes':>6} "
          f"{'recursion outcomes over 21 starts':>38}")
    print("  " + "-" * 76)
    for v in (1, 2, 3):
        bp = haverly(v)
        t = time.perf_counter()
        g = solve_global(bp, SpatialParams(time_limit=60))
        dt = time.perf_counter() - t

        results = []
        for p0 in np.linspace(1.0, 3.0, 21):
            obj, _ = recursion(bp, p0)
            if obj is not None:
                results.append(obj)
        results = np.array(results)
        best = results.max()
        worst = results.min()
        stuck = int(np.sum(results < g.objective - 1e-6))
        print(f"  haverly{v:<3} {g.objective:>9.6g} {g.dual_bound:>9.6g} "
              f"{g.nodes:>6d}   best {best:>7.6g}  worst {worst:>7.6g}  "
              f"stuck below global: {stuck}/{len(results)}")
    print()
    print("  'stuck' counts starting points from which recursion converges to a")
    print("  blend worth strictly less than the proven global optimum.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
