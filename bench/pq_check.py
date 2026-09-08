"""pq-formulation vs plain q on multi-pool networks.

Both encode the same feasible set and the same global optimum. They differ only
in the strength of the McCormick relaxation, which is what the search runs on.
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

from sovopt.core.problem import Status
from sovopt.globalopt.bilinear import build_relaxation
from sovopt.globalopt.spatial import SpatialParams, solve_global
from sovopt.lp.simplex import SimplexParams, solve_simplex
from sovopt.models import random_pooling


def root_bound(bp):
    s = solve_simplex(build_relaxation(bp, bp.linear.col_lb, bp.linear.col_ub),
                      SimplexParams())
    return s.objective if s.status == Status.OPTIMAL else float("nan")


def main():
    print("Multi-pool pooling: q-formulation vs pq (same optimum, different bound)")
    print()
    print(f"  {'network':<16} {'terms':>5} {'root q':>10} {'root pq':>10} "
          f"{'tighter':>8}   {'q: opt/nodes/time':>24}   {'pq: opt/nodes/time':>24}")
    print("  " + "-" * 108)

    for (ns, npl, nk, nw) in [(4, 2, 2, 1), (5, 2, 3, 2), (6, 3, 3, 2)]:
        for seed in (0, 1):
            q = random_pooling(ns, npl, nk, nw, seed=seed, rlt=False)
            pq = random_pooling(ns, npl, nk, nw, seed=seed, rlt=True)
            rq, rpq = root_bound(q), root_bound(pq)

            out = []
            for bp in (q, pq):
                t = time.perf_counter()
                s = solve_global(bp, SpatialParams(time_limit=60))
                out.append((s, time.perf_counter() - t))

            tighten = (rq - rpq) / max(abs(rq), 1.0) * 100.0
            (sq, tq), (spq, tpq) = out
            print(f"  {ns}x{npl}x{nk} s{seed:<8} {len(pq.terms):>5} "
                  f"{rq:>10.5g} {rpq:>10.5g} {tighten:>7.1f}%   "
                  f"{(sq.objective if sq.x is not None else float('nan')):>9.6g}"
                  f"/{sq.nodes:>5d}/{tq:>5.1f}s   "
                  f"{(spq.objective if spq.x is not None else float('nan')):>9.6g}"
                  f"/{spq.nodes:>5d}/{tpq:>5.1f}s")
    print()
    print("  'tighter' is how much the RLT rows pull the root bound toward the")
    print("  true optimum. Both formulations must report the same optimum.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
