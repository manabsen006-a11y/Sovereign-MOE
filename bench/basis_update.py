"""Product form against Forrest-Tomlin, across refactorisation budgets.

The question is not "which update is faster" but "which update, at which
budget". The product form's solve cost grows with every pivot since the last
refactorisation, so it wants a short budget; Forrest-Tomlin's stays flat, so
it can afford a long one and skip the refactorisations. Each cell below is
the whole set solved once, and the two columns that matter are the total
time and the total pivots -- because a different update also means different
rounding at degenerate ties, and a path that takes twice the pivots is a
different measurement, not a slower solve.

    python -m bench.basis_update --set miplib
    python -m bench.basis_update --set netlib --time-limit 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sovopt.core.problem import Status
from sovopt.io.mps import read_mps
from sovopt.lp.simplex import SimplexParams, solve_simplex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIPLIB = ["10teams", "dcmulti", "flugpl", "gr4x6", "gt2", "khb05250", "mas76",
          "misc07", "mod010", "p0201", "qnet1"]


def load(setname, only):
    if setname == "miplib":
        names = only or MIPLIB
        return [(n, read_mps(os.path.join(ROOT, "data", "instances", f"{n}.mps")))
                for n in names]
    from sovopt.io.netlib import read_netlib
    d = os.path.join(ROOT, "data", "netlib")
    names = only or sorted(f for f in os.listdir(d)
                           if "." not in f and not f.startswith("index"))
    out = []
    for n in names:
        path = os.path.join(d, n)
        try:
            out.append((n, read_netlib(path, name=n)))
        except Exception as e:                       # noqa: BLE001
            print(f"  {n}: skipped ({e})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="miplib", choices=["miplib", "netlib"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--budgets", type=int, nargs="+", default=[150, 400, 1000])
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args(argv)

    probs = load(args.set, args.only)
    print(f"basis update x refactorisation budget  set={args.set} "
          f"({len(probs)} instances)  time-limit={args.time_limit:g}s\n")
    print(f"{'budget':>7} {'update':>7} {'optimal':>8} {'pivots':>9} {'factor.':>8} "
          f"{'refused':>8} {'time':>9}")
    for budget in args.budgets:
        for mode in ("pfi", "ft"):
            best = None
            for _ in range(args.repeat):
                tot_t = 0.0
                piv = fac = ref = opt = 0
                for name, p in probs:
                    t = time.perf_counter()
                    s = solve_simplex(p.copy(), SimplexParams(
                        basis_update=mode, refactor_freq=budget,
                        time_limit=args.time_limit))
                    tot_t += time.perf_counter() - t
                    piv += s.iterations
                    fac += s.info.get("factorizations", 0)
                    ref += s.info.get("ft_refused", 0)
                    opt += s.status == Status.OPTIMAL
                if best is None or tot_t < best[0]:
                    best = (tot_t, opt, piv, fac, ref)
            tot_t, opt, piv, fac, ref = best
            print(f"{budget:>7} {mode:>7} {opt:>5}/{len(probs):<2} {piv:>9} {fac:>8} "
                  f"{ref:>8} {tot_t:>8.2f}s")


if __name__ == "__main__":
    main()
