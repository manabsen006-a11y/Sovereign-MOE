"""How far the engines actually go, measured on refinery-shaped models.

The problem statement names a benchmark of "thousands to millions" of variables
and the README has carried "scale is the weakest claim" since the beginning,
because every published figure came from MIPLIB instances that top out at
230x2025. This is the study that replaces the claim with numbers.

Why generated models and not a bigger benchmark library
-------------------------------------------------------
Netlib and Mittelmann would be the obvious answer and are the honest long-term
one. They are also not what this problem statement is about: MRPL blends,
schedules and plans, and a model's *shape* decides an LP's difficulty far more
than its dimensions do. A random sparse matrix of the same size is easier than
a multi-period planning model in every way that matters -- it has no
time-coupling, no near-identical columns and no degeneracy. So the ladder here
is built from :mod:`sovopt.models.refinery`, the same templates the demo uses:

* ``blending``            -- quality-constrained pooling, wide and shallow.
* ``production_planning`` -- multi-period, time-coupled, *deliberately
  degenerate*: several crudes have near-identical yields, so many bases attain
  the same objective. This is the case that punishes a simplex.
* ``unit_scheduling``     -- big-M binaries with minimum up-time, the weak
  relaxation that makes refinery scheduling hard, used for the MILP ladder.

Every point is checked by :mod:`bench.verify`, the independent verifier, so a
size that "solves" but returns an infeasible point is reported as a failure
rather than a time.

    python -m bench.scale --mode lp
    python -m bench.scale --mode mip --time-limit 120
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from bench.verify import verify
from sovopt.core.problem import Status, VarKind
from sovopt.models.refinery import blending, production_planning, unit_scheduling


# --------------------------------------------------------------------------- #
# ladders                                                                      #
# --------------------------------------------------------------------------- #

LP_LADDER = [
    ("blend",   lambda k: blending(n_components=40 * k, n_products=10 * k,
                                   n_properties=4)),
    ("plan",    lambda k: production_planning(n_crudes=12 * k, n_units=8 * k,
                                              n_products=10 * k,
                                              n_periods=6 * k)),
]

MIP_LADDER = [
    ("sched",   lambda k: unit_scheduling(n_units=4 * k, n_periods=6 * k)),
]


def _describe(p):
    return f"{p.m:>7d} {p.n:>8d} {p.nnz:>10d}"


def run_lp(steps, methods, time_limit, tol):
    from sovopt.cli import solve

    print(f"{'model':<10} {'k':>3} {'rows':>7} {'cols':>8} {'nnz':>10} "
          f"{'method':<9} {'status':<12} {'objective':>16} {'time':>9} {'chk':>4}")
    print("-" * 104)
    rows = []
    for name, make in LP_LADDER:
        for k in steps:
            p = make(k)
            p.kind[:] = VarKind.CONTINUOUS
            ref = None
            for method in methods:
                t = time.perf_counter()
                try:
                    s = solve(p.copy(), method=method, time_limit=time_limit,
                              tol=tol)
                except Exception as e:                    # noqa: BLE001
                    print(f"{name:<10} {k:>3} {_describe(p)} {method:<9} "
                          f"{'RAISED':<12} {type(e).__name__:>16} "
                          f"{time.perf_counter() - t:>8.2f}s    -")
                    continue
                dt = time.perf_counter() - t
                chk = "-"
                if s.x is not None:
                    chk = "ok" if verify(p, s.x, feas_tol=1e-6).ok else "BAD"
                obj = s.objective if s.x is not None else float("nan")
                # cross-engine agreement: the first method to produce an
                # answer at this size is the reference for the rest
                if ref is None and np.isfinite(obj):
                    ref = obj
                flag = ""
                if ref is not None and np.isfinite(obj) and \
                        abs(obj - ref) > 1e-5 * max(1.0, abs(ref)):
                    flag = " <- DISAGREES"
                print(f"{name:<10} {k:>3} {_describe(p)} {method:<9} "
                      f"{s.status.name:<12} {obj:>16.8g} {dt:>8.2f}s {chk:>4}{flag}")
                rows.append((name, k, p.m, p.n, p.nnz, method, s.status.name,
                             obj, dt, chk))
            sys.stdout.flush()
    return rows


def run_mip(steps, time_limit, tol, gap):
    from sovopt.cli import solve

    print(f"{'model':<10} {'k':>3} {'rows':>7} {'cols':>8} {'nnz':>10} "
          f"{'int':>7} {'status':<12} {'objective':>16} {'bound':>16} "
          f"{'nodes':>8} {'time':>9} {'chk':>4}")
    print("-" * 124)
    rows = []
    for name, make in MIP_LADDER:
        for k in steps:
            p = make(k)
            t = time.perf_counter()
            try:
                s = solve(p.copy(), method="bnb", time_limit=time_limit,
                          tol=tol, gap=gap)
            except Exception as e:                        # noqa: BLE001
                print(f"{name:<10} {k:>3} {_describe(p)} {p.n_integer:>7d} "
                      f"{'RAISED':<12} {type(e).__name__}")
                continue
            dt = time.perf_counter() - t
            chk = "-"
            if s.x is not None:
                chk = "ok" if verify(p, s.x, feas_tol=1e-6).ok else "BAD"
            obj = s.objective if s.x is not None else float("nan")
            print(f"{name:<10} {k:>3} {_describe(p)} {p.n_integer:>7d} "
                  f"{s.status.name:<12} {obj:>16.8g} {s.dual_bound:>16.8g} "
                  f"{s.nodes:>8d} {dt:>8.2f}s {chk:>4}")
            rows.append((name, k, p.m, p.n, p.nnz, p.n_integer, s.status.name,
                         obj, s.dual_bound, s.nodes, dt, chk))
            sys.stdout.flush()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["lp", "mip"], default="lp")
    ap.add_argument("--steps", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--methods", nargs="*",
                    default=["simplex", "ipm", "pdlp"])
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--gap", type=float, default=1e-4)
    a = ap.parse_args(argv)

    print(f"SOVOPT scaling study  mode={a.mode}  steps={a.steps}  "
          f"time-limit={a.time_limit}s")
    print()
    if a.mode == "lp":
        run_lp(a.steps, a.methods, a.time_limit, a.tol)
    else:
        run_mip(a.steps, a.time_limit, a.tol, a.gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
