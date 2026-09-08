"""Benchmark harness.

Runs the solver over a directory of instances and checks every answer against
the published reference value in the MPS header, then against the independent
verifier. A run that does not check its answers is not a benchmark, it is a
timing exercise.

Reported statistics follow the conventions the optimisation literature uses, so
the numbers are comparable with published tables:

* **shifted geometric mean** of solve times, ``exp(mean(log(t + s))) - s`` with
  ``s = 1s`` for LP and ``10s`` for MIP. The shift stops a handful of very fast
  instances from dominating the mean, which a plain geometric mean does.
* **solved count** at a stated tolerance and time limit.
* per-instance relative error against the published optimum.

    python -m bench.harness --mode lp
    python -m bench.harness --mode mip --time-limit 60
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from bench.fetch import DATA_DIR, read_reference
from bench.verify import verify
from sovopt.core.problem import Status, VarKind
from sovopt.io.mps import read_mps


def shifted_geomean(values, shift):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(np.maximum(v, 0.0) + shift))) - shift)


def run(mode, paths, time_limit, device, tol, gap, verbose=False,
        method=None):
    from sovopt.cli import solve

    rows = []
    print(f"{'instance':<14} {'rows':>6} {'cols':>6} {'nnz':>8} "
          f"{'status':<10} {'objective':>16} {'reference':>16} "
          f"{'relerr':>9} {'time':>8} {'chk':>4}")
    print("-" * 118)

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        ref = read_reference(path)
        try:
            prob = read_mps(path)
        except Exception as e:
            print(f"{name:<14} PARSE FAILED  {type(e).__name__}: {str(e)[:50]}")
            continue

        target = None
        if mode == "lp":
            prob.kind[:] = VarKind.CONTINUOUS       # solve the relaxation
            target = ref.get("lp_soln")
        else:
            target = ref.get("best_soln")

        t = time.perf_counter()
        chosen = method or ("auto" if mode == "lp" else "bnb")
        try:
            sol = solve(prob, method=chosen,
                        device=device, time_limit=time_limit, gap=gap, tol=tol,
                        verbose=verbose)
        except Exception as e:
            print(f"{name:<14} SOLVER FAILED  {type(e).__name__}: {str(e)[:50]}")
            continue
        dt = time.perf_counter() - t

        relerr = float("nan")
        if target is not None and sol.x is not None and np.isfinite(sol.objective):
            relerr = abs(sol.objective - target) / max(1.0, abs(target))

        chk = "-"
        if sol.x is not None:
            v = verify(prob, sol.x, sol.objective,
                       feas_tol=1e-6, int_tol=1e-6)
            chk = "ok" if v.ok else "BAD"

        print(f"{name:<14} {prob.m:>6} {prob.n:>6} {prob.nnz:>8} "
              f"{sol.status.name:<10} "
              f"{sol.objective if sol.x is not None else float('nan'):>16.8g} "
              f"{target if target is not None else float('nan'):>16.8g} "
              f"{relerr:>9.2e} {dt:>7.2f}s {chk:>4}")

        rows.append({
            "name": name, "status": sol.status, "obj": sol.objective,
            "ref": target, "relerr": relerr, "time": dt, "check": chk,
            "nodes": sol.nodes,
        })

    # ---- summary ---------------------------------------------------------- #
    print("-" * 118)
    n = len(rows)
    if not n:
        return rows
    solved = [r for r in rows if r["status"] == Status.OPTIMAL]
    verified = [r for r in rows if r["check"] == "ok"]
    close = [r for r in rows if np.isfinite(r["relerr"]) and r["relerr"] < 1e-4]
    bad = [r for r in rows if r["check"] == "BAD"]

    shift = 1.0 if mode == "lp" else 10.0
    print(f"  instances            {n}")
    print(f"  status OPTIMAL       {len(solved)}/{n}")
    print(f"  verifier accepted    {len(verified)}/{n}")
    print(f"  within 1e-4 of ref   {len(close)}/{sum(1 for r in rows if r['ref'] is not None)}")
    print(f"  shifted geomean time {shifted_geomean([r['time'] for r in rows], shift):.3f}s "
          f"(shift {shift:g}s)")
    print(f"  total time           {sum(r['time'] for r in rows):.1f}s")
    if bad:
        print(f"  !! VERIFIER REJECTED: {', '.join(r['name'] for r in bad)}")
    worst = max((r for r in rows if np.isfinite(r["relerr"])),
                key=lambda r: r["relerr"], default=None)
    if worst:
        print(f"  worst relative error {worst['relerr']:.2e} ({worst['name']})")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lp", "mip"], default="lp")
    ap.add_argument("--method", choices=["auto", "simplex", "pdlp", "bnb"],
                    default=None, help="override the solver choice")
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--max-nnz", type=int, default=10**9)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(a.dir, "*.mps")))
    if a.only:
        keep = set(a.only)
        paths = [p for p in paths
                 if os.path.splitext(os.path.basename(p))[0] in keep]
    if not paths:
        print(f"no instances in {a.dir}; run:  python -m bench.fetch --set small")
        return 1

    print(f"SOVOPT benchmark  mode={a.mode}  method={a.method or 'auto'}  "
          f"device={a.device}  time-limit={a.time_limit}s  tol={a.tol:g}")
    print()
    run(a.mode, paths, a.time_limit, a.device, a.tol, a.gap, a.verbose,
        method=a.method)
    return 0


if __name__ == "__main__":
    sys.exit(main())
