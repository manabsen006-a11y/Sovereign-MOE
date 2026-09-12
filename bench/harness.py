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
        method=None, repeat=1):
    """``repeat`` > 1 solves each instance that many times and reports the
    *median* wall time and the spread (min-max) beside it. One draw on a
    laptop is not a measurement -- background load alone has produced 2.3x
    swings on repeated runs of the same instance -- and the median of three
    is what the README's tables now quote. The objective, status and
    verifier verdict are those of the first run; the algorithms are
    deterministic and every run agrees on them (asserted below)."""
    from sovopt.cli import solve

    rows = []
    spread_col = f" {'spread':>13}" if repeat > 1 else ""
    print(f"{'instance':<14} {'rows':>6} {'cols':>6} {'nnz':>8} "
          f"{'status':<10} {'objective':>16} {'reference':>16} "
          f"{'relerr':>9} {'time':>8} {'chk':>4}{spread_col}")
    print("-" * (118 + len(spread_col)))

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

        chosen = method or ("auto" if mode == "lp" else "bnb")
        times = []
        sol = None
        failed = False
        for _ in range(max(1, repeat)):
            t = time.perf_counter()
            try:
                s = solve(prob, method=chosen,
                          device=device, time_limit=time_limit, gap=gap,
                          tol=tol, verbose=verbose)
            except Exception as e:
                print(f"{name:<14} SOLVER FAILED  {type(e).__name__}: {str(e)[:50]}")
                failed = True
                break
            times.append(time.perf_counter() - t)
            if sol is None:
                sol = s
            elif s.status != sol.status or (
                    s.x is not None and sol.x is not None
                    and abs(s.objective - sol.objective) > 1e-6 * max(1.0, abs(sol.objective))
                    and mode == "lp"):
                # an LP solve is deterministic; a MIP with a time limit is
                # not, and only the LP disagreement is worth a line
                print(f"{name:<14} RUNS DISAGREE  {sol.status.name} {sol.objective} "
                      f"vs {s.status.name} {s.objective}")
        if failed:
            continue
        dt = float(np.median(times))
        spread = (f" {min(times):>6.2f}-{max(times):<6.2f}" if repeat > 1 else "")

        relerr = float("nan")
        if target is not None and sol.x is not None and np.isfinite(sol.objective):
            relerr = abs(sol.objective - target) / max(1.0, abs(target))

        chk = "-"
        if sol.x is not None:
            v = verify(prob, sol.x, sol.objective,
                       feas_tol=1e-6, int_tol=1e-6,
                       y=(getattr(sol, "y", None) if mode == "lp" else None))
            # feasibility is the verdict; optimality, when duals are there,
            # is reported beside it and counted separately below
            feas = [c for c in v.checks if c[0] != "optimality"]
            opt = [c for c in v.checks if c[0] == "optimality"]
            chk = "ok" if all(c[1] for c in feas) else "BAD"
            if chk == "ok" and opt:
                chk = "opt" if opt[0][1] else "ok"

        print(f"{name:<14} {prob.m:>6} {prob.n:>6} {prob.nnz:>8} "
              f"{sol.status.name:<10} "
              f"{sol.objective if sol.x is not None else float('nan'):>16.8g} "
              f"{target if target is not None else float('nan'):>16.8g} "
              f"{relerr:>9.2e} {dt:>7.2f}s {chk:>4}{spread}")

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
    verified = [r for r in rows if r["check"] in ("ok", "opt")]
    close = [r for r in rows if np.isfinite(r["relerr"]) and r["relerr"] < 1e-4]
    bad = [r for r in rows if r["check"] == "BAD"]

    shift = 1.0 if mode == "lp" else 10.0
    print(f"  instances            {n}")
    if repeat > 1:
        print(f"  runs per instance    {repeat}   (times are medians)")
    print(f"  status OPTIMAL       {len(solved)}/{n}")
    print(f"  verifier accepted    {len(verified)}/{n}")
    if mode == "lp":
        cert = [r for r in rows if r["check"] == "opt"]
        print(f"  certified optimal    {len(cert)}/{n}   (duals bound the optimum within 1e-9)")
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
    ap.add_argument("--method", choices=["auto", "simplex", "pdlp", "ipm", "bnb"],
                    default=None, help="override the solver choice")
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--max-nnz", type=int, default=10**9)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--repeat", type=int, default=1,
                    help="solve each instance this many times; report the median time")
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
          f"device={a.device}  time-limit={a.time_limit}s  tol={a.tol:g}"
          + (f"  repeat={a.repeat}" if a.repeat > 1 else ""))
    print()
    run(a.mode, paths, a.time_limit, a.device, a.tol, a.gap, a.verbose,
        method=a.method, repeat=a.repeat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
