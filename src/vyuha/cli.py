"""Command line interface.

    vyuha info    model.mps                 model statistics and a numerical health check
    vyuha solve   model.mps [options]       solve and optionally write a solution file
    vyuha verify  model.mps solution.json   independent feasibility check
    vyuha devices                           what hardware this build can use

The problem statement asks for an API or command line, not a GUI, so this is the
primary surface. ``vyuha.solve()`` is the library equivalent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .core.backend import GPU_ERROR, gpu_available
from .core.problem import ObjSense, Problem, Solution, Status
from .core.tolerances import INF
from .io.mps import read_mps

try:                                    # keep the console honest on Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# library entry point                                                          #
# --------------------------------------------------------------------------- #


#: Above this many nonzeros the first-order method is preferred over the
#: simplex: it is the regime where the GPU pays and where the simplex's
#: sequential factorisation updates stop keeping up. Below it the simplex wins
#: decisively -- measured at 10x to 800x on the MIPLIB LP relaxations -- and it
#: returns a basis, which the first-order method cannot.
SIMPLEX_NNZ_LIMIT = 500_000


def solve(prob: Problem, method: str = "auto", device: str = "auto",
          time_limit: float = 300.0, gap: float = 1e-4,
          tol: float = 1e-8, verbose: bool = False,
          sensitivity: bool = False) -> Solution:
    """Solve an LP or MILP, picking the method automatically by default.

    A quadratic objective is **refused**, not ignored. There is no QP solver
    here yet; every engine below optimises ``c'x`` only. Silently dropping the
    ``Q`` term would return a confident, wrong answer -- on a two-variable test
    it reports -3.0 for a point whose true quadratic objective is -1.875 -- and
    nothing downstream could detect it. Refusing is the only safe behaviour
    until a QP path exists.
    """
    from .lp.pdlp import PDLPParams, solve_pdlp
    from .lp.simplex import SimplexParams, solve_simplex
    from .mip.tree import MIPParams, solve_mip

    if prob.is_qp:
        raise NotImplementedError(
            "this model has a quadratic objective (Q is set) and no QP solver "
            "is implemented; solving it as an LP would silently discard the "
            "quadratic term and report a wrong objective")

    if method == "auto":
        if prob.is_mip:
            method = "bnb"
        elif sensitivity:
            method = "simplex"      # only a basis can answer ranging questions
        else:
            method = "simplex" if prob.nnz <= SIMPLEX_NNZ_LIMIT else "pdlp"

    if method == "simplex":
        return solve_simplex(prob, SimplexParams(time_limit=time_limit,
                                                 feas_tol=max(tol, 1e-9),
                                                 opt_tol=max(tol, 1e-9),
                                                 sensitivity=sensitivity,
                                                 verbose=verbose))
    if method == "pdlp":
        return solve_pdlp(prob, PDLPParams(device=device, eps_abs=tol,
                                           eps_rel=tol, time_limit=time_limit,
                                           verbose=verbose))
    if method == "bnb":
        return solve_mip(prob, MIPParams(device=device, gap_rel=gap,
                                         time_limit=time_limit, verbose=verbose))
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------------- #
# commands                                                                     #
# --------------------------------------------------------------------------- #


def cmd_info(a):
    t = time.perf_counter()
    prob = read_mps(a.model)
    dt = time.perf_counter() - t
    print(prob.summary())
    print(f"  parsed in {dt:.3f}s")

    s = prob.stats()
    print()
    print("  numerical health")
    ratio = s["coeff_ratio"]
    verdict = ("good" if ratio < 1e4 else
               "wide -- scaling will matter" if ratio < 1e8 else
               "severe -- expect trouble without scaling")
    print(f"    coefficient ratio   {ratio:.3e}   {verdict}")
    if s["obj_max"] > 0:
        print(f"    objective range     [{s['obj_min']:.3e}, {s['obj_max']:.3e}]")

    from .numerics.scaling import compute_scaling
    sc = compute_scaling(prob.A, method="auto")
    print(f"    after scaling       {sc.ratio_after:.3e}   ({sc.method})")

    finite_lb = int((prob.col_lb > -INF).sum())
    finite_ub = int((prob.col_ub < INF).sum())
    print(f"    bounded columns     {finite_lb} lower, {finite_ub} upper "
          f"of {prob.n}")
    return 0


def cmd_solve(a):
    prob = read_mps(a.model)
    if a.maximise:
        prob.sense = ObjSense.MAXIMISE
    print(prob.summary())
    print()

    t = time.perf_counter()
    sol = solve(prob, method=a.method, device=a.device,
                time_limit=a.time_limit, gap=a.gap, tol=a.tol,
                verbose=a.verbose, sensitivity=a.sensitivity)
    dt = time.perf_counter() - t

    print()
    print(f"  status      {sol.status.name}")
    if sol.x is not None:
        print(f"  objective   {sol.objective:.12g}")
    if np.isfinite(sol.dual_bound) and sol.nodes:
        print(f"  dual bound  {sol.dual_bound:.12g}")
        print(f"  gap         {sol.gap:.4%}")
        print(f"  nodes       {sol.nodes:,}")
    if sol.iterations:
        print(f"  iterations  {sol.iterations:,}")
    print(f"  method      {sol.method}")
    print(f"  time        {dt:.3f}s")

    if sol.x is not None:
        rv, bv, iv = prob.violation(sol.x)
        print(f"  violation   row {rv:.2e}  bound {bv:.2e}  integrality {iv:.2e}")
    if sol.basis_status is not None:
        from .lp.basis import BASIC
        nb = int((sol.basis_status == BASIC).sum())
        print(f"  basis       {nb} basic variables (duals and reduced costs available)")

    if a.out and sol.x is not None:
        payload = {
            "model": prob.name,
            "status": sol.status.name,
            "objective": sol.objective,
            "dual_bound": sol.dual_bound if np.isfinite(sol.dual_bound) else None,
            "time": dt,
            "method": sol.method,
            "x": {n: float(v) for n, v in zip(
                prob.col_names or [f"C{i}" for i in range(prob.n)], sol.x)}
            if a.named else [float(v) for v in sol.x],
        }
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print(f"  wrote       {a.out}")

    if getattr(sol, "sensitivity", None) is not None:
        print()
        print(sol.sensitivity.report())

    if a.print_solution and sol.x is not None:
        names = prob.col_names or [f"C{i}" for i in range(prob.n)]
        nz = np.flatnonzero(np.abs(sol.x) > 1e-9)
        print(f"\n  nonzero variables ({len(nz)} of {prob.n}):")
        for j in nz[:a.print_solution]:
            print(f"    {names[j]:<24s} {sol.x[j]:.10g}")
        if len(nz) > a.print_solution:
            print(f"    ... {len(nz)-a.print_solution} more")

    return 0 if sol.status in (Status.OPTIMAL,) else 1


def cmd_verify(a):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bench.verify import main as vmain
    return vmain([a.model, a.solution])


def cmd_devices(a):
    from .core._jit import HAVE_NUMBA, NUMBA_THREADS
    print("  CPU")
    print(f"    numba JIT       {'yes' if HAVE_NUMBA else 'NO (slow fallback)'}")
    print(f"    threads         {NUMBA_THREADS}")
    print("  GPU")
    if gpu_available():
        import cupy as cp
        p = cp.cuda.runtime.getDeviceProperties(0)
        free, total = cp.cuda.runtime.memGetInfo()
        print(f"    device          {p['name'].decode()}")
        print(f"    memory          {free//2**20} MB free of {total//2**20} MB")
        print(f"    compute cap     {p['major']}.{p['minor']}")
    else:
        print(f"    unavailable     {GPU_ERROR}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="vyuha",
        description="VYUHA - sovereign GPU-accelerated optimization engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="model statistics and numerical health")
    p.add_argument("model")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("solve", help="solve a model")
    p.add_argument("model")
    p.add_argument("--method", choices=["auto", "simplex", "pdlp", "bnb"],
                   default="auto")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--gap", type=float, default=1e-4, help="relative MIP gap")
    p.add_argument("--tol", type=float, default=1e-8, help="LP tolerance")
    p.add_argument("--maximise", action="store_true")
    p.add_argument("--out", help="write the solution as JSON")
    p.add_argument("--named", action="store_true",
                   help="write the solution keyed by column name")
    p.add_argument("--print-solution", type=int, default=0, metavar="N",
                   help="print the first N nonzero variables")
    p.add_argument("--sensitivity", action="store_true",
                   help="report shadow prices and cost/RHS ranging")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("verify", help="independently verify a solution file")
    p.add_argument("model")
    p.add_argument("solution")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("devices", help="show available compute")
    p.set_defaults(fn=cmd_devices)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
