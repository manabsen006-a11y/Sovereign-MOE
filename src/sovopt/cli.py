"""Command line interface.

    sovopt demo                              end-to-end LP showcase
    sovopt info    model.mps|model.lp        model statistics and a numerical health check
    sovopt solve   model.mps|model.lp        solve; --sensitivity for shadow prices
    sovopt verify  model.mps solution.json   independent feasibility check
    sovopt devices                           what hardware this build can use

The problem statement asks for an API or command line, not a GUI, so this is the
primary surface. ``sovopt.solve()`` is the library equivalent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .core.backend import GPU_ERROR, gpu_available, gpu_selftest
from .core.problem import ObjSense, Problem, Solution, Status
from .core.tolerances import INF
from .io import read_model

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
          sensitivity: bool = False, crossover: bool = False,
          presolve: bool = False) -> Solution:
    """Solve an LP or MILP, picking the method automatically by default.

    A quadratic objective never reaches the LP engines: the simplex and
    branch-and-bound below optimise ``c'x`` only, so dropping the ``Q`` term
    would return a confident, wrong answer -- on a two-variable test it reports
    -3.0 for a point whose true quadratic objective is -1.875 -- and nothing
    downstream could detect it. A **convex** ``Q`` goes to the proximal
    primal-dual QP solver, or to branch-and-bound over it when there are
    integers; a ``Q`` the convexity check rejects goes to spatial
    branch-and-bound over McCormick envelopes
    (:mod:`sovopt.globalopt.nonconvex_qp`), which needs a finite box on every
    variable in a quadratic term and refuses, naming the variable, without
    one.
    """
    from .lp.pdlp import PDLPParams, solve_pdlp
    from .lp.simplex import SimplexParams, solve_simplex
    from .mip.tree import MIPParams, solve_mip

    if presolve and not prob.is_qp and not prob.is_mip:
        # Pure LPs only. The reductions here are value-preserving but none of
        # them rounds an implied bound to an integer, and the tree already runs
        # domain propagation at every node -- so a MIP gets nothing from this
        # and could get an unsound bound. Off by default: measured over both
        # benchmark families it does not pay (docs/NEGATIVE-RESULTS.md).
        from .presolve import postsolve as _postsolve
        from .presolve import presolve as _presolve
        res = _presolve(prob)
        if res.status is not None:
            return Solution(status=res.status, method="presolve")
        reduced = solve(res.problem, method=method, device=device,
                        time_limit=time_limit, gap=gap, tol=tol,
                        verbose=verbose, sensitivity=False,
                        crossover=crossover, presolve=False)
        return _postsolve(res, reduced)

    if prob.is_qp:
        from .qp import NotConvexError, QPParams, solve_qp
        qp_params = QPParams(time_limit=time_limit, eps_abs=max(tol, 1e-10),
                             eps_rel=max(tol, 1e-10), verbose=verbose)
        try:
            if prob.is_mip:
                # branch-and-bound with a convex QP at every node
                from .mip.miqp import MIQPParams, solve_miqp
                return solve_miqp(prob, MIQPParams(
                    time_limit=time_limit, gap_rel=gap, qp=qp_params,
                    verbose=verbose))
            return solve_qp(prob, qp_params)
        except NotConvexError:
            # Convexity is decided by the QP solver, and a Q it rejects is
            # not approximated: its products become variables under McCormick
            # envelopes and the result is solved to a proven global optimum
            # by spatial branch-and-bound. The check fails before any
            # iteration runs, so the time budget arrives here nearly intact.
            from .globalopt.nonconvex_qp import (NonconvexQPParams,
                                                 solve_nonconvex_qp)
            return solve_nonconvex_qp(prob, NonconvexQPParams(
                time_limit=time_limit, gap_rel=gap, verbose=verbose))

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
    if method in ("ipm", "pdlp") and crossover:
        # Cross over to a basis. Neither engine produces one on its own, so
        # without this there is no ranging and no warm start -- see
        # :mod:`sovopt.lp.crossover`.
        from .lp.crossover import crossover as _crossover
        interior = solve(prob, method=method, device=device,
                         time_limit=time_limit, gap=gap, tol=tol,
                         verbose=verbose)
        if interior.x is None:
            return interior
        return _crossover(prob, interior.x, interior.y,
                          SimplexParams(time_limit=time_limit,
                                        feas_tol=max(tol, 1e-9),
                                        opt_tol=max(tol, 1e-9),
                                        sensitivity=sensitivity,
                                        verbose=verbose))
    if method == "ipm":
        from .lp.ipm import IPMParams, solve_ipm
        return solve_ipm(prob, IPMParams(time_limit=time_limit,
                                         feas_cap=max(tol, 1e-9),
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
    prob = read_model(a.model)
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
    prob = read_model(a.model)
    if a.maximise:
        prob.sense = ObjSense.MAXIMISE
    print(prob.summary())
    print()

    t = time.perf_counter()
    sol = solve(prob, method=a.method, device=a.device,
                time_limit=a.time_limit, gap=a.gap, tol=a.tol,
                verbose=a.verbose, sensitivity=a.sensitivity,
                crossover=a.crossover, presolve=a.presolve)
    dt = time.perf_counter() - t

    print()
    print(f"  status      {sol.status.name}")
    if np.isfinite(sol.objective):
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
            "objective": (sol.objective if np.isfinite(sol.objective)
                          else None),
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

    if a.sensitivity and getattr(sol, "sensitivity", None) is None:
        # Asked for and not delivered: say why rather than printing nothing.
        print()
        print("  no sensitivity report: ranging is read off a simplex "
              "basis, and this model")
        print(f"  was solved by {sol.method}, which produces no basis.")
        print("  Add --crossover to cross over to one and get the ranging.")
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


def cmd_demo(a):
    from .demo import run
    return run(size=a.size)


def _cuda_ver(v: int) -> str:
    """CUDA reports 12040 for 12.4."""
    return f"{v // 1000}.{(v % 1000) // 10}"


def cmd_warmup(a):
    """Compile every kernel once, so the first real solve does not pay.

    Numba compiles a kernel the first time it is called and caches the
    machine code next to the source, so on a fresh machine (or after a
    source change) the first solve of each kind carries the compile cost:
    measured cold at 15 s for the node-LP kernel alone and about a minute
    for everything, against a few seconds warm. Before a live demo, run
    this once. Each engine is exercised on a model small enough that the
    solve itself is negligible; what is reported is the compile.
    """
    from .lp.ipm import IPMParams, solve_ipm
    from .lp.pdlp import PDLPParams, solve_pdlp
    from .lp.simplex import SimplexParams, solve_simplex
    from .mip.tree import MIPParams, solve_mip
    from .models.refinery import blending, unit_scheduling
    from .qp import QPParams, solve_qp
    from .globalopt.nonconvex_qp import NonconvexQPParams, solve_nonconvex_qp
    from .core.sparse import SparseMatrix

    lp = blending(n_components=6, n_products=2)
    mip = unit_scheduling(n_units=2, n_periods=3)
    n = 3
    q = Problem(A=SparseMatrix.from_dense(np.ones((1, n))), c=-np.ones(n),
                row_lb=np.array([-INF]), row_ub=np.array([2.0]),
                col_lb=np.zeros(n), col_ub=np.ones(n),
                Q=SparseMatrix.from_dense(np.eye(n)))
    nq = q.copy()
    nq.Q = SparseMatrix.from_dense(np.eye(n) - 2.0 * np.ones((n, n)) / n)
    steps = [
        ("simplex, sensitivity",
         lambda: solve_simplex(lp.copy(), SimplexParams(sensitivity=True))),
        ("interior point",
         lambda: solve_ipm(lp.copy(), IPMParams())),
        ("PDLP (CPU)",
         lambda: solve_pdlp(lp.copy(), PDLPParams(device="cpu", time_limit=5.0))),
        ("branch-and-bound: node kernel, cuts, heuristics, dives",
         lambda: solve_mip(mip, MIPParams(time_limit=20.0, threads=1))),
        ("convex QP: interior point and proximal",
         lambda: (solve_qp(q.copy(), QPParams()),
                  solve_qp(q.copy(), QPParams(method="proximal")))),
        ("non-convex QP: McCormick, spatial branch-and-bound",
         lambda: solve_nonconvex_qp(nq, NonconvexQPParams(time_limit=20.0))),
    ]
    if a.gpu and gpu_available():
        steps.append(("PDLP (GPU)",
                      lambda: solve_pdlp(lp.copy(), PDLPParams(device="gpu",
                                                               time_limit=5.0))))
    total = 0.0
    print("  warming every engine (first call compiles, later calls are cached)")
    for name, fn in steps:
        t0 = time.perf_counter()
        try:
            fn()
            note = ""
        except Exception as e:                       # noqa: BLE001
            note = f"  ({type(e).__name__}: {str(e)[:50]})"
        dt = time.perf_counter() - t0
        total += dt
        print(f"    {name:<56s} {dt:6.1f} s{note}")
    print(f"    {'total':<56s} {total:6.1f} s")
    return 0


def cmd_devices(a):
    from .core._jit import HAVE_NUMBA, NUMBA_THREADS
    print("  CPU")
    print(f"    numba JIT       {'yes' if HAVE_NUMBA else 'NO (slow fallback)'}")
    print(f"    threads         {NUMBA_THREADS}")
    print("  GPU")
    if not gpu_available():
        print(f"    unavailable     {GPU_ERROR}")
        return 0

    import cupy as cp
    p = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"    device          {p['name'].decode()}")
    print(f"    memory          {free//2**20} MB free of {total//2**20} MB")
    print(f"    compute cap     {p['major']}.{p['minor']}")
    print(f"    driver/runtime  {_cuda_ver(cp.cuda.runtime.driverGetVersion())}"
          f" / {_cuda_ver(cp.cuda.runtime.runtimeGetVersion())}")
    # The kernels are compiled by NVRTC and linked by the driver at run time,
    # so "CuPy imported" and "the kernels work" are separate facts. Report the
    # second one, because that is the one a solve depends on.
    ok, err = gpu_selftest()
    print(f"    kernels         {'compiled and verified' if ok else 'FAILED: ' + str(err)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="sovopt",
        description="SOVOPT - sovereign GPU-accelerated optimization engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="model statistics and numerical health")
    p.add_argument("model")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("solve", help="solve a model")
    p.add_argument("model")
    p.add_argument("--method", choices=["auto", "simplex", "pdlp", "ipm", "bnb"],
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
    p.add_argument("--presolve", action="store_true",
                   help="reduce the model before solving, then postsolve "
                        "(LP only; measured as a net loss on the benchmark "
                        "sets, so it is opt-in)")
    p.add_argument("--crossover", action="store_true",
                   help="with --method ipm or pdlp, cross over to a simplex "
                        "basis so ranging and warm starts are available")
    p.add_argument("--sensitivity", action="store_true",
                   help="report shadow prices and cost/RHS ranging")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("verify", help="independently verify a solution file")
    p.add_argument("model")
    p.add_argument("solution")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("demo", help="end-to-end LP showcase")
    p.add_argument("--size", type=int, default=14,
                   help="blending model size (components)")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("devices", help="show available compute")
    p.set_defaults(fn=cmd_devices)

    p = sub.add_parser("warmup", help="compile and cache every kernel (run "
                                      "once on a fresh machine)")
    p.add_argument("--gpu", action="store_true", help="also compile the GPU path")
    p.set_defaults(fn=cmd_warmup)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
