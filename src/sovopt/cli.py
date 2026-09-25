"""Command line interface.

    sovopt demo                              end-to-end LP showcase
    sovopt info    model.mps|model.lp        model statistics and a numerical health check
    sovopt solve   model.mps|model.lp        solve; --sensitivity for shadow prices
    sovopt verify  model.mps solution.json   independent feasibility check
    sovopt blend   components.csv products.csv   a plan from a planner's tables
                   --prices prices.csv           ... over a horizon (purchases, storage, capacities)
                   --pools pools.csv             ... through pools, to proven global optimality
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


#: Above this many nonzeros the simplex's sequential factorisation updates
#: stop keeping up, and ``auto`` takes the interior point on the CPU -- PDLP
#: when a GPU is asked for, or when the interior point declines a model it
#: cannot factorise in time (see ``_solve_large_lp``). Below it the simplex wins
#: decisively -- measured at 10x to 800x on the MIPLIB LP relaxations -- and it
#: returns a basis, which the first-order method cannot.
SIMPLEX_NNZ_LIMIT = 500_000


def _solve_large_lp(prob, device, time_limit, tol, verbose):
    """An LP past the simplex's size: the interior point, and PDLP only
    when the interior point declines.

    This used to be PDLP, and PDLP is a first-order method: it reaches the
    optimum to 1e-4 or so and rarely to the verifier's 1e-9. Measured on
    Kennington's sixteen (8k to 1.4M nonzeros) the interior point certified
    15 in 482 s against PDLP-on-the-GPU's 9 in 657 s; on a month of hourly
    blending (Test_Data/One_Month_2026-10, 1.4M nonzeros) PDLP ran out its
    600 s on the CPU and its iteration limit on the GPU with points the
    verifier rejected, and the interior point returned a feasible plan
    within 7.3e-9 of optimal in 814 s. PDLP needs no factorisation, which is
    what it is kept for: when one factorisation of the KKT system is
    predicted to take longer than the limit, the interior point refuses at
    once, and PDLP gets the time that is left.
    """
    from .lp.ipm import IPMParams, solve_ipm
    from .lp.pdlp import PDLPParams, solve_pdlp
    t0 = time.perf_counter()
    sol = solve_ipm(prob, IPMParams(time_limit=time_limit, verbose=verbose))
    if not (sol.info or {}).get("refused"):
        return sol
    left = max(0.0, time_limit - (time.perf_counter() - t0))
    return solve_pdlp(prob, PDLPParams(device=device, eps_abs=tol, eps_rel=tol,
                                       time_limit=left, verbose=verbose))


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
        elif prob.nnz <= SIMPLEX_NNZ_LIMIT:
            method = "simplex"
        elif device == "gpu":
            method = "pdlp"         # a GPU was asked for, and PDLP is what runs there
        else:
            return _solve_large_lp(prob, device, time_limit, tol, verbose)

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
        # ``tol`` is not passed on: the interior point's convergence tests
        # are relative and already tighter than any tol the CLI is given,
        # and its absolute feasibility cap is the verifier's line, not a
        # tolerance -- passing tol as that cap (1e-8 here) demanded of a
        # model with rows at 1e6 what double precision cannot deliver, and
        # returned NUMERICAL on five Netlib instances the loop had converged
        # on (see IPMParams.feas_cap).
        return solve_ipm(prob, IPMParams(time_limit=time_limit,
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
        if sol.y is not None and len(sol.y) == prob.m:
            # the duals let the verifier certify optimality, not just
            # feasibility: bench.verify computes a bound from them that
            # holds whatever the solver did
            payload["y"] = ({n: float(v) for n, v in zip(
                prob.row_names or [f"R{i}" for i in range(prob.m)], sol.y)}
                if a.named else [float(v) for v in sol.y])
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


def cmd_blend(a):
    """Build a model from a planner's tables, solve it, verify the answer
    against the model built from the tables, and print the plan in the
    planner's terms. Two tables give the single-period blend
    (:mod:`sovopt.models.tabular`); ``--prices`` adds a horizon
    (:mod:`sovopt.models.tabular_planning`); ``--pools`` puts pools between
    components and products and solves to proven global optimality
    (:mod:`sovopt.models.tabular_pooling`)."""
    from .models.tabular import TableError
    if a.pools and a.prices:
        print("--pools and --prices cannot be combined: pooled qualities over a horizon "
              "is not a table-driven model yet")
        return 2
    try:
        if a.pools:
            from .models.tabular_pooling import (pooling_plan, pooling_text, pooling_to_csv,
                                                 read_pooling_csv)
            tables = read_pooling_csv(a.components, a.products, a.pools)
            kind, plan_fn, text_fn, csv_fn = "pooling", pooling_plan, pooling_text, pooling_to_csv
        elif a.prices:
            from .models.tabular_planning import (planning_plan, planning_text, planning_to_csv,
                                                  read_planning_csv)
            tables = read_planning_csv(a.components, a.products, a.prices, a.demand, a.capacity)
            kind, plan_fn, text_fn, csv_fn = "planning", planning_plan, planning_text, planning_to_csv
        else:
            from .models.tabular import blend_plan, plan_text, plan_to_csv, read_blending_csv
            tables = read_blending_csv(a.components, a.products)
            kind, plan_fn, text_fn, csv_fn = "blending", blend_plan, plan_text, plan_to_csv
    except TableError as e:
        print(f"cannot build the model: {e}")
        return 2

    t = time.perf_counter()
    if kind == "pooling":
        from .globalopt.spatial import SpatialParams, solve_global
        bp = tables.problem
        print(f"model: {len(tables.components)} components, {len(tables.pools)} pools, "
              f"{len(tables.products)} products -> {bp.linear.m} rows x {bp.n} columns, "
              f"{len(bp.terms)} bilinear terms; spatial branch-and-bound")
        sol = solve_global(bp, SpatialParams(time_limit=a.time_limit))
        prob = bp.linear
    else:
        prob = tables.problem
        what = (f"{len(tables.periods)} periods, " if kind == "planning" else "")
        print(f"model: {len(tables.components)} components, {len(tables.products)} products, "
              f"{what}{len(tables.qualities)} qualities ({', '.join(tables.qualities)}) -> "
              f"{prob.m} rows x {prob.n} columns")
        sol = solve(prob, method=a.method, device=a.device, time_limit=a.time_limit,
                    verbose=a.verbose)
    dt = time.perf_counter() - t
    plan = plan_fn(tables, sol)
    plan["time"] = dt
    plan["method"] = sol.method
    if sol.x is not None and Status(sol.status).has_solution:
        # the independent check, on the model built from the tables; for a
        # pooling plan the bilinear identities are checked as well
        try:
            from bench.verify import verify as _verify
            v = _verify(prob, sol.x, sol.objective, feas_tol=1e-6,
                        y=(sol.y if kind != "pooling" else None))
            extra = None
            if kind == "pooling":
                bil = tables.problem.max_violation(sol.x)
                extra = ("bilinear identities", bil <= 1e-6, f"max violation {bil:.3e}")
            plan["verified"] = {c[0]: [bool(c[1]), c[2]] for c in v.checks}
            plan["verifier_verdict"], plan["verifier_detail"] = v.headline(extra)
        except ImportError:
            plan["verifier_verdict"] = "not run (bench package not importable)"
    print(text_fn(plan))
    print()
    verdict = ""
    if "verifier_verdict" in plan:
        verdict = f"; independent check: {plan['verifier_verdict']}"
        if plan.get("verifier_detail"):
            verdict += f" ({plan['verifier_detail']})"
    print(f"engine {sol.method}, {dt:.3f} s{verdict}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=2)
        print(f"report written to {a.out}")
    if a.plan:
        with open(a.plan, "w", encoding="utf-8", newline="") as fh:
            fh.write(csv_fn(plan))
        print(f"plan written to {a.plan}")
    return 0 if sol.status == Status.OPTIMAL else 1


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

    p = sub.add_parser("blend", help="a blending plan from components.csv and products.csv")
    p.add_argument("components", help="one row per component: name, cost, available, minimum, qualities...")
    p.add_argument("products", help="one row per product: name, price, demand_min, demand_max, <quality>_min/_max...")
    p.add_argument("--prices", help="prices.csv: one row per period, a column per component -- "
                                    "makes it a multi-period plan with storage")
    p.add_argument("--demand", help="demand.csv: period, product, price, demand_min, demand_max (optional)")
    p.add_argument("--capacity", help="capacity.csv: line, capacity, period (optional)")
    p.add_argument("--pools", help="pools.csv: name, capacity, inputs -- pooled qualities, solved "
                                   "to proven global optimality")
    p.add_argument("--method", choices=["auto", "simplex", "ipm", "pdlp"], default="auto")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--out", help="write the full report as JSON")
    p.add_argument("--plan", help="write the plan as CSV (the recipe, the period table, or the flows)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_blend)

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
