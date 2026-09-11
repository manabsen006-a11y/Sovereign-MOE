"""Quadratic programming.

Two convex engines behind one call. The **interior point** (the LP method in
:mod:`sovopt.lp.ipm` with ``Q`` in its (1,1) block) is the default: it
converges in tens of factorisations regardless of conditioning, and it is what
solves the convex QPLIB instances. The **proximal** first-order method is
``method="proximal"``: matrix-free, so it is the GPU path and the one that
survives a ``Q`` too dense to factorise, and slow in the tail. Both refuse an
indefinite ``Q`` the same way; the non-convex case is
:mod:`sovopt.globalopt.nonconvex_qp`.
"""

from __future__ import annotations

from ..core.problem import Problem, Solution
from .proximal import NotConvexError, QPParams, check_convex
from .proximal import solve_qp as solve_proximal_qp

__all__ = ["QPParams", "solve_qp", "solve_proximal_qp", "NotConvexError",
           "check_convex"]


def solve_qp(prob: Problem, params: QPParams | None = None) -> Solution:
    """Solve a convex QP with the engine ``params.method`` names."""
    params = params or QPParams()
    if params.method == "proximal":
        return solve_proximal_qp(prob, params)
    if params.method != "ipm":
        raise ValueError(f"unknown QP method {params.method!r}")
    if prob.Q is None:
        from ..lp.simplex import solve_simplex
        return solve_simplex(prob)
    from ..lp.ipm import IPMParams, solve_ipm
    from ..core.problem import ObjSense
    Q = prob.Q
    if prob.sense == ObjSense.MAXIMISE:
        Q = Q.copy()                      # a concave maximisation is convex
        Q.cx = -Q.cx
        Q.rx = -Q.rx
    check_convex(Q, prob.n, params.convexity_tol)
    tol = max(params.eps_abs, params.eps_rel)
    sol = solve_ipm(prob, IPMParams(
        time_limit=params.time_limit,
        eps_p=min(tol, 1e-9), eps_d=min(tol, 1e-9), eps_gap=min(tol, 1e-10),
        verbose=params.verbose))
    sol.method = "qp[ipm]"
    return sol
