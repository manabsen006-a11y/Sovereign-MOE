"""Convex quadratic programming by proximal primal-dual hybrid gradient.

    minimise  ½ xᵀQx + cᵀx     subject to   rl <= Ax <= ru,  l <= x <= u,  Q ⪰ 0

Why this method, and why it is the safe one to add
--------------------------------------------------
The simplex cannot do this: it walks vertices, and a quadratic objective's
optimum is generally not at one. A proper QP solver is an active-set method or
an interior-point method over the KKT system, and either is a large piece of
work that would touch the basis machinery the LP path depends on.

The first-order engine, though, extends to QP with one change. PDHG already
solves

    min_{x ∈ [l,u]}  max_y   cᵀx + yᵀAx − δ*_C(y)

by a projected gradient step in ``x``. A convex quadratic is a *smooth* term, so
it joins that gradient step directly:

    x ← proj_[l,u]( x − τ(Qx + c + Aᵀy) )

This is the Condat-Vũ primal-dual method: PDHG with a smooth term folded into
the primal step. It converges for convex ``Q`` provided

    1/τ  ≥  L/2 + σ‖A‖²,        L = ‖Q‖₂

which is the only substantive difference from the LP loop -- the step size must
now also pay for the curvature. ``L`` is estimated by power iteration on ``Q``.

Nothing in :mod:`vyuha.lp.pdlp`, the simplex, or the branch-and-bound tree is
touched by this file. Those paths still *refuse* a quadratic objective rather
than silently dropping it, which is the correct behaviour for engines that
genuinely cannot optimise one.

Honest limits
-------------
* **Convex only.** ``Q`` must be positive semi-definite; a non-convex quadratic
  needs spatial branch-and-bound, which lives in :mod:`vyuha.globalopt`.
* First-order accuracy. This reaches ~1e-8 on well-scaled problems, not the
  1e-12 the simplex reaches on an LP, and it returns no basis -- so no
  sensitivity ranging.
* Convexity is *checked*, not assumed: the smallest eigenvalue is estimated and
  a clearly indefinite ``Q`` is refused rather than silently mis-solved.

References
----------
Condat, "A primal-dual splitting method for convex optimization involving
  Lipschitzian, proximable and linear composite terms", J. Optim. Theory Appl.
  158 (2013) 460-479.
Vũ, "A splitting algorithm for dual monotone inclusions involving cocoercive
  operators", Adv. Comput. Math. 38 (2013) 667-681.
Chambolle & Pock, "A first-order primal-dual algorithm for convex problems",
  J. Math. Imaging Vis. 40 (2011) 120-145.
Applegate et al., "Practical large-scale linear programming using primal-dual
  hybrid gradient", NeurIPS 2021 -- the restart and step-size machinery this
  reuses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..core.problem import ObjSense, Problem, Solution, Status
from ..core.sparse import VAL
from ..core.tolerances import INF
from ..numerics.scaling import scale_problem

__all__ = ["QPParams", "solve_qp", "NotConvexError"]


class NotConvexError(ValueError):
    """``Q`` is not positive semi-definite, so this method does not apply."""


@dataclass
class QPParams:
    eps_abs: float = 1e-8
    eps_rel: float = 1e-8
    max_iter: int = 200_000
    time_limit: float = 300.0
    check_every: int = 64
    restart_beta: float = 0.2
    convexity_tol: float = 1e-7
    """How negative an eigenvalue may be before ``Q`` is called indefinite."""
    scaling: str = "pdlp"
    verbose: bool = False


def _spectral_norm(mat, n, iters=50):
    """Largest eigenvalue magnitude of a symmetric matrix, by power iteration."""
    if mat is None:
        return 0.0
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    nv = np.linalg.norm(v)
    if nv == 0.0:
        return 0.0
    v /= nv
    lam = 0.0
    for _ in range(iters):
        w = mat.matvec(v)
        nw = np.linalg.norm(w)
        if nw <= 1e-300:
            return 0.0
        v = w / nw
        lam = nw
    return float(lam)


def _min_eigenvalue(mat, n, lam_max, iters=60):
    """Smallest eigenvalue, via power iteration on ``lam_max*I - Q``."""
    if mat is None or lam_max <= 0.0:
        return 0.0
    rng = np.random.default_rng(1)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    shift = 0.0
    for _ in range(iters):
        w = lam_max * v - mat.matvec(v)
        nw = np.linalg.norm(w)
        if nw <= 1e-300:
            break
        v = w / nw
        shift = nw
    return float(lam_max - shift)


def solve_qp(prob: Problem, params: QPParams | None = None) -> Solution:
    """Solve a convex QP. Raises :class:`NotConvexError` if ``Q`` is indefinite."""
    params = params or QPParams()
    t0 = time.perf_counter()

    if prob.Q is None:
        from ..lp.simplex import solve_simplex
        return solve_simplex(prob)

    flip = prob.sense == ObjSense.MAXIMISE
    work = prob
    if flip:
        work = prob.copy()
        work.c = -work.c
        work.obj_offset = -work.obj_offset
        work.Q = work.Q.copy()
        work.Q.cx *= -1.0
        if work.Q.rx is not None:
            work.Q.rx *= -1.0
        work.sense = ObjSense.MINIMISE

    scaled, sc = scale_problem(work, method=params.scaling)
    n, m = scaled.n, scaled.m
    A, Q = scaled.A, scaled.Q

    # --- convexity check: refuse rather than silently mis-solve -------------
    lam_max = _spectral_norm(Q, n)
    lam_min = _min_eigenvalue(Q, n, lam_max)
    if lam_min < -params.convexity_tol * max(1.0, lam_max):
        raise NotConvexError(
            f"Q has an eigenvalue near {lam_min:.3e}; this method solves convex "
            f"QP only. A non-convex quadratic needs spatial branch-and-bound "
            f"(vyuha.globalopt).")

    c = scaled.c
    rl, ru = scaled.row_lb, scaled.row_ub
    lo, hi = scaled.col_lb, scaled.col_ub

    # --- step sizes: 1/tau >= L/2 + sigma||A||^2 ---------------------------
    a_norm = 1.0
    v = np.random.default_rng(0).standard_normal(n)
    v /= max(np.linalg.norm(v), 1e-300)
    for _ in range(30):
        w = A.rmatvec(A.matvec(v))
        nw = np.linalg.norm(w)
        if nw <= 1e-300:
            break
        v = w / nw
        a_norm = np.sqrt(nw)
    a_norm = max(a_norm, 1e-12)

    sigma = 1.0 / a_norm
    tau = 1.0 / (lam_max / 2.0 + sigma * a_norm * a_norm + 1e-12)

    x = np.clip(np.zeros(n, dtype=VAL), lo, hi)
    y = np.zeros(m, dtype=VAL)
    sum_x = np.zeros(n, dtype=VAL)
    sum_y = np.zeros(m, dtype=VAL)
    k_avg = 0

    status = Status.ITERATION_LIMIT
    best = None

    for k in range(1, params.max_iter + 1):
        if k % params.check_every == 0 and \
                (time.perf_counter() - t0) > params.time_limit:
            status = Status.TIME_LIMIT
            break

        grad = c + A.rmatvec(y)
        if Q is not None:
            grad = grad + Q.matvec(x)
        x_new = np.clip(x - tau * grad, lo, hi)

        ax_bar = A.matvec(2.0 * x_new - x)
        w = y + sigma * ax_bar
        y_new = w - sigma * np.clip(w / sigma, rl, ru)

        x, y = x_new, y_new
        sum_x += x
        sum_y += y
        k_avg += 1

        if k % params.check_every == 0:
            cands = ((sum_x / k_avg, sum_y / k_avg), (x, y))
            res = None
            for xc, yc in cands:
                r = _kkt(scaled, xc, yc)
                if res is None or r[0] < res[0]:
                    res, pick = r, (xc, yc)
                if best is None or r[0] < best[0]:
                    best = (r[0], xc.copy(), yc.copy())
            if _converged(scaled, res, params):
                x, y = pick[0].copy(), pick[1].copy()
                status = Status.OPTIMAL
                break
            # restart the average once it has clearly improved
            if res[0] < params.restart_beta * (best[0] if best else np.inf):
                sum_x[:] = x
                sum_y[:] = y
                k_avg = 1
            if params.verbose and k % (params.check_every * 40) == 0:
                print(f"  it {k:>7d}  kkt {res[0]:.3e}  obj {res[3]:.10g}")

    if status != Status.OPTIMAL and best is not None:
        x, y = best[1], best[2]

    x_orig = np.clip(sc.unscale_primal(x), work.col_lb, work.col_ub)
    y_orig = -sc.unscale_dual(y)
    obj = float(prob.c @ x_orig) + prob.obj_offset
    if prob.Q is not None:
        obj += 0.5 * float(x_orig @ prob.Q.matvec(x_orig))
    if flip:
        y_orig = -y_orig

    # Convergence above was measured on the *scaled* problem. Row scales here
    # reach 1e-3, so a residual that looks converged there can be a thousand
    # times larger in the model the user handed us. Every correctness bug this
    # project has had was a property checked where the search runs rather than
    # where the answer is read, so the status is earned again in original units.
    ax = prob.A.matvec(x_orig)
    pres_orig = float(max(
        np.maximum(prob.row_lb - ax, 0.0).max(initial=0.0),
        np.maximum(ax - prob.row_ub, 0.0).max(initial=0.0)))
    if status == Status.OPTIMAL and pres_orig > 1e-6:
        status = Status.NUMERICAL

    sol = Solution(status=status, x=x_orig, objective=obj, y=y_orig,
                   iterations=k, time=time.perf_counter() - t0,
                   method="qp[proximal-pdhg]")
    sol.dual_bound = obj
    sol.info = {"lambda_max": lam_max, "lambda_min": lam_min,
                "tau": tau, "sigma": sigma, "primal_residual": pres_orig}
    return sol.drop_objective_if_unsolved()


def _kkt(p, x, y):
    """``(merit, primal residual, dual residual, objective)``.

    The dual residual is the *projected-gradient* residual

        ‖ x − proj_[l,u](x − g) ‖∞ ,      g = Qx + c + Aᵀy

    which is zero exactly at a KKT point. The obvious alternative -- classify
    each variable as at-a-bound or free and check the sign of ``g`` on each set
    -- looks equivalent and is not, because the iterate being tested is a
    *running average*, and an average approaches its bound without ever landing
    on it. Every variable then classifies as free, its non-zero ``g`` counts as
    a violation, and the method converges to the right answer while reporting
    that it has not. The projection form needs no active-set guess.
    """
    ax = p.A.matvec(x)
    pv = np.maximum(p.row_lb - ax, 0.0) + np.maximum(ax - p.row_ub, 0.0)
    pres = float(np.abs(pv).max(initial=0.0))

    g = p.c + p.A.rmatvec(y)
    if p.Q is not None:
        g = g + p.Q.matvec(x)
    dres = float(np.abs(x - np.clip(x - g, p.col_lb, p.col_ub)).max(initial=0.0))

    obj = float(p.c @ x) + p.obj_offset
    if p.Q is not None:
        obj += 0.5 * float(x @ p.Q.matvec(x))
    return max(pres, dres), pres, dres, obj


def _converged(p, res, params):
    _, pres, dres, _ = res
    rhs = max(float(np.abs(np.where(p.row_lb > -INF, p.row_lb, 0.0)).max(initial=0.0)),
              float(np.abs(np.where(p.row_ub < INF, p.row_ub, 0.0)).max(initial=0.0)))
    cn = float(np.abs(p.c).max(initial=0.0))
    return (pres <= min(params.eps_abs + params.eps_rel * rhs, 1e-6)
            and dres <= min(params.eps_abs + params.eps_rel * cn, 1e-6))
