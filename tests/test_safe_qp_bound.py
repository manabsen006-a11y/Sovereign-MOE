"""The certified bound for a convex QP relaxation.

``safe_qp_bound`` is the Neumaier-Shcherbina bound applied to the tangent plane
of a convex quadratic at an arbitrary point. Its only claim is the one that
matters for branch-and-bound: for *any* point and *any* dual vector, it never
lies above the objective of any feasible point. Everything here tests that
claim from a different side -- random points and duals, a half-converged
solve, a converged one (where it must also be tight), and the LP special case
it has to collapse to.
"""

import numpy as np
import pytest

from sovopt.core.problem import Problem
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.mip.safebound import safe_dual_bound, safe_qp_bound
from sovopt.qp import QPParams, solve_qp


def convex_qp(seed, n=5, m=3, cond=1.0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    if cond > 1.0:
        Qm, _ = np.linalg.qr(M)
        H = Qm @ np.diag(np.logspace(0, np.log10(cond), n)) @ Qm.T
    else:
        H = M.T @ M + 0.5 * np.eye(n)
    H = 0.5 * (H + H.T)
    c = rng.standard_normal(n) * 3
    A = SparseMatrix.from_dense(np.round(rng.uniform(-1, 3, (m, n))))
    b = np.abs(A.matvec(np.full(n, 1.5))) + rng.uniform(1, 5, m)
    return Problem(A=A, c=c, Q=SparseMatrix.from_dense(H),
                   row_lb=np.full(m, -INF), row_ub=b,
                   col_lb=np.zeros(n), col_ub=np.full(n, 3.0), name=f"qp{seed}")


def feasible_points(p, rng, count=400):
    pts = rng.uniform(p.col_lb, p.col_ub, (count, p.n))
    keep = []
    for x in pts:
        rv, cv, _ = p.violation(x)
        if max(rv, cv) <= 1e-12:
            keep.append(x)
    return keep


def bound(p, x_hat, y, **kw):
    return safe_qp_bound(p.A, p.c, p.Q, p.row_lb, p.row_ub, p.col_lb,
                         p.col_ub, x_hat, y, **kw) + p.obj_offset


# --------------------------------------------------------------------------- #
# validity: the claim                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("strict", [False, True])
def test_never_above_any_feasible_point_for_any_point_and_any_dual(seed, strict):
    """Random ``x̂`` -- inside the box, outside it, far away -- and random
    ``y`` of either sign. None of that may push the bound above ``f(x)`` for a
    feasible ``x``."""
    p = convex_qp(seed)
    rng = np.random.default_rng(100 + seed)
    feas = feasible_points(p, rng)
    assert feas, "the generator should leave room in the box"
    f_min = min(p.objective(x) for x in feas)
    for trial in range(30):
        x_hat = rng.uniform(-4, 7, p.n) if trial % 3 else rng.uniform(0, 3, p.n)
        y = rng.standard_normal(p.m) * (10.0 ** rng.integers(-2, 3))
        b = bound(p, x_hat, y, strict=strict)
        assert b <= f_min + 1e-9 * max(1.0, abs(f_min)), (
            f"seed {seed} trial {trial}: bound {b} above a feasible value {f_min}")


@pytest.mark.parametrize("seed", range(6))
def test_a_half_converged_solve_still_gives_a_valid_bound(seed):
    """The reason the function exists: a first-order QP stopped early hands
    over an iterate whose *objective* is not a bound (it is a feasible-ish
    point, so it lies above the optimum) -- yet the certified bound from that
    same iterate is below the optimum."""
    p = convex_qp(seed, cond=1e3)
    full = solve_qp(p.copy(), QPParams(time_limit=60))
    early = solve_qp(p.copy(), QPParams(max_iter=64, check_every=64,
                                        eps_abs=1e-1, eps_rel=1e-1))
    b = bound(p, early.x, early.y, strict=True)
    assert b <= full.objective + 1e-7 * max(1.0, abs(full.objective))
    # and the early objective itself is *not* a bound, which is the point
    if early.objective > full.objective + 1e-6:
        assert b < early.objective


# --------------------------------------------------------------------------- #
# tightness: it is not merely valid                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(6))
def test_tight_at_a_converged_optimum(seed):
    """At an optimal primal-dual pair the reduced costs have the signs
    complementarity demands, and the bound is the objective to the solver's
    own accuracy. A bound that were valid but always loose would be useless."""
    p = convex_qp(seed)
    s = solve_qp(p.copy(), QPParams(time_limit=60))
    b = bound(p, s.x, s.y, strict=True)
    assert abs(s.objective - b) <= 1e-5 * max(1.0, abs(s.objective))
    # The bound may sit a hair *above* s.objective: the solver's point can
    # violate a row by ~1e-7, and a slightly infeasible point can be slightly
    # better than the optimum. The bound is over the feasible set, and the
    # first test in this file checks it against genuinely feasible points.


def test_strict_is_never_above_plain():
    p = convex_qp(3)
    rng = np.random.default_rng(3)
    for _ in range(20):
        x_hat = rng.uniform(0, 3, p.n)
        y = rng.standard_normal(p.m)
        assert bound(p, x_hat, y, strict=True) <= bound(p, x_hat, y) + 1e-15


# --------------------------------------------------------------------------- #
# special cases                                                                #
# --------------------------------------------------------------------------- #


def test_reduces_to_the_lp_bound_without_a_quadratic_term():
    p = convex_qp(0)
    rng = np.random.default_rng(0)
    y = -np.abs(rng.standard_normal(p.m))     # the sign that is not vacuous on <= rows
    lp = safe_dual_bound(p.A, p.c, p.row_lb, p.row_ub, p.col_lb, p.col_ub, y)
    qp = safe_qp_bound(p.A, p.c, None, p.row_lb, p.row_ub, p.col_lb, p.col_ub,
                       rng.uniform(0, 3, p.n), y)
    assert np.isfinite(lp) and qp == lp


def test_zero_q_agrees_with_the_lp_bound():
    """A ``Q`` that is present and zero must give the LP bound, whatever
    ``x̂`` is: the tangent plane of a linear function is itself."""
    p = convex_qp(1)
    p.Q = SparseMatrix.from_dense(np.zeros((p.n, p.n)))
    rng = np.random.default_rng(1)
    y = -np.abs(rng.standard_normal(p.m))
    lp = safe_dual_bound(p.A, p.c, p.row_lb, p.row_ub, p.col_lb, p.col_ub, y)
    assert np.isfinite(lp)
    for _ in range(5):
        assert abs(bound(p, rng.uniform(-5, 5, p.n), y) - lp) <= 1e-12 * max(1.0, abs(lp))


def test_vacuous_when_the_tangent_points_at_an_infinite_bound():
    """A gradient component pushing toward an infinite bound gives ``-inf``,
    which prunes nothing and is still correct. The same coordinate with the
    gradient pointing the other way is fine."""
    p = convex_qp(2)
    p.col_ub[0] = INF
    y = np.zeros(p.m)
    # tangent slope on x0 is (Qx̂ + c)_0; find an x̂ that makes it negative
    x_hat = np.zeros(p.n)
    g0 = float(p.Q.matvec(x_hat)[0] + p.c[0])
    if g0 > 0:
        # push x̂_0 negative until the slope flips (Q_00 > 0 for this generator)
        x_hat[0] = -(g0 + 1.0) / float(p.Q.to_dense()[0, 0])
    assert bound(p, x_hat, y) == -np.inf
    # now make the slope positive: the lower bound 0 on x0 is finite
    x_hat[0] = abs(x_hat[0]) + (abs(g0) + 1.0) / float(p.Q.to_dense()[0, 0])
    assert np.isfinite(bound(p, x_hat, y))
