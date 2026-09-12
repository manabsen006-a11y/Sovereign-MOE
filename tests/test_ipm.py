"""Interior-point method: agreement with the exact simplex, and the three
starting/termination bugs that made it look like a slow solver rather than a
broken one.

The IPM is the third LP engine, and the one with no independent notion of
correctness of its own -- it stops when residuals are small, and "small" is a
judgement. So most of what is asserted here is *agreement with the revised
simplex*, which is exact and which the rest of the suite already pins against
published values. Where the two disagree, the simplex is right.
"""

import os

import numpy as np
import pytest

from sovopt.core.problem import ObjSense, Problem, Status, VarKind
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.io.mps import read_mps
from sovopt.lp.ipm import IPMParams, solve_ipm
from sovopt.lp.simplex import solve_simplex

# tests/ is a package, so the shared LP generator and dual-objective
# helper are imported by their qualified name rather than as a top-level
# module.
from tests.test_simplex import _dual_objective, random_lp, toy_lp


# --------------------------------------------------------------------------- #
# agreement with the exact engine                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_the_simplex_objective(seed):
    p = random_lp(seed=seed)
    exact = solve_simplex(p)
    ipm = solve_ipm(p)
    assert ipm.status == Status.OPTIMAL
    assert exact.status == Status.OPTIMAL
    assert abs(ipm.objective - exact.objective) <= 1e-6 * max(1.0, abs(exact.objective))


@pytest.mark.parametrize("seed", [0, 1, 3])
def test_row_duals_match_the_simplex(seed):
    """Not just the objective -- the marginal prices too.

    A sign error in the dual is invisible in the objective and invisible in
    every feasibility check; it shows up only here. The first version of this
    module negated ``y`` on the way out and agreed with the simplex to 1e-15
    in objective while reporting every shadow price backwards.
    """
    p = random_lp(seed=seed)
    exact = solve_simplex(p)
    ipm = solve_ipm(p)
    scale = max(1.0, float(np.abs(exact.y).max(initial=0.0)))
    assert np.abs(exact.y - ipm.y).max() <= 1e-5 * scale


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reduced_costs_obey_the_repo_convention(seed):
    """``d = c - Aᵀy`` on the vectors as returned, the same identity the
    simplex is held to."""
    p = random_lp(seed=seed)
    s = solve_ipm(p)
    d = p.c - p.A.rmatvec(s.y)
    assert np.abs(d - s.reduced_costs).max() <= 1e-7 * max(1.0, np.abs(p.c).max())


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_strong_duality(seed):
    p = random_lp(seed=seed)
    s = solve_ipm(p)
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - _dual_objective(p, s.y, s.reduced_costs)) \
        <= 1e-6 * max(1.0, abs(s.objective))


def test_solves_a_maximisation_the_same_way():
    p = toy_lp()
    s = solve_ipm(p)
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 11.0) < 1e-6
    # the toy is degenerate -- three rows active at a two-dimensional vertex --
    # so the dual is not unique and only the identity can be asserted
    d = p.c - p.A.rmatvec(s.y)
    assert np.abs(d - s.reduced_costs).max() < 1e-7


def test_solves_a_quadratic_objective_and_the_lp_path_is_unchanged():
    """It used to refuse a Q. Now Q joins the (1,1) block: with Q = I the
    optimum is the projection of -c onto the feasible set, which the
    proximal QP computes independently. And Q = 0 must give the LP answer
    to the last digit, since the LP pattern builder is a separate branch."""
    from sovopt.qp import QPParams, solve_qp
    p = random_lp(seed=0)
    lp = solve_ipm(p.copy())
    p.Q = SparseMatrix.from_dense(np.eye(p.n))
    s = solve_ipm(p.copy())
    assert s.status == Status.OPTIMAL
    ref = solve_qp(p.copy(), QPParams(method="proximal", time_limit=60))
    assert abs(s.objective - ref.objective) <= 1e-6 * max(1.0, abs(ref.objective))
    z = p.copy()
    z.Q = SparseMatrix.from_dense(np.zeros((p.n, p.n)))
    s0 = solve_ipm(z)
    assert abs(s0.objective - lp.objective) <= 1e-9 * max(1.0, abs(lp.objective))


# --------------------------------------------------------------------------- #
# row and column shapes the slack formulation has to absorb                    #
# --------------------------------------------------------------------------- #


def test_equality_and_ranged_and_free_rows():
    """One of each row type in a single model.

    Every row becomes a bounded slack: an equality is a slack pinned between
    equal bounds, a ranged row a slack in an interval, a free row a slack with
    no bound at all. If any of the three is mishandled the KKT block for it is
    singular, so this is as much a factorisation test as a modelling one.
    """
    A = SparseMatrix.from_dense(np.array([[1., 1., 0.],
                                          [0., 1., 1.],
                                          [1., 0., 1.],
                                          [1., 1., 1.]]))
    p = Problem(A=A, c=np.array([1., 2., 3.]),
                row_lb=np.array([2.0, 1.0, -INF, -INF]),
                row_ub=np.array([2.0, 4.0, 5.0, INF]),
                col_lb=np.zeros(3), col_ub=np.full(3, 10.0), name="mixed")
    exact = solve_simplex(p)
    s = solve_ipm(p)
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - exact.objective) < 1e-6
    row_v, col_v, _ = p.violation(s.x)
    assert max(row_v, col_v) < 1e-7


def test_free_and_fixed_columns():
    """A free column has no finite Θ⁻¹ and a fixed one has no interior; both
    make a diagonal block singular without regularisation."""
    A = SparseMatrix.from_dense(np.array([[1., 1., 1.],
                                          [1., -1., 0.]]))
    p = Problem(A=A, c=np.array([1., 1., -1.]),
                row_lb=np.array([1.0, -INF]), row_ub=np.array([1.0, 3.0]),
                col_lb=np.array([-INF, 2.0, 0.0]),
                col_ub=np.array([INF, 2.0, 5.0]), name="freefix")
    exact = solve_simplex(p)
    s = solve_ipm(p)
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - exact.objective) < 1e-6
    assert abs(s.x[1] - 2.0) < 1e-9          # the fixed column stayed fixed


# --------------------------------------------------------------------------- #
# regressions                                                                  #
# --------------------------------------------------------------------------- #


def test_pinned_variables_enter_the_dual_objective():
    """REGRESSION: the duality gap must close on a model with equality rows.

    An equality row's slack is *pinned*, and a pinned variable's reduced cost
    is free in sign: it carries no dual residual and it contributes to the dual
    objective directly rather than through a bound multiplier. Leaving that
    term out does not perturb the answer -- it stops the solver recognising it.
    Measured before the fix on flugpl: primal residual 1e-16, dual residual
    1e-16, mu 5e-13, and a duality gap frozen at 3.594e-02 for 180 iterations
    while the objective sat on the published optimum the whole time.
    """
    A = SparseMatrix.from_dense(np.array([[1., 1., 1., 0.],
                                          [0., 1., 1., 1.],
                                          [1., 0., 0., 1.]]))
    p = Problem(A=A, c=np.array([1., 3., 2., 1.]),
                row_lb=np.array([3.0, 2.0, -INF]),    # two equalities
                row_ub=np.array([3.0, 2.0, 5.0]),
                col_lb=np.zeros(4), col_ub=np.full(4, 10.0), name="eqrows")
    exact = solve_simplex(p)
    assert exact.status == Status.OPTIMAL          # the model is feasible
    s = solve_ipm(p, IPMParams(max_iter=60))
    assert s.status == Status.OPTIMAL, "gap never closed on an equality model"
    assert s.iterations < 60
    assert abs(s.objective - exact.objective) < 1e-6


def test_detects_an_infeasible_model():
    """An infeasible model must not come back as "I ran out of iterations".

    The three equalities here determine x uniquely and that unique point has
    x[1] = -1/3, against a lower bound of zero. The signature is distinctive:
    the dual residual falls to 1e-50 and mu to 1e-4 while the primal residual
    sits at 5.556e-02 and never moves again. Detection is a stagnation test
    rather than a Farkas certificate, so INFEASIBLE_OR_UNBOUNDED is the honest
    status -- the simplex, which does produce a certificate, says INFEASIBLE.
    """
    A = SparseMatrix.from_dense(np.array([[1., 1., 1.],
                                          [1., 2., 0.],
                                          [0., 1., 2.]]))
    p = Problem(A=A, c=np.array([1., 3., 2.]),
                row_lb=np.array([3.0, 2.0, 1.0]),
                row_ub=np.array([3.0, 2.0, 1.0]),
                col_lb=np.zeros(3), col_ub=np.full(3, 10.0), name="infeas")
    assert solve_simplex(p).status == Status.INFEASIBLE
    s = solve_ipm(p, IPMParams(max_iter=120))
    assert s.status == Status.INFEASIBLE_OR_UNBOUNDED
    assert s.iterations < 120           # detected, not merely exhausted


def test_a_single_wide_bound_does_not_set_the_starting_scale():
    """REGRESSION: one loose bound must not drag the starting point with it.

    Starting a boxed variable at its midpoint is the obvious choice and it
    fails whenever one box is far wider than the rest: on mas76 a single column
    is bounded by 1e12 while every other is bounded by 1, the midpoint start
    puts it at 5e11 for an initial objective of 1.28e14, and the iteration
    never recovers -- the primal residual stalls at 1.2e-2 with the step length
    pinned at zero. The start clips zero into the box instead, so the scale
    comes from the data rather than from the loosest bound.
    """
    rng = np.random.default_rng(7)
    n, m = 24, 6
    cols = np.repeat(np.arange(n), 3)
    rows = rng.integers(0, m, cols.size)
    A = SparseMatrix.from_triplets(rows, cols, rng.random(cols.size) + 0.5, m, n)
    ub = np.ones(n)
    ub[0] = 1e12                                   # the one wide column
    p = Problem(A=A, c=-rng.random(n), row_lb=np.full(m, -INF),
                row_ub=np.full(m, 5.0), col_lb=np.zeros(n), col_ub=ub,
                name="wide")
    exact = solve_simplex(p)
    s = solve_ipm(p, IPMParams(max_iter=80))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - exact.objective) <= 1e-6 * max(1.0, abs(exact.objective))


def test_starting_point_corrects_a_bounded_point_not_a_minimum_norm_one():
    """REGRESSION: the least-squares start must respect large lower bounds.

    Mehrotra's start takes the minimum-norm point satisfying the rows. With
    two-sided bounds that is the wrong problem: on khb05250 the columns have
    lower bounds up to 1.3e4, the minimum-norm x is near zero, and clipping it
    back into the box drags every column up to its bound again -- the initial
    primal residual is 2.6e4 with the least-squares solve and 2.6e4 without,
    i.e. the work is entirely wasted and the model never solves. Solving for
    the minimum-norm *correction* to a point that already respects the bounds
    keeps the correction small enough that the clip only trims it.
    """
    rng = np.random.default_rng(11)
    n, m = 30, 10
    cols = np.repeat(np.arange(n), 3)
    rows = rng.integers(0, m, cols.size)
    A = SparseMatrix.from_triplets(rows, cols, rng.random(cols.size) + 0.5, m, n)
    lo = np.full(n, 1.0e4)                        # far from the origin
    hi = lo + 5.0e3
    b = A.matvec(lo + 2.5e3)
    p = Problem(A=A, c=rng.standard_normal(n),
                row_lb=b, row_ub=b,               # equality rows, as khb05250
                col_lb=lo, col_ub=hi, name="highlo")
    exact = solve_simplex(p)
    s = solve_ipm(p, IPMParams(max_iter=80))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - exact.objective) <= 1e-6 * max(1.0, abs(exact.objective))


# --------------------------------------------------------------------------- #
# regularisation                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_two_regularisations_reach_the_same_answer(seed):
    """Static regularisation refined away against the unregularised matrix,
    and the proximal-point form that solves the regularised system as the
    system: different directions, one optimum."""
    p = random_lp(seed=seed)
    a = solve_ipm(p, IPMParams(regularisation="static"))
    b = solve_ipm(p, IPMParams(regularisation="pmm"))
    assert a.status == b.status == Status.OPTIMAL
    assert abs(a.objective - b.objective) <= 1e-7 * max(1.0, abs(a.objective))
    scale = max(1.0, float(np.abs(a.y).max(initial=0.0)))
    assert np.abs(a.y - b.y).max() <= 1e-5 * scale


def test_an_unknown_regularisation_is_refused():
    with pytest.raises(ValueError):
        solve_ipm(random_lp(seed=0), IPMParams(regularisation="dynamic"))


def test_a_quadratic_with_no_linear_term_is_scaled_by_its_hessian():
    """The 8559 shape in miniature: ``c = 0``, a diagonal ``Q`` spanning
    four decades, equality rows, a box. The optimum does not depend on the
    scale of the objective; the iteration count did, before the objective
    scale looked at ``Q``."""
    rng = np.random.default_rng(3)
    n, m = 40, 20
    cols = np.repeat(np.arange(n), 2)
    rows = rng.integers(0, m, cols.size)
    A = SparseMatrix.from_triplets(rows, cols, rng.integers(1, 5, cols.size).astype(float), m, n)
    b = A.matvec(rng.uniform(0.5, 5.0, n))
    Q = SparseMatrix.from_dense(np.diag(rng.uniform(4.0, 95000.0, n)))
    p = Problem(A=A, c=np.zeros(n), row_lb=b, row_ub=b,
                col_lb=np.full(n, 0.1), col_ub=np.full(n, 10.0), Q=Q)
    s = solve_ipm(p)
    assert s.status == Status.OPTIMAL
    # the hand-scaled copy has the same argmin
    Qs = Q.copy()
    Qs.cx *= 1e-4
    Qs.rx *= 1e-4
    ps = Problem(A=A, c=np.zeros(n), row_lb=b, row_ub=b,
                 col_lb=np.full(n, 0.1), col_ub=np.full(n, 10.0), Q=Qs)
    ss = solve_ipm(ps)
    assert ss.status == Status.OPTIMAL
    assert abs(s.objective - 1e4 * ss.objective) <= 1e-6 * abs(s.objective)
    assert np.abs(s.x - ss.x).max() <= 1e-5
    assert s.iterations <= ss.iterations + 3


# --------------------------------------------------------------------------- #
# a real instance, end to end                                                  #
# --------------------------------------------------------------------------- #


def test_agrees_with_the_published_lp_value_on_a_benchmark():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "instances", "khb05250.mps")
    if not os.path.exists(path):
        pytest.skip("benchmark instances not fetched")
    p = read_mps(path)
    p.kind[:] = VarKind.CONTINUOUS
    s = solve_ipm(p)
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 95919464.0) <= 1e-6 * 95919464.0
    row_v, col_v, _ = p.violation(s.x)
    assert max(row_v, col_v) < 1e-6
