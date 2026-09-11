"""Convex quadratic programming.

Every expected value here is derived by hand -- from a closed-form optimum or
from solving the KKT system directly -- rather than recorded from a previous run
of this solver. A test that only checks the solver still agrees with itself
cannot catch the bug where it was wrong all along, which is the bug that
matters. The one thing a QP test must establish is that the answer is *not* a
vertex, because that is exactly what the simplex could never have produced.
"""

import numpy as np
import pytest

from sovopt.core.problem import ObjSense, Problem, Status, VarKind
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.qp import NotConvexError, QPParams, solve_qp

TOL = 1e-6

METHODS = ["ipm", "proximal"]
"""Every hand-derived optimum is checked against both engines: the interior
point that is the default, and the proximal method that is the GPU path."""


def qp(Q, c, A, rl, ru, lo, hi, sense=ObjSense.MINIMISE):
    n, m = len(c), len(rl)
    return Problem(
        A=SparseMatrix.from_dense(np.array(A, float).reshape(m, n)),
        c=np.array(c, float),
        row_lb=np.array(rl, float), row_ub=np.array(ru, float),
        col_lb=np.array(lo, float), col_ub=np.array(hi, float),
        kind=np.full(n, VarKind.CONTINUOUS),
        Q=SparseMatrix.from_dense(np.array(Q, float)), sense=sense)


# --------------------------------------------------------------------------- #
# optima derived by hand                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", METHODS)
def test_unconstrained_minimum_inside_the_box(method):
    """min 0.5(x^2+y^2) - x - 2y  ->  gradient zero at (1,2), objective -2.5."""
    s = solve_qp(qp([[1, 0], [0, 1]], [-1, -2], [[1, 1]], [-INF], [100.],
                    [0, 0], [10, 10]), QPParams(method=method))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective + 2.5) < TOL
    assert np.allclose(s.x, [1.0, 2.0], atol=1e-4)


@pytest.mark.parametrize("method", METHODS)
def test_optimum_is_not_at_a_vertex(method):
    """min 0.5(x^2+y^2) s.t. x+y >= 1, x,y in [0,1].

    The optimum is (0.5, 0.5) with objective 0.25. The feasible region's
    vertices are (1,0), (0,1) and (1,1), worth 0.5, 0.5 and 1. A vertex method
    cannot reach the answer, so this is the test that shows the QP path is doing
    something the simplex could not.
    """
    s = solve_qp(qp([[1, 0], [0, 1]], [0, 0], [[1, 1]], [1.], [INF],
                    [0, 0], [1, 1]), QPParams(method=method))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 0.25) < TOL
    assert np.allclose(s.x, [0.5, 0.5], atol=1e-4)

    for vertex in ([1, 0], [0, 1], [1, 1]):
        v = np.array(vertex, float)
        assert 0.5 * v @ v > s.objective + 1e-3


@pytest.mark.parametrize("method", METHODS)
def test_min_variance_portfolio(method):
    """min 0.5 x'diag(1,2,4)x s.t. sum(x)=1, x>=0  ->  x prop. 1/d."""
    s = solve_qp(qp(np.diag([1.0, 2, 4]), [0, 0, 0], [[1, 1, 1]], [1.], [1.],
                    [0, 0, 0], [1, 1, 1]), QPParams(method=method))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 1.0 / 3.5) < TOL
    assert np.allclose(s.x, np.array([1, 0.5, 0.25]) / 1.75, atol=1e-4)


@pytest.mark.parametrize("method", METHODS)
def test_coupled_hessian_with_an_equality_against_a_direct_kkt_solve(method):
    """The KKT system is small enough to solve exactly, so do that."""
    Q, c = np.array([[2.0, 1], [1, 2]]), np.array([-4.0, -6])
    z = np.linalg.solve(np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 0]]),
                        np.array([4.0, 6, 1]))
    x_star = z[:2]
    want = 0.5 * x_star @ Q @ x_star + c @ x_star

    s = solve_qp(qp(Q, c, [[1, 1]], [1.], [1.], [-100, -100], [100, 100]), QPParams(method=method))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - want) < TOL
    assert np.allclose(s.x, x_star, atol=1e-4)


@pytest.mark.parametrize("method", METHODS)
def test_optimum_pushed_onto_a_bound(method):
    """min 0.5(x-5)^2 with x <= 2 -> x=2. The gradient is non-zero there."""
    s = solve_qp(qp([[1.0]], [-5.0], [[1.0]], [-INF], [100.], [0.], [2.]), QPParams(method=method))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective + 8.0) < TOL
    assert abs(s.x[0] - 2.0) < 1e-4


@pytest.mark.parametrize("method", METHODS)
def test_maximise_a_concave_objective(method):
    """max -0.5x^2 + 3x -> x=3, objective 4.5. Exercises the sign flip on Q."""
    s = solve_qp(qp([[-1.0]], [3.0], [[1.0]], [-INF], [10.], [0.], [10.],
                    sense=ObjSense.MAXIMISE), QPParams(method=method))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 4.5) < TOL
    assert abs(s.x[0] - 3.0) < 1e-4


@pytest.mark.parametrize("method", METHODS)
def test_answer_is_feasible_in_the_original_unscaled_model(method):
    """Feasibility is re-checked where the answer is read, not where it is found."""
    s = solve_qp(qp(np.diag([1.0, 3.0]), [-2.0, -1.0], [[1e4, 1e-3]],
                    [-INF], [1e4], [0, 0], [10, 10]), QPParams(method=method))
    assert s.x[0] * 1e4 + s.x[1] * 1e-3 <= 1e4 + 1e-6
    # each engine reports its violation under its own name
    assert s.info.get("primal_residual", s.info.get("worst_violation")) <= 1e-6


# --------------------------------------------------------------------------- #
# what it refuses, and why                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", METHODS)
def test_indefinite_q_is_refused_not_mis_solved(method):
    with pytest.raises(NotConvexError):
        solve_qp(qp([[1, 0], [0, -1]], [0, 0], [[1, 1]], [-INF], [1.],
                    [-5, -5], [5, 5]), QPParams(method=method))


def test_simplex_still_refuses_a_quadratic_objective():
    """The LP engines must keep refusing rather than dropping Q silently."""
    from sovopt.lp.simplex import solve_simplex
    with pytest.raises(NotImplementedError):
        solve_simplex(qp([[1, 0], [0, 1]], [-1, -2], [[1, 1]], [-INF], [100.],
                         [0, 0], [10, 10]))


def test_miqp_is_solved_now_and_its_answer_is_integral():
    """This used to assert a refusal, and the refusal was the right answer
    while branch-and-bound over QP nodes did not exist. It does now
    (:mod:`sovopt.mip.miqp`), so what is asserted is the thing the refusal was
    protecting: the quadratic term is honoured and the answer is integral.
    Solving a MIQP as a MILP would drop Q and return a confident wrong number,
    which is what the old message warned about.
    """
    from sovopt.cli import solve
    p = qp([[1, 0], [0, 1]], [-1, -2], [[1, 1]], [-INF], [100.], [0, 0], [10, 10])
    p.kind = np.full(2, VarKind.INTEGER)
    s = solve(p, time_limit=60)
    assert s.status == Status.OPTIMAL
    assert s.method == "miqp-bb"
    assert np.abs(s.x - np.round(s.x)).max() < 1e-6
    # the reported value must include the quadratic term, not just c'x
    assert abs(s.objective - p.objective(s.x)) < 1e-9
    assert abs(s.objective - (-2.5)) < 1e-6      # optimum at (1, 2)


def test_dispatcher_routes_a_convex_qp_to_the_qp_solver():
    from sovopt.cli import solve
    s = solve(qp([[1, 0], [0, 1]], [-1, -2], [[1, 1]], [-INF], [100.],
                 [0, 0], [10, 10]))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective + 2.5) < TOL
    assert "qp" in s.method


def test_an_lp_passed_to_the_qp_solver_still_works():
    p = qp([[1.0]], [1.0], [[1.0]], [1.], [INF], [0.], [10.])
    p.Q = None
    s = solve_qp(p)
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 1.0) < TOL


def test_the_two_engines_agree_and_say_which_they_are():
    p = qp([[2, 0.5], [0.5, 1]], [-1, -2], [[1, 1]], [-INF], [1.5], [0, 0], [10, 10])
    a = solve_qp(p.copy(), QPParams(method="ipm"))
    b = solve_qp(p.copy(), QPParams(method="proximal"))
    assert a.status == b.status == Status.OPTIMAL
    assert abs(a.objective - b.objective) < 1e-6
    assert a.method == "qp[ipm]" and b.method.startswith("qp[proximal")
    assert np.allclose(a.x, b.x, atol=1e-5)
    with pytest.raises(ValueError):
        solve_qp(p.copy(), QPParams(method="magic"))


def test_the_interior_point_reaches_the_simplex_digits():
    """The reason it is the default: 1e-9 on the KKT residuals against the
    proximal method's 1e-6 to 1e-8, checked on the portfolio model whose
    answer is known in closed form: x = (4/7, 2/7, 1/7), f = 2/7."""
    s = solve_qp(qp(np.diag([1.0, 2, 4]), [0, 0, 0], [[1, 1, 1]], [1.], [1.],
                    [0, 0, 0], [1, 1, 1]), QPParams(method="ipm"))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 2.0 / 7) < 1e-9
    assert np.allclose(s.x, [4 / 7, 2 / 7, 1 / 7], atol=1e-8)
