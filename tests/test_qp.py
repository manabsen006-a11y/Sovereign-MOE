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
from sovopt.qp import NotConvexError, solve_qp

TOL = 1e-6


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


def test_unconstrained_minimum_inside_the_box():
    """min 0.5(x^2+y^2) - x - 2y  ->  gradient zero at (1,2), objective -2.5."""
    s = solve_qp(qp([[1, 0], [0, 1]], [-1, -2], [[1, 1]], [-INF], [100.],
                    [0, 0], [10, 10]))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective + 2.5) < TOL
    assert np.allclose(s.x, [1.0, 2.0], atol=1e-4)


def test_optimum_is_not_at_a_vertex():
    """min 0.5(x^2+y^2) s.t. x+y >= 1, x,y in [0,1].

    The optimum is (0.5, 0.5) with objective 0.25. The feasible region's
    vertices are (1,0), (0,1) and (1,1), worth 0.5, 0.5 and 1. A vertex method
    cannot reach the answer, so this is the test that shows the QP path is doing
    something the simplex could not.
    """
    s = solve_qp(qp([[1, 0], [0, 1]], [0, 0], [[1, 1]], [1.], [INF],
                    [0, 0], [1, 1]))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 0.25) < TOL
    assert np.allclose(s.x, [0.5, 0.5], atol=1e-4)

    for vertex in ([1, 0], [0, 1], [1, 1]):
        v = np.array(vertex, float)
        assert 0.5 * v @ v > s.objective + 1e-3


def test_min_variance_portfolio():
    """min 0.5 x'diag(1,2,4)x s.t. sum(x)=1, x>=0  ->  x prop. 1/d."""
    s = solve_qp(qp(np.diag([1.0, 2, 4]), [0, 0, 0], [[1, 1, 1]], [1.], [1.],
                    [0, 0, 0], [1, 1, 1]))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 1.0 / 3.5) < TOL
    assert np.allclose(s.x, np.array([1, 0.5, 0.25]) / 1.75, atol=1e-4)


def test_coupled_hessian_with_an_equality_against_a_direct_kkt_solve():
    """The KKT system is small enough to solve exactly, so do that."""
    Q, c = np.array([[2.0, 1], [1, 2]]), np.array([-4.0, -6])
    z = np.linalg.solve(np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 0]]),
                        np.array([4.0, 6, 1]))
    x_star = z[:2]
    want = 0.5 * x_star @ Q @ x_star + c @ x_star

    s = solve_qp(qp(Q, c, [[1, 1]], [1.], [1.], [-100, -100], [100, 100]))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - want) < TOL
    assert np.allclose(s.x, x_star, atol=1e-4)


def test_optimum_pushed_onto_a_bound():
    """min 0.5(x-5)^2 with x <= 2 -> x=2. The gradient is non-zero there."""
    s = solve_qp(qp([[1.0]], [-5.0], [[1.0]], [-INF], [100.], [0.], [2.]))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective + 8.0) < TOL
    assert abs(s.x[0] - 2.0) < 1e-4


def test_maximise_a_concave_objective():
    """max -0.5x^2 + 3x -> x=3, objective 4.5. Exercises the sign flip on Q."""
    s = solve_qp(qp([[-1.0]], [3.0], [[1.0]], [-INF], [10.], [0.], [10.],
                    sense=ObjSense.MAXIMISE))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 4.5) < TOL
    assert abs(s.x[0] - 3.0) < 1e-4


def test_answer_is_feasible_in_the_original_unscaled_model():
    """Feasibility is re-checked where the answer is read, not where it is found."""
    s = solve_qp(qp(np.diag([1.0, 3.0]), [-2.0, -1.0], [[1e4, 1e-3]],
                    [-INF], [1e4], [0, 0], [10, 10]))
    assert s.x[0] * 1e4 + s.x[1] * 1e-3 <= 1e4 + 1e-6
    assert s.info["primal_residual"] <= 1e-6


# --------------------------------------------------------------------------- #
# what it refuses, and why                                                     #
# --------------------------------------------------------------------------- #


def test_indefinite_q_is_refused_not_mis_solved():
    with pytest.raises(NotConvexError):
        solve_qp(qp([[1, 0], [0, -1]], [0, 0], [[1, 1]], [-INF], [1.],
                    [-5, -5], [5, 5]))


def test_simplex_still_refuses_a_quadratic_objective():
    """The LP engines must keep refusing rather than dropping Q silently."""
    from sovopt.lp.simplex import solve_simplex
    with pytest.raises(NotImplementedError):
        solve_simplex(qp([[1, 0], [0, 1]], [-1, -2], [[1, 1]], [-INF], [100.],
                         [0, 0], [10, 10]))


def test_miqp_is_refused():
    from sovopt.cli import solve
    p = qp([[1, 0], [0, 1]], [-1, -2], [[1, 1]], [-INF], [100.], [0, 0], [10, 10])
    p.kind = np.full(2, VarKind.INTEGER)
    with pytest.raises(NotImplementedError, match="MIQP"):
        solve(p)


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
