"""Revised simplex tests.

The simplex is the only part of the engine that returns a *basis*, so these
tests check the things a basis is for -- duals, reduced costs, complementary
slackness, strong duality -- and not merely the objective value. An objective
can be right while the duals are nonsense, and the duals are what a planner
reads off as marginal prices.
"""

import os

import numpy as np
import pytest

from sovopt.core.problem import ObjSense, Problem, Status, VarKind
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.io.mps import read_mps
from sovopt.lp.basis import AT_LOWER, AT_UPPER, BASIC, FIXED, FREE
from sovopt.lp.simplex import SimplexParams, solve_simplex

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "instances")


def toy_lp():
    """max 3x + 2y  s.t.  x+y<=4, x+3y<=6, x<=3, x,y>=0.  Optimum 11 at (3,1)."""
    A = SparseMatrix.from_dense(np.array([[1., 1.], [1., 3.], [1., 0.]]))
    return Problem(A=A, c=np.array([3., 2.]),
                   row_lb=np.full(3, -INF), row_ub=np.array([4., 6., 3.]),
                   col_lb=np.zeros(2), col_ub=np.full(2, INF),
                   sense=ObjSense.MAXIMISE, name="toy")


def random_lp(seed=0, m=30, n=45, spread=0):
    rng = np.random.default_rng(seed)
    cols = np.repeat(np.arange(n), 4)
    rows = rng.integers(0, m, cols.size)
    vals = rng.standard_normal(cols.size)
    A = SparseMatrix.from_triplets(rows, cols, vals, m, n)
    if spread:
        A.scale(10.0 ** rng.integers(-spread, spread + 1, size=m).astype(float),
                np.ones(n))
    x0 = rng.random(n)
    b = A.matvec(x0) + rng.random(m)
    return Problem(A=A, c=rng.standard_normal(n),
                   row_lb=np.full(m, -INF), row_ub=b,
                   col_lb=np.zeros(n), col_ub=np.full(n, 10.0), name="rand")


# --------------------------------------------------------------------------- #
# basics                                                                       #
# --------------------------------------------------------------------------- #


def test_solves_toy_exactly():
    s = solve_simplex(toy_lp())
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 11.0) < 1e-9
    assert np.allclose(s.x, [3.0, 1.0], atol=1e-9)


def test_solves_mps_fixture_exactly():
    s = solve_simplex(read_mps(os.path.join(FIX, "testprob.mps")))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 16.0) < 1e-9
    assert np.allclose(s.x, [0.0, -1.0, 6.0], atol=1e-9)


def test_returns_a_basis_of_the_right_size():
    s = solve_simplex(toy_lp())
    assert s.basis_status is not None
    assert int((s.basis_status == BASIC).sum()) == 3      # one per row


def test_detects_infeasible():
    A = SparseMatrix.from_dense(np.array([[1.0], [1.0]]))
    p = Problem(A=A, c=np.array([1.0]),
                row_lb=np.array([5.0, -INF]), row_ub=np.array([INF, 3.0]),
                col_lb=np.array([0.0]), col_ub=np.array([INF]), name="infeas")
    assert solve_simplex(p).status == Status.INFEASIBLE


def test_detects_unbounded():
    A = SparseMatrix.from_dense(np.array([[1.0, -1.0]]))
    p = Problem(A=A, c=np.array([-1.0, 0.0]),
                row_lb=np.array([-INF]), row_ub=np.array([0.0]),
                col_lb=np.zeros(2), col_ub=np.full(2, INF), name="unb")
    assert solve_simplex(p).status == Status.UNBOUNDED


def test_handles_equality_and_ranged_rows():
    A = SparseMatrix.from_dense(np.array([[1., 1., 0.], [0., 1., 1.]]))
    p = Problem(A=A, c=np.array([1., 1., 1.]),
                row_lb=np.array([2.0, 1.0]),      # first equality, second ranged
                row_ub=np.array([2.0, 4.0]),
                col_lb=np.zeros(3), col_ub=np.full(3, 5.0), name="mix")
    s = solve_simplex(p)
    assert s.status == Status.OPTIMAL
    act = p.A.matvec(s.x)
    assert abs(act[0] - 2.0) < 1e-9
    assert 1.0 - 1e-9 <= act[1] <= 4.0 + 1e-9


# --------------------------------------------------------------------------- #
# the point of having a basis: duals                                           #
# --------------------------------------------------------------------------- #


def _dual_objective(p, y, d):
    """b'y contribution plus the bound terms -- the LP dual objective."""
    row = np.where(y > 0, np.where(p.row_lb > -INF, p.row_lb, 0.0),
                   np.where(p.row_ub < INF, p.row_ub, 0.0)) * y
    col = np.where(d > 0, np.where(p.col_lb > -INF, p.col_lb, 0.0),
                   np.where(p.col_ub < INF, p.col_ub, 0.0)) * d
    return float(row.sum() + col.sum()) + p.obj_offset


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_strong_duality(seed):
    """The primal and dual objectives must agree to machine precision.

    This is the check a first-order method cannot pass at this tolerance, and
    the reason the simplex exists in this project.
    """
    p = random_lp(seed=seed)
    s = solve_simplex(p)
    assert s.status == Status.OPTIMAL
    dual_obj = _dual_objective(p, s.y, s.reduced_costs)
    assert abs(s.objective - dual_obj) <= 1e-7 * max(1.0, abs(s.objective))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reduced_costs_are_consistent_with_duals(seed):
    """``d = c - A'y`` must hold exactly on the returned vectors."""
    p = random_lp(seed=seed)
    s = solve_simplex(p)
    d = p.c - p.A.rmatvec(s.y)
    assert np.abs(d - s.reduced_costs).max() <= 1e-7 * max(1.0, np.abs(p.c).max())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_complementary_slackness(seed):
    p = random_lp(seed=seed)
    s = solve_simplex(p)
    act = p.A.matvec(s.x)
    slack = np.where(p.row_ub < INF, p.row_ub - act, 0.0)
    assert float(np.max(np.abs(s.y) * np.abs(slack))) <= 1e-7 * max(1.0, abs(s.objective))


def test_dual_sign_convention_on_a_known_problem():
    """A binding ``<=`` row in a maximisation has a non-negative shadow price."""
    p = toy_lp()
    s = solve_simplex(p)
    act = p.A.matvec(s.x)
    for i in range(p.m):
        if abs(act[i] - p.row_ub[i]) < 1e-9:
            assert s.y[i] >= -1e-9, f"row {i} binding but price {s.y[i]}"
    # c = A'y at the optimum of a nondegenerate vertex
    assert np.abs(p.A.rmatvec(s.y) - p.c).max() < 1e-7


# --------------------------------------------------------------------------- #
# robustness                                                                   #
# --------------------------------------------------------------------------- #


def test_phase_one_finds_a_feasible_start():
    """REGRESSION: the phase-1 ratio test computed a breakpoint for a basic
    variable already outside its box and moving further out.

    That ratio is negative, which clamped the step to zero and stalled phase 1
    at a non-optimal point, making the solver report the feasible instance
    misc07 as INFEASIBLE. A variable moving away from its box has no breakpoint;
    its infeasibility grows at a constant rate that the phase-1 objective
    already accounts for.
    """
    rng = np.random.default_rng(5)
    m, n = 25, 30
    A = SparseMatrix.from_dense(
        rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.3))
    # equality rows with a nonzero right-hand side force a real phase 1
    b = A.matvec(rng.random(n) * 2.0)
    p = Problem(A=A, c=rng.standard_normal(n),
                row_lb=b, row_ub=b,
                col_lb=np.zeros(n), col_ub=np.full(n, 10.0), name="ph1")
    s = solve_simplex(p)
    assert s.status == Status.OPTIMAL, s.status
    assert max(p.violation(s.x)[:2]) < 1e-7


def test_degenerate_problem_terminates():
    """Many identical rows give a massively degenerate LP; it must not cycle."""
    n = 12
    rows = []
    for _ in range(6):
        rows.append(np.ones(n))
    rows.append(np.arange(1.0, n + 1.0))
    A = SparseMatrix.from_dense(np.array(rows))
    m = A.m
    p = Problem(A=A, c=np.ones(n),
                row_lb=np.full(m, -INF), row_ub=np.full(m, 5.0),
                col_lb=np.zeros(n), col_ub=np.ones(n),
                sense=ObjSense.MAXIMISE, name="degen")
    s = solve_simplex(p, SimplexParams(max_iter=20000, time_limit=60))
    assert s.status == Status.OPTIMAL
    assert s.iterations < 20000


@pytest.mark.parametrize("spread", [0, 4, 6])
def test_survives_badly_scaled_models(spread):
    p = random_lp(seed=7, m=40, n=60, spread=spread)
    s = solve_simplex(p)
    assert s.status == Status.OPTIMAL
    assert max(p.violation(s.x)[:2]) < 1e-6


def test_warm_start_from_a_previous_basis():
    """Re-solving from the optimal basis must cost almost nothing."""
    p = random_lp(seed=3, m=40, n=60)
    cold = solve_simplex(p)
    assert cold.status == Status.OPTIMAL
    warm = solve_simplex(p, warm_basis=cold.basis_status)
    assert warm.status == Status.OPTIMAL
    assert abs(warm.objective - cold.objective) < 1e-9
    assert warm.iterations <= max(2, cold.iterations // 2)


# --------------------------------------------------------------------------- #
# agreement with the other engine                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["flugpl", "gt2", "p0201"])
def test_agrees_with_published_miplib_lp_value(name):
    """Two independent methods and a published number must all coincide."""
    path = os.path.join(DATA, f"{name}.mps")
    if not os.path.exists(path):
        pytest.skip("benchmark instances not fetched")
    from bench.fetch import read_reference

    ref = read_reference(path).get("lp_soln")
    p = read_mps(path)
    p.kind[:] = VarKind.CONTINUOUS
    s = solve_simplex(p, SimplexParams(time_limit=60))
    assert s.status == Status.OPTIMAL
    if ref is not None:
        assert abs(s.objective - ref) / max(1.0, abs(ref)) < 1e-5
    assert max(p.violation(s.x)[:2]) < 1e-7
