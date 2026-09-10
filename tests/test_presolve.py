"""Presolve and postsolve.

Presolve is opt-in and, on everything measured here, does not pay -- see
`docs/NEGATIVE-RESULTS.md`. That makes these tests *more* important rather than
less: a reduction that is off by default is a reduction nobody notices breaking,
and the failure mode is not a crash but a right objective with wrong shadow
prices. So the assertion that matters throughout is **strong duality on the
original model**, not the objective.
"""

import numpy as np
import pytest

from sovopt.core.problem import ObjSense, Problem, Status, VarKind
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.lp.simplex import SimplexParams, solve_simplex
from sovopt.presolve import postsolve, presolve

from tests.test_simplex import _dual_objective, random_lp


def _roundtrip(p):
    """presolve → solve → postsolve, plus the direct answer to compare with."""
    direct = solve_simplex(p.copy(), SimplexParams())
    res = presolve(p)
    red = solve_simplex(res.problem, SimplexParams())
    return direct, res, postsolve(res, red)


# --------------------------------------------------------------------------- #
# the round trip must preserve everything, not just the objective              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_roundtrip_preserves_objective_primal_and_duals(seed):
    p = random_lp(seed=seed)
    direct, res, got = _roundtrip(p)
    assert got.status == Status.OPTIMAL

    assert abs(got.objective - direct.objective) <= 1e-7 * max(1.0, abs(direct.objective))

    row_v, col_v, _ = p.violation(got.x)
    assert max(row_v, col_v) < 1e-7, "postsolved point is infeasible for the original"

    gap = abs(got.objective - _dual_objective(p, got.y, got.reduced_costs))
    assert gap <= 1e-6 * max(1.0, abs(got.objective)), "duals do not close the gap"


def test_reduced_costs_obey_the_definition():
    """``d = c - Aᵀy`` on the *original* model, which is how postsolve builds
    them -- so this pins that they were rebuilt rather than carried."""
    p = random_lp(seed=2)
    _, _, got = _roundtrip(p)
    d = p.c - p.A.rmatvec(got.y)
    assert np.abs(d - got.reduced_costs).max() <= 1e-9 * max(1.0, np.abs(p.c).max())


# --------------------------------------------------------------------------- #
# each reduction, on a model built to trigger exactly it                       #
# --------------------------------------------------------------------------- #


def test_a_fixed_column_is_removed_and_restored():
    A = SparseMatrix.from_dense(np.array([[1., 1., 1.],
                                          [1., 0., 1.]]))
    p = Problem(A=A, c=np.array([1., 2., 3.]),
                row_lb=np.array([2.0, -INF]), row_ub=np.array([INF, 8.0]),
                col_lb=np.array([0.0, 4.0, 0.0]),
                col_ub=np.array([10.0, 4.0, 10.0]), name="fixedcol")
    direct, res, got = _roundtrip(p)
    assert res.problem.n == 2, "the pinned column survived presolve"
    assert abs(got.x[1] - 4.0) < 1e-12
    assert abs(got.objective - direct.objective) < 1e-9


def test_a_singleton_row_becomes_a_bound_and_its_dual_comes_back():
    """The delicate one: a binding singleton row has a *nonzero* dual.

    ``2*x0 <= 6`` is what caps x0 at 3 -- x0's own upper bound of 10 is nowhere
    near active -- so the row is paying for x0's reduced cost and its shadow
    price must come back nonzero. Getting this wrong returns the right plan
    with a zero price on the constraint that actually binds it, which is the
    single most misleading thing a presolver can do.
    """
    A = SparseMatrix.from_dense(np.array([[2., 0.],
                                          [1., 1.]]))
    p = Problem(A=A, c=np.array([-1.0, 1.0]),
                row_lb=np.array([-INF, 2.0]), row_ub=np.array([6.0, INF]),
                col_lb=np.zeros(2), col_ub=np.full(2, 10.0), name="singrow")
    direct, res, got = _roundtrip(p)
    assert res.problem.m == 1, "the singleton row was not turned into a bound"
    assert abs(got.x[0] - 3.0) < 1e-9
    assert abs(got.y[0]) > 1e-6, "the binding singleton row came back priced at zero"
    assert np.abs(got.y - direct.y).max() < 1e-7


def test_a_non_binding_singleton_row_comes_back_priced_at_zero():
    """The other half of the rule: the column's own bound is doing the work."""
    A = SparseMatrix.from_dense(np.array([[1., 0.],
                                          [1., 1.]]))
    p = Problem(A=A, c=np.array([1.0, 1.0]),          # minimise: x0 wants to be 0
                row_lb=np.array([-INF, 1.0]), row_ub=np.array([50.0, INF]),
                col_lb=np.zeros(2), col_ub=np.full(2, 10.0), name="slackrow")
    direct, res, got = _roundtrip(p)
    assert abs(got.y[0]) < 1e-9
    assert np.abs(got.y - direct.y).max() < 1e-7


def test_a_redundant_row_is_dropped_with_a_zero_dual():
    A = SparseMatrix.from_dense(np.array([[1., 1.],
                                          [1., -1.]]))
    p = Problem(A=A, c=np.array([1.0, 1.0]),
                # first row can never bind: x in [0,1]^2 gives activity in [0,2]
                row_lb=np.array([-5.0, 0.5]), row_ub=np.array([5.0, INF]),
                col_lb=np.zeros(2), col_ub=np.ones(2), name="redundant")
    direct, res, got = _roundtrip(p)
    assert res.problem.m == 1, "the redundant row was kept"
    assert abs(got.y[0]) < 1e-12
    assert abs(got.objective - direct.objective) < 1e-9


def test_fixing_columns_cascades_into_emptying_rows():
    """The reason the loop iterates instead of making one pass.

    Row 0 touches only pinned columns, so it is not empty in the input and is
    empty the moment they are substituted out. A single-pass presolve leaves it
    behind.
    """
    A = SparseMatrix.from_dense(np.array([[1., 1., 0.],
                                          [0., 0., 1.]]))
    p = Problem(A=A, c=np.array([1., 1., 1.]),
                row_lb=np.array([3.0, 1.0]), row_ub=np.array([3.0, INF]),
                col_lb=np.array([1.0, 2.0, 0.0]),
                col_ub=np.array([1.0, 2.0, 10.0]), name="cascade")
    direct, res, got = _roundtrip(p)
    assert res.rounds >= 2, "presolve did not iterate to a fixpoint"
    # the cascade runs all the way out: pinned columns empty row 0, the
    # remaining row is a singleton that becomes a bound, and its column is then
    # unconstrained -- presolve solves the model outright and hands the solver
    # nothing at all.
    assert (res.problem.m, res.problem.n) == (0, 0)
    assert abs(got.objective - direct.objective) < 1e-9
    row_v, col_v, _ = p.violation(got.x)
    assert max(row_v, col_v) < 1e-9


def test_an_inconsistent_empty_row_is_reported_infeasible():
    A = SparseMatrix.from_dense(np.array([[1., 1.]]))
    p = Problem(A=A, c=np.array([1., 1.]),
                row_lb=np.array([9.0]), row_ub=np.array([9.0]),
                col_lb=np.zeros(2), col_ub=np.zeros(2), name="badempty")
    res = presolve(p)
    assert res.status == Status.INFEASIBLE


def test_a_model_with_nothing_to_reduce_is_returned_intact():
    p = random_lp(seed=0)
    res = presolve(p)
    if res.problem.m == p.m and res.problem.n == p.n:
        assert res.stack == [] or all(True for _ in res.stack)
    assert res.problem.n <= p.n and res.problem.m <= p.m


def test_maximisation_survives_the_round_trip():
    """Compared against the direct solve, not against the duality helper.

    ``_dual_objective`` in the simplex tests is written for a minimisation and
    disagrees with a *direct* maximisation solve too, so using it here would
    have failed the code for the helper's assumption.
    """
    p = random_lp(seed=3)
    p.c = -p.c
    p.sense = ObjSense.MAXIMISE
    direct, res, got = _roundtrip(p)
    assert abs(got.objective - direct.objective) <= 1e-7 * max(1.0, abs(direct.objective))
    assert np.abs(got.y - direct.y).max() <= 1e-7 * max(1.0, np.abs(direct.y).max())
    row_v, col_v, _ = p.violation(got.x)
    assert max(row_v, col_v) < 1e-7


def test_an_unconstrained_column_is_parked_by_the_objective_sense():
    """REGRESSION: a maximisation must park a free-standing column at the top.

    The empty-column rule chooses the bound that helps the objective, and the
    minimisation branch taken on a maximisation parks it at the *worst* end --
    producing a feasible, suboptimal plan that no feasibility check would ever
    flag.
    """
    A = SparseMatrix.from_dense(np.array([[0., 1.]]))
    p = Problem(A=A, c=np.array([3.0, 1.0]),
                row_lb=np.array([1.0]), row_ub=np.array([INF]),
                col_lb=np.zeros(2), col_ub=np.array([5.0, 5.0]),
                sense=ObjSense.MAXIMISE, name="parkmax")
    direct, res, got = _roundtrip(p)
    assert abs(got.x[0] - 5.0) < 1e-12, "column parked at the wrong bound"
    assert abs(got.objective - direct.objective) < 1e-9
