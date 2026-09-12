"""The verifier's optimality certificate.

Feasibility is the verifier's old job. With duals it now also bounds the
optimum -- Neumaier-Shcherbina on the tangent plane, valid for any dual
vector -- and reports the gap to the point's own objective. The point of the
tests: a simplex optimum certifies with a gap at rounding level; a point that
is feasible but not optimal does not, and the verdict says so rather than
passing it; a maximisation certifies in its own sense; the perturbation the
certificate needed is reported and small; and duals that certify nothing are
reported as certifying nothing.
"""

import numpy as np
import pytest

from bench.verify import certified_bound, verify
from sovopt.core.problem import ObjSense, Problem, Status
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.lp.ipm import IPMParams, solve_ipm
from sovopt.lp.simplex import SimplexParams, solve_simplex
from tests.test_simplex import random_lp


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_simplex_optimum_certifies_at_rounding_level(seed):
    p = random_lp(seed=seed)
    s = solve_simplex(p)
    assert s.status == Status.OPTIMAL
    v = verify(p, s.x, s.objective, y=s.y)
    assert v.ok, v.report()
    opt = [c for c in v.checks if c[0] == "optimality"][0]
    assert opt[1]
    bnd, pert = certified_bound(p, s.x, s.y)
    assert bnd <= s.objective + 1e-9 * max(1.0, abs(s.objective))
    assert s.objective - bnd <= 1e-9 * max(1.0, abs(s.objective))
    assert pert <= 1e-9


@pytest.mark.parametrize("seed", [0, 1])
def test_an_interior_point_optimum_certifies_too(seed):
    p = random_lp(seed=seed)
    s = solve_ipm(p, IPMParams())
    assert s.status == Status.OPTIMAL
    v = verify(p, s.x, s.objective, y=s.y, opt_tol=1e-7)
    assert v.ok, v.report()


def test_a_feasible_but_suboptimal_point_is_not_certified():
    """Take an optimal pair, then move the point to another feasible vertex
    of the box: feasibility passes, the certificate does not."""
    p = random_lp(seed=5)
    s = solve_simplex(p)
    assert s.status == Status.OPTIMAL
    # a second, worse feasible point: the optimum of a perturbed objective
    q = p.copy()
    rng = np.random.default_rng(1)
    q.c = p.c + 3.0 * rng.standard_normal(p.n) * (1.0 + np.abs(p.c))
    w = solve_simplex(q)
    assert w.status == Status.OPTIMAL
    worse = float(p.c @ w.x) + p.obj_offset
    assert worse > s.objective + 1e-6 * max(1.0, abs(s.objective))
    v = verify(p, w.x, worse, y=s.y)          # the true duals, the wrong point
    feas = [c for c in v.checks if c[0] != "optimality"]
    assert all(c[1] for c in feas)
    opt = [c for c in v.checks if c[0] == "optimality"][0]
    assert not opt[1]
    assert not v.ok


def test_a_maximisation_certifies_in_its_own_sense():
    p = random_lp(seed=2)
    p.sense = ObjSense.MAXIMISE
    s = solve_simplex(p)
    assert s.status == Status.OPTIMAL
    bnd, _ = certified_bound(p, s.x, s.y)
    assert bnd >= s.objective - 1e-9 * max(1.0, abs(s.objective))   # an upper bound
    v = verify(p, s.x, s.objective, y=s.y)
    assert v.ok, v.report()


def test_duals_that_point_at_an_infinite_bound_certify_nothing():
    """A large reduced cost on a free column cannot be absorbed: the bound
    is vacuous and the verdict says so instead of passing."""
    A = SparseMatrix.from_dense(np.array([[1.0, 1.0]]))
    p = Problem(A=A, c=np.array([1.0, 2.0]), row_lb=np.array([1.0]),
                row_ub=np.array([1.0]), col_lb=np.array([0.0, -INF]),
                col_ub=np.array([INF, INF]))
    x = np.array([1.0, 0.0])
    y_wrong = np.array([5.0])     # d = c - Aᵀy = (-4, -3): both point at an infinite bound
    bnd, pert = certified_bound(p, x, y_wrong)
    assert bnd == -np.inf and pert == 4.0
    v = verify(p, x, 1.0, y=y_wrong)
    opt = [c for c in v.checks if c[0] == "optimality"][0]
    assert not opt[1] and "certify nothing" in opt[2]


def test_integer_models_get_no_optimality_line():
    p = random_lp(seed=0)
    p.kind[:] = 1
    s = solve_simplex(p)
    v = verify(p, s.x, s.objective, y=s.y)
    assert not any(c[0] == "optimality" for c in v.checks)
