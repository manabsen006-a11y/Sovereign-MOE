"""Crossover from an interior point to an optimal basis.

Correctness is not the interesting property here -- the simplex finishes every
crossover, and it terminates only at an optimal basis, so the answer is exact
however bad the identification was. What is worth asserting is the two things
that are *not* automatic: that the identified basis is non-singular by
construction rather than by repair, and that crossing over is cheaper than
starting cold. A crossover that quietly degenerates to the all-logical basis
would pass every correctness check and be worth nothing, and that is exactly
the bug the first implementation had.
"""

import os

import numpy as np
import pytest

from sovopt.core.problem import ObjSense, Problem, Status, VarKind
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.io.mps import read_mps
from sovopt.lp.basis import BASIC, Basis
from sovopt.lp.crossover import basis_from_point, crossover
from sovopt.lp.ipm import IPMParams, solve_ipm
from sovopt.lp.pdlp import PDLPParams, solve_pdlp
from sovopt.lp.simplex import SimplexParams, solve_simplex

from tests.test_simplex import random_lp, toy_lp

INSTANCES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "instances")


def _relaxation(name):
    path = os.path.join(INSTANCES, f"{name}.mps")
    if not os.path.exists(path):
        pytest.skip("benchmark instances not fetched")
    p = read_mps(path)
    p.kind[:] = VarKind.CONTINUOUS
    return p


# --------------------------------------------------------------------------- #
# the basis itself                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_identified_basis_is_nonsingular_without_repair(seed):
    """REGRESSION: the candidate set must be built with a rank test.

    The first version ranked variables by how far they sat from their nearest
    bound and took the best ``m``. That set is almost always singular -- at an
    optimum only a handful of variables are strictly between bounds, so the
    remaining picks are ties at zero and arbitrary -- and the basis
    factorisation quietly repaired it back to the all-logical basis. Every
    correctness assertion still passed, and the crossover was worth exactly
    nothing: identical pivot counts to a cold start on all seven instances
    tried, and one of them eighteen times slower for the wasted repair.

    ``n_repairs == 0`` is the assertion that would have caught it.
    """
    p = random_lp(seed=seed)
    ip = solve_ipm(p.copy(), IPMParams())
    status = basis_from_point(p, ip.x, ip.y)

    assert status.shape[0] == p.n + p.m
    assert int((status == BASIC).sum()) == p.m, "not exactly m basic variables"

    B = Basis(p)
    B.status[:] = status
    B.basic[:] = np.flatnonzero(status == BASIC)
    B.factorize()
    assert B.n_repairs == 0, "the identified basis was singular"


def test_a_fixed_variable_is_never_made_basic():
    """A variable pinned between equal bounds has no room to be basic in."""
    A = SparseMatrix.from_dense(np.array([[1., 1., 1.],
                                          [1., -1., 0.]]))
    p = Problem(A=A, c=np.array([1., 1., -1.]),
                row_lb=np.array([1.0, -INF]), row_ub=np.array([1.0, 3.0]),
                col_lb=np.array([0.0, 2.0, 0.0]),
                col_ub=np.array([5.0, 2.0, 5.0]), name="pinned")
    ip = solve_ipm(p.copy(), IPMParams())
    status = basis_from_point(p, ip.x, ip.y)
    assert status[1] != BASIC


# --------------------------------------------------------------------------- #
# it must not change the answer                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_agrees_with_a_cold_simplex(seed):
    p = random_lp(seed=seed)
    cold = solve_simplex(p.copy(), SimplexParams())
    ip = solve_ipm(p.copy(), IPMParams())
    xo = crossover(p.copy(), ip.x, ip.y)
    assert xo.status == Status.OPTIMAL
    assert abs(xo.objective - cold.objective) <= 1e-6 * max(1.0, abs(cold.objective))


def test_crosses_over_from_the_first_order_point_too():
    """The README's gap was PDLP specifically: no basis, so no handover."""
    p = random_lp(seed=1)
    cold = solve_simplex(p.copy(), SimplexParams())
    fo = solve_pdlp(p.copy(), PDLPParams(device="cpu", eps_abs=1e-8,
                                         eps_rel=1e-8))
    xo = crossover(p.copy(), fo.x, fo.y)
    assert xo.status == Status.OPTIMAL
    assert abs(xo.objective - cold.objective) <= 1e-6 * max(1.0, abs(cold.objective))


def test_handles_a_maximisation():
    p = toy_lp()
    ip = solve_ipm(p.copy(), IPMParams())
    xo = crossover(p.copy(), ip.x, ip.y)
    assert xo.status == Status.OPTIMAL
    assert abs(xo.objective - 11.0) < 1e-6


# --------------------------------------------------------------------------- #
# the capability it exists to provide                                          #
# --------------------------------------------------------------------------- #


def test_ranging_becomes_available_on_an_interior_point_solve():
    """The whole point: an interior point cannot answer a ranging question.

    Before crossover ``sensitivity`` is None on an interior-point solve however
    it is asked for, because ranging is read off a basis inverse and there is
    no basis. This is the assertion that the gap is closed.
    """
    p = _relaxation("gt2")
    ip = solve_ipm(p.copy(), IPMParams())
    assert ip.basis_status is None
    assert ip.sensitivity is None

    xo = crossover(p.copy(), ip.x, ip.y, SimplexParams(sensitivity=True))
    assert xo.status == Status.OPTIMAL
    assert xo.basis_status is not None
    assert int((xo.basis_status == BASIC).sum()) == p.m
    assert xo.sensitivity is not None
    rep = xo.sensitivity.report()
    assert isinstance(rep, str) and rep.strip()


def test_saves_pivots_against_a_cold_start():
    """Crossing over has to be cheaper than solving from scratch.

    Asserted in aggregate rather than per instance: misc07 costs *more* pivots
    from a crossover than cold, and one instance going the wrong way is a
    normal property of a heuristic identification, not a failure. Over the set
    the saving measured about 59%.
    """
    names = ["gt2", "p0201", "khb05250", "mod010"]
    cold_total = xover_total = 0
    for name in names:
        p = _relaxation(name)
        cold = solve_simplex(p.copy(), SimplexParams())
        ip = solve_ipm(p.copy(), IPMParams())
        xo = crossover(p.copy(), ip.x, ip.y)
        assert xo.status == Status.OPTIMAL
        assert abs(xo.objective - cold.objective) <= 1e-6 * max(1.0, abs(cold.objective))
        cold_total += cold.iterations
        xover_total += xo.iterations
    assert xover_total < cold_total, (
        f"crossover used {xover_total} pivots against a cold {cold_total}; "
        f"it is not paying for itself")
