"""INFEASIBLE is the one verdict a caller could not check: there is no
point to hand a verifier. Now there is a ray. The simplex returns the
phase-1 duals that price the rows into a contradiction, and the verifier
checks them with the same Farkas arithmetic the tree prunes on -- so an
infeasible model's answer is verified like a feasible one's point.

Netlib's infeasible set (Chinneck 1993) is the measurement: 29 models
published as infeasible, every one recognised by the simplex, and every
ray certified once the verifier learnt to drop rounding at infinite
bounds (the first pass refused 20 of 29 for terms of 1e-19 to 1e-11 of the
ray's largest entry). Three of the smallest are fetched by the suite.
"""

import os

import numpy as np
import pytest

from bench.verify import verify_infeasible
from sovopt.core.problem import Problem, Status, VarKind
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.io.mps import read_mps
from sovopt.lp.simplex import SimplexParams, solve_simplex
from tests.test_simplex import random_lp

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "netlib-infeasible")


def _two_rows():
    """x1 + x2 <= 1 and x1 + x2 >= 3 over x >= 0."""
    A = SparseMatrix.from_triplets(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]),
                                   np.array([1.0, 1.0, 1.0, 1.0]), 2, 2)
    return Problem(A=A, c=np.array([1.0, 1.0]),
                   row_lb=np.array([-INF, 3.0]), row_ub=np.array([1.0, INF]),
                   col_lb=np.zeros(2), col_ub=np.full(2, INF), name="two-rows")


def test_the_simplex_returns_a_ray_the_verifier_certifies():
    p = _two_rows()
    s = solve_simplex(p)
    assert s.status == Status.INFEASIBLE
    assert s.farkas is not None and s.farkas.shape == (2,)
    v = verify_infeasible(p, s.farkas)
    assert v.ok, v.report()


def test_a_feasible_model_returns_no_ray():
    s = solve_simplex(random_lp(seed=0))
    assert s.status == Status.OPTIMAL
    assert s.farkas is None


def test_a_ray_that_proves_nothing_is_refused():
    p = _two_rows()
    # y = (1, 1) prices the <= row positively, which is unbounded on that side
    assert not verify_infeasible(p, np.array([1.0, 1.0])).ok
    # the zero ray and a ray of the wrong length
    assert not verify_infeasible(p, np.zeros(2)).ok
    assert not verify_infeasible(p, np.zeros(3)).ok
    assert not verify_infeasible(p, None).ok


def test_rounding_at_an_infinite_bound_is_dropped_and_reported():
    """A phase-1 basis leaves 1e-16 residues on its basic columns; a
    residue on a column with no bound on that side made the value -inf and
    refused 20 of Netlib's 29 valid rays. Below rounding it is dropped and
    reported; above it the ray certifies nothing."""
    p = _two_rows()
    s = solve_simplex(p)
    y = s.farkas.copy()
    # perturb the ray so d = -A'y turns negative by 2e-13 on both columns,
    # whose upper bound is +inf: a residue pointing at an infinite bound
    ok_noise = verify_infeasible(p, y * (1.0 - 1e-13 * np.array([1.0, -1.0])))
    assert ok_noise.ok, ok_noise.report()
    assert "dropped" in ok_noise.checks[0][2]
    bad = verify_infeasible(p, y * (1.0 - 1e-3 * np.array([1.0, -1.0])))
    assert not bad.ok


def test_the_ray_survives_scaling_and_a_maximisation():
    """The ray is unscaled by the row factors alone and never flipped with
    the objective: a badly scaled maximisation certifies too."""
    from sovopt.core.problem import ObjSense
    p = _two_rows()
    p.A.scale(np.array([1e4, 1e-3]), np.array([1.0, 1e3]))
    p.row_ub = p.row_ub * np.array([1e4, 1.0])
    p.row_lb = p.row_lb * np.array([1.0, 1e-3])
    p.col_ub = p.col_ub / np.array([1.0, 1e3])
    p.sense = ObjSense.MAXIMISE
    p.c = -p.c
    s = solve_simplex(p, SimplexParams())
    assert s.status == Status.INFEASIBLE
    v = verify_infeasible(p, s.farkas)
    assert v.ok, v.report()


@pytest.mark.parametrize("name", ["galenet", "woodinfe", "refinery"])
def test_netlibs_infeasible_models_are_recognised_and_certified(name):
    """galenet (a network), woodinfe (Greenberg's forestry example) and
    refinery (a doctored petrochemical plant model, Chesapeake Decision
    Sciences) from netlib/lp/infeas."""
    path = os.path.join(DATA, f"{name}.mps")
    if not os.path.exists(path):
        pytest.skip("Netlib's infeasible set not fetched")
    p = read_mps(path)
    p.kind[:] = VarKind.CONTINUOUS
    s = solve_simplex(p)
    assert s.status == Status.INFEASIBLE
    v = verify_infeasible(p, s.farkas)
    assert v.ok, v.report()
