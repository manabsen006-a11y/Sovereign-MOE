"""Multi-pool pooling in the pq-formulation.

Two formulations of one physical network must agree on the answer and disagree
only on the bound. Both halves of that are tested: the optima must coincide (or
one encoding is wrong), and the pq bound must be at least as tight as the plain
q bound (or the RLT rows are not doing their job).
"""

import numpy as np
import pytest

from vyuha.core.problem import Status
from vyuha.globalopt.bilinear import build_relaxation
from vyuha.globalopt.spatial import SpatialParams, solve_global
from vyuha.lp.simplex import SimplexParams, solve_simplex
from vyuha.models import haverly, haverly_pq, random_pooling
from vyuha.models.pooling_pq import PQIndex, PoolingData, pooling_pq

PUBLISHED = {1: 400.0, 2: 600.0, 3: 750.0}


def root_bound(bp):
    s = solve_simplex(build_relaxation(bp, bp.linear.col_lb, bp.linear.col_ub),
                      SimplexParams())
    assert s.status == Status.OPTIMAL
    return s.objective


# --------------------------------------------------------------------------- #
# the two encodings must agree                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_pq_matches_published_optimum(variant):
    s = solve_global(haverly_pq(variant), SpatialParams(time_limit=120))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - PUBLISHED[variant]) < 1e-4, (
        f"pq gave {s.objective}, published {PUBLISHED[variant]}")


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_p_and_pq_formulations_agree(variant):
    """Two independent encodings of the same network."""
    a = solve_global(haverly(variant), SpatialParams(time_limit=120))
    b = solve_global(haverly_pq(variant), SpatialParams(time_limit=120))
    assert a.status == Status.OPTIMAL and b.status == Status.OPTIMAL
    assert abs(a.objective - b.objective) < 1e-4


@pytest.mark.parametrize("seed", [0, 1])
def test_q_and_pq_reach_the_same_optimum(seed):
    """RLT rows change the bound, never the feasible set."""
    # A 2x2x2 network, where *both* encodings prove optimality in well under a
    # second. The 4x2x2 used here before never let the q-formulation close --
    # that is the whole point of the pq-formulation and it is measured in
    # bench/pq_check.py -- so this test skipped every time and never once
    # compared the two optima it exists to compare.
    q = random_pooling(2, 2, 2, 1, seed=seed, rlt=False)
    pq = random_pooling(2, 2, 2, 1, seed=seed, rlt=True)
    sq = solve_global(q, SpatialParams(time_limit=60))
    spq = solve_global(pq, SpatialParams(time_limit=60))
    assert sq.status == Status.OPTIMAL and spq.status == Status.OPTIMAL, (
        f"q={sq.status.name} pq={spq.status.name}; both must close for the "
        f"objectives to be comparable")
    assert abs(sq.objective - spq.objective) < 1e-4 * max(1.0, abs(spq.objective))


# --------------------------------------------------------------------------- #
# the RLT rows must actually tighten                                           #
# --------------------------------------------------------------------------- #


def test_pq_bound_is_at_least_as_tight_as_q():
    """Maximisation, so a *lower* root bound is tighter."""
    checked = 0
    for seed in range(4):
        for shape in [(4, 2, 2, 1), (5, 2, 3, 2)]:
            q = random_pooling(*shape, seed=seed, rlt=False)
            pq = random_pooling(*shape, seed=seed, rlt=True)
            rq, rpq = root_bound(q), root_bound(pq)
            assert rpq <= rq + 1e-6, (
                f"pq bound {rpq} is looser than q bound {rq}")
            checked += 1
    assert checked >= 6


def test_rlt_rows_hold_for_every_bilinear_feasible_point():
    """``sum_i v_ijk = y_jk`` follows from ``v = q*y`` and ``sum_i q_ij = 1``.

    Checked directly on the solved point: if it failed, the rows would be
    cutting off genuine solutions rather than tightening a relaxation.
    """
    bp = random_pooling(4, 2, 2, 1, seed=0, rlt=True)
    s = solve_global(bp, SpatialParams(time_limit=90))
    assert s.x is not None
    d = PoolingData(cost=np.zeros(4), avail=np.zeros(4),
                    quality=np.zeros((4, 1)), pool_cap=np.zeros(2),
                    price=np.zeros(2), demand=np.zeros(2),
                    spec=np.zeros((2, 1)))
    ix = PQIndex(d)
    for j in range(2):
        for k in range(2):
            lhs = sum(s.x[ix.v(i, j, k)] for i in range(4))
            assert abs(lhs - s.x[ix.y(j, k)]) < 1e-6


# --------------------------------------------------------------------------- #
# network structure                                                            #
# --------------------------------------------------------------------------- #


def test_arc_masks_are_respected():
    """REGRESSION: every source was allowed to bypass every pool.

    That does not make the model infeasible, it makes it *easier*: with a
    complete bypass network there is no pooling decision at all, the problem
    collapses to an LP, and the solver confidently reports an objective better
    than the true optimum. The first version of this builder "solved" Haverly
    to 500 against a published 400 by routing everything around the pool.
    """
    bp = haverly_pq(1)
    hi = bp.linear.col_ub
    d = PoolingData(cost=np.zeros(3), avail=np.zeros(3),
                    quality=np.zeros((3, 1)), pool_cap=np.zeros(1),
                    price=np.zeros(2), demand=np.zeros(2),
                    spec=np.zeros((2, 1)))
    ix = PQIndex(d)

    # C (index 2) does not feed the pool
    assert hi[ix.q(2, 0)] == 0.0
    for k in range(2):
        assert hi[ix.v(2, 0, k)] == 0.0
    # A and B do
    assert hi[ix.q(0, 0)] == 1.0 and hi[ix.q(1, 0)] == 1.0
    # only C may ship direct
    for k in range(2):
        assert hi[ix.z(0, k)] == 0.0 and hi[ix.z(1, k)] == 0.0
        assert hi[ix.z(2, k)] > 0.0

    s = solve_global(bp, SpatialParams(time_limit=120))
    assert s.x is not None
    # the pool must actually be used; a solution that bypasses it entirely is
    # the signature of the bug above
    pooled = sum(s.x[ix.v(i, 0, k)] for i in (0, 1) for k in range(2))
    assert pooled > 1e-6, "solution bypasses the pool entirely"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_solutions_are_feasible(seed):
    bp = random_pooling(5, 2, 3, 2, seed=seed, rlt=True)
    s = solve_global(bp, SpatialParams(time_limit=90))
    if s.x is None:
        pytest.skip("no solution within the limit")
    lin, bil = bp.feasible(s.x)
    assert lin < 1e-6 and bil < 1e-6


def test_bound_never_understates_a_maximisation():
    for seed in (0, 1, 2):
        bp = random_pooling(4, 2, 2, 1, seed=seed, rlt=True)
        s = solve_global(bp, SpatialParams(time_limit=90))
        if s.x is None:
            continue
        assert root_bound(bp) >= s.objective - 1e-6


# --------------------------------------------------------------------------- #
# robustness                                                                   #
# --------------------------------------------------------------------------- #


def test_simplex_never_raises_on_a_degenerate_pooling_relaxation():
    """REGRESSION: a zero pivot escaped as an exception.

    ``alpha_row[q]`` and ``alpha[r]`` are the same number computed two ways --
    from the pivot row and from the FTRAN'd entering column. When the
    factorisation has drifted they disagree, the ratio test passes on a pivot
    the basis update then finds to be zero, and LUSingular propagated out of the
    solver. A caller inside a spatial branch-and-bound cannot do anything with
    an exception, and one bad node must not abort the search: numerical failure
    is now a status.
    """
    for seed in range(6):
        for shape in [(4, 2, 2, 1), (5, 2, 3, 2), (6, 3, 3, 2)]:
            bp = random_pooling(*shape, seed=seed, rlt=False)
            s = solve_global(bp, SpatialParams(time_limit=15))
            assert s.status in (Status.OPTIMAL, Status.TIME_LIMIT,
                                Status.NODE_LIMIT, Status.NUMERICAL,
                                Status.INFEASIBLE)
