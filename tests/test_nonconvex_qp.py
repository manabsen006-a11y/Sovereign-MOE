"""Non-convex QP through the McCormick reformulation -- the default route.

The oracles are the ones :mod:`tests.test_alphabb` uses, because the point of
having two relaxations is that they must agree with the enumeration *and*
with each other. What is specific here: the reformulation must evaluate to
the same objective as the quadratic at every point, the product bounds must
contain every product the box allows, and the spatial tree's new integer
branching and certified node bounds must hold on their own.
"""

import itertools

import numpy as np
import pytest

from sovopt.core.problem import ObjSense, Status
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.globalopt.alphabb import AlphaBBParams, solve_alphabb
from sovopt.globalopt.nonconvex_qp import (NonconvexQPParams, qp_to_bilinear,
                                           solve_nonconvex_qp)
from sovopt.globalopt.spatial import SpatialParams, solve_global

from tests.test_alphabb import (brute_integers, concave_box, grid_min,
                                indefinite, lipschitz)


# --------------------------------------------------------------------------- #
# the reformulation                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(5))
def test_the_bilinear_objective_equals_the_quadratic_everywhere(seed):
    p = indefinite(seed, n=6)
    bp = qp_to_bilinear(p)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        x = rng.uniform(p.col_lb, p.col_ub)
        ext = np.concatenate([x, [x[t.x] * x[t.y] for t in bp.terms]])
        assert abs(bp.linear.objective(ext) - p.objective(x)) <= 1e-9 * max(1.0, abs(p.objective(x)))


def test_only_nonzero_pairs_become_products():
    p = indefinite(0, n=5)
    Qd = p.Q.to_dense()
    Qd[0, :] = Qd[:, 0] = 0.0                    # x0 leaves the quadratic
    Qd[2, 3] = Qd[3, 2] = 0.0
    p.Q = SparseMatrix.from_dense(Qd)
    bp = qp_to_bilinear(p)
    pairs = {(t.x, t.y) for t in bp.terms}
    assert not any(0 in pr for pr in pairs)
    assert (2, 3) not in pairs
    assert len(pairs) == sum(1 for i in range(5) for j in range(i, 5) if Qd[i, j] != 0.0)


def test_an_asymmetric_q_is_symmetrised_by_the_reformulation():
    """``½xᵀQx`` only sees ``(Q + Qᵀ)/2``, and so must the products."""
    p = indefinite(1, n=3)
    Qd = p.Q.to_dense()
    Qd[0, 1] += 1.0                               # now asymmetric
    p.Q = SparseMatrix.from_dense(Qd)
    bp = qp_to_bilinear(p)
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.uniform(p.col_lb, p.col_ub)
        ext = np.concatenate([x, [x[t.x] * x[t.y] for t in bp.terms]])
        assert abs(bp.linear.objective(ext) - p.objective(x)) <= 1e-9


@pytest.mark.parametrize("seed", range(4))
def test_product_bounds_contain_every_product_the_box_allows(seed):
    p = concave_box(seed, n=5)                    # boxes straddle zero
    bp = qp_to_bilinear(p)
    rng = np.random.default_rng(seed)
    n = p.n
    for _ in range(200):
        x = rng.uniform(p.col_lb, p.col_ub)
        for t in bp.terms:
            w = x[t.x] * x[t.y]
            assert bp.linear.col_lb[t.w] - 1e-12 <= w <= bp.linear.col_ub[t.w] + 1e-12
    # and a square's lower bound is exactly zero when the box straddles zero
    for t in bp.terms:
        if t.x == t.y and p.col_lb[t.x] < 0 < p.col_ub[t.x]:
            assert bp.linear.col_lb[t.w] == 0.0
    assert bp.linear.n == n + len(bp.terms)


def test_integrality_and_sense_carry_over_and_products_are_continuous():
    p = indefinite(0, n=4, integer=True)
    p.sense = ObjSense.MAXIMISE
    bp = qp_to_bilinear(p)
    assert bp.linear.sense == ObjSense.MAXIMISE
    assert bp.linear.integer_mask[:4].all()
    assert not bp.linear.integer_mask[4:].any()


# --------------------------------------------------------------------------- #
# the oracle, again, on the default route                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(6))
def test_concave_box_qp_finds_the_best_vertex(seed):
    p = concave_box(seed)
    corners = itertools.product(*[(p.col_lb[j], p.col_ub[j]) for j in range(p.n)])
    best = min(p.objective(np.array(v)) for v in corners)
    s = solve_nonconvex_qp(p.copy(), NonconvexQPParams(time_limit=120))
    assert s.status == Status.OPTIMAL, s
    assert abs(s.objective - best) <= 1e-4 * max(1.0, abs(best))
    assert s.dual_bound <= best + 1e-6 * max(1.0, abs(best))


@pytest.mark.parametrize("seed", range(8))
def test_integer_nonconvex_qp_matches_brute_force(seed):
    """Integer branching in the spatial tree is new; this is its oracle."""
    p = indefinite(seed, n=4, integer=True)
    best = brute_integers(p, 3)
    if best is None:
        pytest.skip("no feasible integer point")
    s = solve_nonconvex_qp(p.copy(), NonconvexQPParams(time_limit=120))
    assert s.status == Status.OPTIMAL, s
    assert abs(s.objective - best[0]) <= 1e-6 * max(1.0, abs(best[0])), (
        f"seed {seed}: brute {best[0]} tree {s.objective}")
    _, _, int_v = p.violation(s.x)
    assert int_v <= 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_continuous_nonconvex_qp_agrees_with_a_fine_grid(seed):
    p = indefinite(seed, n=2, m=1)
    g, _, h = grid_min(p, 401)
    s = solve_nonconvex_qp(p.copy(), NonconvexQPParams(time_limit=120))
    assert s.status == Status.OPTIMAL, s
    assert s.dual_bound <= g + 1e-6 * max(1.0, abs(g))
    assert s.objective <= g + 1e-4 * max(1.0, abs(g)) + 1e-9
    assert s.objective >= g - lipschitz(p) * h * np.sqrt(p.n) - 1e-9


def test_mixed_integer_and_continuous():
    p = indefinite(31, n=3, m=2)
    p.kind = np.array([1, 1, 0], dtype=np.uint8)
    best = np.inf
    for a in range(4):
        for b in range(4):
            q = p.copy()
            q.col_lb[:2] = q.col_ub[:2] = (a, b)
            g, _, _ = grid_min(q, 2001)
            best = min(best, g)
    s = solve_nonconvex_qp(p.copy(), NonconvexQPParams(time_limit=120))
    assert s.status == Status.OPTIMAL, s
    assert abs(s.objective - best) <= 1e-4 * max(1.0, abs(best)) + 1e-6


@pytest.mark.parametrize("seed", range(4))
def test_the_two_relaxations_agree(seed):
    """Different relaxations, different node solvers, one answer."""
    p = indefinite(seed + 40, n=5, m=2)
    a = solve_nonconvex_qp(p.copy(), NonconvexQPParams(time_limit=120))
    b = solve_nonconvex_qp(p.copy(), NonconvexQPParams(time_limit=120,
                                                       relaxation="alphabb"))
    assert a.status == b.status == Status.OPTIMAL
    assert abs(a.objective - b.objective) <= 2e-4 * max(1.0, abs(a.objective))


# --------------------------------------------------------------------------- #
# what the answer carries                                                      #
# --------------------------------------------------------------------------- #


def test_the_answer_is_validated_against_the_quadratic_model():
    p = indefinite(2, n=4, m=2)
    s = solve_nonconvex_qp(p.copy(), NonconvexQPParams(time_limit=120))
    assert s.x is not None and s.x.size == p.n     # products stripped
    row_v, col_v, _ = p.violation(s.x)
    assert max(row_v, col_v) <= 1e-6
    assert abs(s.objective - p.objective(s.x)) <= 1e-9
    assert s.method == "nonconvex-qp[mccormick]"
    assert s.info["products"] == 10                # 4 squares + 6 cross terms
    assert s.info["bound_is_rigorous"] is True


def test_maximisation_is_solved_in_the_users_sense():
    p = concave_box(0)
    p.Q = SparseMatrix.from_dense(-p.Q.to_dense())
    p.c = -p.c
    p.sense = ObjSense.MAXIMISE
    corners = itertools.product(*[(p.col_lb[j], p.col_ub[j]) for j in range(p.n)])
    best = max(p.objective(np.array(v)) for v in corners)
    s = solve_nonconvex_qp(p.copy(), NonconvexQPParams(time_limit=120))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - best) <= 1e-4 * max(1.0, abs(best))
    assert s.dual_bound >= best - 1e-6 * max(1.0, abs(best))


def test_unbounded_variable_in_a_product_is_refused_by_name():
    p = indefinite(0, n=3, m=1)
    p.col_names = ["flow", "temp", "yield"]
    p.col_ub[1] = INF
    with pytest.raises(ValueError, match="temp"):
        solve_nonconvex_qp(p, NonconvexQPParams(time_limit=10))


def test_unknown_relaxation_is_refused():
    with pytest.raises(ValueError):
        solve_nonconvex_qp(indefinite(0), NonconvexQPParams(relaxation="magic"))


# --------------------------------------------------------------------------- #
# the spatial tree's new pieces, on the pooling models it already had          #
# --------------------------------------------------------------------------- #


def test_pooling_nodes_are_certified_once_obbt_has_boxed_the_variables():
    from sovopt.models.pooling import haverly
    s = solve_global(haverly(1), SpatialParams(time_limit=60))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 400.0) <= 1e-6 * 400
    assert s.info["uncertified_nodes"] == 0
    assert s.info["bound_is_rigorous"] is True


def test_a_pooling_model_with_an_integer_decision():
    """Haverly with a binary that must equal 1 for any flow to be
    profitable, at a cost of 0.5: the tree has to branch on it and find
    400 − 0.5 with the switch exactly on."""
    bp = haverly_with_switch()
    s = solve_global(bp, SpatialParams(time_limit=60))
    assert s.status == Status.OPTIMAL, s
    assert abs(s.objective - 399.5) <= 1e-6 * 400
    z = s.x[-1]
    assert abs(z - 1.0) <= 1e-6


def haverly_with_switch():
    """Haverly(1) plus a binary ``z`` with every flow bounded by ``z * big``.

    Off, the plant makes nothing; on, it makes the usual 400. The relaxation
    with ``z`` fractional is profitable, so the integer branch is exercised.
    """
    from sovopt.core.problem import Problem
    from sovopt.core.sparse import SparseMatrix
    from sovopt.globalopt.bilinear import BilinearProblem, BilinearTerm
    from sovopt.models.pooling import haverly
    base = haverly(1)
    lin = base.linear
    n = lin.n
    Ad = lin.A.to_dense()
    big = 1000.0
    # every original column bounded by z*big: x_j - big z <= 0
    extra = np.zeros((n, n + 1))
    for j in range(n):
        extra[j, j] = 1.0
        extra[j, n] = -big
    A = SparseMatrix.from_dense(np.vstack([np.hstack([Ad, np.zeros((lin.m, 1))]), extra]))
    p = Problem(A=A, c=np.concatenate([lin.c, [-0.5]]),   # a small cost for switching on
                row_lb=np.concatenate([lin.row_lb, np.full(n, -INF)]),
                row_ub=np.concatenate([lin.row_ub, np.zeros(n)]),
                col_lb=np.concatenate([lin.col_lb, [0.0]]),
                col_ub=np.concatenate([lin.col_ub, [1.0]]),
                kind=np.concatenate([lin.kind, [1]]).astype(np.uint8),
                obj_offset=lin.obj_offset, sense=lin.sense, name="haverly+z")
    return BilinearProblem(linear=p, terms=list(base.terms), name="haverly+z")
