"""Non-convex QP to global optimality by αBB.

Two certificates and one oracle. The certificates: Gershgorin's shift really
makes the relaxation convex (checked against a dense eigenvalue solver, which
the engine itself never calls), and the αBB term really underestimates
(checked pointwise). The oracle: exhaustive enumeration -- every vertex of a
box for a concave objective, every integer point for an integer model, a fine
grid for a continuous one -- because a global optimiser that agrees with a
local one has proved nothing.
"""

import itertools

import numpy as np
import pytest

from sovopt.core.problem import ObjSense, Problem, Status
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.globalopt.alphabb import (AlphaBBParams, gershgorin_alpha,
                                      solve_alphabb)
from sovopt.qp import NotConvexError, QPParams, solve_qp


# --------------------------------------------------------------------------- #
# generators                                                                   #
# --------------------------------------------------------------------------- #


def indefinite(seed, n=4, m=2, ub=3.0, integer=False, rows=True):
    """A random indefinite ``Q`` with ``c`` pointing into the box, a few
    ``<=`` rows that leave it mostly open, and a ``[0, ub]`` box."""
    rng = np.random.default_rng(700 + seed)
    M = rng.standard_normal((n, n))
    H = 0.5 * (M + M.T)
    H -= (np.linalg.eigvalsh(H).max() * 0.6) * np.eye(n)   # push it indefinite
    c = rng.standard_normal(n)
    if rows and m:
        A = SparseMatrix.from_dense(np.round(rng.uniform(-1, 3, (m, n))))
        b = np.abs(A.matvec(np.full(n, ub / 2.0))) + rng.uniform(1, 5, m)
        rl, ru = np.full(m, -INF), b
    else:
        A = SparseMatrix.from_dense(np.zeros((1, n)))
        rl, ru = np.array([-INF]), np.array([INF])
    return Problem(A=A, c=c, Q=SparseMatrix.from_dense(H), row_lb=rl, row_ub=ru,
                   col_lb=np.zeros(n), col_ub=np.full(n, ub),
                   kind=np.ones(n, dtype=np.uint8) if integer else None,
                   name=f"ncqp{seed}")


def concave_box(seed, n=6):
    rng = np.random.default_rng(900 + seed)
    M = rng.standard_normal((n, n))
    H = -(M.T @ M + 0.2 * np.eye(n))                     # negative definite
    c = rng.standard_normal(n)
    lo = rng.uniform(-2, 0, n)
    hi = lo + rng.uniform(0.5, 3, n)
    A = SparseMatrix.from_dense(np.zeros((1, n)))
    return Problem(A=A, c=c, Q=SparseMatrix.from_dense(H),
                   row_lb=np.array([-INF]), row_ub=np.array([INF]),
                   col_lb=lo, col_ub=hi, name=f"concave{seed}")


def brute_integers(p, ub):
    best = None
    for combo in itertools.product(range(int(ub) + 1), repeat=p.n):
        x = np.array(combo, dtype=float)
        rv, cv, _ = p.violation(x)
        if max(rv, cv) > 1e-9:
            continue
        v = p.objective(x)
        if best is None or v < best[0]:
            best = (v, x)
    return best


def grid_min(p, points_per_axis):
    axes = [np.linspace(p.col_lb[j], p.col_ub[j], points_per_axis)
            if p.col_ub[j] > p.col_lb[j] else np.array([p.col_lb[j]])
            for j in range(p.n)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, p.n)
    Qd = p.Q.to_dense()
    f = 0.5 * np.einsum("ij,jk,ik->i", G, Qd, G) + G @ p.c + p.obj_offset
    act = G @ p.A.to_dense().T
    ok = ((act >= p.row_lb - 1e-9) & (act <= p.row_ub + 1e-9)).all(axis=1)
    f = np.where(ok, f, np.inf)
    k = int(np.argmin(f))
    h = max(float((p.col_ub[j] - p.col_lb[j]) / (points_per_axis - 1))
            for j in range(p.n))                      # 0 for a fixed axis
    return float(f[k]), G[k], h


def lipschitz(p):
    """An upper bound on ‖∇f‖₂ over the box, for the grid-spacing allowance."""
    Qd = p.Q.to_dense()
    big = np.maximum(np.abs(p.col_lb), np.abs(p.col_ub))
    return float(np.linalg.norm(np.abs(Qd) @ big + np.abs(p.c)))


# --------------------------------------------------------------------------- #
# the certificates                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(10))
def test_gershgorin_shift_makes_q_positive_semidefinite(seed):
    """The whole method rests on ``Q + 2 diag(α)`` being PSD on the free
    variables. Checked with a dense eigenvalue solver, which is the oracle the
    engine deliberately does not use."""
    rng = np.random.default_rng(seed)
    n = 7
    M = rng.standard_normal((n, n))
    H = 0.5 * (M + M.T)
    lo = rng.uniform(-3, 0, n)
    hi = lo + rng.uniform(0.1, 5, n)
    if seed % 3 == 0:
        hi[1] = lo[1]                                   # a fixed variable
    Q = SparseMatrix.from_dense(H)
    a = gershgorin_alpha(Q, lo, hi)
    assert (a >= 0.0).all()
    free = np.flatnonzero(hi > lo)
    Hs = (H + 2.0 * np.diag(a))[np.ix_(free, free)]
    assert np.linalg.eigvalsh(Hs).min() >= -1e-9, "the certificate is false"
    assert a[hi <= lo].max(initial=0.0) == 0.0, "a fixed variable needs no shift"


def test_gershgorin_is_zero_on_a_diagonally_dominant_convex_q():
    H = np.array([[4.0, 1.0, -1.0], [1.0, 3.0, 0.5], [-1.0, 0.5, 2.0]])
    a = gershgorin_alpha(SparseMatrix.from_dense(H), np.zeros(3), np.ones(3))
    assert (a == 0.0).all()


def test_gershgorin_scaling_moves_curvature_onto_the_wide_variable():
    """With ``d = hi − lo``, the shift a narrow variable needs from a wide
    neighbour shrinks by their width ratio: the term it costs in the
    underestimator is ``α d²``, so this is where the scaling pays."""
    H = np.array([[0.0, 1.0], [1.0, 0.0]])
    Q = SparseMatrix.from_dense(H)
    a_equal = gershgorin_alpha(Q, np.zeros(2), np.array([1.0, 1.0]))
    a_skew = gershgorin_alpha(Q, np.zeros(2), np.array([1.0, 10.0]))
    assert np.allclose(a_equal, 0.5, atol=1e-9)
    assert a_skew[0] > a_skew[1]                          # narrow one takes more
    assert a_skew[0] * 1.0 ** 2 + a_skew[1] * 10.0 ** 2 < 2 * 0.5 * 10.0 ** 2 + 1e-9


def test_gershgorin_refuses_an_unbounded_coupled_variable():
    H = np.array([[1.0, 2.0], [2.0, 1.0]])
    a = gershgorin_alpha(SparseMatrix.from_dense(H), np.zeros(2),
                        np.array([1.0, INF]))
    assert not np.isfinite(a).all()


def test_gershgorin_admits_an_unbounded_uncoupled_convex_variable():
    H = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    a = gershgorin_alpha(SparseMatrix.from_dense(H), np.zeros(3),
                        np.array([1.0, 1.0, INF]))
    assert np.isfinite(a).all() and a[2] == 0.0


@pytest.mark.parametrize("seed", range(6))
def test_the_alpha_term_underestimates_everywhere_in_the_box(seed):
    p = indefinite(seed, n=5)
    a = gershgorin_alpha(p.Q, p.col_lb, p.col_ub)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        x = rng.uniform(p.col_lb, p.col_ub)
        f = p.objective(x)
        fa = f + float(np.sum(a * (x - p.col_lb) * (x - p.col_ub)))
        assert fa <= f + 1e-12


# --------------------------------------------------------------------------- #
# the oracle: exhaustive enumeration                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(6))
def test_concave_box_qp_finds_the_best_vertex(seed):
    """A concave function over a box is minimised at a vertex, so all 2^n of
    them are the whole answer. A local method stops at whichever one its
    start point rolls to."""
    p = concave_box(seed)
    corners = itertools.product(*[(p.col_lb[j], p.col_ub[j]) for j in range(p.n)])
    best = min(p.objective(np.array(v)) for v in corners)
    s = solve_alphabb(p.copy(), AlphaBBParams(time_limit=120))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - best) <= 1e-4 * max(1.0, abs(best))
    assert s.dual_bound <= best + 1e-9                    # the bound is a bound


@pytest.mark.parametrize("seed", range(8))
def test_integer_nonconvex_qp_matches_brute_force(seed):
    p = indefinite(seed, n=4, integer=True)
    best = brute_integers(p, 3)
    if best is None:
        pytest.skip("no feasible integer point")
    s = solve_alphabb(p.copy(), AlphaBBParams(time_limit=120))
    assert s.status == Status.OPTIMAL, s
    assert abs(s.objective - best[0]) <= 1e-6 * max(1.0, abs(best[0])), (
        f"seed {seed}: brute {best[0]} tree {s.objective}")
    assert s.dual_bound <= best[0] + 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_continuous_nonconvex_qp_agrees_with_a_fine_grid(seed):
    """Two variables, 401 points per axis. The grid's minimum is a feasible
    point, so the certified dual bound must lie below it; and the incumbent
    is feasible, so it cannot lie further below the grid than one grid step
    times the gradient can carry it."""
    p = indefinite(seed, n=2, m=1)
    g, _, h = grid_min(p, 401)
    s = solve_alphabb(p.copy(), AlphaBBParams(time_limit=120))
    assert s.status == Status.OPTIMAL, s
    assert s.dual_bound <= g + 1e-9
    assert s.objective <= g + 1e-4 * max(1.0, abs(g)) + 1e-9
    assert s.objective >= g - lipschitz(p) * h * np.sqrt(p.n) - 1e-9


@pytest.mark.parametrize("seed", range(3))
def test_three_variables_against_a_grid(seed):
    p = indefinite(seed + 20, n=3, m=2)
    g, _, h = grid_min(p, 121)
    s = solve_alphabb(p.copy(), AlphaBBParams(time_limit=120))
    assert s.status == Status.OPTIMAL, s
    assert s.dual_bound <= g + 1e-9
    assert s.objective <= g + 1e-4 * max(1.0, abs(g)) + 1e-9
    assert s.objective >= g - lipschitz(p) * h * np.sqrt(p.n) - 1e-9


def test_mixed_integer_and_continuous():
    """Two integers and one continuous: enumerate the integers, grid the
    rest."""
    p = indefinite(31, n=3, m=2)
    p.kind = np.array([1, 1, 0], dtype=np.uint8)
    best = np.inf
    for a in range(4):
        for b in range(4):
            q = p.copy()
            q.col_lb[:2] = q.col_ub[:2] = (a, b)
            g, _, _ = grid_min(q, 2001)
            best = min(best, g)
    s = solve_alphabb(p.copy(), AlphaBBParams(time_limit=120))
    assert s.status == Status.OPTIMAL, s
    assert s.dual_bound <= best + 1e-9
    assert abs(s.objective - best) <= 1e-4 * max(1.0, abs(best)) + 1e-6


# --------------------------------------------------------------------------- #
# senses, routing, refusals                                                    #
# --------------------------------------------------------------------------- #


def test_maximising_a_convex_quadratic_is_a_nonconvex_problem_and_is_solved():
    """``max ½x'Hx`` with ``H ≻ 0`` over a box is a concave minimisation in
    disguise; the answer is a vertex."""
    p = concave_box(0)
    p.Q = SparseMatrix.from_dense(-p.Q.to_dense())        # now convex ...
    p.c = -p.c
    p.sense = ObjSense.MAXIMISE                            # ... and maximised
    corners = itertools.product(*[(p.col_lb[j], p.col_ub[j]) for j in range(p.n)])
    best = max(p.objective(np.array(v)) for v in corners)
    s = solve_alphabb(p.copy(), AlphaBBParams(time_limit=120))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - best) <= 1e-4 * max(1.0, abs(best))
    assert s.dual_bound >= best - 1e-9


def test_a_convex_q_is_recognised_and_solved_in_one_node():
    p = indefinite(3, n=4)
    Hd = p.Q.to_dense()
    Hd = Hd + (abs(np.linalg.eigvalsh(Hd).min()) + 3.0) * np.eye(4)   # dominant
    p.Q = SparseMatrix.from_dense(Hd)
    ref = solve_qp(p.copy(), QPParams(time_limit=60))
    s = solve_alphabb(p.copy(), AlphaBBParams(time_limit=60))
    assert s.info["root_convex_by_gershgorin"] is True
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - ref.objective) <= 1e-5 * max(1.0, abs(ref.objective))


def test_the_cli_routes_a_nonconvex_q_instead_of_refusing_it():
    """The CLI's default route is the McCormick reformulation; αBB is the
    opt-in one. Both must agree with the convex solver's refusal being the
    wrong final answer."""
    from sovopt.cli import solve
    p = indefinite(0, n=3, m=1)
    with pytest.raises(NotConvexError):
        solve_qp(p.copy(), QPParams(time_limit=10))
    s = solve(p.copy(), time_limit=120)
    assert s.status == Status.OPTIMAL
    assert s.method == "nonconvex-qp[mccormick]"
    a = solve_alphabb(p.copy(), AlphaBBParams(time_limit=120))
    assert abs(a.objective - s.objective) <= 1e-4 * max(1.0, abs(s.objective))


def test_unbounded_coupled_variable_is_refused_by_name():
    p = indefinite(0, n=3, m=1)
    p.col_names = ["flow", "temp", "yield"]
    p.col_ub[1] = INF
    with pytest.raises(ValueError, match="temp"):
        solve_alphabb(p, AlphaBBParams(time_limit=10))


def test_no_quadratic_term_is_refused():
    p = indefinite(0)
    p.Q = None
    with pytest.raises(ValueError):
        solve_alphabb(p)


# --------------------------------------------------------------------------- #
# the reduction: same answer, fewer nodes                                      #
# --------------------------------------------------------------------------- #


def test_marginal_reduction_changes_the_node_count_and_not_the_answer():
    """On the two-variable case in the module's smoke test the reduction took
    the search from 319 nodes to 7. It is derived from the certified bound's
    own pieces, so it is as valid as the bound; this checks the answer did not
    move and that the switch is doing something."""
    H = np.array([[1.0, 3.0], [3.0, 1.0]])
    A = SparseMatrix.from_dense(np.array([[1.0, 1.0]]))
    p = Problem(A=A, c=np.array([-1.0, -2.0]), Q=SparseMatrix.from_dense(H),
                row_lb=np.array([-INF]), row_ub=np.array([3.0]),
                col_lb=np.zeros(2), col_ub=np.full(2, 3.0))
    on = solve_alphabb(p.copy(), AlphaBBParams(time_limit=120))
    off = solve_alphabb(p.copy(), AlphaBBParams(time_limit=120,
                                                     marginal_reduction=False))
    assert on.status == off.status == Status.OPTIMAL
    assert abs(on.objective - (-2.0)) <= 1e-6 and abs(off.objective - (-2.0)) <= 1e-6
    assert on.nodes < off.nodes
    assert on.info["marginal_reductions"] > 0
