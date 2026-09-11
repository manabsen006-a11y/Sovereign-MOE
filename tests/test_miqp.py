"""Mixed-integer quadratic programming.

The engine had a convex QP solver and a branch-and-bound tree and no way to put
them together, so a MIQP was refused. This is the join, and the standard it is
held to is the one the cuts and the batched bounder are held to: **brute force
over every integer point of a model small enough to enumerate**. The node
bound is certified by ``safe_qp_bound`` -- Neumaier-Shcherbina on the tangent
plane -- so the tree is rigorous on paper; agreement with an exhaustive search
is what establishes it in practice, and the test that solves every relaxation
deliberately badly is what shows the certificate, not the solver's accuracy,
is doing the work.
"""

import itertools

import numpy as np
import pytest

from sovopt.core.problem import ObjSense, Problem, Status
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.mip.miqp import MIQPParams, solve_miqp
from sovopt.qp import QPParams

UB = 3


def target_miqp(seed=0, n=4, m=3, maximise=False):
    """``min ½‖x−t‖²_H`` over a box and some rows, for a non-integer target t.

    Built this way on purpose: a random ``c`` with a positive-definite ``H``
    usually puts the optimum at the origin, and an optimum at zero tests
    nothing -- rounding the relaxation would find it. A target strictly inside
    the box forces the answer to a non-trivial integer point.
    """
    rng = np.random.default_rng(1000 + seed)
    M = rng.standard_normal((n, n))
    H = M.T @ M + 0.5 * np.eye(n)                 # positive definite
    t = rng.uniform(0.4, UB - 0.4, n)
    c = -(H @ t)
    A = SparseMatrix.from_dense(np.round(rng.uniform(-1, 3, (m, n))))
    b = np.abs(A.matvec(np.full(n, UB / 2.0))) + rng.uniform(1, 5, m)

    Q, cc, sense = SparseMatrix.from_dense(H), c, ObjSense.MINIMISE
    if maximise:
        Q, cc, sense = SparseMatrix.from_dense(-H), -c, ObjSense.MAXIMISE
    return Problem(A=A, c=cc, Q=Q, row_lb=np.full(m, -INF), row_ub=b,
                   col_lb=np.zeros(n), col_ub=np.full(n, float(UB)),
                   kind=np.ones(n, dtype=np.uint8), sense=sense,
                   name=f"miqp{seed}")


def ill_conditioned_miqp(seed=0, n=5, m=3, cond=1e3):
    """Same shape as :func:`target_miqp` with eigenvalues spread over ``cond``.

    A first-order method crawls on this, so a relaxation stopped after a few
    dozen iterations is genuinely far from optimal -- which is what a test of
    the certified bound needs, since on a well-conditioned model even a "bad"
    solve is close enough to hide an unsafe prune.
    """
    rng = np.random.default_rng(5000 + seed)
    Qm, _ = np.linalg.qr(rng.standard_normal((n, n)))
    H = Qm @ np.diag(np.logspace(0, np.log10(cond), n)) @ Qm.T
    H = 0.5 * (H + H.T)
    t = rng.uniform(0.4, UB - 0.4, n)
    c = -(H @ t)
    A = SparseMatrix.from_dense(np.round(rng.uniform(-1, 3, (m, n))))
    b = np.abs(A.matvec(np.full(n, UB / 2.0))) + rng.uniform(1, 5, m)
    return Problem(A=A, c=c, Q=SparseMatrix.from_dense(H),
                   row_lb=np.full(m, -INF), row_ub=b,
                   col_lb=np.zeros(n), col_ub=np.full(n, float(UB)),
                   kind=np.ones(n, dtype=np.uint8), name=f"ill{seed}")


def brute_force(p):
    """Every integer point of the box, checked against the original model."""
    best = None
    better = ((lambda a, b: a > b) if p.sense == ObjSense.MAXIMISE
              else (lambda a, b: a < b))
    for combo in itertools.product(range(UB + 1), repeat=p.n):
        x = np.array(combo, dtype=float)
        row_v, col_v, _ = p.violation(x)
        if max(row_v, col_v) > 1e-9:
            continue
        v = p.objective(x)
        if best is None or better(v, best[0]):
            best = (v, x)
    return best


# --------------------------------------------------------------------------- #
# the standard: agreement with an exhaustive search                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force(seed):
    p = target_miqp(seed)
    best = brute_force(p)
    if best is None:
        pytest.skip("no feasible integer point")
    s = solve_miqp(p.copy(), MIQPParams(time_limit=120))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - best[0]) <= 1e-6 * max(1.0, abs(best[0]))


@pytest.mark.parametrize("seed", range(4))
def test_matches_brute_force_on_a_maximisation(seed):
    """A maximisation is converted by negating both ``c`` and ``Q``, so it is
    solvable exactly when the negated ``Q`` is positive semidefinite."""
    p = target_miqp(seed, maximise=True)
    best = brute_force(p)
    if best is None:
        pytest.skip("no feasible integer point")
    s = solve_miqp(p.copy(), MIQPParams(time_limit=120))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - best[0]) <= 1e-6 * max(1.0, abs(best[0]))


@pytest.mark.parametrize("seed", range(4))
def test_the_answer_is_integral_and_feasible_for_the_original_model(seed):
    """Every incumbent is validated against the original model before it is
    accepted, so this is what that validation is worth."""
    p = target_miqp(seed)
    s = solve_miqp(p.copy(), MIQPParams(time_limit=120))
    if s.x is None:
        pytest.skip("no incumbent")
    row_v, col_v, int_v = p.violation(s.x)
    assert max(row_v, col_v) < 1e-6
    assert int_v < 1e-6
    # the reported objective must be the one the model computes for that point
    assert abs(s.objective - p.objective(s.x)) < 1e-9


# --------------------------------------------------------------------------- #
# the certificate is what makes it right, not the QP solver's accuracy         #
# --------------------------------------------------------------------------- #

BADLY = QPParams(max_iter=64, check_every=64, eps_abs=1e-1, eps_rel=1e-1)
"""One convergence check and then stop, whatever the residuals say."""


@pytest.mark.parametrize("seed", range(20))
def test_exact_even_when_every_relaxation_is_solved_badly(seed):
    """Every node relaxation is stopped after 64 iterations on a model with a
    1e3 eigenvalue spread, so the QP solver's objective at a node is nowhere
    near the node's optimum. Pruning on that objective returned a wrong answer
    marked OPTIMAL on 5 of these 20 models; pruning on the certified bound
    returns the brute-force optimum on all of them, at the price of more nodes.
    """
    p = ill_conditioned_miqp(seed)
    best = brute_force(p)
    if best is None:
        pytest.skip("no feasible integer point")
    s = solve_miqp(p.copy(), MIQPParams(time_limit=120, qp=BADLY))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - best[0]) <= 1e-6 * max(1.0, abs(best[0])), (
        f"seed {seed}: brute force {best[0]}, tree {s.objective}")


@pytest.mark.parametrize("seed", range(4))
def test_the_reported_dual_bound_is_below_the_optimum_and_closes_the_gap(seed):
    p = target_miqp(seed)
    s = solve_miqp(p.copy(), MIQPParams(time_limit=120))
    assert s.status == Status.OPTIMAL
    assert np.isfinite(s.dual_bound)
    assert s.dual_bound <= s.objective + 1e-9
    assert s.gap <= 1e-4


def test_the_dual_bound_follows_the_sense_on_a_maximisation():
    p = target_miqp(0, maximise=True)
    s = solve_miqp(p.copy(), MIQPParams(time_limit=120))
    assert s.status == Status.OPTIMAL
    assert s.dual_bound >= s.objective - 1e-9
    assert s.gap <= 1e-4


# --------------------------------------------------------------------------- #
# the shapes it has to refuse or route                                         #
# --------------------------------------------------------------------------- #


def test_a_model_with_no_quadratic_term_is_refused():
    """A MILP belongs in the MILP tree, which has cuts, conflicts and a safe
    bound; routing it here would silently give up all three."""
    p = target_miqp(0)
    p.Q = None
    with pytest.raises(ValueError):
        solve_miqp(p)


def test_a_continuous_model_falls_through_to_the_qp_solver():
    p = target_miqp(1)
    p.kind = np.zeros(p.n, dtype=p.kind.dtype)
    s = solve_miqp(p, MIQPParams(time_limit=120))
    assert s.status == Status.OPTIMAL
    # the continuous optimum is at least as good as any integer point
    best = brute_force(target_miqp(1))
    if best is not None:
        assert s.objective <= best[0] + 1e-6


def test_a_non_convex_objective_is_still_refused():
    """Refusing is the correct answer, and it must survive the new path:
    branch-and-bound over a non-convex relaxation would return a confident
    wrong number rather than fail."""
    from sovopt.qp import QPParams
    rng = np.random.default_rng(0)
    n = 3
    H = np.diag([1.0, -2.0, 1.0])          # indefinite
    A = SparseMatrix.from_dense(np.ones((1, n)))
    p = Problem(A=A, c=rng.standard_normal(n),
                Q=SparseMatrix.from_dense(H),
                row_lb=np.array([-INF]), row_ub=np.array([5.0]),
                col_lb=np.zeros(n), col_ub=np.full(n, 3.0),
                kind=np.ones(n, dtype=np.uint8), name="nonconvex")
    with pytest.raises(Exception):
        solve_miqp(p, MIQPParams(time_limit=30,
                                 qp=QPParams(time_limit=30)))


def test_the_cli_routes_a_miqp_instead_of_refusing_it():
    """It used to raise NotImplementedError; the message said branch-and-bound
    over QP nodes was not implemented."""
    from sovopt.cli import solve

    p = target_miqp(2)
    s = solve(p.copy(), time_limit=120)
    assert s.status == Status.OPTIMAL
    assert s.method == "miqp-bb"


def test_the_bound_is_reported_as_rigorous():
    """It was reported as *not* rigorous when the tree pruned on the QP
    solver's own objective. The flag flipped when the pruning did; it is kept
    so that a caller can tell which tree they are looking at."""
    p = target_miqp(0)
    s = solve_miqp(p.copy(), MIQPParams(time_limit=120))
    assert s.info["bound_is_rigorous"] is True
    # the generator's H = MᵀM + ½I is not diagonally dominant, so convexity
    # here is estimated, not certified, and the flag must say so
    assert s.info["convexity_certified"] is False
    q = p.copy()
    q.Q = SparseMatrix.from_dense(np.diag([1.0, 2.0, 3.0, 4.0]))
    s = solve_miqp(q, MIQPParams(time_limit=120))
    assert s.info["convexity_certified"] is True
