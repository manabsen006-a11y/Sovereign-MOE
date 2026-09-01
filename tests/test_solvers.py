"""End-to-end solver tests, including regressions for two real bugs.

Both bugs below were found by the independent verifier and by comparison against
published reference values, not by unit tests -- which is the argument for
having both.
"""

import numpy as np
import pytest

from vyuha.core.problem import ObjSense, Problem, Status, VarKind
from vyuha.core.sparse import SparseMatrix
from vyuha.core.tolerances import INF
from vyuha.lp.pdlp import PDLPParams, solve_pdlp
from vyuha.mip.safebound import safe_dual_bound
from vyuha.mip.tree import MIPParams, solve_mip
from vyuha.numerics.scaling import scale_problem


def toy_lp():
    """max 3x + 2y  s.t.  x+y<=4, x+3y<=6, x<=3, x,y>=0.  Optimum 11 at (3,1)."""
    A = SparseMatrix.from_dense(np.array([[1., 1.], [1., 3.], [1., 0.]]))
    return Problem(A=A, c=np.array([3., 2.]),
                   row_lb=np.full(3, -INF), row_ub=np.array([4., 6., 3.]),
                   col_lb=np.zeros(2), col_ub=np.full(2, INF),
                   sense=ObjSense.MAXIMISE, name="toy")


def badly_scaled_lp(seed=0, m=40, n=60, spread=6):
    """An LP whose rows span many orders of magnitude -- the refinery case."""
    rng = np.random.default_rng(seed)
    cols = np.repeat(np.arange(n), 4)
    rows = rng.integers(0, m, cols.size)
    vals = rng.standard_normal(cols.size)
    A = SparseMatrix.from_triplets(rows, cols, vals, m, n)
    A.scale(10.0 ** rng.integers(-spread, spread + 1, size=m).astype(float),
            np.ones(n))
    x0 = rng.random(n)
    b = A.matvec(x0) + rng.random(m)
    return Problem(A=A, c=rng.standard_normal(n),
                   row_lb=np.full(m, -INF), row_ub=b,
                   col_lb=np.zeros(n), col_ub=np.full(n, 10.0), name="scaled")


# --------------------------------------------------------------------------- #
# LP                                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("device", ["cpu"])
def test_pdlp_solves_toy_lp(device):
    s = solve_pdlp(toy_lp(), PDLPParams(device=device))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 11.0) < 1e-6
    assert np.allclose(s.x, [3.0, 1.0], atol=1e-5)


def test_pdlp_matches_known_mps_optimum():
    from vyuha.io.mps import read_mps
    import os
    path = os.path.join(os.path.dirname(__file__), "fixtures", "testprob.mps")
    s = solve_pdlp(read_mps(path), PDLPParams(device="cpu"))
    assert s.status == Status.OPTIMAL
    # min x1+2x2+3x3 with x3 = 7+x2 gives x1=0, x2=-1, x3=6, objective 16
    assert abs(s.objective - 16.0) < 1e-7
    assert np.allclose(s.x, [0.0, -1.0, 6.0], atol=1e-6)


def test_strong_duality_holds_after_unscaling():
    """REGRESSION: the dual unscaling multiplied by the objective scale where it
    must divide, leaving every dual wrong by a factor of ``obj_scale**2``.

    Residuals still looked converged and the sign conditions still held, so the
    only visible symptom was a duality gap pinned at a large constant: the
    solver burned 30,000 iterations on an 18x18 model instead of 1,920, and the
    marginal prices handed to a planner would have been silently wrong.

    The test drives a model whose objective scale is genuinely not 1, then
    checks that the reported gap actually closes and that complementary
    slackness holds against the *unscaled* duals.
    """
    prob = badly_scaled_lp(seed=1, m=30, n=40, spread=4)
    _, sc = scale_problem(prob, method="pdlp")
    assert sc.obj != 1.0, "test model does not exercise objective scaling"

    s = solve_pdlp(prob, PDLPParams(device="cpu", max_iter=60_000,
                                    time_limit=60))
    assert s.status == Status.OPTIMAL
    # the bug left this pinned at a large constant
    assert s.info.gap <= 1e-4 * max(1.0, abs(s.objective))

    # Complementary slackness is a condition on the *product* y_i * slack_i,
    # not on either factor alone: a row can sit 1e-4 from its bound and still
    # carry a meaningful price. Testing the factors separately fails on
    # perfectly correct output.
    act = prob.A.matvec(s.x)
    slack = np.where(prob.row_ub < INF, prob.row_ub - act, 0.0)
    comp = float(np.max(np.abs(s.y) * np.abs(slack)))
    assert comp <= 1e-4 * max(1.0, abs(s.objective))


def test_reported_optimal_always_passes_the_verifier():
    """REGRESSION: termination used a purely relative feasibility test.

    On a model with a large right-hand side that permitted absolute row
    violations far beyond what the independent verifier accepts, so the solver
    reported OPTIMAL for points the checker rejected.
    """
    from bench.verify import verify

    for seed in range(4):
        prob = badly_scaled_lp(seed=seed)
        s = solve_pdlp(prob, PDLPParams(device="cpu", max_iter=60_000,
                                        time_limit=60))
        if s.status != Status.OPTIMAL:
            continue
        v = verify(prob, s.x, s.objective, feas_tol=1e-6)
        assert v.ok, f"seed {seed} claimed OPTIMAL but:\n{v.report()}"


def test_maximise_and_minimise_agree():
    p = toy_lp()
    q = p.copy()
    q.c = -q.c
    q.sense = ObjSense.MINIMISE
    a = solve_pdlp(p, PDLPParams(device="cpu"))
    b = solve_pdlp(q, PDLPParams(device="cpu"))
    assert abs(a.objective + b.objective) < 1e-5


# --------------------------------------------------------------------------- #
# safe bounds                                                                  #
# --------------------------------------------------------------------------- #


def test_safe_bound_never_exceeds_the_true_optimum():
    """The property the whole batched search rests on.

    For an arbitrary ``y`` -- feasible or not -- the Neumaier-Shcherbina bound
    must not exceed the true optimum, or branch-and-bound would prune away the
    answer.
    """
    prob = toy_lp()
    minimise = prob.copy()
    minimise.c = -prob.c
    minimise.sense = ObjSense.MINIMISE
    true_opt = -11.0

    rng = np.random.default_rng(0)
    for _ in range(300):
        y = rng.standard_normal(prob.m) * rng.choice([0.01, 1.0, 100.0])
        b = safe_dual_bound(minimise.A, minimise.c,
                            minimise.row_lb, minimise.row_ub,
                            minimise.col_lb, minimise.col_ub, y)
        assert b <= true_opt + 1e-7, f"invalid bound {b} > {true_opt}"


def test_safe_bound_is_tight_at_the_optimal_dual():
    """With finite variable bounds the bound recovers the optimum.

    Finiteness matters: if a reduced cost comes out on the wrong side of zero
    for a variable with an infinite bound -- which numerical noise alone can
    cause -- the corresponding term is unbounded and the whole bound collapses
    to -inf. That is safe but useless, and it is why the tree propagates bounds
    before bounding a node.
    """
    prob = toy_lp()
    minimise = prob.copy()
    minimise.c = -prob.c
    minimise.sense = ObjSense.MINIMISE
    minimise.col_ub = np.array([100.0, 100.0])       # finite box

    s = solve_pdlp(minimise, PDLPParams(device="cpu"))
    b = safe_dual_bound(minimise.A, minimise.c, minimise.row_lb, minimise.row_ub,
                        minimise.col_lb, minimise.col_ub, s.y)
    assert b <= -11.0 + 1e-6
    assert abs(b - (-11.0)) < 1e-3


def test_safe_bound_is_vacuous_not_wrong_on_unbounded_columns():
    """A wrong-signed reduced cost on an infinite bound must give -inf.

    Returning a finite number there would be an *invalid* bound and could prune
    the optimum away. Vacuous is the correct failure mode.
    """
    prob = toy_lp()
    minimise = prob.copy()
    minimise.c = -prob.c
    minimise.sense = ObjSense.MINIMISE
    y = np.zeros(prob.m)                  # d = c < 0 with u = +inf
    b = safe_dual_bound(minimise.A, minimise.c, minimise.row_lb, minimise.row_ub,
                        minimise.col_lb, minimise.col_ub, y)
    assert b == -np.inf


# --------------------------------------------------------------------------- #
# MILP                                                                         #
# --------------------------------------------------------------------------- #


def test_mip_matches_exact_knapsack_dp():
    rng = np.random.default_rng(0)
    n = 18
    w = rng.integers(1, 40, n).astype(float)
    v = rng.integers(1, 60, n).astype(float)
    cap = float(w.sum() * 0.4)

    A = SparseMatrix.from_dense(w.reshape(1, n))
    p = Problem(A=A, c=v, row_lb=np.array([-INF]), row_ub=np.array([cap]),
                col_lb=np.zeros(n), col_ub=np.ones(n),
                kind=np.ones(n, dtype=np.uint8),
                sense=ObjSense.MAXIMISE, name="knap01")

    cap_i = int(cap)
    dp = np.zeros(cap_i + 1)
    wi, vi = w.astype(int), v.astype(int)
    for i in range(n):
        dp[wi[i]:] = np.maximum(dp[wi[i]:], dp[:cap_i + 1 - wi[i]] + vi[i])

    s = solve_mip(p, MIPParams(device="cpu", time_limit=120))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - dp[cap_i]) < 1e-6
    rv, bv, iv = p.violation(s.x)
    assert max(rv, bv, iv) < 1e-6


def test_mip_detects_infeasibility():
    A = SparseMatrix.from_dense(np.array([[1.0, 1.0]]))
    p = Problem(A=A, c=np.ones(2), row_lb=np.array([5.0]), row_ub=np.array([INF]),
                col_lb=np.zeros(2), col_ub=np.ones(2),
                kind=np.ones(2, dtype=np.uint8), name="infeas")
    s = solve_mip(p, MIPParams(device="cpu", time_limit=30))
    assert s.status == Status.INFEASIBLE


def test_integer_solution_is_actually_integral():
    from vyuha.models import unit_scheduling
    p = unit_scheduling(n_units=3, n_periods=5, seed=0)
    s = solve_mip(p, MIPParams(device="cpu", time_limit=90))
    if s.x is None:
        pytest.skip("no incumbent within the time limit")
    mask = p.integer_mask
    assert np.abs(s.x[mask] - np.round(s.x[mask])).max() < 1e-6


def test_mip_incumbent_is_feasible_in_the_original_space():
    """REGRESSION: incumbents were validated in the *scaled* space.

    The search runs on a scaled model, but a row scaled by 1e-3 turns a scaled
    violation of 1e-6 into an unscaled one of 1e-3 -- a thousand times past what
    the independent verifier accepts. The tree therefore adopted infeasible
    points as incumbents and reported them as the answer (caught on MIPLIB
    mas76). Acceptance now unscales the candidate and tests it against the
    original model at the verifier's own tolerance.
    """
    from bench.verify import verify
    from vyuha.core.sparse import SparseMatrix
    from vyuha.mip.tree import MIPParams, solve_mip

    rng = np.random.default_rng(17)
    m, n = 18, 24
    A = SparseMatrix.from_dense(
        rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.4))
    # heavy row scaling is what makes the two spaces disagree
    A.scale(10.0 ** rng.integers(-3, 4, size=m).astype(float), np.ones(n))
    b = A.matvec(rng.random(n) * 3.0) + np.abs(A.matvec(np.ones(n))) * 0.1

    kind = np.zeros(n, dtype=np.uint8)
    kind[::2] = VarKind.INTEGER
    p = Problem(A=A, c=rng.standard_normal(n),
                row_lb=np.full(m, -INF), row_ub=b,
                col_lb=np.zeros(n), col_ub=np.full(n, 4.0), kind=kind,
                name="scaled_mip")

    s = solve_mip(p, MIPParams(device="cpu", time_limit=45))
    if s.x is None:
        pytest.skip("no incumbent found within the limit")
    v = verify(p, s.x, s.objective, feas_tol=1e-6, int_tol=1e-6)
    assert v.ok, f"reported an incumbent the verifier rejects:\n{v.report()}"
