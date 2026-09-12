"""Regression tests for the numerical kernel.

These are the tests that decide whether the solver can be trusted. They compare
against exact rational arithmetic rather than against another floating-point
implementation, because "agrees with NumPy" is not a correctness argument when
NumPy is subject to the same cancellation.
"""

from fractions import Fraction

import numpy as np
import pytest

from sovopt.core.sparse import SparseMatrix
from sovopt.numerics.lu import lu_factor, LUSingular
from sovopt.numerics.refine import (compensated_residual, estimate_condition,
                                   refine_solve)
from sovopt.numerics.scaling import compute_scaling


def _exact_solve(B, b):
    """Gaussian elimination in exact rational arithmetic."""
    n = B.shape[0]
    M = [[Fraction(B[i, j]) for j in range(n)] + [Fraction(b[i])] for i in range(n)]
    for k in range(n):
        p = max(range(k, n), key=lambda r: abs(M[r][k]))
        M[k], M[p] = M[p], M[k]
        pk = M[k][k]
        assert pk != 0, "singular matrix in exact solve"
        for r in range(k + 1, n):
            if M[r][k]:
                f = M[r][k] / pk
                for c in range(k, n + 1):
                    M[r][c] -= f * M[k][c]
    x = [Fraction(0)] * n
    for k in range(n - 1, -1, -1):
        s = M[k][n] - sum(M[k][c] * x[c] for c in range(k + 1, n))
        x[k] = s / M[k][k]
    return np.array([float(v) for v in x])


def _random_sparse(n, per_col=4, seed=0, spread=0):
    rng = np.random.default_rng(seed)
    B = np.zeros((n, n))
    for j in range(n):
        rows = rng.choice(n, per_col, replace=False)
        B[rows, j] = rng.standard_normal(per_col)
    B[np.arange(n), np.arange(n)] += 8.0
    if spread:
        B = B * (10.0 ** rng.integers(-spread, spread + 1, size=n))[:, None]
    return B, rng.standard_normal(n)


# --------------------------------------------------------------------------- #
# LU                                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n,per_col", [(20, 20), (200, 6), (800, 6)])
def test_lu_ftran_btran_accuracy(n, per_col):
    B, b = _random_sparse(n, min(per_col, n), seed=n)
    S = SparseMatrix.from_dense(B)
    F = lu_factor(S.cp, S.ci, S.cx, n)

    x = F.ftran(b.copy())
    assert np.abs(B @ x - b).max() / np.abs(b).max() < 1e-10

    c = np.linspace(-1.0, 1.0, n)
    y = F.btran(c.copy())
    assert np.abs(B.T @ y - c).max() / np.abs(c).max() < 1e-10


def test_lu_detects_singular():
    B = np.eye(50)
    B[10, 10] = 0.0
    S = SparseMatrix.from_dense(B)
    with pytest.raises(LUSingular):
        lu_factor(S.cp, S.ci, S.cx, 50)


def test_lu_fill_stays_low_on_simplex_like_basis():
    """A basis that is mostly identity must factor with almost no fill.

    This is what the singleton-peeling column order is for; if it regresses,
    every simplex iteration gets slower.
    """
    rng = np.random.default_rng(5)
    n = 1000
    B = np.eye(n)
    for j in rng.choice(n, 120, replace=False):
        rows = rng.choice(n, 5, replace=False)
        B[:, j] = 0.0
        B[rows, j] = rng.standard_normal(5)
        B[j, j] += 5.0
    S = SparseMatrix.from_dense(B)
    F = lu_factor(S.cp, S.ci, S.cx, n)
    assert F.nnz / S.nnz < 3.0


def test_hypersparse_ftran_matches_dense_and_stays_sparse():
    rng = np.random.default_rng(9)
    n = 1000
    B = np.eye(n)
    for j in rng.choice(n, 100, replace=False):
        rows = rng.choice(n, 4, replace=False)
        B[rows, j] = rng.standard_normal(4)
        B[j, j] += 5.0
    S = SparseMatrix.from_dense(B)
    F = lu_factor(S.cp, S.ci, S.cx, n)

    for _ in range(10):
        idx = np.sort(rng.choice(n, 3, replace=False)).astype(np.int32)
        val = rng.standard_normal(3)
        dense_rhs = np.zeros(n)
        dense_rhs[idx] = val

        xd = F.ftran(dense_rhs.copy())
        si, sv = F.ftran_sparse(idx, val)
        xs = np.zeros(n)
        xs[si] = sv

        assert np.abs(xd - xs).max() < 1e-12
        # the whole point: the solution must stay sparse
        assert len(si) < 0.05 * n


# --------------------------------------------------------------------------- #
# refinement                                                                   #
# --------------------------------------------------------------------------- #


def test_compensated_residual_beats_naive_under_cancellation():
    """On a badly scaled matrix the naive residual is meaningless.

    ``b - A @ x`` loses every significant digit to cancellation when the rows of
    ``A`` span twelve orders of magnitude. Refinement built on it converges to
    the wrong answer -- which is why this test exists.
    """
    B, b = _random_sparse(60, 4, seed=3, spread=6)
    S = SparseMatrix.from_dense(B)
    F = lu_factor(S.cp, S.ci, S.cx, 60)
    x = F.ftran(b.copy())

    Bq = [[Fraction(B[i, j]) for j in range(60)] for i in range(60)]
    exact = np.array([float(Fraction(b[i]) - sum(Bq[i][j] * Fraction(x[j])
                                                 for j in range(60)))
                      for i in range(60)])

    naive = b - B @ x
    comp = compensated_residual(S.rp, S.ri, S.rx, x, b)

    scale = np.abs(exact).max()
    err_naive = np.abs(naive - exact).max() / scale
    err_comp = np.abs(comp - exact).max() / scale

    assert err_comp < 1e-15
    assert err_comp < err_naive / 1e6


def test_refinement_reaches_machine_precision_on_ill_conditioned():
    B, b = _random_sparse(60, 4, seed=3, spread=6)
    assert np.linalg.cond(B, 1) > 1e10

    S = SparseMatrix.from_dense(B)
    F = lu_factor(S.cp, S.ci, S.cx, 60)
    exact = _exact_solve(B, b)

    plain = F.ftran(b.copy())
    refined = refine_solve(F, S.rp, S.ri, S.rx, b).x

    def fwd(x):
        return np.abs(x - exact).max() / np.abs(exact).max()

    def omega(x):
        r = B @ x - b
        den = np.abs(B) @ np.abs(x) + np.abs(b)
        return np.max(np.abs(r) / np.where(den > 0, den, 1.0))

    assert fwd(refined) <= fwd(plain)
    assert omega(refined) < 1e-14


def test_condition_estimate_is_a_reasonable_lower_bound():
    B, _ = _random_sparse(200, 5, seed=4, spread=4)
    S = SparseMatrix.from_dense(B)
    F = lu_factor(S.cp, S.ci, S.cx, 200)
    true = np.linalg.cond(B, 1)
    est = estimate_condition(F, float(np.abs(B).sum(axis=0).max()), 200)
    assert 0.1 * true <= est <= 1.05 * true


# --------------------------------------------------------------------------- #
# scaling                                                                      #
# --------------------------------------------------------------------------- #


def test_curtis_reid_shrinks_coefficient_spread():
    B, _ = _random_sparse(300, 4, seed=11, spread=6)
    S = SparseMatrix.from_dense(B)
    sc = compute_scaling(S, method="curtis_reid")
    assert sc.ratio_after < sc.ratio_before / 1e6


def test_scale_factors_are_exact_powers_of_two():
    """Power-of-two factors make scaling lossless in binary floating point."""
    B, _ = _random_sparse(100, 4, seed=2, spread=5)
    S = SparseMatrix.from_dense(B)
    sc = compute_scaling(S, method="ruiz")
    for arr in (sc.row, sc.col):
        assert np.all(np.abs(np.log2(arr) - np.rint(np.log2(arr))) < 1e-12)


def test_integer_columns_are_never_scaled():
    """Scaling an integer column would destroy integrality of x' = x / c."""
    from sovopt.core.problem import Problem, VarKind
    from sovopt.numerics.scaling import scale_problem

    B, _ = _random_sparse(40, 4, seed=8, spread=4)
    A = SparseMatrix.from_dense(B)
    kind = np.zeros(40, dtype=np.uint8)
    kind[::3] = VarKind.INTEGER
    p = Problem(A=A, c=np.ones(40), row_lb=np.zeros(40), row_ub=np.ones(40),
                col_lb=np.zeros(40), col_ub=np.full(40, 10.0), kind=kind)
    _, sc = scale_problem(p)
    assert np.all(sc.col[kind == VarKind.INTEGER] == 1.0)


def test_the_objective_scale_sees_the_hessian():
    """QPLIB_8559 has ``c = 0`` and a ``Q`` whose diagonal runs to 95,000. An
    objective scale taken from ``c`` alone is 1 there, so the interior point
    started with a dual residual of 1.7e5, drove the iterate to its bounds
    before that was gone, and crawled: 87 iterations in 600 s without
    converging. The scale of a quadratic objective is the scale of its
    gradient ``c + Qx``, and the diagonal of the column-scaled ``Q`` is the
    curvature along each scaled unit direction, so it joins the geometric
    mean: the same instance then takes 15 iterations."""
    from sovopt.core.problem import Problem
    from sovopt.numerics.scaling import scale_problem

    n = 30
    B, _ = _random_sparse(n, 3, seed=5, spread=1)
    A = SparseMatrix.from_dense(B)
    Q = SparseMatrix.from_dense(np.diag(np.geomspace(4.0, 95000.0, n)))
    p = Problem(A=A, c=np.zeros(n), row_lb=np.ones(n), row_ub=np.ones(n),
                col_lb=np.full(n, 0.1), col_ub=np.full(n, 10.0), Q=Q)
    scaled, sc = scale_problem(p)
    assert sc.obj < 1e-2                          # c alone would say 1
    d = scaled.Q.diagonal()
    assert 0.5 <= np.sqrt(d.max() * d.min()) <= 2.0
    # the same model without its Hessian is an LP with no objective at all,
    # and the scale of nothing is 1
    lp = Problem(A=A, c=np.zeros(n), row_lb=np.ones(n), row_ub=np.ones(n),
                 col_lb=np.full(n, 0.1), col_ub=np.full(n, 10.0))
    assert scale_problem(lp)[1].obj == 1.0


def test_sparse_diagonal_is_read_off_the_columns():
    M = np.zeros((3, 5))
    M[0, 0], M[1, 1], M[2, 2], M[0, 3], M[2, 4] = 1.5, -2.0, 0.25, 9.0, 9.0
    S = SparseMatrix.from_dense(M)
    assert np.array_equal(S.diagonal(), [1.5, -2.0, 0.25])
    assert np.array_equal(SparseMatrix.from_dense(M.T).diagonal(), [1.5, -2.0, 0.25])


def test_bound_scaling_changes_the_unit_and_leaves_a_lone_wide_column_alone():
    """QPLIB_9002's shape: every variable lives at 1e11 and the barrier's
    1e-12 complementarity floor is rounding noise there. The unit is the
    log-mean of the finite bound magnitudes; one wide column among unit
    ones (mas76's shape) barely moves it."""
    from sovopt.core.problem import Problem
    from sovopt.numerics.scaling import scale_problem
    from sovopt.core.tolerances import INF

    n = 30
    B, _ = _random_sparse(n, 3, seed=9, spread=1)
    A = SparseMatrix.from_dense(B)
    big = Problem(A=A, c=np.zeros(n), row_lb=np.zeros(n), row_ub=np.zeros(n),
                  col_lb=np.full(n, -1e11), col_ub=np.full(n, 1e11))
    off, sc = scale_problem(big, method="ruiz")
    on, sc_on = scale_problem(big, method="ruiz", bound_scaling=True)
    assert np.abs(off.col_ub).max() > 1e9
    assert np.abs(on.col_ub).max() < 1e2
    # the equilibration is unchanged by a uniform unit: the scaled matrix
    # entries are the same
    assert np.allclose(np.sort(np.abs(on.A.cx)), np.sort(np.abs(off.A.cx)))
    # and a point maps back where it came from
    x = np.linspace(-1e10, 1e10, n)
    assert np.allclose(sc_on.unscale_primal(sc_on.scale_primal(x)), x)

    ub = np.ones(n)
    ub[0] = 1e12
    lone = Problem(A=A, c=np.zeros(n), row_lb=np.zeros(n), row_ub=np.zeros(n),
                   col_lb=np.zeros(n), col_ub=ub)
    a, _ = scale_problem(lone, method="ruiz")
    b, _ = scale_problem(lone, method="ruiz", bound_scaling=True)
    assert b.col_ub[0] >= a.col_ub[0] / 4          # the unit is still ~1

