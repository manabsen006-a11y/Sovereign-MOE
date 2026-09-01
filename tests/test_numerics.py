"""Regression tests for the numerical kernel.

These are the tests that decide whether the solver can be trusted. They compare
against exact rational arithmetic rather than against another floating-point
implementation, because "agrees with NumPy" is not a correctness argument when
NumPy is subject to the same cancellation.
"""

from fractions import Fraction

import numpy as np
import pytest

from vyuha.core.sparse import SparseMatrix
from vyuha.numerics.lu import lu_factor, LUSingular
from vyuha.numerics.refine import (compensated_residual, estimate_condition,
                                   refine_solve)
from vyuha.numerics.scaling import compute_scaling


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
    from vyuha.core.problem import Problem, VarKind
    from vyuha.numerics.scaling import scale_problem

    B, _ = _random_sparse(40, 4, seed=8, spread=4)
    A = SparseMatrix.from_dense(B)
    kind = np.zeros(40, dtype=np.uint8)
    kind[::3] = VarKind.INTEGER
    p = Problem(A=A, c=np.ones(40), row_lb=np.zeros(40), row_ub=np.ones(40),
                col_lb=np.zeros(40), col_ub=np.full(40, 10.0), kind=kind)
    _, sc = scale_problem(p)
    assert np.all(sc.col[kind == VarKind.INTEGER] == 1.0)
