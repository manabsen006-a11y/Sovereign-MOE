"""The symmetric LDLᵀ for quasi-definite matrices.

Three things it claims. It factorises under *any* permutation without
pivoting (Vanderbei) -- so random permutations are as good a test as the
ordering it was built for -- and the inertia is exactly the block sizes. Its
fill is exactly what the symbolic count predicted, because it takes the
pivots the ordering names. And inside the interior point it reaches the
LU's answer on every instance, falling back to the LU when a pivot has to be
corrected rather than returning a factorisation of the wrong matrix.
"""

import os

import numpy as np
import pytest

from sovopt.core.problem import Problem, Status
from sovopt.core.sparse import SparseMatrix, coo_to_csc
from sovopt.io.mps import read_mps
from sovopt.lp.ipm import IPMParams, solve_ipm
from sovopt.numerics.ldl import LDLSingular, LDLSymbolic
from sovopt.numerics.ordering import amd_order, rcm_order, symbolic_fill

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "instances")


def quasi_definite(rng, n, m, density=0.15, e_lo=1e-6):
    A = np.where(rng.random((m, n)) < density, rng.standard_normal((m, n)), 0.0)
    E = np.diag(rng.uniform(e_lo, 10.0, n))
    F = np.diag(rng.uniform(e_lo, 10.0, m))
    K = np.block([[-E, A.T], [A, F]])
    N = n + m
    r, c = np.nonzero(K)
    cp, ci, cx = coo_to_csc(r, c, K[r, c], N, N)
    return K, cp, ci, cx


# --------------------------------------------------------------------------- #
# the factorisation                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(6))
def test_solves_under_any_permutation_with_the_right_inertia(seed):
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(5, 50)), int(rng.integers(3, 40))
    K, cp, ci, cx = quasi_definite(rng, n, m)
    N = n + m
    b = rng.standard_normal(N)
    for perm in (np.arange(N), rng.permutation(N), amd_order(cp, ci, N),
                 rcm_order(cp, ci, N)):
        sym = LDLSymbolic(cp, ci, N, perm)
        f = sym.factor(cx)
        x = f.ftran(b.copy())
        assert np.abs(K @ x - b).max() <= 1e-10 * np.abs(b).max()
        assert f.n_neg == n                       # Vanderbei's inertia
        assert f.n_corrected == 0


def test_fill_is_exactly_the_symbolic_prediction():
    """No pivoting means the pattern is the pattern that was counted."""
    rng = np.random.default_rng(7)
    K, cp, ci, cx = quasi_definite(rng, 60, 40)
    N = 100
    for perm in (np.arange(N), amd_order(cp, ci, N), rng.permutation(N)):
        sym = LDLSymbolic(cp, ci, N, perm)
        assert sym.lnz + N == symbolic_fill(cp, ci, N, perm)
        f = sym.factor(cx)
        assert np.count_nonzero(f.Lx != 0.0) <= sym.lnz


def test_the_symbolic_part_is_reused_across_values():
    """Same pattern, new diagonals: one analysis, many factorisations."""
    rng = np.random.default_rng(3)
    K, cp, ci, cx = quasi_definite(rng, 30, 20)
    N = 50
    sym = LDLSymbolic(cp, ci, N, amd_order(cp, ci, N))
    b = rng.standard_normal(N)
    for _ in range(4):
        d = rng.uniform(0.1, 5.0, N)
        K2 = K.copy()
        K2[np.arange(N), np.arange(N)] = np.concatenate([-d[:30], d[30:]])
        r, c = np.nonzero(K2)
        cp2, ci2, cx2 = coo_to_csc(r, c, K2[r, c], N, N)
        assert np.array_equal(cp2, cp) and np.array_equal(ci2, ci)
        x = sym.factor(cx2).ftran(b.copy())
        assert np.abs(K2 @ x - b).max() <= 1e-10 * np.abs(b).max()


def test_a_zero_pivot_is_refused_when_no_sign_is_known():
    K = np.array([[0.0, 1.0], [1.0, 1.0]])
    r, c = np.nonzero(K)
    cp, ci, cx = coo_to_csc(r, c, K[r, c], 2, 2)
    sym = LDLSymbolic(cp, ci, 2, np.arange(2))
    with pytest.raises(LDLSingular):
        sym.factor(cx)
    # ordered the other way the pivots are 1 and -1: fine
    sym = LDLSymbolic(cp, ci, 2, np.array([1, 0]))
    f = sym.factor(cx)
    assert f.n_neg == 1


def test_a_wrong_sided_pivot_is_corrected_and_counted():
    """A block that is not definite enough: with the signs known, the pivot
    is replaced by ``sign * delta`` and the count says so."""
    K = np.array([[1e-12, 2.0], [2.0, 1.0]])       # (1,1) should be negative
    r, c = np.nonzero(K)
    cp, ci, cx = coo_to_csc(r, c, K[r, c], 2, 2)
    sym = LDLSymbolic(cp, ci, 2, np.arange(2))
    f = sym.factor(cx, sign=np.array([-1, 1]), delta=1e-8)
    assert f.n_corrected == 1
    assert f.D[0] == -1e-8
    assert f.n_neg == 1


def test_matches_a_dense_ldl_on_a_kkt_from_a_real_model():
    path = os.path.join(DATA, "gt2.mps")
    if not os.path.exists(path):
        pytest.skip("instance not fetched")
    from sovopt.lp.ipm import _KKT
    from sovopt.numerics.scaling import scale_problem
    p = read_mps(path)
    scaled, _ = scale_problem(p, method="ruiz")
    k = _KKT(scaled.A, ordering="none", factorisation="lu")
    n, m = scaled.n, scaled.m
    k.kx[k.diag_x] = -(k.q_diag + 1.0)
    k.kx[k.diag_s] = 1.0
    sym = LDLSymbolic(k.kp, k.ki, k.nk, amd_order(k.kp, k.ki, k.nk))
    f = sym.factor(k.kx)
    assert f.n_neg == n
    rng = np.random.default_rng(0)
    b = rng.standard_normal(k.nk)
    x = f.ftran(b.copy())
    K = np.zeros((k.nk, k.nk))
    for j in range(k.nk):
        for q in range(k.kp[j], k.kp[j + 1]):
            K[k.ki[q], j] = k.kx[q]
    assert np.abs(K @ x - b).max() <= 1e-9 * np.abs(b).max()


# --------------------------------------------------------------------------- #
# inside the interior point                                                    #
# --------------------------------------------------------------------------- #


def _lp(seed, m=40, n=60):
    rng = np.random.default_rng(seed)
    cols = np.repeat(np.arange(n), 3)
    rows = rng.integers(0, m, cols.size)
    A = SparseMatrix.from_triplets(rows, cols, rng.random(cols.size) + 0.5, m, n)
    b = A.matvec(np.full(n, 0.5))
    return Problem(A=A, c=rng.standard_normal(n), row_lb=b, row_ub=b,
                   col_lb=np.zeros(n), col_ub=np.ones(n), name="ldl")


@pytest.mark.parametrize("seed", range(4))
def test_the_interior_point_reaches_the_same_answer_with_either_factorisation(seed):
    p = _lp(seed)
    lu = solve_ipm(p, IPMParams(factorisation="lu"))
    ldl = solve_ipm(p, IPMParams(factorisation="ldl"))
    assert lu.status == ldl.status == Status.OPTIMAL
    assert abs(lu.objective - ldl.objective) <= 1e-7 * max(1.0, abs(lu.objective))
    assert ldl.info["kkt_ordering"].startswith("ldl-")


def test_auto_falls_back_to_the_lu_when_a_pivot_needs_correcting():
    """mod010 is the instance: 18 iterations on the LDLᵀ, then a corrected
    pivot, then the LU for the last one -- and the LU's answer."""
    path = os.path.join(DATA, "mod010.mps")
    if not os.path.exists(path):
        pytest.skip("instance not fetched")
    p = read_mps(path)
    s = solve_ipm(p, IPMParams(factorisation="auto"))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 6532.083333) <= 1e-5 * 6532
    assert s.info["kkt_ordering"].startswith("lu-")
    assert s.info["ldl_iterations"] >= 10
    forced = solve_ipm(p, IPMParams(factorisation="ldl"))
    assert forced.status != Status.OPTIMAL       # the reason auto exists


def test_a_qp_goes_through_the_ldl_too():
    from tests.test_safe_qp_bound import convex_qp
    p = convex_qp(2)
    s = solve_ipm(p, IPMParams())
    assert s.status == Status.OPTIMAL
    assert s.info["kkt_ordering"].startswith("ldl-")
