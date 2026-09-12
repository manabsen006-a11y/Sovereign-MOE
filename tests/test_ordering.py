"""Fill-reducing orderings.

An ordering cannot make a factorisation wrong -- any permutation still factors
the same matrix -- so the failure modes here are of two kinds: returning
something that is not a permutation at all, and reducing fill on the cases it
was adopted for. Both are tested, and the first is fuzzed, because the bug this
module actually had returned a *short* array rather than a wrong one.
"""

import numpy as np
import pytest

from sovopt.core.problem import Problem, VarKind
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.numerics.lu import LUSingular, lu_factor
from sovopt.numerics.ordering import (amd_order, rcm_order, symbolic_fill,
                                      symmetric_pattern)


def _chain(n):
    """Tridiagonal: the banded case RCM exists for."""
    A = np.eye(n) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    return SparseMatrix.from_dense(A)


# --------------------------------------------------------------------------- #
# it must be a permutation, always                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(6))
def test_returns_a_permutation_on_random_patterns(seed):
    rng = np.random.default_rng(seed)
    for _ in range(30):
        n = int(rng.integers(3, 45))
        M = (rng.random((n, n)) < 0.06).astype(float) + np.eye(n)
        S = SparseMatrix.from_dense(M)
        q = rcm_order(S.cp, S.ci, n)
        assert sorted(q.tolist()) == list(range(n))


def test_covers_every_node_of_a_disconnected_graph():
    """REGRESSION: the ordering must not come back short.

    The first version chose its breadth-first root by scanning every remaining
    node for the lowest degree. That root can lie in a *different* component
    from the one the outer loop is standing on, so that component is laid out,
    the loop moves past the node it was on, and nothing ever returns to it.
    Measured on a mod010 simplex basis: 85 nodes placed of 146, and the caller
    got a RuntimeError rather than a silent corruption only because the length
    is checked. Each component is now found before its root is chosen.
    """
    n = 11
    A = np.eye(n)
    for i in (0, 1, 2):                       # one chain
        A[i, i + 1] = A[i + 1, i] = 1.0
    for i in (5, 6, 7, 8):                    # another, plus an isolated node
        A[i, i + 1] = A[i + 1, i] = 1.0
    S = SparseMatrix.from_dense(A)
    q = rcm_order(S.cp, S.ci, n)
    assert sorted(q.tolist()) == list(range(n))


def test_symmetric_pattern_is_symmetric_and_has_no_diagonal():
    rng = np.random.default_rng(0)
    n = 20
    M = (rng.random((n, n)) < 0.2).astype(float) + np.eye(n)
    S = SparseMatrix.from_dense(M)
    ap, ai = symmetric_pattern(S.cp, S.ci, n)
    edges = {(i, j) for i in range(n) for j in ai[ap[i]:ap[i + 1]]}
    assert all(i != j for i, j in edges), "diagonal survived"
    assert all((j, i) in edges for i, j in edges), "pattern not symmetric"


# --------------------------------------------------------------------------- #
# it must actually reduce fill where it is used                                #
# --------------------------------------------------------------------------- #


def test_reduces_fill_on_a_matrix_whose_band_is_hidden():
    """A chain permuted at random is dense-looking to a naive order.

    The point of an ordering is that it recovers structure the input does not
    present in order. A tridiagonal matrix with its rows and columns shuffled
    is still tridiagonal underneath, and RCM finds that back.
    """
    n = 300
    S = _chain(n)
    rng = np.random.default_rng(4)
    p = rng.permutation(n)
    D = S.to_dense()[np.ix_(p, p)]
    Sp = SparseMatrix.from_dense(D)

    plain = lu_factor(Sp.cp, Sp.ci, Sp.cx, n, tol=0.01)
    q = rcm_order(Sp.cp, Sp.ci, n)
    ordered = lu_factor(Sp.cp, Sp.ci, Sp.cx, n, tol=0.01, order=q)
    assert ordered.nnz < plain.nnz


def test_ordering_does_not_change_the_solution():
    """Any permutation factors the same matrix; only sparsity may differ."""
    n = 120
    S = _chain(n)
    rng = np.random.default_rng(1)
    b = rng.standard_normal(n)
    q = rcm_order(S.cp, S.ci, n)
    x0 = lu_factor(S.cp, S.ci, S.cx, n, tol=0.01).ftran(b.copy())
    x1 = lu_factor(S.cp, S.ci, S.cx, n, tol=0.01, order=q).ftran(b.copy())
    assert np.abs(x0 - x1).max() < 1e-9 * max(1.0, np.abs(x0).max())
    assert np.abs(S.matvec(x1) - b).max() < 1e-9


# --------------------------------------------------------------------------- #
# the fill cap that makes racing orderings affordable                          #
# --------------------------------------------------------------------------- #


def test_max_nnz_gives_up_instead_of_filling_in():
    """``max_nnz`` is what lets a losing trial ordering be abandoned early.

    Without it the trial runs to completion: on the mod010 KKT the losing
    candidate reaches 277x fill in 24 s while the winner takes 0.004 s, so the
    choice costs far more than it saves. The cap turns that into an exception
    as soon as the candidate stops being competitive.
    """
    n = 60
    rng = np.random.default_rng(2)
    M = (rng.random((n, n)) < 0.25).astype(float) + n * np.eye(n)
    S = SparseMatrix.from_dense(M)
    full = lu_factor(S.cp, S.ci, S.cx, n, tol=0.01)
    with pytest.raises(LUSingular):
        lu_factor(S.cp, S.ci, S.cx, n, tol=0.01, max_nnz=full.nnz // 4)
    # generous cap still succeeds and agrees
    ok = lu_factor(S.cp, S.ci, S.cx, n, tol=0.01, max_nnz=full.nnz * 4)
    assert ok.nnz == full.nnz


# --------------------------------------------------------------------------- #
# the interior point's use of it                                               #
# --------------------------------------------------------------------------- #


def test_ipm_reports_and_honours_its_ordering_choice():
    """The race must pick one, report it, and reach the same answer either way."""
    from sovopt.lp.ipm import IPMParams, solve_ipm

    rng = np.random.default_rng(5)
    m, n = 40, 60
    cols = np.repeat(np.arange(n), 3)
    rows = rng.integers(0, m, cols.size)
    A = SparseMatrix.from_triplets(rows, cols, rng.random(cols.size) + 0.5, m, n)
    b = A.matvec(np.full(n, 0.5))
    p = Problem(A=A, c=rng.standard_normal(n), row_lb=b, row_ub=b,
                col_lb=np.zeros(n), col_ub=np.ones(n), name="ord")

    auto = solve_ipm(p, IPMParams(ordering="auto"))
    none = solve_ipm(p, IPMParams(ordering="none"))
    rcm = solve_ipm(p, IPMParams(ordering="rcm"))
    assert auto.info["kkt_ordering"] in ("amd", "rcm", "none")
    for s in (auto, none, rcm):
        assert s.status.name == "OPTIMAL"
        assert abs(s.objective - none.objective) <= 1e-6 * max(1.0, abs(none.objective))


# --------------------------------------------------------------------------- #
# approximate minimum degree                                                   #
# --------------------------------------------------------------------------- #


def _grid(k):
    """The 5-point Laplacian on a k x k grid: the textbook ordering test."""
    n = k * k
    D = np.zeros((n, n))
    for i in range(k):
        for j in range(k):
            u = i * k + j
            D[u, u] = 4.0
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < k and 0 <= b < k:
                    D[u, a * k + b] = -1.0
    return SparseMatrix.from_dense(D)


def _random_symmetric(rng, n, density):
    M = (rng.random((n, n)) < density).astype(float)
    M = np.maximum(M, M.T) + 4.0 * np.eye(n) * n
    return SparseMatrix.from_dense(M)


@pytest.mark.parametrize("seed", range(6))
def test_amd_returns_a_permutation(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(1, 90))
    S = _random_symmetric(rng, n, rng.uniform(0.02, 0.4))
    q = amd_order(S.cp, S.ci, n)
    assert q.shape == (n,)
    assert sorted(q.tolist()) == list(range(n))


def test_amd_handles_the_degenerate_shapes():
    assert amd_order(np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32), 0).size == 0
    S = SparseMatrix.from_dense(np.eye(5))             # no edges at all
    assert sorted(amd_order(S.cp, S.ci, 5).tolist()) == list(range(5))
    S = SparseMatrix.from_dense(np.ones((7, 7)))       # one clique
    assert sorted(amd_order(S.cp, S.ci, 7).tolist()) == list(range(7))


def test_amd_beats_rcm_and_the_natural_order_on_a_grid():
    """Minimum degree's home ground: on a 2-D grid the band is wide and the
    nested structure is what matters. 40x40: natural 26x, RCM 12x, AMD 5x."""
    k = 40
    S = _grid(k)
    n = k * k
    nat = lu_factor(S.cp, S.ci, S.cx, n, tol=0.01).nnz
    rcm = lu_factor(S.cp, S.ci, S.cx, n, tol=0.01, order=rcm_order(S.cp, S.ci, n)).nnz
    amd = lu_factor(S.cp, S.ci, S.cx, n, tol=0.01, order=amd_order(S.cp, S.ci, n)).nnz
    assert amd < 0.7 * rcm
    assert amd < 0.4 * nat


def test_amd_matches_exact_minimum_degree_fill_on_small_cases():
    """The approximate degree is an upper bound the paper shows is usually
    tight; on these it must not cost more than an exact minimum degree by
    more than a few per cent."""
    import heapq

    def exact_md(ap, ai, n):
        adj = [set(ai[ap[i]:ap[i + 1]].tolist()) for i in range(n)]
        alive = np.ones(n, dtype=bool)
        heap = [(len(adj[i]), i) for i in range(n)]
        heapq.heapify(heap)
        order = []
        while heap:
            d, k = heapq.heappop(heap)
            if not alive[k]:
                continue
            if d != len(adj[k]):
                heapq.heappush(heap, (len(adj[k]), k))
                continue
            alive[k] = False
            order.append(k)
            nb = adj[k]
            for i in nb:
                adj[i].discard(k)
                adj[i] |= nb
                adj[i].discard(i)
                heapq.heappush(heap, (len(adj[i]), i))
            adj[k] = set()
        return np.array(order)

    rng = np.random.default_rng(9)
    for trial in range(5):
        n = 80
        S = _random_symmetric(rng, n, 0.05) if trial % 2 else _grid(9)
        n = S.n
        ap, ai = symmetric_pattern(S.cp, S.ci, n)
        f_amd = symbolic_fill(S.cp, S.ci, n, amd_order(S.cp, S.ci, n))
        f_md = symbolic_fill(S.cp, S.ci, n, exact_md(ap, ai, n))
        assert f_amd <= 1.1 * f_md + 5, (f_amd, f_md)


def test_amd_sets_dense_rows_aside_and_orders_them_last():
    """A row touching everything would otherwise sit in every clique; it is
    set aside and comes last, where it costs nothing extra."""
    n = 200
    rng = np.random.default_rng(3)
    S = _random_symmetric(rng, n, 0.02)
    D = S.to_dense()
    D[0, :] = 1.0
    D[:, 0] = 1.0
    S = SparseMatrix.from_dense(D)
    q = amd_order(S.cp, S.ci, n)
    assert q[-1] == 0
    assert sorted(q.tolist()) == list(range(n))


def test_amd_ordering_does_not_change_the_solution():
    k = 12
    S = _grid(k)
    n = k * k
    rng = np.random.default_rng(1)
    b = rng.standard_normal(n)
    x0 = lu_factor(S.cp, S.ci, S.cx, n, tol=0.01).ftran(b.copy())
    x1 = lu_factor(S.cp, S.ci, S.cx, n, tol=0.01,
                   order=amd_order(S.cp, S.ci, n)).ftran(b.copy())
    assert np.abs(x0 - x1).max() < 1e-9 * max(1.0, np.abs(x0).max())


# --------------------------------------------------------------------------- #
# symbolic fill                                                                #
# --------------------------------------------------------------------------- #


def test_symbolic_fill_counts_a_pivot_free_elimination_exactly():
    """On a tridiagonal chain in its natural order nothing fills: nnz(L) is
    exactly 2n - 1. Reversed, still nothing."""
    n = 50
    S = _chain(n)
    assert symbolic_fill(S.cp, S.ci, n, np.arange(n)) == 2 * n - 1
    assert symbolic_fill(S.cp, S.ci, n, np.arange(n)[::-1]) == 2 * n - 1
    # a star ordered centre-first fills completely; leaves-first does not
    D = np.eye(n)
    D[0, :] = 1.0
    D[:, 0] = 1.0
    S = SparseMatrix.from_dense(D)
    assert symbolic_fill(S.cp, S.ci, n, np.arange(n)) == n + (n - 1) * n // 2
    assert symbolic_fill(S.cp, S.ci, n, np.arange(n)[::-1]) == 2 * n - 1


def test_symbolic_fill_respects_its_cap():
    n = 50
    D = np.eye(n)
    D[0, :] = 1.0
    D[:, 0] = 1.0
    S = SparseMatrix.from_dense(D)
    assert symbolic_fill(S.cp, S.ci, n, np.arange(n), cap=100) is None
    assert symbolic_fill(S.cp, S.ci, n, np.arange(n)[::-1], cap=100) == 2 * n - 1


def test_the_ipm_race_ranks_by_symbolic_fill_and_reports_the_pick():
    """The candidate factorised in full is the one predicted to win, and the
    prediction and the pick are both on the solution."""
    from sovopt.lp.ipm import IPMParams, solve_ipm

    rng = np.random.default_rng(11)
    m, n = 60, 90
    cols = np.repeat(np.arange(n), 3)
    rows = rng.integers(0, m, cols.size)
    A = SparseMatrix.from_triplets(rows, cols, rng.random(cols.size) + 0.5, m, n)
    b = A.matvec(np.full(n, 0.5))
    p = Problem(A=A, c=rng.standard_normal(n), row_lb=b, row_ub=b,
                col_lb=np.zeros(n), col_ub=np.ones(n), name="ord")
    s = solve_ipm(p, IPMParams(ordering="auto"))
    assert s.status.name == "OPTIMAL"
    assert s.info["kkt_ordering"] in ("amd", "rcm", "none")
    a = solve_ipm(p, IPMParams(ordering="amd"))
    assert a.info["kkt_ordering"] == "amd"
    assert abs(a.objective - s.objective) <= 1e-7 * max(1.0, abs(s.objective))
