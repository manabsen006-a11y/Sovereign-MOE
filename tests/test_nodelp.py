"""The node-LP dual simplex kernel, and the threads it enables.

The claim is exactness of a different kind from the rest of the suite: the
kernel is the *same* algorithm as the Python dual loop and must take the
same pivots -- status, iteration count and objective identical on every node
solve, warm or cold -- so that every property already proved of the loop
transfers to it. Then the tree: any thread count must return the same
objective and the same node count, because node ``t`` of a slab always goes
to solver ``t mod threads`` and results are applied in slab order.
"""

import os

import numpy as np
import pytest

from sovopt.core.problem import Status
from sovopt.io.mps import read_mps
from sovopt.lp.basis import BASIC
from sovopt.lp.simplex import NodeSolver, SimplexParams, solve_simplex
from sovopt.mip.tree import MIPParams, solve_mip
from sovopt.numerics.scaling import scale_problem

from tests.test_simplex import random_lp

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "instances")


def _pair(prob, refactor_freq=60):
    py = NodeSolver(prob, SimplexParams(refactor_freq=refactor_freq, node_kernel=False))
    kn = NodeSolver(prob, SimplexParams(refactor_freq=refactor_freq, node_kernel=True))
    return py, kn


def _same(a, b):
    if a.status != b.status or a.iterations != b.iterations:
        return False
    if a.status == Status.OPTIMAL:
        return abs(a.objective - b.objective) <= 1e-9 * max(1.0, abs(a.objective))
    return True


# --------------------------------------------------------------------------- #
# pivot-for-pivot parity with the Python loop                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(6))
def test_cold_solves_take_the_same_pivots(seed):
    p = random_lp(seed=seed, m=25, n=60)
    py, kn = _pair(p)
    a = py.solve(p.col_lb.copy(), p.col_ub.copy())
    b = kn.solve(p.col_lb.copy(), p.col_ub.copy())
    assert _same(a, b), (a, b)


@pytest.mark.parametrize("seed", range(4))
def test_warm_node_solves_take_the_same_pivots(seed):
    """Tighten random bounds from the root basis, as the tree does; every
    result -- optimal, infeasible, and the pivots to get there -- must match."""
    p = random_lp(seed=seed, m=30, n=70)
    p.kind[:] = 1
    p.col_lb[:] = 0.0
    p.col_ub[:] = 3.0
    root = solve_simplex(p.copy(), SimplexParams())
    if root.status != Status.OPTIMAL:
        pytest.skip("root not solvable")
    py, kn = _pair(p)
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(40):
        lo, hi = p.col_lb.copy(), p.col_ub.copy()
        for j in rng.choice(p.n, size=int(rng.integers(1, 6)), replace=False):
            v = float(rng.integers(0, 4))
            if rng.random() < 0.5:
                hi[j] = min(hi[j], v)
            else:
                lo[j] = max(lo[j], v)
        if (lo > hi).any():
            continue
        a = py.solve(lo, hi, warm_basis=root.basis_status)
        b = kn.solve(lo, hi, warm_basis=root.basis_status)
        assert _same(a, b), (a, b)
        if a.status == Status.INFEASIBLE:
            assert b.farkas is not None
            assert np.allclose(a.farkas, b.farkas, atol=1e-12)
        checked += 1
    assert checked > 20


@pytest.mark.parametrize("name", ["p0201", "gt2", "khb05250"])
def test_miplib_node_solves_match(name):
    path = os.path.join(DATA, f"{name}.mps")
    if not os.path.exists(path):
        pytest.skip("instance not fetched")
    p = read_mps(path)
    scaled, _ = scale_problem(p, method="pdlp")
    root = NodeSolver(scaled, SimplexParams()).solve(scaled.col_lb.copy(),
                                                    scaled.col_ub.copy())
    assert root.status == Status.OPTIMAL
    py, kn = _pair(scaled)
    rng = np.random.default_rng(0)
    ints = np.flatnonzero(scaled.integer_mask)
    for _ in range(30):
        lo, hi = scaled.col_lb.copy(), scaled.col_ub.copy()
        for j in rng.choice(ints, size=int(rng.integers(1, 8)), replace=False):
            v = float(np.floor(rng.uniform(lo[j], hi[j] + 1)))
            if rng.random() < 0.5:
                hi[j] = min(hi[j], v)
            else:
                lo[j] = max(lo[j], v)
        if (lo > hi).any():
            continue
        a = py.solve(lo, hi, warm_basis=root.basis)
        b = kn.solve(lo, hi, warm_basis=root.basis)
        assert _same(a, b), (a, b)


def test_the_basis_carried_out_of_the_kernel_is_the_basis():
    """The kernel hands back its LU and eta file; the Python-side basis must
    solve with them exactly as a fresh factorisation of the same basis would."""
    p = random_lp(seed=7, m=25, n=60)
    kn = NodeSolver(p, SimplexParams(node_kernel=True))
    r = kn.solve(p.col_lb.copy(), p.col_ub.copy())
    assert r.status == Status.OPTIMAL
    B = kn.S.B
    rng = np.random.default_rng(7)
    b = rng.standard_normal(B.m)
    x_carried = B.ftran(b.copy())
    B.factorize()
    x_fresh = B.ftran(b.copy())
    assert np.abs(x_carried - x_fresh).max() <= 1e-10 * (1.0 + np.abs(x_fresh).max())


# --------------------------------------------------------------------------- #
# the tree: same answer at any thread count                                    #
# --------------------------------------------------------------------------- #


def _knapsack_mip(seed=0, n=18):
    from sovopt.core.problem import Problem
    from sovopt.core.sparse import SparseMatrix
    from sovopt.core.tolerances import INF
    rng = np.random.default_rng(seed)
    w = rng.integers(5, 40, n).astype(float)
    v = rng.integers(5, 40, n).astype(float)
    A = SparseMatrix.from_dense(np.vstack([w, rng.integers(0, 3, n).astype(float)]))
    return Problem(A=A, c=-v, row_lb=np.array([-INF, -INF]),
                   row_ub=np.array([w.sum() / 2.5, 12.0]),
                   col_lb=np.zeros(n), col_ub=np.ones(n),
                   kind=np.ones(n, dtype=np.uint8), name="knap")


@pytest.mark.parametrize("seed", range(3))
def test_thread_count_does_not_change_the_search(seed):
    p = _knapsack_mip(seed)
    ref = solve_mip(p.copy(), MIPParams(time_limit=60, threads=1, symmetry=False))
    assert ref.status == Status.OPTIMAL
    for th in (2, 4):
        s = solve_mip(p.copy(), MIPParams(time_limit=60, threads=th, symmetry=False))
        assert s.status == Status.OPTIMAL
        assert abs(s.objective - ref.objective) <= 1e-9 * max(1.0, abs(ref.objective))
        assert s.nodes == ref.nodes, (th, s.nodes, ref.nodes)


def test_the_python_loop_is_still_there_and_agrees():
    p = _knapsack_mip(1)
    import sovopt.lp.simplex as sx
    d = list(sx.SimplexParams.__init__.__defaults__)
    names = list(sx.SimplexParams.__dataclass_fields__)
    k = names.index("node_kernel")
    saved = d[k]
    try:
        d[k] = False
        sx.SimplexParams.__init__.__defaults__ = tuple(d)
        a = solve_mip(p.copy(), MIPParams(time_limit=60, symmetry=False))
    finally:
        d[k] = saved
        sx.SimplexParams.__init__.__defaults__ = tuple(d)
    b = solve_mip(p.copy(), MIPParams(time_limit=60, symmetry=False))
    assert a.status == b.status == Status.OPTIMAL
    assert abs(a.objective - b.objective) <= 1e-9 * max(1.0, abs(a.objective))
