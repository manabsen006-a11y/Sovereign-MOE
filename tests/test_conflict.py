"""Conflict analysis.

A learned clause is a globally valid constraint asserted on the basis of a
*local* infeasibility proof. If the reasoning is wrong the clause removes real
solutions and the solver returns a worse objective with complete confidence.
So the central test here enumerates every integer-feasible point of a small
model and checks that no learned clause rejects any of them.
"""

import itertools

import numpy as np
import pytest

from vyuha.core.problem import ObjSense, Problem, Status
from vyuha.core.sparse import SparseMatrix
from vyuha.core.tolerances import INF
from vyuha.lp.simplex import NodeSolver, SimplexParams
from vyuha.mip.conflict import ConflictAnalyzer, farkas_value
from vyuha.mip.tree import MIPParams, solve_mip


def binary_mip(seed=0, n=9, m=5):
    """A small 0/1 model whose feasible set can be enumerated exactly."""
    rng = np.random.default_rng(seed)
    D = np.round(rng.uniform(-3, 4, (m, n))) * (rng.random((m, n)) < 0.6)
    A = SparseMatrix.from_dense(D)
    b = np.abs(A.matvec(np.full(n, 0.5))) + rng.uniform(0.5, 3.0, m)
    return Problem(A=A, c=rng.standard_normal(n),
                   row_lb=np.full(m, -INF), row_ub=b,
                   col_lb=np.zeros(n), col_ub=np.ones(n),
                   kind=np.ones(n, dtype=np.uint8), name="binmip")


def all_feasible(p):
    pts = []
    for combo in itertools.product((0.0, 1.0), repeat=p.n):
        x = np.array(combo)
        rv, bv, _ = p.violation(x)
        if max(rv, bv) <= 1e-9:
            pts.append(x)
    return pts


def infeasible_nodes(p, tries=60, seed=0):
    """Fix random subsets of binaries until subproblems come out infeasible."""
    rng = np.random.default_rng(seed)
    ns = NodeSolver(p, SimplexParams())
    out = []
    for _ in range(tries):
        lo = p.col_lb.copy()
        hi = p.col_ub.copy()
        k = rng.integers(2, max(3, p.n // 2))
        for j in rng.choice(p.n, size=k, replace=False):
            v = float(rng.integers(0, 2))
            lo[j] = hi[j] = v
        r = ns.solve(lo, hi)
        if r.status == Status.INFEASIBLE and r.farkas is not None:
            out.append((lo, hi, r.farkas))
    return out


# --------------------------------------------------------------------------- #
# the certificate                                                              #
# --------------------------------------------------------------------------- #


def test_farkas_certificate_is_positive_exactly_when_infeasible():
    A = SparseMatrix.from_dense(np.array([[1.0], [1.0]]))
    infeas = Problem(A=A, c=np.array([1.0]),
                     row_lb=np.array([5.0, -INF]), row_ub=np.array([INF, 3.0]),
                     col_lb=np.array([0.0]), col_ub=np.array([10.0]),
                     name="infeas")
    ns = NodeSolver(infeas, SimplexParams())
    r = ns.solve(infeas.col_lb, infeas.col_ub)
    assert r.status == Status.INFEASIBLE
    assert r.farkas is not None
    best = max(farkas_value(A, infeas.row_lb, infeas.row_ub,
                            infeas.col_lb, infeas.col_ub, s * r.farkas)
               for s in (1.0, -1.0))
    assert best > 0.0, "dual ray does not certify the infeasibility"


def test_certificate_is_not_positive_on_a_feasible_model():
    """No vector may certify emptiness of a non-empty set."""
    p = binary_mip(seed=1)
    assert all_feasible(p), "test model has no feasible points"
    rng = np.random.default_rng(0)
    for _ in range(200):
        y = rng.standard_normal(p.m)
        v = farkas_value(p.A, p.row_lb, p.row_ub, p.col_lb, p.col_ub, y)
        assert v <= 1e-7, f"certified a feasible model as empty: L={v}"


# --------------------------------------------------------------------------- #
# validity of what is learned                                                  #
# --------------------------------------------------------------------------- #


def test_learned_clauses_never_exclude_a_feasible_point():
    """The property everything else rests on, checked by brute force."""
    checked_models = 0
    checked_clauses = 0

    for seed in range(12):
        p = binary_mip(seed=seed)
        feasible = all_feasible(p)
        if len(feasible) < 2:
            continue
        nodes = infeasible_nodes(p, seed=seed)
        if not nodes:
            continue

        ca = ConflictAnalyzer(root_lo=p.col_lb.copy(), root_hi=p.col_ub.copy(),
                              binary=p.integer_mask.copy())
        got = []
        for lo, hi, ray in nodes:
            cl = ca.analyse(p.A, p.row_lb, p.row_ub, lo, hi, ray)
            if cl is not None:
                got.append(cl)
        if not got:
            continue

        checked_models += 1
        for idx, val, rhs in got:
            checked_clauses += 1
            for x in feasible:
                lhs = float(val @ x[idx])
                assert lhs >= rhs - 1e-6, (
                    f"seed {seed}: learned clause excludes a feasible point\n"
                    f"  idx={idx} val={val} rhs={rhs} x={x}")

    assert checked_models >= 3, (
        f"only {checked_models} models produced clauses; the validity sweep is "
        f"not exercising anything")
    assert checked_clauses >= 5, f"only {checked_clauses} clauses checked"


def test_reason_is_a_subset_that_still_certifies():
    """Minimisation must not weaken the proof it is shrinking."""
    # seed 6 yields 9 infeasible nodes; seed 3, used here originally, yields
    # none at all, so this test skipped on every run and checked nothing.
    p = binary_mip(seed=6)
    nodes = infeasible_nodes(p, seed=6)
    assert nodes, "fixture produced no infeasible nodes; nothing to minimise"

    ca = ConflictAnalyzer(root_lo=p.col_lb.copy(), root_hi=p.col_ub.copy(),
                          binary=p.integer_mask.copy())
    found = 0
    for lo, hi, ray in nodes:
        cl = ca.analyse(p.A, p.row_lb, p.row_ub, lo, hi, ray)
        if cl is None:
            continue
        found += 1
        idx, val, rhs = cl
        # rebuild the bounds named by the clause; that subproblem must be empty
        lo2 = p.col_lb.copy()
        hi2 = p.col_ub.copy()
        for j, v in zip(idx, val):
            if v < 0:
                lo2[j] = hi2[j] = 1.0     # was branched up
            else:
                lo2[j] = hi2[j] = 0.0     # was branched down
        ns = NodeSolver(p, SimplexParams())
        assert ns.solve(lo2, hi2).status == Status.INFEASIBLE, (
            "the minimised reason does not actually imply infeasibility")
    if found == 0:
        pytest.skip("no clauses emitted for this draw")


def test_clauses_only_mention_binaries():
    """A general integer's conflict is a disjunction, not a linear clause."""
    rng = np.random.default_rng(5)
    n, m = 8, 4
    A = SparseMatrix.from_dense(
        np.round(rng.uniform(-2, 3, (m, n))) * (rng.random((m, n)) < 0.6))
    b = np.abs(A.matvec(np.full(n, 1.0))) + 1.0
    kind = np.ones(n, dtype=np.uint8)
    ub = np.full(n, 3.0)                  # general integers, not binaries
    p = Problem(A=A, c=rng.standard_normal(n), row_lb=np.full(m, -INF),
                row_ub=b, col_lb=np.zeros(n), col_ub=ub, kind=kind,
                name="genint")
    binary = p.integer_mask & (p.col_lb >= -1e-9) & (p.col_ub <= 1 + 1e-9)
    assert not binary.any()

    ca = ConflictAnalyzer(root_lo=p.col_lb.copy(), root_hi=p.col_ub.copy(),
                          binary=binary)
    for lo, hi, ray in infeasible_nodes(p, seed=5):
        assert ca.analyse(p.A, p.row_lb, p.row_ub, lo, hi, ray) is None


# --------------------------------------------------------------------------- #
# end to end                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 2, 4])
def test_conflict_preserves_the_optimum(seed):
    p = binary_mip(seed=seed, n=10, m=6)
    off = solve_mip(p, MIPParams(device="cpu", time_limit=60, conflict=False))
    on = solve_mip(p, MIPParams(device="cpu", time_limit=60, conflict=True))
    if off.status != Status.OPTIMAL or on.status != Status.OPTIMAL:
        pytest.skip("not both solved to optimality")
    assert abs(off.objective - on.objective) < 1e-6 * max(1.0, abs(off.objective))


def test_analyzer_reports_its_work():
    p = binary_mip(seed=7)
    ca = ConflictAnalyzer(root_lo=p.col_lb.copy(), root_hi=p.col_ub.copy(),
                          binary=p.integer_mask.copy())
    for lo, hi, ray in infeasible_nodes(p, seed=7):
        ca.analyse(p.A, p.row_lb, p.row_ub, lo, hi, ray)
    st = ca.stats()
    assert st["analysed"] >= st["certified"] >= st["clauses"]
