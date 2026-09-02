"""Cut validity and strength.

The first test here is the one that matters. A cut that removes an
integer-feasible point is worse than no cut: the solver returns a confidently
wrong answer, and nothing downstream can detect it. Every other property --
strength, efficacy, orthogonality -- is negotiable. Validity is not.
"""

import itertools

import numpy as np
import pytest

from sovopt.core.problem import ObjSense, Problem, Status, VarKind
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import DEFAULT, INF
from sovopt.lp.simplex import NodeSolver, SimplexParams
from sovopt.mip.cuts import (Cut, CutPool, append_cuts, generate_cover,
                            generate_gomory)
from sovopt.numerics.scaling import scale_problem


def small_mip(seed=0, n=8, m=5, ub=3):
    """A small bounded integer program whose feasible set can be enumerated."""
    rng = np.random.default_rng(seed)
    A = SparseMatrix.from_dense(
        np.round(rng.uniform(-3, 4, (m, n))) * (rng.random((m, n)) < 0.6))
    b = np.abs(A.matvec(np.full(n, ub / 2.0))) + rng.uniform(1, 6, m)
    return Problem(A=A, c=rng.standard_normal(n),
                   row_lb=np.full(m, -INF), row_ub=b,
                   col_lb=np.zeros(n), col_ub=np.full(n, float(ub)),
                   kind=np.ones(n, dtype=np.uint8), name="small")


def small_knapsack(seed=0, n=9, m=3):
    """Binary rows with positive coefficients -- where cover cuts apply."""
    rng = np.random.default_rng(1000 + seed)
    A = SparseMatrix.from_dense(rng.integers(1, 9, (m, n)).astype(float))
    b = A.matvec(np.ones(n)) * rng.uniform(0.35, 0.6, m)
    return Problem(A=A, c=-rng.uniform(1, 10, n),
                   row_lb=np.full(m, -INF), row_ub=b,
                   col_lb=np.zeros(n), col_ub=np.ones(n),
                   kind=np.ones(n, dtype=np.uint8), name="knap")


def enumerate_feasible(p, ub=3):
    """Every integer-feasible point, by brute force. Only for tiny models."""
    pts = []
    for combo in itertools.product(range(ub + 1), repeat=p.n):
        x = np.array(combo, dtype=float)
        rv, bv, _ = p.violation(x)
        if max(rv, bv) <= 1e-9:
            pts.append(x)
    return pts


def _root_cuts(p, params_rounds=1):
    """Solve the root LP and separate one round of cuts, in the scaled space."""
    scaled, sc = scale_problem(p, method="pdlp")
    node = NodeSolver(scaled, SimplexParams())
    r = node.solve(scaled.col_lb, scaled.col_ub)
    assert r.status == Status.OPTIMAL, r.status
    int_mask = scaled.integer_mask
    int_full = np.concatenate([int_mask, np.zeros(scaled.m, dtype=bool)])
    cuts = generate_gomory(node.S.B, node.S.zB, int_full, max_cuts=40)
    cuts += generate_cover(scaled, r.x, int_mask,
                           scaled.col_lb, scaled.col_ub, max_cuts=40)
    return scaled, sc, node, r.x, cuts


# --------------------------------------------------------------------------- #
# validity -- the property everything else depends on                          #
# --------------------------------------------------------------------------- #


def test_cuts_never_remove_an_integer_feasible_point():
    """Brute-force every feasible point and check it satisfies every cut.

    Swept over many random instances rather than parametrised, so the test
    cannot pass by quietly skipping: it asserts a minimum number of instances
    actually produced cuts, then checks every cut against every feasible point.

    Integer columns are pinned to a scale factor of 1, so a feasible point of
    the original model is the same vector in the scaled space and can be tested
    against the cuts directly.
    """
    checked_instances = 0
    checked_cuts = 0
    by_kind: dict[str, int] = {}

    draws = ([(small_mip(seed=s, n=7, m=4, ub=2), 2) for s in range(80)]
             + [(small_knapsack(seed=s, n=9, m=3), 1) for s in range(40)])

    for p, ub in draws:
        try:
            scaled, sc, node, x_lp, cuts = _root_cuts(p)
        except AssertionError:
            continue                       # infeasible or unbounded draw
        if not cuts:
            continue
        feasible = enumerate_feasible(p, ub=ub)
        if not feasible:
            continue
        seed = p.name

        checked_instances += 1
        for c in cuts:
            checked_cuts += 1
            by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
            for xf in feasible:
                lhs = float(c.val @ xf[c.idx])
                assert lhs >= c.rhs - 1e-6, (
                    f"seed {seed}: {c.kind} cut removes a feasible point: "
                    f"lhs={lhs:.10g} < rhs={c.rhs:.10g} at x={xf}")

    assert checked_instances >= 10, (
        f"only {checked_instances} instances produced cuts; the validity sweep "
        f"is not exercising anything")
    assert checked_cuts >= 40, f"only {checked_cuts} cuts checked"
    assert "gmi" in by_kind, f"no GMI cuts exercised, only {by_kind}"
    assert "cover" in by_kind, f"no cover cuts exercised, only {by_kind}"


@pytest.mark.parametrize("seed", [0, 2, 7])
def test_cuts_are_violated_by_the_relaxation_point(seed):
    """A cut that does not cut is wasted work.

    Seeds chosen because they actually separate cuts (1, 2 and 3 of them).
    Seed 1, previously in this list, separates none, so a third of this test
    skipped on every run.
    """
    p = small_mip(seed=seed, n=8, m=5, ub=3)
    scaled, sc, node, x_lp, cuts = _root_cuts(p)
    assert cuts, "fixture separated no cuts; nothing to check"
    for c in cuts:
        assert c.violation(x_lp) > 1e-9, f"{c.kind} cut is not violated"


def test_gomory_cut_matches_hand_derivation():
    """A one-row model where the GMI cut is known in closed form.

    max x  s.t.  2x <= 3, x integer >= 0. The relaxation gives x = 1.5 and the
    only valid cut of the form ax >= b that removes it is x <= 1.
    """
    A = SparseMatrix.from_dense(np.array([[2.0]]))
    p = Problem(A=A, c=np.array([1.0]), row_lb=np.array([-INF]),
                row_ub=np.array([3.0]), col_lb=np.array([0.0]),
                col_ub=np.array([5.0]), kind=np.array([1], dtype=np.uint8),
                sense=ObjSense.MAXIMISE, name="onerow")
    work = p.copy()
    work.c = -work.c
    work.sense = ObjSense.MINIMISE

    scaled, sc, node, x_lp, cuts = _root_cuts(work)
    assert abs(x_lp[0] - 1.5) < 1e-9, x_lp
    assert cuts, "expected at least one cut"
    for c in cuts:
        # every cut must admit x=0 and x=1 and exclude x=1.5
        for good in (0.0, 1.0):
            assert float(c.val @ np.array([good])[c.idx]) >= c.rhs - 1e-9
        assert c.violation(x_lp) > 1e-9


def test_append_cuts_preserves_the_original_rows():
    p = small_mip(seed=9, n=6, m=4, ub=2)
    cuts = [Cut(np.array([0, 2], dtype=np.int32), np.array([1.0, -1.0]), -1.0)]
    q = append_cuts(p, cuts)
    assert q.m == p.m + 1
    assert q.n == p.n
    assert np.allclose(q.A.to_dense()[: p.m], p.A.to_dense())
    assert q.row_lb[-1] == -1.0 and q.row_ub[-1] >= INF


# --------------------------------------------------------------------------- #
# selection                                                                    #
# --------------------------------------------------------------------------- #


def test_pool_rejects_ill_conditioned_and_parallel_cuts():
    n = 5
    x = np.zeros(n)
    good = Cut(np.array([0, 1], dtype=np.int32), np.array([1.0, 1.0]), 1.0)
    parallel = Cut(np.array([0, 1], dtype=np.int32), np.array([2.0, 2.0]), 2.0)
    wild = Cut(np.array([0, 1], dtype=np.int32), np.array([1.0, 1e12]), 1.0)

    pool = CutPool(min_efficacy=1e-6, max_dynamism=1e8, min_orthogonality=0.1)
    chosen = pool.select([good, parallel, wild], x, n)
    kinds = [id(c) for c in chosen]
    assert id(good) in kinds or id(parallel) in kinds
    assert len(chosen) == 1, "near-parallel cuts should not both be kept"
    assert id(wild) not in kinds, "dynamism filter should reject 1e12 spread"


def test_pool_rejects_unviolated_cuts():
    n = 3
    x = np.array([1.0, 1.0, 0.0])
    satisfied = Cut(np.array([0], dtype=np.int32), np.array([1.0]), 0.5)
    pool = CutPool(min_efficacy=1e-4)
    assert pool.select([satisfied], x, n) == []


# --------------------------------------------------------------------------- #
# end to end                                                                   #
# --------------------------------------------------------------------------- #


def test_cuts_improve_the_root_bound_and_keep_the_optimum():
    """Cuts must raise the bound without changing the answer."""
    from sovopt.mip.tree import MIPParams, solve_mip

    p = small_mip(seed=11, n=9, m=6, ub=3)
    without = solve_mip(p, MIPParams(device="cpu", time_limit=60, cut_rounds=0))
    with_cuts = solve_mip(p, MIPParams(device="cpu", time_limit=60,
                                       cut_rounds=10))
    assert without.status == Status.OPTIMAL
    assert with_cuts.status == Status.OPTIMAL
    assert abs(without.objective - with_cuts.objective) < 1e-6
    assert with_cuts.info.get("root_cut_bound_gain", 0.0) >= -1e-9


# --------------------------------------------------------------------------- #
# mixed-integer rounding                                                       #
# --------------------------------------------------------------------------- #


def test_mir_cuts_never_remove_an_integer_feasible_point():
    """Same brute-force standard the other cut families are held to.

    MIR is separated from the model's own rows rather than the tableau, so it
    has its own shift-and-complement logic and its own way to be wrong: a row
    containing a free variable has no finite bound to complement against, and a
    cut derived without one is not valid. Those rows are skipped, and this sweep
    is what proves it.
    """
    from sovopt.mip.cuts import generate_mir

    checked_instances = 0
    checked_cuts = 0
    for seed in range(40):
        for gen, ub in ((lambda s: small_mip(seed=s, n=7, m=4, ub=2), 2),
                        (lambda s: small_knapsack(seed=s, n=9, m=3), 1)):
            p = gen(seed)
            scaled, sc = scale_problem(p, method="pdlp")
            node = NodeSolver(scaled, SimplexParams())
            r = node.solve(scaled.col_lb, scaled.col_ub)
            if r.status != Status.OPTIMAL:
                continue
            cuts = generate_mir(scaled, r.x, scaled.integer_mask,
                                scaled.col_lb, scaled.col_ub, max_cuts=20)
            if not cuts:
                continue
            feasible = enumerate_feasible(p, ub=ub)
            if not feasible:
                continue
            checked_instances += 1
            for c in cuts:
                checked_cuts += 1
                for xf in feasible:
                    lhs = float(c.val @ xf[c.idx])
                    assert lhs >= c.rhs - 1e-6, (
                        f"seed {seed}: MIR cut removes a feasible point: "
                        f"lhs={lhs:.10g} < rhs={c.rhs:.10g}")

    assert checked_instances >= 15, (
        f"only {checked_instances} instances produced MIR cuts")
    assert checked_cuts >= 40, f"only {checked_cuts} MIR cuts checked"


def test_mir_skips_rows_with_a_free_variable():
    """No finite bound to complement against means no valid shift."""
    from sovopt.mip.cuts import generate_mir

    A = SparseMatrix.from_dense(np.array([[1.0, 1.0]]))
    p = Problem(A=A, c=np.array([1.0, 1.0]), row_lb=np.array([-INF]),
                row_ub=np.array([2.5]),
                col_lb=np.array([-INF, 0.0]), col_ub=np.array([INF, 5.0]),
                kind=np.ones(2, dtype=np.uint8), name="freevar")
    assert generate_mir(p, np.array([0.5, 0.5]), p.integer_mask,
                        p.col_lb, p.col_ub) == []
