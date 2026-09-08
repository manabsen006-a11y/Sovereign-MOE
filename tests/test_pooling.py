"""Global bilinear pooling.

The claim being tested is *global* optimality, which is stronger than anything
else in this project asserts. So the optimum is checked three ways: against an
independent brute-force computation, against the published literature values,
and against the solver's own dual bound. A "global" optimum that is only a local
one is the failure this whole module exists to prevent.
"""

import numpy as np
import pytest

from sovopt.core.problem import ObjSense, Problem, Status
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.globalopt.bilinear import (BilinearProblem, BilinearTerm,
                                      build_relaxation, mccormick_rows)
from sovopt.globalopt.spatial import SpatialParams, solve_global
from sovopt.lp.simplex import SimplexParams, solve_simplex
from sovopt.models import haverly

P_INDEX = 6

#: Published global optima (Haverly 1978; Adhya et al. 1999).
PUBLISHED = {1: 400.0, 2: 600.0, 3: 750.0}


def brute_force_optimum(bp, steps=4001):
    """True global optimum by scanning the single complicating variable.

    With one pool there is one pool quality. Fix it and the rest is a linear
    program, so sweeping it finely and solving an LP at each point computes the
    global optimum without using the spatial search at all.
    """
    lo = bp.linear.col_lb.copy()
    hi = bp.linear.col_hi.copy() if hasattr(bp.linear, "col_hi") \
        else bp.linear.col_ub.copy()
    best = -np.inf
    for pv in np.linspace(lo[P_INDEX], hi[P_INDEX], steps):
        l, h = lo.copy(), hi.copy()
        l[P_INDEX] = h[P_INDEX] = pv
        s = solve_simplex(build_relaxation(bp, l, h), SimplexParams())
        if s.status == Status.OPTIMAL and s.objective > best:
            best = s.objective
    return best


# --------------------------------------------------------------------------- #
# McCormick envelopes                                                          #
# --------------------------------------------------------------------------- #


def test_envelopes_contain_every_true_product():
    """No point of ``w = x*y`` inside the box may violate an envelope."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        xl, xu = np.sort(rng.uniform(-4, 4, 2))
        yl, yu = np.sort(rng.uniform(-4, 4, 2))
        term = BilinearTerm(w=2, x=0, y=1)
        lo = np.array([xl, yl, -1e30])
        hi = np.array([xu, yu, 1e30])
        rows = mccormick_rows([term], lo, hi)
        assert len(rows) == 4

        for _ in range(60):
            x = rng.uniform(xl, xu)
            y = rng.uniform(yl, yu)
            vec = np.array([x, y, x * y])
            for idx, val, lb, ub in rows:
                a = float(val @ vec[idx])
                assert a >= lb - 1e-9 and a <= ub + 1e-9, (
                    "envelope cuts off a genuine product")


def test_envelopes_are_exact_when_a_factor_is_fixed():
    """Collapsing a range to a point makes the relaxation exact.

    This is what makes the recursion heuristic sound: fixing one factor turns
    the relaxation into the true problem, so its answer is feasible by
    construction rather than by luck.
    """
    term = BilinearTerm(w=2, x=0, y=1)
    lo = np.array([2.0, 0.0, -1e30])
    hi = np.array([2.0, 5.0, 1e30])
    rows = mccormick_rows([term], lo, hi)
    rng = np.random.default_rng(1)
    for _ in range(50):
        y = rng.uniform(0.0, 5.0)
        w = rng.uniform(-20.0, 20.0)
        vec = np.array([2.0, y, w])
        ok = all(lb - 1e-9 <= float(val @ vec[idx]) <= ub + 1e-9
                 for idx, val, lb, ub in rows)
        assert ok == (abs(w - 2.0 * y) < 1e-9), (
            "with x fixed the envelopes must admit exactly w = x*y")


def test_relaxation_bound_never_cuts_off_the_optimum():
    """The relaxation must over-estimate a maximisation, never under."""
    for v in (1, 2, 3):
        bp = haverly(v)
        rel = solve_simplex(build_relaxation(bp, bp.linear.col_lb,
                                             bp.linear.col_ub),
                            SimplexParams())
        assert rel.status == Status.OPTIMAL
        assert rel.objective >= PUBLISHED[v] - 1e-6, (
            f"haverly{v}: relaxation bound {rel.objective} is below the known "
            f"optimum {PUBLISHED[v]}")


# --------------------------------------------------------------------------- #
# global optimality                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_matches_published_global_optimum(variant):
    bp = haverly(variant)
    s = solve_global(bp, SpatialParams(time_limit=120))
    assert s.status == Status.OPTIMAL, s.status
    assert abs(s.objective - PUBLISHED[variant]) < 1e-4, (
        f"got {s.objective}, published {PUBLISHED[variant]}")


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_matches_independent_brute_force(variant):
    """Checked against a computation that does not use the spatial search."""
    bp = haverly(variant)
    s = solve_global(bp, SpatialParams(time_limit=120))
    assert s.status == Status.OPTIMAL
    bf = brute_force_optimum(bp, steps=2001)
    assert abs(s.objective - bf) < 1e-3, f"B&B {s.objective}, scan {bf}"


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_bound_proves_optimality(variant):
    """Objective and dual bound must meet: that is what 'proven' means."""
    bp = haverly(variant)
    s = solve_global(bp, SpatialParams(time_limit=120))
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - s.dual_bound) <= 1e-5 * max(1.0, abs(s.objective))


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_reported_solution_is_actually_feasible(variant):
    """Linear rows, bounds, and every bilinear identity."""
    bp = haverly(variant)
    s = solve_global(bp, SpatialParams(time_limit=120))
    assert s.x is not None
    lin, bil = bp.feasible(s.x)
    assert lin < 1e-6, f"linear violation {lin}"
    assert bil < 1e-6, f"bilinear violation {bil}"


# --------------------------------------------------------------------------- #
# why this exists                                                              #
# --------------------------------------------------------------------------- #


def test_recursion_gets_stuck_where_global_search_does_not():
    """Distributive recursion -- the industry workaround -- has no guarantee.

    Haverly's original point. Recursion is a fixed-point iteration on the pool
    qualities; from most starting guesses it converges to a blend worth less
    than the global optimum, and cannot tell that it has.
    """
    bp = haverly(1)
    lo, hi = bp.linear.col_lb.copy(), bp.linear.col_ub.copy()

    def recursion(p0, iters=200):
        p = float(np.clip(p0, lo[P_INDEX], hi[P_INDEX]))
        obj = None
        for _ in range(iters):
            l, h = lo.copy(), hi.copy()
            l[P_INDEX] = h[P_INDEX] = p
            s = solve_simplex(build_relaxation(bp, l, h), SimplexParams())
            if s.status != Status.OPTIMAL or s.x is None:
                return obj
            obj = s.objective
            flow = s.x[2] + s.x[3]
            if flow <= 1e-9:
                return obj
            new_p = float(np.clip((3.0 * s.x[0] + 1.0 * s.x[1]) / flow,
                                  lo[P_INDEX], hi[P_INDEX]))
            if abs(new_p - p) < 1e-10:
                return obj
            p = new_p
        return obj

    outcomes = [recursion(p0) for p0 in np.linspace(1.0, 3.0, 21)]
    outcomes = [o for o in outcomes if o is not None]
    assert outcomes

    glob = PUBLISHED[1]
    stuck = sum(1 for o in outcomes if o < glob - 1e-6)
    assert stuck >= len(outcomes) // 2, (
        "recursion was expected to get stuck from most starting points; if this "
        "fails the demonstration is no longer showing what it claims")

    s = solve_global(bp, SpatialParams(time_limit=120))
    assert s.objective >= max(outcomes) - 1e-6, (
        "global search must be at least as good as the best recursion run")


def test_bilinear_violation_is_reported():
    bp = haverly(2)
    s = solve_global(bp, SpatialParams(time_limit=120))
    assert s.info["terms"] == 2
    assert s.info["max_bilinear_violation"] < 1e-6
