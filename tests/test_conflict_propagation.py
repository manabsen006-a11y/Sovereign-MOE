"""Conflict analysis on nodes that *propagation* refuted.

The LP-based analyser needs a dual ray, and a node propagation kills never
reaches the LP that would produce one. This is the other half: the required set
is found by deletion filtering -- relax a decision back to its root bound,
re-propagate, and keep the decision only if the node became feasible without
it.

Soundness is the whole game. A clause that excludes a feasible point turns a
solver into one that returns a confident wrong answer, and no feasibility check
downstream can catch it because the point it would have found is simply never
visited. So the central test here is brute force over every integer point of a
small model, the same standard the cuts are held to.
"""

import itertools

import numpy as np
import pytest

from sovopt.core.problem import Problem, Status
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.mip.conflict import ConflictAnalyzer
from sovopt.mip.propagate import propagate
from sovopt.mip.tree import MIPParams, solve_mip

from tests.test_conflict import all_feasible, binary_mip


def _propagation_conflicts(p, tries=120, seed=0):
    """Fix random subsets of binaries until propagation alone refutes them."""
    rng = np.random.default_rng(seed)
    is_int = p.integer_mask
    out = []
    for _ in range(tries):
        lo = p.col_lb.copy()
        hi = p.col_ub.copy()
        k = int(rng.integers(2, max(3, p.n // 2)))
        for j in rng.choice(p.n, size=k, replace=False):
            v = float(rng.integers(0, 2))
            lo[j] = hi[j] = v
        r = propagate(p.A, p.row_lb, p.row_ub, lo, hi, is_int,
                      max_rounds=2, feas_tol=1e-9)
        if r.infeasible:
            out.append((lo, hi))
    return out


def _analyzer(p):
    binary = ((p.col_lb >= -1e-9) & (p.col_ub <= 1.0 + 1e-9)
              & p.integer_mask)
    return ConflictAnalyzer(root_lo=p.col_lb.copy(), root_hi=p.col_ub.copy(),
                            binary=binary)


# --------------------------------------------------------------------------- #
# soundness                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_learned_clauses_never_exclude_a_feasible_point(seed):
    """The property everything else rests on, checked by brute force.

    A clause derived from a deletion filter is only valid if every decision it
    keeps is genuinely required. If the filter relaxes one decision too many --
    say by mistaking a propagation round limit for infeasibility -- the clause
    forbids an assignment that was actually fine, and the search silently loses
    the optimum.
    """
    p = binary_mip(seed=seed)
    feas = all_feasible(p)
    if not feas:
        pytest.skip("no feasible points to protect")

    ca = _analyzer(p)
    made = 0
    for lo, hi in _propagation_conflicts(p, seed=seed):
        cl = ca.analyse_propagation(p.A, p.row_lb, p.row_ub, lo, hi,
                                    p.integer_mask, max_rounds=2,
                                    feas_tol=1e-9)
        if cl is None:
            continue
        made += 1
        for x in feas:
            assert not ca.excludes(cl, x), (
                f"seed {seed}: a propagation clause excludes a feasible point\n"
                f"  clause {cl}\n  point  {x}")
    if made == 0:
        pytest.skip("this seed produced no propagation conflicts")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_kept_decision_is_actually_required(seed):
    """The filter must be irreducible: put a kept decision back and the node
    has to become feasible again.

    This is what separates a *reason* from the whole path. Returning the entire
    set of branching decisions is always sound and always useless -- it
    excludes only the node that was already being pruned.
    """
    p = binary_mip(seed=seed)
    ca = _analyzer(p)
    is_int = p.integer_mask
    checked = 0
    for lo, hi in _propagation_conflicts(p, seed=seed):
        cl = ca.analyse_propagation(p.A, p.row_lb, p.row_ub, lo, hi, is_int,
                                    max_rounds=2, feas_tol=1e-9)
        if cl is None:
            continue
        idx, _, _ = cl
        # rebuild the node from only the kept decisions; it must still die
        klo, khi = p.col_lb.copy(), p.col_ub.copy()
        for j in idx:
            klo[j], khi[j] = lo[j], hi[j]
        assert propagate(p.A, p.row_lb, p.row_ub, klo, khi, is_int,
                         max_rounds=2, feas_tol=1e-9).infeasible, (
            "the clause keeps a set that is not infeasible on its own")
        checked += 1
    if checked == 0:
        pytest.skip("this seed produced no propagation conflicts")


def test_a_feasible_node_is_never_analysed():
    """The caller says a node is dead; the analyser verifies it rather than
    believing it, because a clause built from a feasible node is unsound."""
    p = binary_mip(seed=0)
    ca = _analyzer(p)
    lo, hi = p.col_lb.copy(), p.col_ub.copy()
    lo[0] = hi[0] = 0.0                     # a decision, but not a refutation
    assert not propagate(p.A, p.row_lb, p.row_ub, lo, hi, p.integer_mask,
                         max_rounds=2, feas_tol=1e-9).infeasible
    assert ca.analyse_propagation(p.A, p.row_lb, p.row_ub, lo, hi,
                                  p.integer_mask) is None
    assert ca.certified == 0


def test_the_root_itself_is_never_analysed():
    p = binary_mip(seed=0)
    ca = _analyzer(p)
    assert ca.analyse_propagation(p.A, p.row_lb, p.row_ub, p.col_lb.copy(),
                                  p.col_ub.copy(), p.integer_mask) is None


def test_deep_nodes_are_skipped_rather_than_analysed_forever():
    """The filter costs one propagation per decision, so it needs a bound."""
    p = binary_mip(seed=1)
    ca = _analyzer(p)
    ca.max_propagation_vars = 1
    conflicts = _propagation_conflicts(p, seed=1)
    if not conflicts:
        pytest.skip("no propagation conflicts for this model")
    lo, hi = conflicts[0]
    assert ca.analyse_propagation(p.A, p.row_lb, p.row_ub, lo, hi,
                                  p.integer_mask) is None
    assert ca.prop_analysed == 0


# --------------------------------------------------------------------------- #
# integration                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_optimum_is_unchanged_with_it_on(seed):
    """Whatever it learns, it must not change the answer."""
    p = binary_mip(seed=seed)
    off = solve_mip(p.copy(), MIPParams(device="cpu", time_limit=60,
                                        propagation_conflicts=False))
    on = solve_mip(p.copy(), MIPParams(device="cpu", time_limit=60,
                                       propagation_conflicts=True))
    assert off.status == on.status
    if off.status == Status.OPTIMAL:
        assert abs(off.objective - on.objective) <= 1e-6 * max(1.0, abs(off.objective))


def test_the_switch_actually_switches():
    """Measured over the MIPLIB set the two settings are a wash -- 6/11 either
    way, 375.9 s against 379.6 s -- so the switch exists to make that
    measurable rather than to be tuned."""
    assert MIPParams().propagation_conflicts is True
    p = binary_mip(seed=0)
    off = solve_mip(p.copy(), MIPParams(device="cpu", time_limit=60,
                                        propagation_conflicts=False))
    cf = getattr(off, "info", {}).get("conflict", {}) or {}
    assert cf.get("propagation_analysed", 0) == 0
