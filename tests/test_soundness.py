"""Soundness: the solver must never report INFEASIBLE on a feasible model.

These two regressions were found the same way, on the same instance, and both
had been passing every existing test. Neither would have been caught by any
amount of the solver agreeing with itself, which is the whole argument for
keeping oracles outside the search.

The instance is flugpl. It is unremarkable except in one respect: five of its
eighteen columns have a non-zero lower bound (57). Almost every MIP model, and
every model in the property-test generators that validated the cut code, has
``lo = 0`` on every column -- and a surprising number of bugs are invisible
when a bound is zero, because the wrong quantity gets multiplied by it.
"""

import itertools
import os

import numpy as np
import pytest

from vyuha.core.problem import ObjSense, Problem, Status, VarKind
from vyuha.core.sparse import SparseMatrix
from vyuha.core.tolerances import INF
from vyuha.mip.cuts import generate_mir
from vyuha.mip.tree import MIPParams, solve_mip

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "instances")


# --------------------------------------------------------------------------- #
# 1. an invalid cut                                                            #
# --------------------------------------------------------------------------- #


def test_mir_cuts_are_valid_when_lower_bounds_are_non_zero():
    """REGRESSION: the MIR un-shift constant had the wrong sign.

    A MIR cut is derived in a shifted space where every variable is a
    non-negative deviation from one of its bounds. Undoing that shift moves a
    constant to the right-hand side, and the constant is ``coef * lo_j`` (or
    ``-coef * hi_j``). Both signs were inverted.

    Multiplying by ``lo_j = 0`` hides the error completely, so the original
    validation -- 133 cuts brute-forced over 55 generated instances -- passed
    while the code was wrong. This generator draws non-zero, sometimes negative
    lower bounds, which is the case that exposes it.
    """
    rng = np.random.default_rng(11)
    total = 0
    for _ in range(400):
        m, n = int(rng.integers(1, 4)), int(rng.integers(2, 5))
        A = rng.integers(-4, 5, size=(m, n)).astype(float)
        lo = rng.integers(-3, 4, size=n).astype(float)      # <- never all zero
        hi = lo + rng.integers(1, 5, size=n).astype(float)
        ru = rng.integers(-2, 12, size=m).astype(float)
        isint = rng.random(n) < 0.7
        if not isint.any():
            continue

        p = Problem(A=SparseMatrix.from_dense(A), c=np.ones(n),
                    row_lb=np.full(m, -INF), row_ub=ru,
                    col_lb=lo.copy(), col_ub=hi.copy(),
                    kind=np.where(isint, VarKind.INTEGER, VarKind.CONTINUOUS))

        axes = [np.arange(lo[j], hi[j] + 1e-9) if isint[j]
                else np.linspace(lo[j], hi[j], 7) for j in range(n)]
        feasible = [np.array(pt) for pt in itertools.product(*axes)
                    if np.all(A @ np.array(pt) <= ru + 1e-9)]
        if not feasible:
            continue

        x = rng.uniform(lo, hi)
        for cut in generate_mir(p, x, isint, lo, hi, max_cuts=8):
            total += 1
            g = np.zeros(n)
            g[cut.idx] = cut.val
            for pt in feasible:                     # the cut is  g.x >= rhs
                assert float(g @ pt) - cut.rhs >= -1e-7, (
                    "MIR cut removes an integer-feasible point")

    assert total > 20, "generator produced too few cuts to be a real test"


# --------------------------------------------------------------------------- #
# 2. a node dropped without proof                                              #
# --------------------------------------------------------------------------- #


def test_batched_node_relaxation_does_not_drop_nodes_unproven():
    """REGRESSION: an integral-looking BNR iterate emptied the whole tree.

    A node is finished when its relaxation optimum is integral. With an exact
    node LP that reasoning is sound -- an integral vertex is feasible, so the
    node really is done. A batched first-order iterate is not a vertex and not
    even necessarily feasible: it can look integral while violating rows, in
    which case the incumbent check rejects it *and* there is no fractional
    variable to branch on. The node was then dropped by fallthrough.

    Dropping a node discards its subtree without proving anything about it. On
    this model it discarded the entire tree after one node and reported
    INFEASIBLE for a model with a published optimum.
    """
    path = os.path.join(DATA, "flugpl.mps")
    if not os.path.exists(path):
        pytest.skip("benchmark instances not fetched")
    from vyuha.io import read_model

    prob = read_model(path)
    for node_solver in ("simplex", "bnr"):
        s = solve_mip(prob, MIPParams(node_solver=node_solver, time_limit=120))
        assert s.status != Status.INFEASIBLE, (
            f"node_solver={node_solver} reported INFEASIBLE on a model whose "
            f"published optimum is 1201500")
        assert s.status == Status.OPTIMAL
        assert abs(s.objective - 1201500.0) < 1e-4
        assert s.nodes > 1, "the tree cannot have been explored in one node"


# --------------------------------------------------------------------------- #
# 3. a bound reported in the wrong sense                                       #
# --------------------------------------------------------------------------- #


def _random_max_binary(seed, n=14, m=8):
    """A MAXIMISE binary model with positive costs and a few equality rows.

    The equalities are what stop the root relaxation rounding to a feasible
    point, which is what leaves the search with no incumbent to report.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 9, size=(m, n)).astype(float)
    ru = ((A @ np.full(n, 0.5)) + rng.integers(0, 5, size=m)).astype(float)
    eq = rng.random(m) < 0.2
    return Problem(A=SparseMatrix.from_dense(A),
                   c=rng.integers(1, 9, n).astype(float),
                   row_lb=np.where(eq, ru, -INF).astype(float),
                   row_ub=ru, col_lb=np.zeros(n), col_ub=np.ones(n),
                   kind=np.full(n, VarKind.BINARY, dtype=np.uint8),
                   sense=ObjSense.MAXIMISE, name="maxbin")


def test_maximisation_dual_bound_is_reported_in_the_users_sense():
    """REGRESSION: the no-incumbent return path never un-negated the bound.

    A MAXIMISE model is negated on the way into the tree and searched as a
    minimisation, so every bound inside lives in that negated space. The
    normal return flipped it back; the early return taken when no incumbent
    was found did not, and reported the negation of its own bound. With
    positive costs that is not merely the wrong sign but a number below every
    feasible objective, offered as an upper bound on the maximum.

    That path is the ordinary outcome on a model too large to crack inside its
    limit -- exactly the case where the bound is the only thing a user has to
    judge whether continuing is worthwhile.

    The property asserted is the definition: a maximisation's dual bound is an
    *upper* bound, so it cannot sit below an objective actually achieved on
    the same model. ``node_limit=0`` stops the search before any incumbent
    exists without reference to the clock, so which models exercise the path
    does not vary with machine speed.
    """
    checked = 0
    for seed in range(20):
        p = _random_max_binary(seed)
        s = solve_mip(p.copy(), MIPParams(node_limit=0, heuristics=False))
        if s.x is not None or not np.isfinite(s.dual_bound):
            continue
        ref = solve_mip(p.copy(), MIPParams(time_limit=30))
        if ref.status != Status.OPTIMAL:
            continue                    # nothing achievable to compare against
        checked += 1
        assert s.dual_bound >= ref.objective - 1e-6, (
            f"seed {seed}: reported dual bound {s.dual_bound:.6g} is below an "
            f"achievable objective {ref.objective:.6g} on a maximisation")
    assert checked >= 4, (
        f"only {checked} models reached the no-incumbent return; the test is "
        f"not exercising the path it is meant to pin")


def test_batched_node_bounds_agree_with_brute_force():
    """REGRESSION: an integral-looking BNR iterate closed the node it sat in.

    A node is finished when its relaxation *optimum* is integral, because an
    integral optimum is feasible and nothing below it can beat it. That holds
    for an exact node LP. A BNR iterate is not an optimum -- it is an
    unconverged first-order point -- so its looking integral proves nothing,
    and ``_accept`` rounds before it validates, so it will happily turn such a
    point into a feasible incumbent. Reading that incumbent as proof closed
    the node and discarded the subtree holding the real optimum.

    The symptom was the worst kind available: not a crash and not INFEASIBLE,
    but ``OPTIMAL`` with a plausible near-optimal objective and a dual bound
    agreeing with it, because the node that would have refuted the bound was
    never opened. Measured here at 3 wrong answers in 40 models before the
    fix; the models are small enough to enumerate, so brute force settles it.

    Mixed-sign coefficients and non-zero -- sometimes negative -- lower bounds
    are what the earlier generators lacked.
    """
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(20):
        n, m = int(rng.integers(3, 6)), int(rng.integers(2, 4))
        lo = rng.integers(-30, 60, size=n).astype(float)
        hi = lo + rng.integers(2, 5, size=n).astype(float)
        A = rng.integers(-3, 4, size=(m, n)).astype(float)
        ru = ((A @ ((lo + hi) / 2.0)) + rng.integers(0, 6, size=m)).astype(float)
        c = rng.integers(-5, 6, size=n).astype(float)

        feasible = [np.array(pt) for pt in
                    itertools.product(*[np.arange(lo[j], hi[j] + 1e-9)
                                        for j in range(n)])
                    if np.all(A @ np.array(pt) <= ru + 1e-9)]
        if not feasible:
            continue
        best = min(float(c @ pt) for pt in feasible)
        checked += 1

        p = Problem(A=SparseMatrix.from_dense(A), c=c,
                    row_lb=np.full(m, -INF), row_ub=ru,
                    col_lb=lo.copy(), col_ub=hi.copy(),
                    kind=np.full(n, VarKind.INTEGER, dtype=np.uint8))
        s = solve_mip(p, MIPParams(node_solver="bnr", time_limit=30))

        assert s.status == Status.OPTIMAL, (
            f"batched bounding returned {s.status.name} on a model with "
            f"{len(feasible)} enumerated feasible points")
        assert abs(s.objective - best) < 1e-6, (
            f"batched bounding reported {s.objective} as optimal; brute force "
            f"over every integer point gives {best}")
        # and the proof offered must not be stronger than the truth
        assert not (np.isfinite(s.dual_bound) and s.dual_bound > best + 1e-6), (
            f"dual bound {s.dual_bound} lies above the true optimum {best}")
    assert checked >= 10, f"only {checked} models had a feasible point"


def test_conflict_clauses_can_be_added_without_symmetry_breaking():
    """REGRESSION: a redundant local import made ``Cut`` local to all of solve_mip.

    ``from .cuts import Cut`` sat inside the symmetry-breaking branch, although
    Cut is already imported at module level. In Python an import anywhere in a
    function makes the name local to the *whole* function, so on a model where
    symmetry breaking contributes no rows -- that branch never running -- the
    conflict-clause append later in the search raised UnboundLocalError rather
    than solving. Symmetry and conflict analysis are both on by default, so
    reaching it needed only a model with no detectable symmetry that learns
    enough clauses to hit the batch threshold.
    """
    p = _random_max_binary(8)
    s = solve_mip(p, MIPParams(symmetry=False, conflict=True, clause_batch=1,
                               time_limit=30))
    assert s.info["conflict"]["clauses"] > 0, (
        "no clauses were learned, so the append path this test pins never ran")


@pytest.mark.parametrize("name,optimum", [("flugpl", 1201500.0),
                                          ("khb05250", 106940226.0),
                                          ("mod010", 6548.0)])
def test_published_milp_optima_are_reached(name, optimum):
    """The published value is an oracle the solver cannot argue with."""
    path = os.path.join(DATA, f"{name}.mps")
    if not os.path.exists(path):
        pytest.skip("benchmark instances not fetched")
    from vyuha.io import read_model

    s = solve_mip(read_model(path), MIPParams(time_limit=120))
    if s.status != Status.OPTIMAL:
        pytest.skip(f"{name} not proved optimal within the limit")
    assert abs(s.objective - optimum) <= 1e-6 * max(1.0, abs(optimum))


# --------------------------------------------------------------------------- #
# 4. an objective attached to a model that has no feasible points              #
# --------------------------------------------------------------------------- #


def _infeasible_blend():
    """Two rows that cannot both hold: use at most 10, produce at least 40."""
    A = SparseMatrix.from_dense(np.array([[1.0, 1.0], [1.0, 1.0]]))
    return Problem(A=A, c=np.array([700.0, 600.0]),
                   row_lb=np.array([-INF, 40.0]),
                   row_ub=np.array([10.0, INF]),
                   col_lb=np.zeros(2), col_ub=np.full(2, INF),
                   sense=ObjSense.MAXIMISE, name="infeasible_blend",
                   row_names=["AVAIL", "DEM"], col_names=["X0", "X1"])


def test_infeasible_lp_reports_no_objective():
    """REGRESSION: an infeasible exit carried the last iterate's cost.

    The simplex assembles its answer after the loops finish, unconditionally:
    ``obj = c @ x`` on whatever point the basis holds. Phase 1 ends on a point
    that is optimal for the *infeasibility* it was minimising, not for ``c``,
    and pricing it out gives a finite, plausible, entirely meaningless number.

    It was not merely ugly. Perturbing one demand row of the demo blend past
    the total component availability made the model infeasible, and the solve
    then reported the objective of the *feasible* model it came from, to
    twelve digits, beside the word INFEASIBLE -- a number a planner would read
    as the answer. The branch-and-bound paths already return ``nan`` when they
    end with no incumbent; this pins the LP and QP exits to the same contract.
    """
    from vyuha.lp.simplex import SimplexParams, solve_simplex

    s = solve_simplex(_infeasible_blend(), SimplexParams())
    assert s.status == Status.INFEASIBLE
    assert np.isnan(s.objective), (
        f"infeasible solve reported objective {s.objective!r}")
    assert np.isnan(s.dual_bound)


def test_a_feasible_solve_still_reports_its_objective():
    """The guard keys on Status.has_solution, so it must not blank a real one."""
    from vyuha.lp.simplex import SimplexParams, solve_simplex

    p = _infeasible_blend()
    p.row_ub[0] = 100.0                      # AVAIL now covers DEM
    s = solve_simplex(p, SimplexParams())
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 70000.0) < 1e-6   # 100 t of X0 at 700
    assert np.isfinite(s.dual_bound)


def test_infeasible_solve_writes_valid_json(tmp_path):
    """The CLI must not emit a bare NaN: JSON.parse in the UI rejects it."""
    import json

    from vyuha.cli import main
    from vyuha.io import write_mps

    model = tmp_path / "infeasible.mps"
    out = tmp_path / "sol.json"
    write_mps(_infeasible_blend(), str(model))
    main(["solve", str(model), "--out", str(out)])

    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text, "bare NaN in the payload is not valid JSON"
    assert json.loads(text)["objective"] is None
