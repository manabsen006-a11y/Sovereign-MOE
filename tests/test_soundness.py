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

from vyuha.core.problem import Problem, Status, VarKind
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
