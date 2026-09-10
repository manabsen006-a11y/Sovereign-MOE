"""The refactorisation budget, and why the two paths disagree about it.

The product-form update appends an eta per pivot, so the budget trades a longer
eta chain on every solve against a full LU refactorisation. The README blamed
the product form itself for 10teams and named Forrest-Tomlin as the fix;
measured, the budget was set too low, and raising it for the LP path is worth
1.4-1.9x with no accuracy cost at all.

The tree wants the opposite, and that is the part worth pinning: a node LP does
a handful of pivots from a warm basis and never gets far enough down a long
chain to amortise it. Nothing about the code makes that obvious, so a future
tidy-up that "unifies the default" would silently cost the MIP set an instance.
"""

import numpy as np
import pytest

from sovopt.core.problem import Status, VarKind
from sovopt.lp.simplex import SimplexParams, solve_simplex
from sovopt.mip.tree import NODE_REFACTOR_FREQ

from tests.test_simplex import random_lp


def test_the_two_paths_keep_different_budgets():
    """REGRESSION: the tree must not inherit the LP path's budget.

    Measured over the MIPLIB set at a 60 s limit, the LP default of 150 proves
    5/11 against 6/11 at 60, and takes 414.5 s against 377.4 s. The two
    settings are deliberate and opposite.
    """
    assert SimplexParams().refactor_freq == 150, "the LP default moved"
    assert NODE_REFACTOR_FREQ == 60, "the tree default moved"
    assert NODE_REFACTOR_FREQ != SimplexParams().refactor_freq


def test_the_tree_actually_passes_its_own_budget():
    """The constant is only worth anything if every solver is built with it."""
    import inspect

    from sovopt.mip import tree as T

    src = inspect.getsource(T)
    built = src.count("NodeSolver(scaled, SimplexParams(")
    passed = src.count("refactor_freq=NODE_REFACTOR_FREQ")
    assert built > 0
    assert passed == built, (
        f"{built} node solvers built but only {passed} given the tree's "
        f"refactorisation budget")


@pytest.mark.parametrize("freq", [20, 60, 150, 400])
def test_the_answer_does_not_depend_on_the_budget(freq):
    """Whatever the budget, the optimum is the optimum.

    The budget is a speed knob and must never be an accuracy one. A long eta
    chain accumulates error, so this is the assertion that the refactorisation
    trigger is doing its job rather than the budget quietly buying speed with
    precision.
    """
    p = random_lp(seed=1)
    ref = solve_simplex(p.copy(), SimplexParams(refactor_freq=60))
    got = solve_simplex(p.copy(), SimplexParams(refactor_freq=freq))
    assert got.status == Status.OPTIMAL
    assert abs(got.objective - ref.objective) <= 1e-9 * max(1.0, abs(ref.objective))
    row_v, col_v, _ = p.violation(got.x)
    assert max(row_v, col_v) < 1e-7


def test_a_long_budget_stays_accurate_on_a_model_that_pivots_a_lot():
    """The case the budget was raised for: many pivots from one basis."""
    p = random_lp(seed=4, m=60, n=90)
    ref = solve_simplex(p.copy(), SimplexParams(refactor_freq=60))
    got = solve_simplex(p.copy(), SimplexParams(refactor_freq=150))
    assert abs(got.objective - ref.objective) <= 1e-9 * max(1.0, abs(ref.objective))
    # the duals have to survive it too, not just the objective
    scale = max(1.0, float(np.abs(ref.y).max(initial=0.0)))
    assert np.abs(got.y - ref.y).max() <= 1e-6 * scale
