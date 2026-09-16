"""The refinery models: the literature one against its published optimum,
and the generated ones for the shape they promise."""

from __future__ import annotations

import numpy as np
import pytest

from sovopt.cli import solve
from sovopt.core.problem import Status
from sovopt.models import (WILLIAMS_OPTIMUM, blending, production_planning,
                           unit_scheduling, williams_refinery)

from bench.verify import verify


@pytest.mark.parametrize("method", ["simplex", "ipm", "pdlp"])
def test_williams_refinery_reaches_the_published_profit(method):
    """Williams 12.6: optimal profit 211,365.13 per day, printed in the
    book's solution 13.6 with the plan that earns it. All three engines
    must land on it, to the penny the book rounds to."""
    p = williams_refinery()
    s = solve(p, method=method, device="cpu", time_limit=60, tol=1e-9)
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - WILLIAMS_OPTIMUM) < 0.01
    v = verify(p, s.x, s.objective, y=s.y)
    assert v.ok, v.report()


def test_williams_refinery_plan_is_the_books():
    """The book's plan: crude 1 15,000 and crude 2 30,000 barrels (the
    distillation cap binds), premium 6,817.78, regular 17,044.45, jet fuel
    15,156, no fuel oil, lube at its floor of 500."""
    p = williams_refinery()
    s = solve(p, method="simplex", device="cpu", time_limit=60, tol=1e-9)
    x = dict(zip(p.col_names, s.x))
    want = {"CR1": 15000.0, "CR2": 30000.0, "PMF": 6817.78, "RMF": 17044.45,
            "JF": 15156.0, "FO": 0.0, "LBO": 500.0}
    for k, v in want.items():
        assert abs(x[k] - v) < 0.01, (k, x[k], v)


@pytest.mark.parametrize("make", [
    lambda: blending(n_components=10, n_products=4, n_properties=3, seed=3),
    lambda: production_planning(n_crudes=4, n_units=3, n_products=4,
                                n_periods=4, seed=3),
])
def test_generated_lps_solve_and_verify(make):
    p = make()
    s = solve(p, method="auto", device="cpu", time_limit=60, tol=1e-8)
    assert s.status == Status.OPTIMAL
    v = verify(p, s.x, s.objective, y=s.y)
    assert v.ok, v.report()


def test_unit_scheduling_is_a_mip_that_the_tree_closes():
    p = unit_scheduling(n_units=3, n_periods=6, seed=1)
    assert p.n_integer > 0
    s = solve(p, method="bnb", device="cpu", time_limit=120, gap=1e-6)
    assert s.status == Status.OPTIMAL
    v = verify(p, s.x, s.objective)
    assert v.ok, v.report()
    assert np.all(np.abs(s.x[p.integer_mask] - np.round(s.x[p.integer_mask])) < 1e-6)
