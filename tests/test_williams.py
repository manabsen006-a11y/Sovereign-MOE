"""Williams' textbook models against the book's published optima.

Six models transcribed from *Model Building in Mathematical Programming*
-- blending with storage (12.1, 12.2), factory planning (12.3), a
distribution network (12.19), a day of unit commitment (12.15) and a mine
schedule (12.7) -- each checked against the value printed in chapter 13.
A transcription that reaches the book's number to its rounding is the
book's model; every one of the six does, on the first solve, which is the
check the models exist for.
"""

import numpy as np
import pytest

from bench.verify import verify
from sovopt.core.problem import Status, VarKind
from sovopt.lp.ipm import solve_ipm
from sovopt.lp.pdlp import PDLPParams, solve_pdlp
from sovopt.lp.simplex import solve_simplex
from sovopt.mip.tree import MIPParams, solve_mip
from sovopt.models.williams import MODELS, PUBLISHED

LPS = ["food_manufacture_1", "factory_planning_1", "distribution_1"]
MILPS = ["food_manufacture_2", "tariff_rates", "mining"]


def _close(value, name):
    """To the penny -- the book prints pounds to two decimals, and its
    values are recomputed from plans it prints rounded (12.2: the book's
    100,278.71 against an exact 100,278.7037); mining is in millions to
    three decimals, so to five hundred pounds."""
    pub = PUBLISHED[name]
    return abs(value - pub) <= (0.0005 if name == "mining" else 0.01)


@pytest.mark.parametrize("name", LPS)
def test_the_lps_reach_the_book_with_every_engine(name):
    p = MODELS[name]()
    assert p.n_integer == 0
    for engine in (solve_simplex, solve_ipm,
                   lambda q: solve_pdlp(q, PDLPParams(eps_abs=1e-9, eps_rel=1e-9, time_limit=60))):
        s = engine(p)
        assert s.status == Status.OPTIMAL, (name, engine, s.status)
        assert _close(s.objective, name), (name, s.objective, PUBLISHED[name])
        v = verify(p, s.x, s.objective, feas_tol=1e-6)
        assert v.ok, v.report()


@pytest.mark.parametrize("name", MILPS)
def test_the_milps_reach_the_book_and_prove_it(name):
    p = MODELS[name]()
    assert p.n_integer > 0
    s = solve_mip(p, MIPParams(time_limit=120, gap_rel=1e-6))
    assert s.status == Status.OPTIMAL, (name, s.status)
    assert _close(s.objective, name), (name, s.objective, PUBLISHED[name])
    v = verify(p, s.x, s.objective, feas_tol=1e-6, int_tol=1e-6)
    assert v.ok, v.report()


def test_the_logical_conditions_cost_what_the_book_says():
    """12.2 is 12.1 with three logical conditions; the book's two values
    are 7,563.88 apart, which is what the conditions cost."""
    a = solve_simplex(MODELS["food_manufacture_1"]()).objective
    b = solve_mip(MODELS["food_manufacture_2"](), MIPParams(time_limit=120, gap_rel=1e-6)).objective
    assert abs((a - b) - (PUBLISHED["food_manufacture_1"] - PUBLISHED["food_manufacture_2"])) < 0.02


def test_the_plans_are_the_books():
    """Three facts from the printed solutions: distribution 1 sends all of
    C1 from Liverpool directly; the tariff plan starts no type-3 generator
    in the night period; the mine schedule never works four mines."""
    p = MODELS["distribution_1"]()
    s = solve_simplex(p)
    x = dict(zip(p.col_names, s.x))
    assert x["LIVERPOOL_C1"] == pytest.approx(50000.0, abs=1e-6)
    p = MODELS["tariff_rates"]()
    s = solve_mip(p, MIPParams(time_limit=120, gap_rel=1e-6))
    x = dict(zip(p.col_names, s.x))
    assert x["ON_T3_0-6"] == pytest.approx(0.0, abs=1e-6)
    p = MODELS["mining"]()
    s = solve_mip(p, MIPParams(time_limit=120, gap_rel=1e-6))
    x = dict(zip(p.col_names, s.x))
    for t in range(1, 6):
        assert sum(x[f"WORK_M{m}_Y{t}"] for m in range(1, 5)) <= 3 + 1e-6
