"""Blending from a planner's tables (:mod:`sovopt.models.tabular`).

The layer that turns two CSV files into a model and a solution back into
a plan. What is pinned here: a hand-solvable case, the specifications
actually binding, every engine reaching the same plan, the shadow prices
matching a re-solve, the CSV reader's error messages, the example under
examples/blending, and an infeasible set of specifications coming back
as a proved infeasibility rather than a plan.
"""

import copy
import os

import numpy as np
import pytest

from bench.verify import verify, verify_infeasible
from sovopt.core.problem import Status
from sovopt.lp.ipm import solve_ipm
from sovopt.lp.pdlp import PDLPParams, solve_pdlp
from sovopt.lp.simplex import solve_simplex
from sovopt.models.tabular import (TableError, blend_plan, blending_from_tables,
                                   parse_blending_csv, plan_text, plan_to_csv,
                                   read_blending_csv)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples", "blending")

COMPONENTS = """name,cost,available,minimum,sulfur
Sweet,100,50,,0.1
Sour,60,100,,0.5
"""
PRODUCTS = """name,price,demand_min,demand_max,sulfur_max
Fuel,120,,80,0.3
"""


def _two_by_one():
    return parse_blending_csv(COMPONENTS, PRODUCTS)


def test_a_hand_solvable_blend():
    """80 units of fuel at sulfur <= 0.3 from Sweet (0.1, cost 100) and Sour
    (0.5, cost 60): the cheapest blend meeting the spec is half and half,
    40 + 40, margin 80*120 - 40*100 - 40*60 = 3,200."""
    t = _two_by_one()
    assert t.problem.m == 4 and t.problem.n == 3         # AVAIL x2, BAL, SPEC
    s = solve_simplex(t.problem)
    assert s.status == Status.OPTIMAL
    assert abs(s.objective - 3200.0) < 1e-9
    plan = blend_plan(t, s)
    fuel = plan["products"][0]
    assert abs(fuel["volume"] - 80.0) < 1e-9
    assert abs(fuel["qualities"]["sulfur"]["value"] - 0.3) < 1e-9
    assert fuel["qualities"]["sulfur"]["binding"] == "max"
    recipe = {r["component"]: r["quantity"] for r in plan["recipe"]}
    assert abs(recipe["Sweet"] - 40.0) < 1e-9 and abs(recipe["Sour"] - 40.0) < 1e-9
    assert plan["verifier_verdict"] if "verifier_verdict" in plan else True
    assert verify(t.problem, s.x, s.objective, y=s.y).ok


def test_every_engine_reaches_the_same_plan():
    t = _two_by_one()
    for engine in (solve_simplex, solve_ipm,
                   lambda p: solve_pdlp(p, PDLPParams(eps_abs=1e-9, eps_rel=1e-9, time_limit=60))):
        s = engine(t.problem)
        assert s.status == Status.OPTIMAL, engine
        assert abs(s.objective - 3200.0) < 1e-6


def test_shadow_prices_match_a_re_solve():
    """A component's shadow price is the margin per extra unit available;
    a specification's is per unit of quality the limit is relaxed, which is
    the row's dual times the product's volume."""
    t = read_blending_csv(os.path.join(EXAMPLES, "components.csv"),
                          os.path.join(EXAMPLES, "products.csv"))
    s = solve_simplex(t.problem)
    plan = blend_plan(t, s)
    binding = [c for c in plan["components"] if c["at_limit"]]
    assert binding
    co = binding[0]
    comps = copy.deepcopy(t.components)
    next(c for c in comps if c["name"] == co["name"])["available"] += 1.0
    s2 = solve_simplex(blending_from_tables(comps, copy.deepcopy(t.products), t.qualities).problem)
    assert abs((s2.objective - s.objective) - co["shadow_price"]) < 1e-6
    spec = next(x for x in plan["specs"] if x["binding"])
    prods = copy.deepcopy(t.products)
    row = spec["row"]                       # SPEC[<product>.<quality><=<limit>]
    inner = row[5:-1]
    product, rest = inner.split(".", 1)
    quality = rest.split("<=")[0].split(">=")[0]
    side = "max" if "<=" in rest else "min"
    pr = next(p for p in prods if p["name"] == product)
    pr["specs"][quality][side] += 1e-3 if side == "max" else -1e-3
    s3 = solve_simplex(blending_from_tables(copy.deepcopy(t.components), prods, t.qualities).problem)
    assert abs((s3.objective - s.objective) - 1e-3 * spec["per_unit_of_quality"]) < 1e-6 * abs(s.objective)


def test_the_example_solves_to_a_verified_plan_with_every_engine():
    t = read_blending_csv(os.path.join(EXAMPLES, "components.csv"),
                          os.path.join(EXAMPLES, "products.csv"))
    assert t.qualities == ["aromatics", "density", "sulfur"]
    ref = solve_simplex(t.problem).objective
    for engine in (solve_simplex, solve_ipm,
                   lambda p: solve_pdlp(p, PDLPParams(eps_abs=1e-9, eps_rel=1e-9, time_limit=60))):
        s = engine(t.problem)
        assert s.status == Status.OPTIMAL
        assert abs(s.objective - ref) <= 1e-6 * abs(ref)
        assert verify(t.problem, s.x, s.objective, feas_tol=1e-6).ok
    plan = blend_plan(t, solve_simplex(t.problem))
    # the plan respects what the tables said
    for pr in plan["products"]:
        spec = next(p for p in t.products if p["name"] == pr["name"])
        for q, v in pr["qualities"].items():
            if v["value"] is None:
                continue
            if v["max"] is not None:
                assert v["value"] <= v["max"] + 1e-9
            if v["min"] is not None:
                assert v["value"] >= v["min"] - 1e-9
        if spec["demand_max"] is not None:
            assert pr["volume"] <= spec["demand_max"] + 1e-9
        assert pr["volume"] >= spec["demand_min"] - 1e-9
    for co in plan["components"]:
        if co["available"] is not None:
            assert co["used"] <= co["available"] + 1e-9
        assert co["used"] >= co["minimum"] - 1e-9
    assert "margin" in plan_text(plan)
    csv_text = plan_to_csv(plan)
    assert csv_text.splitlines()[0] == "product,component,quantity,fraction"
    assert len(csv_text.splitlines()) == 1 + len(plan["recipe"])


def test_specifications_that_cannot_be_met_are_proved_infeasible():
    """A sulfur limit below every component's sulfur, with a demand that
    must be met: no plan, and the simplex's ray certifies it."""
    t = parse_blending_csv(COMPONENTS, "name,price,demand_min,sulfur_max\nFuel,120,10,0.05\n")
    s = solve_simplex(t.problem)
    assert s.status == Status.INFEASIBLE
    assert verify_infeasible(t.problem, s.farkas).ok
    plan = blend_plan(t, s)
    assert plan["objective"] is None and "no feasible plan" in plan["message"]


@pytest.mark.parametrize("comps,prods,message", [
    ("cost,sulfur\n1,0.1\n", PRODUCTS, "no 'name' column"),
    ("name,cost,sulfur\nA,1,0.1\nA,2,0.2\n", PRODUCTS, "appears twice"),
    ("name,cost,sulfur\nA,1,\n", PRODUCTS, "has no value for sulfur"),
    ("name,cost,sulfur\nA,abc,0.1\n", PRODUCTS, "is not a number"),
    ("name,cost,available,minimum,sulfur\nA,1,5,10,0.1\n", PRODUCTS, "minimum 10 above available 5"),
    (COMPONENTS, "name,price,octane_min\nP,2,90\n", "no component has a 'octane' column"),
    (COMPONENTS, "name,price,colour\nP,2,red\n", "is not price, demand_min, demand_max or a"),
    (COMPONENTS, "name,price,demand_min,demand_max\nP,2,10,5\n", "demand_min 10 above demand_max 5"),
    (COMPONENTS, "", "the file is empty"),
])
def test_a_table_the_model_cannot_be_built_from_says_why(comps, prods, message):
    with pytest.raises(TableError) as e:
        parse_blending_csv(comps, prods)
    assert message in str(e.value)


def test_blank_cells_mean_unlimited_or_zero_and_headers_are_case_insensitive():
    t = parse_blending_csv("Name,Cost,Available,Sulfur\nA,1,,0.1\n",
                           "NAME,PRICE,Demand_Max,Sulfur_Max\nP,5,,\n")
    co, pr = t.components[0], t.products[0]
    assert co["available"] is None and co["minimum"] == 0.0
    assert pr["demand_max"] is None and pr["demand_min"] == 0.0 and pr["specs"] == {}
    # unlimited on both sides with a positive margin: unbounded, reported as such
    s = solve_simplex(t.problem)
    assert s.status == Status.UNBOUNDED
