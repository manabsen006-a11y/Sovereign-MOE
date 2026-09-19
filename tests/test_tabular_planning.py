"""Multi-period planning from tables (:mod:`sovopt.models.tabular_planning`).

The answer key is Williams' food manufacture 1: written as the tables
under examples/planning it must reach the book's 107,842.59, and the plan
must be the book's (every month at the vegetable-oil capacity, hardness at
its upper limit, 450 tons a month).
"""

import os

import pytest

from bench.verify import verify
from sovopt.core.problem import Status
from sovopt.lp.ipm import solve_ipm
from sovopt.lp.simplex import solve_simplex
from sovopt.models.tabular import TableError
from sovopt.models.tabular_planning import (parse_planning_csv, planning_plan, planning_text,
                                            planning_to_csv, read_planning_csv)
from sovopt.models.williams import PUBLISHED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples", "planning")


def _williams():
    return read_planning_csv(os.path.join(EX, "components.csv"), os.path.join(EX, "products.csv"),
                             os.path.join(EX, "prices.csv"),
                             capacity_path=os.path.join(EX, "capacity.csv"))


def test_williams_food_manufacture_from_the_tables_reaches_the_book():
    t = _williams()
    assert t.periods == ["January", "February", "March", "April", "May", "June"]
    for engine in (solve_simplex, solve_ipm):
        s = engine(t.problem)
        assert s.status == Status.OPTIMAL
        assert abs(s.objective - PUBLISHED["food_manufacture_1"]) <= 0.01
        assert verify(t.problem, s.x, s.objective, feas_tol=1e-6, y=s.y).ok
    plan = planning_plan(t, solve_simplex(t.problem))
    # the book's plan: the vegetable line full every month, hardness at 6
    for c in plan["capacity"]:
        if c["line"] == "VEG":
            assert c["at_limit"] and c["shadow_price"] > 0
    for per in plan["periods"]:
        prod = per["products"][0]
        assert prod["qualities"]["hardness"]["binding"] == "max"
        assert abs(prod["volume"] - 450.0) < 1e-6
    # closing stocks honoured, storage cost charged on every month's stock
    last = plan["periods"][-1]
    for co in last["components"]:
        assert co["store"] >= 500.0 - 1e-6
    assert abs(plan["totals"]["storage"] - 55000.0) < 1e-6
    assert "margin over the horizon" in planning_text(plan)
    rows = planning_to_csv(plan).splitlines()
    assert rows[0] == "period,item,kind,buy,use,store,make,revenue"
    assert len(rows) == 1 + 6 * (5 + 1)


def test_demand_and_capacity_tables_override_per_period():
    comps = "name,cost,line,hardness\nA,10,L,2\nB,20,L,6\n"
    prods = "name,price,hardness_min,hardness_max\nP,50,3,5\n"
    prices = "period,A,B\nT1,10,20\nT2,12,\n"          # blank: B keeps its cost
    demand = "period,product,price,demand_max\nT1,P,50,100\nT2,P,60,50\n"
    capacity = "line,capacity,period\nL,80,T1\nL,200,T2\n"
    t = parse_planning_csv(comps, prods, prices, demand, capacity)
    assert t.prices[("B", "T2")] == 20.0 and t.prices[("A", "T2")] == 12.0
    assert t.demand[("P", "T2")]["price"] == 60.0
    s = solve_simplex(t.problem)
    assert s.status == Status.OPTIMAL
    plan = planning_plan(t, s)
    by = {(p["period"], q["name"]): q for p in plan["periods"] for q in p["products"]}
    assert abs(by[("T1", "P")]["volume"] - 80.0) < 1e-9        # capacity binds
    assert abs(by[("T2", "P")]["volume"] - 50.0) < 1e-9        # demand binds
    cap = {(c["line"], c["period"]): c for c in plan["capacity"]}
    assert cap[("L", "T1")]["at_limit"] and not cap[("L", "T2")]["at_limit"]


@pytest.mark.parametrize("prices,demand,capacity,message", [
    ("period,A,Z\nT1,1,2\n", None, None, "column 'z' is not a component"),
    ("period,A\nT1,1\nT1,2\n", None, None, "period 'T1' appears twice"),
    ("period,A\nT1,1\n", "period,product,price\nT9,P,1\n", None, "period 'T9' is not in prices.csv"),
    ("period,A\nT1,1\n", "period,product,price\nT1,Q,1\n", None, "product 'Q' is not in products.csv"),
    ("period,A\nT1,1\n", None, "line,capacity\nM,5\n", "line 'M' is on no component"),
    ("period,A\nT1,1\n", None, "line,capacity\nL,\n", "no capacity for L"),
])
def test_a_horizon_table_the_model_cannot_be_built_from_says_why(prices, demand, capacity, message):
    comps = "name,cost,line,hardness\nA,10,L,2\n"
    prods = "name,price,hardness_max\nP,50,5\n"
    with pytest.raises(TableError) as e:
        parse_planning_csv(comps, prods, prices, demand, capacity)
    assert message in str(e.value)
