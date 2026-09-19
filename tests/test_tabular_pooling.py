"""Pooled qualities from tables (:mod:`sovopt.models.tabular_pooling`).

The answer key is Haverly: written as the three tables under
examples/pooling it must reach the published global optimum of 400 with
the search closed, and its two published variants -- demand for X raised
to 600 (600) and B's cost cut to 13 (750) -- must follow from editing the
tables alone.
"""

import os

import pytest

from bench.verify import verify
from sovopt.core.problem import Status
from sovopt.globalopt.spatial import SpatialParams, solve_global
from sovopt.models.pooling import HAVERLY_VARIANTS
from sovopt.models.tabular import TableError
from sovopt.models.tabular_pooling import (parse_pooling_csv, pooling_plan, pooling_text,
                                           pooling_to_csv, read_pooling_csv)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples", "pooling")

COMPS = "name,cost,available,direct,sulfur\nA,6,600,,3\nB,{b},600,,1\nC,10,600,*,2\n"
POOLS = "name,capacity,inputs\nPool,600,A;B\n"
PRODS = "name,price,demand_max,sulfur_max\nX,9,{x},2.5\nY,15,200,1.5\n"


def _solve(t):
    s = solve_global(t.problem, SpatialParams(time_limit=60))
    assert s.status == Status.OPTIMAL, s.status
    assert t.problem.max_violation(s.x) <= 1e-6
    assert verify(t.problem.linear, s.x, s.objective, feas_tol=1e-6).ok
    return s


def test_haverly_from_the_tables_is_proved_to_its_published_optimum():
    t = read_pooling_csv(os.path.join(EX, "components.csv"), os.path.join(EX, "products.csv"),
                         os.path.join(EX, "pools.csv"))
    s = _solve(t)
    assert abs(s.objective - 400.0) < 1e-6
    plan = pooling_plan(t, s)
    assert plan["proved_global"]
    assert abs(plan["objective"] - 400.0) < 1e-6
    # the known plan: B through the pool, C direct, all of it into Y
    y = next(p for p in plan["products"] if p["name"] == "Y")
    assert abs(y["volume"] - 200.0) < 1e-6 and y["qualities"]["sulfur"]["binding"] == "max"
    flows = {(f["from"], f["to"]): f["quantity"] for f in plan["flows"]}
    assert abs(flows[("B", "Pool")] - 100.0) < 1e-6 and abs(flows[("C", "Y")] - 100.0) < 1e-6
    assert ("A", "Pool") not in flows
    assert "proved globally optimal" in pooling_text(plan)
    assert pooling_to_csv(plan).splitlines()[0] == "from,to,quantity"


@pytest.mark.parametrize("variant,b_cost,x_demand", [(2, 16, 600), (3, 13, 100)])
def test_the_published_variants_follow_from_the_tables(variant, b_cost, x_demand):
    t = parse_pooling_csv(COMPS.format(b=b_cost), PRODS.format(x=x_demand), POOLS)
    s = _solve(t)
    assert abs(s.objective - HAVERLY_VARIANTS[variant]) < 1e-6


def test_a_minimum_specification_is_the_maximum_on_the_negated_quality():
    """Y must have sulfur at least 1.2 as well as at most 1.5: the optimum
    is still 400 (Y's blend sits at 1.5), and the plan reports the band."""
    prods = "name,price,demand_max,sulfur_min,sulfur_max\nX,9,100,,2.5\nY,15,200,1.2,1.5\n"
    t = parse_pooling_csv(COMPS.format(b=16), prods, POOLS)
    assert t.data.spec.shape == (2, 2)                  # sulfur, and -sulfur
    s = _solve(t)
    assert abs(s.objective - 400.0) < 1e-6
    plan = pooling_plan(t, s)
    y = next(p for p in plan["products"] if p["name"] == "Y")
    assert y["qualities"]["sulfur"]["min"] == 1.2 and y["qualities"]["sulfur"]["max"] == 1.5


@pytest.mark.parametrize("comps,prods,pools,message", [
    ("name,cost,sulfur\nA,6,3\n", PRODS.format(x=100), "name,capacity,inputs\nP,600,B\n",
     "'B' is not a known name"),
    ("name,cost,direct,sulfur\nA,6,,3\nD,1,,1\n", PRODS.format(x=100),
     "name,capacity,inputs\nP,600,A\n", "D is in no pool's inputs and has no direct products"),
    (COMPS.format(b=16), "name,price,demand_min,sulfur_max\nX,9,10,2.5\n", POOLS,
     "the pooling model takes demand_max only"),
    (COMPS.format(b=16), PRODS.format(x=100), "name,capacity,inputs\nP,600,A\nP,600,B\n",
     "'P' appears twice"),
])
def test_a_pool_table_the_model_cannot_be_built_from_says_why(comps, prods, pools, message):
    with pytest.raises(TableError) as e:
        parse_pooling_csv(comps, prods, pools)
    assert message in str(e.value)
