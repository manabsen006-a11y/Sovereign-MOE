"""Multi-period planning from a planner's tables: purchases, storage,
processing capacities and blending over a horizon.

The shape is Williams' food manufacture (12.1): components bought each
period at that period's price, held in store at a cost between periods,
processed on lines with a capacity per period, and blended into products
under quality bands -- the crude-purchase-and-storage plan a refinery
runs month by month. The single-period tables of :mod:`tabular` are
reused; a horizon is added by a prices table whose rows are the periods.

The tables
----------
``components.csv`` -- as in :mod:`tabular`, plus optional columns::

    line           the processing line the component runs on (capacity
                   below applies per line and period); blank = no line
    storage_max    the most that can be held in store; blank = unlimited
    storage_cost   cost per unit held at the end of a period; blank = 0
    opening_stock  in store before the first period; blank = 0
    closing_stock  the least that must be in store after the last period;
                   blank = no requirement

    ``cost`` is the purchase price where ``prices.csv`` leaves a cell
    blank; ``available`` the most that can be bought in a period;
    ``minimum`` the least that must be used in a period.

``products.csv`` -- as in :mod:`tabular`; ``price``, ``demand_min`` and
``demand_max`` are per period, overridable below.

``prices.csv`` -- one row per period, in order; a column per component::

    period, <component>, <component>, ...

``demand.csv`` (optional) -- overrides per period and product::

    period, product, price, demand_min, demand_max

``capacity.csv`` (optional) -- capacity per line, and per period::

    line, capacity, period      (period blank or the column absent: every period)

The model
---------
For each period ``t``, component ``i``, product ``j``::

    max  Σ_t [ Σ_j price_jt MAKE_jt − Σ_i price_it BUY_it − Σ_i store_i STORE_it ]
    s.t. STORE_it = STORE_i,t-1 + BUY_it − Σ_j USE_ijt        (STOCK, STORE_i0 = opening)
         STORE_iT >= closing_i
         Σ_j USE_ijt in [minimum_i, ∞),  BUY_it <= available_i,  STORE_it <= storage_max_i
         Σ_{i on line} Σ_j USE_ijt <= capacity_line,t          (CAP)
         Σ_i USE_ijt = MAKE_jt,  demand_min_jt <= MAKE_jt <= demand_max_jt   (BAL)
         Σ_i q_i USE_ijt <= q_max_j MAKE_jt,  >= q_min_j MAKE_jt       (SPEC)

Checked against the book: Williams 12.1 written as these tables
(``examples/planning``) solves to the published 107,842.59.

References
----------
Williams, H.P., "Model Building in Mathematical Programming", 5th ed.,
  Wiley (2013), 12.1 "Food manufacture 1" -- purchases, storage and
  blending over a horizon, and its solution 13.1.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

from ..core.problem import ObjSense, Problem, Status
from ..core.tolerances import INF
from ._builder import Builder
from .tabular import TableError, _num, _plain, _rows, may_enter, parse_blending_csv

__all__ = ["PlanTables", "parse_planning_csv", "read_planning_csv",
           "planning_from_tables", "planning_plan", "planning_text", "planning_to_csv"]


@dataclass
class PlanTables:
    components: list[dict]
    products: list[dict]
    qualities: list[str]
    periods: list[str]
    prices: dict                      # (component, period) -> price
    demand: dict                      # (product, period) -> {price, demand_min, demand_max}
    capacity: dict                    # (line, period) -> capacity
    problem: Problem
    buy: dict = field(default_factory=dict)
    use: dict = field(default_factory=dict)
    store: dict = field(default_factory=dict)
    make: dict = field(default_factory=dict)


def parse_planning_csv(components_text, products_text, prices_text,
                       demand_text=None, capacity_text=None) -> PlanTables:
    """Build the planning model from the tables' text."""
    base = parse_blending_csv(components_text, products_text)
    comps_raw = _rows(components_text, "components")
    extra = {}
    for r in comps_raw:
        name = r["name"]
        extra[name] = {
            "line": (r.get("line") or "").strip() or None,
            "storage_max": _num(r.get("storage_max"), f"components {name} storage_max"),
            "storage_cost": _num(r.get("storage_cost"), f"components {name} storage_cost") or 0.0,
            "opening_stock": _num(r.get("opening_stock"), f"components {name} opening_stock") or 0.0,
            "closing_stock": _num(r.get("closing_stock"), f"components {name} closing_stock"),
        }
    components = []
    for co in base.components:
        rec = dict(co)
        rec.update(extra[co["name"]])
        components.append(rec)
    names = {c["name"] for c in components}
    pnames = {p["name"] for p in base.products}

    # prices: periods in row order, a column per component
    prows = _rows_by(prices_text, "prices", "period")
    periods, prices = [], {}
    for r in prows:
        per = (r.get("period") or r.get("name") or "").strip()
        if not per:
            raise TableError(f"prices line {r['_line']}: a row without a period")
        if per in periods:
            raise TableError(f"prices: period {per!r} appears twice")
        periods.append(per)
        for key, val in r.items():
            if key in ("period", "name") or key.startswith("_"):
                continue
            match = [c for c in names if c.lower() == key]
            if not match:
                raise TableError(f"prices: column {key!r} is not a component "
                                 f"(components: {', '.join(sorted(names))})")
            v = _num(val, f"prices {per} {key}")
            if v is not None:
                prices[match[0], per] = v
    for co in components:
        for per in periods:
            prices.setdefault((co["name"], per), co["cost"])

    demand = {}
    if demand_text and demand_text.strip():
        for r in _rows_by(demand_text, "demand", "product"):
            per = (r.get("period") or "").strip()
            prod = (r.get("product") or r.get("name") or "").strip()
            if per not in periods:
                raise TableError(f"demand line {r['_line']}: period {per!r} is not in prices.csv")
            if prod not in pnames:
                raise TableError(f"demand line {r['_line']}: product {prod!r} is not in products.csv")
            demand[prod, per] = {
                "price": _num(r.get("price"), f"demand {prod} {per} price"),
                "demand_min": _num(r.get("demand_min"), f"demand {prod} {per} demand_min"),
                "demand_max": _num(r.get("demand_max"), f"demand {prod} {per} demand_max")}

    lines = {c["line"] for c in components if c["line"]}
    capacity = {}
    if capacity_text and capacity_text.strip():
        for r in _rows_by(capacity_text, "capacity", "line"):
            line = (r.get("line") or r.get("name") or "").strip()
            if line not in lines:
                raise TableError(f"capacity line {r['_line']}: line {line!r} is on no component "
                                 f"(lines: {', '.join(sorted(lines)) or 'none'})")
            cap = _num(r.get("capacity"), f"capacity {line}")
            if cap is None:
                raise TableError(f"capacity line {r['_line']}: no capacity for {line}")
            per = (r.get("period") or "").strip()
            if per and per not in periods:
                raise TableError(f"capacity line {r['_line']}: period {per!r} is not in prices.csv")
            for p in ([per] if per else periods):
                capacity[line, p] = cap
    return planning_from_tables(components, base.products, base.qualities, periods,
                                prices, demand, capacity)


def _rows_by(text, what, key):
    """Rows of a table whose key column is ``key`` rather than ``name``."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise TableError(f"{what}: the file is empty")
    fields = [f.strip().lower() for f in reader.fieldnames]
    if key not in fields and "name" not in fields:
        raise TableError(f"{what}: no {key!r} column (columns: {', '.join(fields)})")
    rows = []
    for k, raw in enumerate(reader, 2):
        row = {kk.strip().lower(): (v.strip() if isinstance(v, str) else v)
               for kk, v in raw.items() if kk is not None}
        if not any(v for v in row.values()):
            continue
        row["_line"] = k
        rows.append(row)
    if not rows:
        raise TableError(f"{what}: no rows")
    return rows


def read_planning_csv(components_path, products_path, prices_path,
                      demand_path=None, capacity_path=None) -> PlanTables:
    def rd(p):
        if p is None:
            return None
        with open(p, encoding="utf-8-sig") as fh:
            return fh.read()
    return parse_planning_csv(rd(components_path), rd(products_path), rd(prices_path),
                              rd(demand_path), rd(capacity_path))


def planning_from_tables(components, products, qualities, periods, prices,
                         demand=None, capacity=None) -> PlanTables:
    demand = demand or {}
    capacity = capacity or {}
    b = Builder("plan", ObjSense.MAXIMISE)
    buy, use, store, make = {}, {}, {}, {}
    for t in periods:
        for pr in products:
            ov = demand.get((pr["name"], t), {})
            price = ov.get("price") if ov.get("price") is not None else pr["price"]
            dmin = ov.get("demand_min") if ov.get("demand_min") is not None else pr["demand_min"]
            dmax = ov.get("demand_max") if ov.get("demand_max") is not None else pr["demand_max"]
            make[pr["name"], t] = b.col(f"MAKE[{pr['name']}@{t}]", lo=dmin,
                                        hi=INF if dmax is None else dmax, c=price)
        for co in components:
            buy[co["name"], t] = b.col(f"BUY[{co['name']}@{t}]",
                                       hi=INF if co["available"] is None else co["available"],
                                       c=-prices[co["name"], t])
            store[co["name"], t] = b.col(
                f"STORE[{co['name']}@{t}]",
                hi=INF if co.get("storage_max") is None else co["storage_max"],
                c=-co.get("storage_cost", 0.0))
            for pr in products:
                use[co["name"], pr["name"], t] = b.col(
                    f"USE[{co['name']}->{pr['name']}@{t}]",
                    hi=INF if may_enter(co, pr["name"]) else 0.0)
    lines = sorted({c["line"] for c in components if c.get("line")})
    for k, t in enumerate(periods):
        for co in components:
            terms = {store[co["name"], t]: 1.0, buy[co["name"], t]: -1.0,
                     **{use[co["name"], pr["name"], t]: 1.0 for pr in products}}
            rhs = 0.0
            if k > 0:
                terms[store[co["name"], periods[k - 1]]] = -1.0
            else:
                rhs = co.get("opening_stock", 0.0)
            b.eq(f"STOCK[{co['name']}@{t}]", terms, rhs)
            if co["minimum"]:
                b.ge(f"MINUSE[{co['name']}@{t}]",
                     {use[co["name"], pr["name"], t]: 1.0 for pr in products}, co["minimum"])
        for line in lines:
            cap = capacity.get((line, t))
            if cap is not None:
                b.le(f"CAP[{line}@{t}]",
                     {use[co["name"], pr["name"], t]: 1.0
                      for co in components if co.get("line") == line for pr in products}, cap)
        for pr in products:
            b.eq(f"BAL[{pr['name']}@{t}]",
                 {**{use[co["name"], pr["name"], t]: 1.0 for co in components},
                  make[pr["name"], t]: -1.0})
            for q, spec in pr["specs"].items():
                if "max" in spec:
                    b.le(f"SPEC[{pr['name']}.{q}<={spec['max']:g}@{t}]",
                         {**{use[co["name"], pr["name"], t]: co["qualities"][q] for co in components},
                          make[pr["name"], t]: -spec["max"]}, 0.0)
                if "min" in spec:
                    b.ge(f"SPEC[{pr['name']}.{q}>={spec['min']:g}@{t}]",
                         {**{use[co["name"], pr["name"], t]: co["qualities"][q] for co in components},
                          make[pr["name"], t]: -spec["min"]}, 0.0)
    last = periods[-1]
    for co in components:
        if co.get("closing_stock") is not None:
            b.ge(f"CLOSE[{co['name']}]", {store[co["name"], last]: 1.0}, co["closing_stock"])
    return PlanTables(components, products, list(qualities), list(periods), dict(prices),
                      dict(demand), dict(capacity), b.problem(), buy, use, store, make)


# --------------------------------------------------------------------------- #
# the plan                                                                    #
# --------------------------------------------------------------------------- #

def planning_plan(tables: PlanTables, sol) -> dict:
    p = tables.problem
    out = {"status": sol.status.name, "objective": None, "periods": [], "totals": {},
           "capacity": []}
    if sol.x is None or not Status(sol.status).has_solution:
        out["message"] = ("no feasible plan: the demands, specifications and capacities "
                          "cannot be met over the horizon" if sol.status == Status.INFEASIBLE
                          else f"no plan returned ({sol.status.name})")
        return out
    x = dict(zip(p.col_names, sol.x))
    y = dict(zip(p.row_names, sol.y)) if getattr(sol, "y", None) is not None else {}
    out["objective"] = float(sol.objective)
    tol = 1e-7
    rev = spend = hold = 0.0
    for t in tables.periods:
        per = {"period": t, "products": [], "components": [], "recipe": []}
        for pr in tables.products:
            vol = x[tables.make[pr["name"], t]]
            ov = tables.demand.get((pr["name"], t), {})
            price = ov.get("price") if ov.get("price") is not None else pr["price"]
            rec = {"name": pr["name"], "volume": vol, "revenue": price * vol, "qualities": {}}
            rev += price * vol
            for q in tables.qualities:
                carried = sum(co["qualities"][q] * x[tables.use[co["name"], pr["name"], t]]
                              for co in tables.components)
                spec = pr["specs"].get(q, {})
                val = carried / vol if vol > tol else None
                binding = None
                if val is not None:
                    if "max" in spec and abs(val - spec["max"]) <= 1e-6 * max(1.0, abs(spec["max"])):
                        binding = "max"
                    elif "min" in spec and abs(val - spec["min"]) <= 1e-6 * max(1.0, abs(spec["min"])):
                        binding = "min"
                rec["qualities"][q] = {"value": val, "min": spec.get("min"), "max": spec.get("max"),
                                       "binding": binding}
            per["products"].append(rec)
            for co in tables.components:
                q = x[tables.use[co["name"], pr["name"], t]]
                if q > tol:
                    per["recipe"].append({"product": pr["name"], "component": co["name"],
                                          "quantity": q, "fraction": q / vol if vol > tol else None})
        for co in tables.components:
            bought = x[tables.buy[co["name"], t]]
            stored = x[tables.store[co["name"], t]]
            used = sum(x[tables.use[co["name"], pr["name"], t]] for pr in tables.products)
            price = tables.prices[co["name"], t]
            spend += price * bought
            hold += co.get("storage_cost", 0.0) * stored
            per["components"].append({"name": co["name"], "buy": bought, "price": price,
                                      "use": used, "store": stored,
                                      "storage_cost": co.get("storage_cost", 0.0)})
        out["periods"].append(per)
    # by line, then along the horizon -- not the period names' lexical order
    order = {t: i for i, t in enumerate(tables.periods)}
    for (line, t), cap in sorted(tables.capacity.items(),
                                 key=lambda kv: (kv[0][0], order.get(kv[0][1], 0))):
        used = sum(x[tables.use[co["name"], pr["name"], t]]
                   for co in tables.components if co.get("line") == line for pr in tables.products)
        out["capacity"].append({"line": line, "period": t, "capacity": cap, "used": used,
                                "at_limit": abs(used - cap) <= 1e-6 * max(1.0, cap),
                                "shadow_price": y.get(f"CAP[{line}@{t}]")})
    out["totals"] = {"revenue": rev, "purchases": spend, "storage": hold}
    return _plain(out)


def planning_to_csv(plan: dict) -> str:
    """The plan as CSV, one row per period and component, then products."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["period", "item", "kind", "buy", "use", "store", "make", "revenue"])
    for per in plan.get("periods", []):
        for co in per["components"]:
            w.writerow([per["period"], co["name"], "component", f"{co['buy']:.6g}",
                        f"{co['use']:.6g}", f"{co['store']:.6g}", "", ""])
        for pr in per["products"]:
            w.writerow([per["period"], pr["name"], "product", "", "", "",
                        f"{pr['volume']:.6g}", f"{pr['revenue']:.6g}"])
    return buf.getvalue()


def planning_text(plan: dict) -> str:
    lines = [f"status: {plan['status']}"]
    if plan.get("objective") is None:
        lines.append("  " + plan.get("message", ""))
        return "\n".join(lines)
    tt = plan["totals"]
    lines.append(f"margin over the horizon: {plan['objective']:,.2f}   (revenue {tt['revenue']:,.2f}"
                 f" - purchases {tt['purchases']:,.2f} - storage {tt['storage']:,.2f})")
    for per in plan["periods"]:
        lines.append("")
        lines.append(f"  [{per['period']}]")
        lines.append(f"    {'component':<16}{'buy':>10}{'@price':>9}{'use':>10}{'store':>10}")
        for co in per["components"]:
            lines.append(f"    {co['name']:<16}{co['buy']:>10.3f}{co['price']:>9.4g}"
                         f"{co['use']:>10.3f}{co['store']:>10.3f}")
        for pr in per["products"]:
            parts = []
            for q, v in pr["qualities"].items():
                val = "-" if v["value"] is None else f"{v['value']:.4g}"
                parts.append(f"{q} {val}" + (f" *{v['binding']}" if v["binding"] else ""))
            lines.append(f"    make {pr['name']:<11}{pr['volume']:>10.3f}  revenue "
                         f"{pr['revenue']:>12,.2f}  {'; '.join(parts)}")
    binding = [c for c in plan["capacity"] if c["at_limit"]]
    if binding:
        lines.append("")
        lines.append("  capacities at their limit (shadow price per unit)")
        for c in binding:
            sp = "-" if c["shadow_price"] is None else f"{c['shadow_price']:.5g}"
            lines.append(f"    {c['line']:<12}{c['period']:<10}{c['used']:>10.3f} / {c['capacity']:<10.6g}{sp:>12}")
    return "\n".join(lines)
