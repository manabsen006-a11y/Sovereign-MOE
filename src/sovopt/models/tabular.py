"""Blending from a planner's tables: components and products as CSV in, a
named plan out.

The repository solves models; this is the layer that builds one from the
data a planner actually has -- a crude slate or a set of blend stocks with
costs, availabilities and qualities, and a set of products with prices,
demands and specifications -- and reads the answer back in the planner's
own terms: how much of each product to make, from what, at what quality,
and what one more tonne of each stock would be worth.

The tables
----------
Two CSV files with a header row (column names are case-insensitive;
Excel's "save as CSV" produces exactly this):

``components.csv`` -- one row per stock (crude, intermediate, purchased
component)::

    name, cost, available, minimum, <quality>, <quality>, ...

    cost       cost per unit (any consistent unit: £/t, $/bbl)
    available  the most that can be used; blank = unlimited
    minimum    the least that must be used (a must-run stream); blank = 0
    qualities  every other numeric column is a quality -- sulfur, density,
               aromatics, a blending index -- carried into the blend in
               proportion to the quantity (linear blending); the columns
               the planning and pooling layers add (line, storage_max,
               storage_cost, opening_stock, closing_stock, direct) are
               not qualities and are ignored here

``products.csv`` -- one row per product::

    name, price, demand_min, demand_max, <quality>_min, <quality>_max, ...

    price       revenue per unit
    demand_min  the least to make; blank = 0
    demand_max  the most that can be sold; blank = unlimited
    q_min/q_max the specification on quality q; blank = no limit on that
                side; a product may specify any subset of the qualities

The model
---------
For every component ``i`` and product ``j`` a flow ``x[i,j] >= 0``, and for
every product its volume ``P[j]``::

    max   Σ_j price_j P_j  −  Σ_i cost_i Σ_j x_ij
    s.t.  minimum_i <= Σ_j x_ij <= available_i          (AVAIL_i)
          Σ_i x_ij = P_j,  demand_min_j <= P_j <= demand_max_j   (BAL_j)
          Σ_i q_i x_ij <= q_max_j P_j,   Σ_i q_i x_ij >= q_min_j P_j   (SPEC)

The quality rows are the linear blending assumption -- the blend's
quality is the quantity-weighted mean of the components' -- which holds
for sulfur, density, aromatics and anything measured per unit of mass or
volume, and is what refinery LPs assume for octane and vapour pressure
through their blending indices. It is not a pooling model: a quality
that must pass through a shared pool is :mod:`pooling`'s business.

The plan
--------
:func:`blend_plan` turns a solution into a report in the planner's terms:
per product the volume, revenue and every quality achieved beside its
specification, with the binding ones marked; per component the use
against its availability and its **shadow price** -- the change in margin
per extra unit of availability, from the duals the verifier certifies;
the recipe as (product, component, quantity, fraction); and per
specification what relaxing it by one unit of quality would be worth.

    python -m sovopt.cli blend components.csv products.csv
    python -m sovopt.cli blend components.csv products.csv --plan plan.csv --out report.json

References
----------
Williams, H.P., "Model Building in Mathematical Programming", 5th ed.,
  Wiley (2013), 12.1 "Food manufacture" and 12.6 "Refinery optimisation"
  -- blending under quality bands with linear blending.
Gary, Handwerk & Kaiser, "Petroleum Refining: Technology and Economics",
  5th ed., CRC Press (2007), ch. 14 -- product blending, and the blending
  indices under which octane and vapour pressure blend linearly.
Chvátal, "Linear Programming", Freeman (1983), ch. 10 -- the shadow price
  as the dual of a binding constraint.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field

import numpy as np

from ..core.problem import ObjSense, Problem, Status
from ..core.tolerances import INF
from ._builder import Builder

__all__ = ["BlendTables", "read_blending_csv", "parse_blending_csv",
           "blending_from_tables", "blend_plan", "plan_to_csv", "plan_text"]

# the planning layer's columns are reserved here too, so a components table
# written for a horizon reads as a single-period table without them
_RESERVED_COMPONENT = {"name", "cost", "available", "minimum", "line", "storage_max",
                       "storage_cost", "opening_stock", "closing_stock", "direct"}
_RESERVED_PRODUCT = {"name", "price", "demand_min", "demand_max"}


class TableError(ValueError):
    """A table the model cannot be built from, with the reason."""


@dataclass
class BlendTables:
    """The two tables, parsed, and the model built from them."""

    components: list[dict]
    products: list[dict]
    qualities: list[str]
    problem: Problem
    flow: dict = field(default_factory=dict)      # (component, product) -> column name
    volume: dict = field(default_factory=dict)    # product -> column name


# --------------------------------------------------------------------------- #
# reading                                                                     #
# --------------------------------------------------------------------------- #

def _num(v, what):
    """A cell as a number; blank is None."""
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s == "" or s.lower() in ("na", "n/a", "-", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        raise TableError(f"{what}: {v!r} is not a number") from None


def _rows(text: str, what: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise TableError(f"{what}: the file is empty")
    fields = [f.strip().lower() for f in reader.fieldnames]
    if "name" not in fields:
        raise TableError(f"{what}: no 'name' column (columns: {', '.join(fields)})")
    rows = []
    for k, raw in enumerate(reader, 2):
        row = {}
        for key, val in raw.items():
            if key is None:
                continue
            row[key.strip().lower()] = val.strip() if isinstance(val, str) else val
        if not row.get("name"):
            if any(v for v in row.values()):
                raise TableError(f"{what} line {k}: a row without a name")
            continue                                   # a blank line
        row["_line"] = k
        rows.append(row)
    if not rows:
        raise TableError(f"{what}: no rows")
    return rows


def parse_blending_csv(components_text: str, products_text: str) -> BlendTables:
    """Build the model from the two tables' text."""
    comps = _rows(components_text, "components")
    prods = _rows(products_text, "products")

    qualities = sorted({k for r in comps for k in r
                        if k not in _RESERVED_COMPONENT and not k.startswith("_")})
    components = []
    seen = set()
    for r in comps:
        name = r["name"]
        if name in seen:
            raise TableError(f"components: {name!r} appears twice")
        seen.add(name)
        rec = {"name": name,
               "cost": _num(r.get("cost"), f"components {name} cost") or 0.0,
               "available": _num(r.get("available"), f"components {name} available"),
               "minimum": _num(r.get("minimum"), f"components {name} minimum") or 0.0,
               "qualities": {}}
        for q in qualities:
            v = _num(r.get(q), f"components {name} {q}")
            if v is None:
                raise TableError(f"components: {name} has no value for {q}; every "
                                 f"component needs every quality, or the blend's "
                                 f"{q} cannot be computed")
            rec["qualities"][q] = v
        if rec["available"] is not None and rec["available"] < rec["minimum"]:
            raise TableError(f"components: {name} has minimum {rec['minimum']:g} above "
                             f"available {rec['available']:g}")
        components.append(rec)

    products = []
    seen = set()
    for r in prods:
        name = r["name"]
        if name in seen:
            raise TableError(f"products: {name!r} appears twice")
        seen.add(name)
        rec = {"name": name,
               "price": _num(r.get("price"), f"products {name} price") or 0.0,
               "demand_min": _num(r.get("demand_min"), f"products {name} demand_min") or 0.0,
               "demand_max": _num(r.get("demand_max"), f"products {name} demand_max"),
               "specs": {}}
        for key, val in r.items():
            if key in _RESERVED_PRODUCT or key.startswith("_"):
                continue
            if key.endswith("_min") or key.endswith("_max"):
                q, side = key[:-4], key[-3:]
                if q not in qualities:
                    raise TableError(f"products: {name} specifies {key}, and no component "
                                     f"has a {q!r} column (qualities: {', '.join(qualities) or 'none'})")
                v = _num(val, f"products {name} {key}")
                if v is not None:
                    rec["specs"].setdefault(q, {})[side] = v
            else:
                raise TableError(f"products: column {key!r} is not price, demand_min, "
                                 f"demand_max or a <quality>_min / <quality>_max")
        if rec["demand_max"] is not None and rec["demand_max"] < rec["demand_min"]:
            raise TableError(f"products: {name} has demand_min {rec['demand_min']:g} above "
                             f"demand_max {rec['demand_max']:g}")
        products.append(rec)

    return blending_from_tables(components, products, qualities)


def read_blending_csv(components_path, products_path) -> BlendTables:
    """Build the model from two CSV files."""
    with open(components_path, encoding="utf-8-sig") as fh:
        c = fh.read()
    with open(products_path, encoding="utf-8-sig") as fh:
        p = fh.read()
    return parse_blending_csv(c, p)


# --------------------------------------------------------------------------- #
# the model                                                                   #
# --------------------------------------------------------------------------- #

def blending_from_tables(components, products, qualities=None) -> BlendTables:
    """The blending LP from parsed rows (the dicts :func:`parse_blending_csv`
    produces; ``qualities`` defaults to the keys of the first component)."""
    if qualities is None:
        qualities = sorted(components[0]["qualities"]) if components else []
    b = Builder("blend", ObjSense.MAXIMISE)
    flow, volume = {}, {}
    for pr in products:
        volume[pr["name"]] = b.col(
            f"MAKE[{pr['name']}]", lo=pr["demand_min"],
            hi=INF if pr["demand_max"] is None else pr["demand_max"], c=pr["price"])
    for co in components:
        for pr in products:
            flow[co["name"], pr["name"]] = b.col(
                f"USE[{co['name']}->{pr['name']}]", c=-co["cost"])
    for co in components:
        b.row(f"AVAIL[{co['name']}]",
              {flow[co["name"], pr["name"]]: 1.0 for pr in products},
              co["minimum"], INF if co["available"] is None else co["available"])
    for pr in products:
        b.eq(f"BAL[{pr['name']}]",
             {**{flow[co["name"], pr["name"]]: 1.0 for co in components},
              volume[pr["name"]]: -1.0})
        for q, spec in pr["specs"].items():
            if "max" in spec:
                b.le(f"SPEC[{pr['name']}.{q}<={spec['max']:g}]",
                     {**{flow[co["name"], pr["name"]]: co["qualities"][q] for co in components},
                      volume[pr["name"]]: -spec["max"]}, 0.0)
            if "min" in spec:
                b.ge(f"SPEC[{pr['name']}.{q}>={spec['min']:g}]",
                     {**{flow[co["name"], pr["name"]]: co["qualities"][q] for co in components},
                      volume[pr["name"]]: -spec["min"]}, 0.0)
    return BlendTables(components, products, list(qualities), b.problem(), flow, volume)


# --------------------------------------------------------------------------- #
# the plan                                                                    #
# --------------------------------------------------------------------------- #

def blend_plan(tables: BlendTables, sol) -> dict:
    """The solution in the planner's terms. Shadow prices are the duals as
    the engine reports them (``d = c − Aᵀy``, the sign the simplex uses
    and the verifier certifies): for a maximisation, a component's is the
    margin gained per extra unit of availability, a specification's the
    margin gained per unit of quality the limit is moved *toward*
    feasibility, and a product's demand bound's the margin per extra unit
    that could be sold."""
    p = tables.problem
    out = {"status": sol.status.name, "objective": None, "products": [], "components": [],
           "recipe": [], "specs": [], "demand": []}
    if sol.x is None or not Status(sol.status).has_solution:
        # an INFEASIBLE or UNBOUNDED exit keeps its last iterate for
        # diagnostics; it is not a plan
        out["message"] = ("no feasible plan: the specifications and demands cannot be met "
                          "from these components" if sol.status == Status.INFEASIBLE
                          else "the margin is unbounded: some product has no demand limit "
                               "and a component with no availability limit that improves it"
                          if sol.status in (Status.UNBOUNDED, Status.INFEASIBLE_OR_UNBOUNDED)
                          else f"no plan returned ({sol.status.name})")
        return out
    x = dict(zip(p.col_names, sol.x))
    y = dict(zip(p.row_names, sol.y)) if getattr(sol, "y", None) is not None else {}
    d = (dict(zip(p.col_names, sol.reduced_costs))
         if getattr(sol, "reduced_costs", None) is not None else {})
    out["objective"] = float(sol.objective)
    tol = 1e-7

    for pr in tables.products:
        vol = x[tables.volume[pr["name"]]]
        rec = {"name": pr["name"], "volume": vol, "price": pr["price"],
               "revenue": pr["price"] * vol, "qualities": {}}
        for q in tables.qualities:
            carried = sum(co["qualities"][q] * x[tables.flow[co["name"], pr["name"]]]
                          for co in tables.components)
            value = carried / vol if vol > tol else None
            spec = pr["specs"].get(q, {})
            binding = None
            if value is not None:
                if "max" in spec and abs(value - spec["max"]) <= 1e-6 * max(1.0, abs(spec["max"])):
                    binding = "max"
                elif "min" in spec and abs(value - spec["min"]) <= 1e-6 * max(1.0, abs(spec["min"])):
                    binding = "min"
            rec["qualities"][q] = {"value": value, "min": spec.get("min"),
                                   "max": spec.get("max"), "binding": binding}
        out["products"].append(rec)
        dm = pr["demand_max"]
        at_max = dm is not None and abs(vol - dm) <= 1e-6 * max(1.0, dm)
        at_min = abs(vol - pr["demand_min"]) <= 1e-6 * max(1.0, pr["demand_min"]) and pr["demand_min"] > 0
        out["demand"].append({"product": pr["name"], "volume": vol,
                              "demand_min": pr["demand_min"], "demand_max": dm,
                              "at": "max" if at_max else ("min" if at_min else None),
                              "shadow_price": d.get(tables.volume[pr["name"]])})

    for co in tables.components:
        used = sum(x[tables.flow[co["name"], pr["name"]]] for pr in tables.products)
        avail = co["available"]
        at_avail = avail is not None and abs(used - avail) <= 1e-6 * max(1.0, avail)
        out["components"].append({
            "name": co["name"], "used": used, "available": avail, "minimum": co["minimum"],
            "cost": co["cost"], "spend": co["cost"] * used, "at_limit": at_avail,
            "shadow_price": y.get(f"AVAIL[{co['name']}]")})

    for pr in tables.products:
        vol = x[tables.volume[pr["name"]]]
        for co in tables.components:
            q = x[tables.flow[co["name"], pr["name"]]]
            if q > tol:
                out["recipe"].append({"product": pr["name"], "component": co["name"],
                                      "quantity": q, "fraction": q / vol if vol > tol else None})

    # A specification's dual is per unit of the row's right-hand side; the
    # planner's question is per unit of *quality*, and moving the limit by
    # one unit of quality moves the right-hand side by the product's
    # volume: value per unit of quality = dual x volume (checked against a
    # re-solve on the example: 520 x 2,500 x 0.001 = 1,300 for one
    # thousandth of sulfur on Premium). A product that is not made has a
    # tight row at zero volume, which is degeneracy, not a binding spec.
    for name in p.row_names:
        if name.startswith("SPEC["):
            inner = name[5:name.index(".")]
            vol = x[tables.volume[inner]] if inner in tables.volume else 0.0
            dual = y.get(name)
            out["specs"].append({
                "row": name, "product": inner, "shadow_price": dual,
                "per_unit_of_quality": None if dual is None else dual * vol,
                "binding": bool(dual) and abs(dual) > 1e-9 and vol > tol})

    out["revenue"] = sum(r["revenue"] for r in out["products"])
    out["cost"] = sum(c["spend"] for c in out["components"])
    return _plain(out)


def _plain(obj):
    """numpy scalars to Python ones, so the plan serialises as JSON."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def plan_to_csv(plan: dict) -> str:
    """The recipe as CSV: product, component, quantity, fraction."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["product", "component", "quantity", "fraction"])
    for r in plan.get("recipe", []):
        w.writerow([r["product"], r["component"], f"{r['quantity']:.6g}",
                    "" if r["fraction"] is None else f"{r['fraction']:.4f}"])
    return buf.getvalue()


def _f(v, digits=6):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{digits}g}"


def plan_text(plan: dict) -> str:
    """The plan as the CLI prints it."""
    lines = [f"status: {plan['status']}"]
    if plan.get("objective") is None:
        lines.append("  " + plan.get("message", ""))
        return "\n".join(lines)
    lines.append(f"margin: {plan['objective']:,.2f}   (revenue {plan['revenue']:,.2f} "
                 f"- component cost {plan['cost']:,.2f})")
    lines.append("")
    lines.append(f"  {'product':<16}{'volume':>12}{'revenue':>14}  qualities (value | spec)")
    for pr in plan["products"]:
        qs = "; ".join(
            f"{q} {_f(v['value'], 4)}"
            + (f" [{_f(v['min'], 4)}..{_f(v['max'], 4)}]" if (v["min"] is not None or v["max"] is not None) else "")
            + (f" *{v['binding']}" if v["binding"] else "")
            for q, v in pr["qualities"].items())
        lines.append(f"  {pr['name']:<16}{pr['volume']:>12.4f}{pr['revenue']:>14,.2f}  {qs}")
    lines.append("")
    lines.append(f"  {'component':<16}{'used':>12}{'available':>12}{'cost':>10}{'shadow price':>14}")
    for co in plan["components"]:
        lines.append(f"  {co['name']:<16}{co['used']:>12.4f}{_f(co['available']):>12}"
                     f"{co['cost']:>10.4g}{_f(co['shadow_price'], 5):>14}"
                     + ("  at limit" if co["at_limit"] else ""))
    lines.append("")
    lines.append("  recipe")
    for r in plan["recipe"]:
        lines.append(f"    {r['product']:<16}<- {r['component']:<16}{r['quantity']:>12.4f}"
                     f"  ({100 * r['fraction']:.1f}%)" if r["fraction"] is not None else "")
    binding = [s for s in plan["specs"] if s.get("binding")]
    if binding:
        lines.append("")
        lines.append("  binding specifications (margin per unit the limit is relaxed)")
        for s in binding:
            lines.append(f"    {s['row']:<40}{s['per_unit_of_quality']:>14,.5g}")
    return "\n".join(lines)
