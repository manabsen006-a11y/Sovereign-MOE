"""Pooled qualities from a planner's tables: components through pools to
products, solved to proven global optimality.

A blending table assumes every component reaches every product in its
own pipe. A refinery has tanks and headers -- pools -- in the way: the
crudes into a pool mix, the pool has one quality, and that quality is
carried to every product the pool feeds in proportion to the flow. The
carried quality is then a *product of decisions* (the pool's quality
times the flow), the model is bilinear, and the fixed-point iteration
industry practice uses for it can stop at a blend worth less than the
best one. This layer builds the pq-formulation of :mod:`pooling_pq` from
three tables and sends it to the spatial branch-and-bound, which proves
its answer.

The tables
----------
``components.csv`` -- as in :mod:`tabular` (name, cost, available,
qualities), plus::

    direct    the products this component may ship to straight, without a
              pool: names separated by ";", "*" for every product, blank
              for none (it reaches products through pools only)

``pools.csv``::

    name, capacity, inputs        inputs: component names separated by ";",
                                  "*" or blank for every component

``products.csv`` -- as in :mod:`tabular`: name, price, demand_max (the
most that can be sold; ``demand_min`` is not part of the pooling model
and is refused), and ``<quality>_max`` / ``<quality>_min``
specifications (a minimum is carried as a maximum on the negated quality,
which is the same linear statement).

Every pool feeds every product. A component that appears in no pool's
inputs and has no direct products is refused: it could go nowhere.

Checked against the literature: Haverly's problem written as these
tables (``examples/pooling``) solves to the published global optimum of
400, with the search closed.

References
----------
Haverly, "Studies of the behaviour of recursion for the pooling problem",
  ACM SIGMAP Bulletin 25 (1978) 19-28.
Ben-Tal, Eiger & Gershovitz, "Global minimization by reducing the duality
  gap", Math. Prog. 63 (1994) 193-212 -- the q-formulation the tables are
  written into.
Tawarmalani & Sahinidis, *Convexification and Global Optimization*, Kluwer
  2002, ch. 9 -- the pq-formulation.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import numpy as np

from ..core.problem import Status
from ..globalopt.bilinear import BilinearProblem
from .pooling_pq import PQIndex, PoolingData, pooling_pq
from .tabular import TableError, _num, _plain, _rows, parse_blending_csv

__all__ = ["PoolTables", "parse_pooling_csv", "read_pooling_csv",
           "pooling_from_tables", "pooling_plan", "pooling_text", "pooling_to_csv"]


@dataclass
class PoolTables:
    components: list[dict]
    pools: list[dict]
    products: list[dict]
    qualities: list[str]              # the planner's qualities
    data: PoolingData
    problem: BilinearProblem
    index: PQIndex = None
    columns: list[str] = field(default_factory=list)


def _names(cell, universe, what, allow_star=True):
    """A ';'-separated list of names, checked against ``universe``."""
    s = (cell or "").strip()
    if s == "":
        return []
    if s == "*" and allow_star:
        return list(universe)
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        if part not in universe:
            raise TableError(f"{what}: {part!r} is not a known name "
                             f"(known: {', '.join(universe)})")
        if part not in out:
            out.append(part)
    return out


def parse_pooling_csv(components_text, products_text, pools_text) -> PoolTables:
    """Build the pooling model from the three tables' text."""
    base = parse_blending_csv(components_text, products_text)
    comps_raw = {r["name"]: r for r in _rows(components_text, "components")}
    cnames = [c["name"] for c in base.components]
    pnames = [p["name"] for p in base.products]
    for pr in base.products:
        if pr["demand_min"]:
            raise TableError(f"products: {pr['name']} has demand_min {pr['demand_min']:g}; the "
                             f"pooling model takes demand_max only")
    components = []
    for co in base.components:
        if co.get("allowed") is not None:
            # a pool mixes its inputs and feeds every product, so which
            # products one component may reach is not the pool model's to
            # restrict -- refused rather than dropped without a word
            raise TableError(f"components: {co['name']} has an 'allowed' list; the pooling "
                             f"model routes by 'direct' and the pools' inputs instead")
        rec = dict(co)
        rec["direct"] = _names(comps_raw[co["name"]].get("direct"), pnames,
                               f"components {co['name']} direct")
        components.append(rec)
    pools = []
    seen = set()
    for r in _rows(pools_text, "pools"):
        name = r["name"]
        if name in seen:
            raise TableError(f"pools: {name!r} appears twice")
        seen.add(name)
        cap = _num(r.get("capacity"), f"pools {name} capacity")
        inputs = _names(r.get("inputs"), cnames, f"pools {name} inputs") or list(cnames)
        pools.append({"name": name, "capacity": cap, "inputs": inputs})
    fed = {c for pl in pools for c in pl["inputs"]}
    for co in components:
        if co["name"] not in fed and not co["direct"]:
            raise TableError(f"components: {co['name']} is in no pool's inputs and has no "
                             f"direct products; it could go nowhere")
    return pooling_from_tables(components, pools, base.products, base.qualities)


def read_pooling_csv(components_path, products_path, pools_path) -> PoolTables:
    def rd(p):
        with open(p, encoding="utf-8-sig") as fh:
            return fh.read()
    return parse_pooling_csv(rd(components_path), rd(products_path), rd(pools_path))


def pooling_from_tables(components, pools, products, qualities) -> PoolTables:
    """The pq-formulation from parsed rows. A ``q_min`` specification is a
    ``q_max`` on ``-q``; a product with no specification on a quality gets
    an unreachable one."""
    ns, npl, nk = len(components), len(pools), len(products)
    # the quality columns the pq model sees: q for every max spec anywhere,
    # -q for every min spec anywhere
    cols = []
    for q in qualities:
        if any("max" in pr["specs"].get(q, {}) for pr in products):
            cols.append((q, 1.0))
        if any("min" in pr["specs"].get(q, {}) for pr in products):
            cols.append((q, -1.0))
    if not cols:
        cols = [(qualities[0], 1.0)] if qualities else []
    big = 0.0
    for co in components:
        for q, sgn in cols:
            big = max(big, abs(sgn * co["qualities"][q]))
    slack = 10.0 * big + 1.0
    quality = np.array([[sgn * co["qualities"][q] for q, sgn in cols] for co in components],
                       dtype=np.float64).reshape(ns, len(cols))
    spec = np.empty((nk, len(cols)), dtype=np.float64)
    for k, pr in enumerate(products):
        for c, (q, sgn) in enumerate(cols):
            side = "max" if sgn > 0 else "min"
            v = pr["specs"].get(q, {}).get(side)
            spec[k, c] = sgn * v if v is not None else slack
    feed = np.array([[co["name"] in pl["inputs"] for pl in pools] for co in components],
                    dtype=bool).reshape(ns, npl)
    direct = np.array([[pr["name"] in co["direct"] for pr in products] for co in components],
                      dtype=bool).reshape(ns, nk)
    inf = 1e30
    d = PoolingData(
        cost=np.array([co["cost"] for co in components], dtype=np.float64),
        avail=np.array([inf if co["available"] is None else co["available"]
                        for co in components], dtype=np.float64),
        quality=quality,
        pool_cap=np.array([inf if pl["capacity"] is None else pl["capacity"] for pl in pools],
                          dtype=np.float64),
        price=np.array([pr["price"] for pr in products], dtype=np.float64),
        demand=np.array([inf if pr["demand_max"] is None else pr["demand_max"]
                         for pr in products], dtype=np.float64),
        spec=spec, bypass=bool(direct.any()), feed=feed, direct=direct)
    bp = pooling_pq(d, rlt=True, name="pool")
    idx = PQIndex(d)
    names = [None] * idx.n
    for i, co in enumerate(components):
        for j, pl in enumerate(pools):
            names[idx.q(i, j)] = f"SHARE[{co['name']}@{pl['name']}]"
            for k, pr in enumerate(products):
                names[idx.v(i, j, k)] = f"VIA[{co['name']}@{pl['name']}->{pr['name']}]"
        if d.bypass:
            for k, pr in enumerate(products):
                names[idx.z(i, k)] = f"DIRECT[{co['name']}->{pr['name']}]"
    for j, pl in enumerate(pools):
        for k, pr in enumerate(products):
            names[idx.y(j, k)] = f"POOL[{pl['name']}->{pr['name']}]"
    return PoolTables(components, pools, products, list(qualities), d, bp, idx, names)


# --------------------------------------------------------------------------- #
# the plan                                                                    #
# --------------------------------------------------------------------------- #

def pooling_plan(tables: PoolTables, sol) -> dict:
    """The solution in the planner's terms: pool compositions and qualities,
    flows on every arc, products with their qualities against the specs,
    and whether the search proved the plan globally optimal."""
    out = {"status": sol.status.name, "objective": None, "proved_global": False,
           "pools": [], "products": [], "components": [], "flows": []}
    if sol.x is None or not Status(sol.status).has_solution:
        out["message"] = ("no feasible plan: the specifications and demands cannot be met "
                          "through these pools" if sol.status == Status.INFEASIBLE
                          else f"no plan returned ({sol.status.name})")
        return out
    x = np.asarray(sol.x, dtype=np.float64)
    idx, d = tables.index, tables.data
    comps, pools, prods = tables.components, tables.pools, tables.products
    tol = 1e-7
    out["objective"] = float(sol.objective)
    out["proved_global"] = sol.status == Status.OPTIMAL
    out["dual_bound"] = float(sol.dual_bound) if np.isfinite(getattr(sol, "dual_bound", np.nan)) else None
    out["nodes"] = int(getattr(sol, "nodes", 0))

    into_pool = np.zeros((len(comps), len(pools)))
    for j, pl in enumerate(pools):
        total = sum(x[idx.y(j, k)] for k in range(len(prods)))
        comp = {}
        for i, co in enumerate(comps):
            share = x[idx.q(i, j)] if d.feed_mask()[i, j] else 0.0
            into_pool[i, j] = share * total
            if share > tol:
                comp[co["name"]] = share
        qual = {q: sum(x[idx.q(i, j)] * co["qualities"][q] for i, co in enumerate(comps)
                       if d.feed_mask()[i, j]) for q in tables.qualities} if total > tol else {}
        out["pools"].append({"name": pl["name"], "throughput": total, "capacity": pl["capacity"],
                             "composition": comp, "qualities": qual,
                             "at_capacity": pl["capacity"] is not None
                             and abs(total - pl["capacity"]) <= 1e-6 * max(1.0, pl["capacity"])})
        for i, co in enumerate(comps):
            if into_pool[i, j] > tol:
                out["flows"].append({"from": co["name"], "to": pl["name"], "quantity": into_pool[i, j]})
        for k, pr in enumerate(prods):
            if x[idx.y(j, k)] > tol:
                out["flows"].append({"from": pl["name"], "to": pr["name"], "quantity": x[idx.y(j, k)]})
    for i, co in enumerate(comps):
        if d.bypass:
            for k, pr in enumerate(prods):
                if d.direct_mask()[i, k] and x[idx.z(i, k)] > tol:
                    out["flows"].append({"from": co["name"], "to": pr["name"],
                                         "quantity": x[idx.z(i, k)]})
    for k, pr in enumerate(prods):
        vol = sum(x[idx.y(j, k)] for j in range(len(pools)))
        carried = {q: 0.0 for q in tables.qualities}
        for j in range(len(pools)):
            for i, co in enumerate(comps):
                if d.feed_mask()[i, j]:
                    for q in tables.qualities:
                        carried[q] += co["qualities"][q] * x[idx.v(i, j, k)]
        if d.bypass:
            for i, co in enumerate(comps):
                if d.direct_mask()[i, k]:
                    vol += x[idx.z(i, k)]
                    for q in tables.qualities:
                        carried[q] += co["qualities"][q] * x[idx.z(i, k)]
        rec = {"name": pr["name"], "volume": vol, "revenue": pr["price"] * vol, "qualities": {}}
        for q in tables.qualities:
            spec = pr["specs"].get(q, {})
            val = carried[q] / vol if vol > tol else None
            binding = None
            if val is not None:
                if "max" in spec and abs(val - spec["max"]) <= 1e-6 * max(1.0, abs(spec["max"])):
                    binding = "max"
                elif "min" in spec and abs(val - spec["min"]) <= 1e-6 * max(1.0, abs(spec["min"])):
                    binding = "min"
            rec["qualities"][q] = {"value": val, "min": spec.get("min"), "max": spec.get("max"),
                                   "binding": binding}
        out["products"].append(rec)
    for i, co in enumerate(comps):
        used = float(into_pool[i].sum())
        if d.bypass:
            used += sum(x[idx.z(i, k)] for k in range(len(prods)) if d.direct_mask()[i, k])
        out["components"].append({"name": co["name"], "used": used, "available": co["available"],
                                  "cost": co["cost"], "spend": co["cost"] * used,
                                  "at_limit": co["available"] is not None
                                  and abs(used - co["available"]) <= 1e-6 * max(1.0, co["available"])})
    out["revenue"] = sum(r["revenue"] for r in out["products"])
    out["cost"] = sum(c["spend"] for c in out["components"])
    return _plain(out)


def pooling_to_csv(plan: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["from", "to", "quantity"])
    for f in plan.get("flows", []):
        w.writerow([f["from"], f["to"], f"{f['quantity']:.6g}"])
    return buf.getvalue()


def pooling_text(plan: dict) -> str:
    lines = [f"status: {plan['status']}" + ("  (proved globally optimal)" if plan.get("proved_global")
                                              else "")]
    if plan.get("objective") is None:
        lines.append("  " + plan.get("message", ""))
        return "\n".join(lines)
    lines.append(f"margin: {plan['objective']:,.2f}   (revenue {plan['revenue']:,.2f} - "
                 f"component cost {plan['cost']:,.2f})"
                 + (f"; bound {plan['dual_bound']:,.2f}, {plan['nodes']} nodes"
                    if plan.get("dual_bound") is not None else ""))
    lines.append("")
    lines.append("  pools")
    for pl in plan["pools"]:
        comp = ", ".join(f"{c} {100 * s:.1f}%" for c, s in pl["composition"].items()) or "idle"
        qs = "; ".join(f"{q} {v:.4g}" for q, v in pl["qualities"].items())
        cap = "-" if pl["capacity"] is None else f"{pl['capacity']:g}"
        lines.append(f"    {pl['name']:<12}{pl['throughput']:>10.3f} / {cap:<10}{comp}"
                     + (f"  [{qs}]" if qs else "") + ("  at capacity" if pl["at_capacity"] else ""))
    lines.append("")
    lines.append("  products")
    for pr in plan["products"]:
        parts = []
        for q, v in pr["qualities"].items():
            val = "-" if v["value"] is None else f"{v['value']:.4g}"
            band = f" [{'-' if v['min'] is None else v['min']}..{'-' if v['max'] is None else v['max']}]" \
                if (v["min"] is not None or v["max"] is not None) else ""
            parts.append(f"{q} {val}{band}" + (f" *{v['binding']}" if v["binding"] else ""))
        lines.append(f"    {pr['name']:<12}{pr['volume']:>10.3f}  revenue {pr['revenue']:>12,.2f}  "
                     + "; ".join(parts))
    lines.append("")
    lines.append("  flows")
    for f in plan["flows"]:
        lines.append(f"    {f['from']:<14}-> {f['to']:<14}{f['quantity']:>10.3f}")
    return "\n".join(lines)
