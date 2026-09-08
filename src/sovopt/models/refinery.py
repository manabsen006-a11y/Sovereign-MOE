"""Refinery model generators.

Representative structures from the open literature, parameterised by size so the
solver can be exercised on models that *behave* like MRPL's without needing
proprietary data. These are the shapes that make refinery optimisation hard, and
each one stresses a different part of the engine:

``blending``
    Product blending with linear quality specifications. A dense-ish LP with a
    wide coefficient spread -- volumes in kilotonnes beside sulfur in ppm --
    which is the canonical badly-scaled matrix the numerics were built for.

``production_planning``
    Multi-period crude selection, unit throughput and product slates. Network
    structured and highly degenerate: many bases give the same objective, which
    is what makes naive simplex implementations stall.

``unit_scheduling``
    Discrete run/idle decisions per unit per period with minimum up and down
    times. Big-M constrained, so the LP relaxation is weak -- the case where a
    solver's cuts and heuristics decide everything.

The quality constraints in ``blending`` use the linear blending assumption
(properties combine by volume fraction), which holds for sulfur and density and
is what refinery LPs assume in practice. Octane and viscosity blend
non-linearly; those need the bilinear pooling treatment and are deliberately not
faked here.

References
----------
Haverly, "Studies of the behaviour of recursion for the pooling problem",
  ACM SIGMAP Bulletin 25 (1978) 19-28.
Lee, Pinto, Grossmann & Park, "Mixed-integer linear programming model for
  refinery short-term scheduling of crude oil unloading with inventory
  management", Ind. Eng. Chem. Res. 35 (1996) 1630-1641.
Pinto, Joly & Moro, "Planning and scheduling models for refinery operations",
  Computers & Chem. Eng. 24 (2000) 2259-2276.
"""

from __future__ import annotations

import numpy as np

from ..core.problem import ObjSense, Problem, VarKind
from ..core.sparse import SparseMatrix
from ..core.tolerances import INF

__all__ = ["blending", "production_planning", "unit_scheduling", "TEMPLATES"]


def blending(n_components: int = 12, n_products: int = 4,
             n_properties: int = 3, seed: int = 0) -> Problem:
    """Blend components into products subject to quality specs.

    Variables ``x[i, j]`` -- tonnes of component ``i`` going into product ``j``.
    Maximises margin (product value minus component cost).
    """
    rng = np.random.default_rng(seed)
    nc, npd, npr = n_components, n_products, n_properties
    n = nc * npd

    cost = rng.uniform(300, 700, nc)
    avail = rng.uniform(50, 400, nc)
    price = rng.uniform(600, 1100, npd)
    demand = rng.uniform(20, 120, npd)

    # property values per component; scales differ wildly on purpose
    props = np.empty((npr, nc))
    prop_scale = [1.0, 1e-4, 1e3][:npr] + [1.0] * max(0, npr - 3)
    for k in range(npr):
        props[k] = rng.uniform(0.2, 3.0, nc) * prop_scale[k]

    rows_i, rows_j, vals = [], [], []
    row_lb, row_ub, row_names = [], [], []

    def var(i, j):
        return i * npd + j

    r = 0
    # component availability:  sum_j x[i,j] <= avail_i
    for i in range(nc):
        for j in range(npd):
            rows_i.append(r); rows_j.append(var(i, j)); vals.append(1.0)
        row_lb.append(-INF); row_ub.append(float(avail[i]))
        row_names.append(f"AVAIL_{i}")
        r += 1

    # product demand: sum_i x[i,j] >= demand_j
    for j in range(npd):
        for i in range(nc):
            rows_i.append(r); rows_j.append(var(i, j)); vals.append(1.0)
        row_lb.append(float(demand[j])); row_ub.append(INF)
        row_names.append(f"DEM_{j}")
        r += 1

    # quality: lo * sum_i x <= sum_i p_ik x <= hi * sum_i x, rearranged linear
    for j in range(npd):
        for k in range(npr):
            lo = float(np.percentile(props[k], 25))
            hi = float(np.percentile(props[k], 80))
            for i in range(nc):
                rows_i.append(r); rows_j.append(var(i, j))
                vals.append(float(props[k, i]) - lo)
            row_lb.append(0.0); row_ub.append(INF)
            row_names.append(f"QLO_{j}_{k}")
            r += 1
            for i in range(nc):
                rows_i.append(r); rows_j.append(var(i, j))
                vals.append(hi - float(props[k, i]))
            row_lb.append(0.0); row_ub.append(INF)
            row_names.append(f"QHI_{j}_{k}")
            r += 1

    A = SparseMatrix.from_triplets(rows_i, rows_j, vals, r, n)
    c = np.empty(n)
    for i in range(nc):
        for j in range(npd):
            c[var(i, j)] = price[j] - cost[i]

    names = [f"X_{i}_{j}" for i in range(nc) for j in range(npd)]
    return Problem(A=A, c=c,
                   row_lb=np.array(row_lb), row_ub=np.array(row_ub),
                   col_lb=np.zeros(n), col_ub=np.full(n, INF),
                   sense=ObjSense.MAXIMISE,
                   name=f"blend_{nc}x{npd}x{npr}",
                   col_names=names, row_names=row_names)


def production_planning(n_crudes: int = 6, n_units: int = 4,
                        n_products: int = 5, n_periods: int = 6,
                        seed: int = 1) -> Problem:
    """Multi-period crude purchase, unit throughput and product output.

    Deliberately degenerate: several crudes have near-identical yields, so many
    bases attain the same objective.
    """
    rng = np.random.default_rng(seed)
    ncr, nu, npd, T = n_crudes, n_units, n_products, n_periods

    # variables: buy[cr,t] | run[u,t] | make[p,t] | stock[p,t]
    nbuy = ncr * T
    nrun = nu * T
    nmake = npd * T
    nstock = npd * T
    n = nbuy + nrun + nmake + nstock
    o_buy, o_run, o_make, o_stock = 0, nbuy, nbuy + nrun, nbuy + nrun + nmake

    yield_up = rng.uniform(0.05, 0.45, (npd, nu))
    unit_cap = rng.uniform(80, 260, nu)
    crude_cost = rng.uniform(380, 560, ncr)
    price = rng.uniform(520, 980, npd)
    hold = rng.uniform(1.0, 4.0, npd)
    dem = rng.uniform(10, 70, (npd, T))
    feed_share = rng.dirichlet(np.ones(ncr) * 3.0, nu)

    ri, rj, rv = [], [], []
    rlo, rhi, rn = [], [], []
    r = 0

    # unit capacity
    for u in range(nu):
        for t in range(T):
            ri.append(r); rj.append(o_run + u * T + t); rv.append(1.0)
            rlo.append(-INF); rhi.append(float(unit_cap[u]))
            rn.append(f"CAP_{u}_{t}")
            r += 1

    # crude balance: sum_u share[u,cr]*run[u,t] == buy[cr,t]
    for cr in range(ncr):
        for t in range(T):
            for u in range(nu):
                ri.append(r); rj.append(o_run + u * T + t)
                rv.append(float(feed_share[u, cr]))
            ri.append(r); rj.append(o_buy + cr * T + t); rv.append(-1.0)
            rlo.append(0.0); rhi.append(0.0)
            rn.append(f"CRB_{cr}_{t}")
            r += 1

    # production: sum_u yield[p,u]*run[u,t] == make[p,t]
    for p in range(npd):
        for t in range(T):
            for u in range(nu):
                ri.append(r); rj.append(o_run + u * T + t)
                rv.append(float(yield_up[p, u]))
            ri.append(r); rj.append(o_make + p * T + t); rv.append(-1.0)
            rlo.append(0.0); rhi.append(0.0)
            rn.append(f"PRD_{p}_{t}")
            r += 1

    # inventory: stock[p,t-1] + make[p,t] - dem[p,t] == stock[p,t]
    for p in range(npd):
        for t in range(T):
            ri.append(r); rj.append(o_make + p * T + t); rv.append(1.0)
            ri.append(r); rj.append(o_stock + p * T + t); rv.append(-1.0)
            if t > 0:
                ri.append(r); rj.append(o_stock + p * T + (t - 1)); rv.append(1.0)
            rlo.append(float(dem[p, t])); rhi.append(float(dem[p, t]))
            rn.append(f"INV_{p}_{t}")
            r += 1

    A = SparseMatrix.from_triplets(ri, rj, rv, r, n)
    c = np.zeros(n)
    for cr in range(ncr):
        for t in range(T):
            c[o_buy + cr * T + t] = -crude_cost[cr]
    for p in range(npd):
        for t in range(T):
            c[o_make + p * T + t] = price[p]
            c[o_stock + p * T + t] = -hold[p]

    names = ([f"BUY_{i}_{t}" for i in range(ncr) for t in range(T)]
             + [f"RUN_{u}_{t}" for u in range(nu) for t in range(T)]
             + [f"MAKE_{p}_{t}" for p in range(npd) for t in range(T)]
             + [f"STK_{p}_{t}" for p in range(npd) for t in range(T)])
    return Problem(A=A, c=c,
                   row_lb=np.array(rlo), row_ub=np.array(rhi),
                   col_lb=np.zeros(n), col_ub=np.full(n, INF),
                   sense=ObjSense.MAXIMISE,
                   name=f"plan_{ncr}c_{nu}u_{npd}p_{T}t",
                   col_names=names, row_names=rn)


def unit_scheduling(n_units: int = 5, n_periods: int = 10,
                    seed: int = 2, identical_units: bool = False) -> Problem:
    """Run/idle scheduling with minimum up-time -- a weak-relaxation MILP.

    Binary ``on[u,t]`` linked to continuous throughput by big-M, which is
    exactly the formulation that makes refinery scheduling relaxations weak.

    ``identical_units`` makes every unit the same size and cost. That is the
    ordinary situation in a refinery -- parallel trains, identical tanks -- and
    it makes the model **symmetric**: any relabelling of the units maps a
    schedule to an equally good one, so a plain branch-and-bound re-derives the
    same plan ``n_units!`` times. See :mod:`sovopt.mip.symmetry`.
    """
    rng = np.random.default_rng(seed)
    nu, T = n_units, n_periods

    if identical_units:
        cap = np.full(nu, 120.0)
        startup = np.full(nu, 800.0)
        varcost = np.full(nu, 12.0)
    else:
        cap = rng.uniform(60, 180, nu)
        startup = rng.uniform(400, 1200, nu)
        varcost = rng.uniform(6, 18, nu)
    demand = rng.uniform(0.45, 0.8, T) * cap.sum()

    non = nu * T
    nq = nu * T
    n = non + nq
    o_on, o_q = 0, non

    ri, rj, rv, rlo, rhi, rn = [], [], [], [], [], []
    r = 0

    # throughput bounded by capacity when on:  q[u,t] - cap*on[u,t] <= 0
    for u in range(nu):
        for t in range(T):
            ri.append(r); rj.append(o_q + u * T + t); rv.append(1.0)
            ri.append(r); rj.append(o_on + u * T + t); rv.append(-float(cap[u]))
            rlo.append(-INF); rhi.append(0.0)
            rn.append(f"UB_{u}_{t}")
            r += 1

    # minimum turndown when running: q[u,t] - 0.4*cap*on[u,t] >= 0
    for u in range(nu):
        for t in range(T):
            ri.append(r); rj.append(o_q + u * T + t); rv.append(1.0)
            ri.append(r); rj.append(o_on + u * T + t)
            rv.append(-0.4 * float(cap[u]))
            rlo.append(0.0); rhi.append(INF)
            rn.append(f"LB_{u}_{t}")
            r += 1

    # meet demand each period
    for t in range(T):
        for u in range(nu):
            ri.append(r); rj.append(o_q + u * T + t); rv.append(1.0)
        rlo.append(float(demand[t])); rhi.append(INF)
        rn.append(f"DEM_{t}")
        r += 1

    # minimum up time of 2 periods: on[u,t] - on[u,t-1] <= on[u,t+1]
    for u in range(nu):
        for t in range(1, T - 1):
            ri.append(r); rj.append(o_on + u * T + t); rv.append(1.0)
            ri.append(r); rj.append(o_on + u * T + t - 1); rv.append(-1.0)
            ri.append(r); rj.append(o_on + u * T + t + 1); rv.append(-1.0)
            rlo.append(-INF); rhi.append(0.0)
            rn.append(f"MINUP_{u}_{t}")
            r += 1

    A = SparseMatrix.from_triplets(ri, rj, rv, r, n)
    c = np.zeros(n)
    for u in range(nu):
        for t in range(T):
            c[o_on + u * T + t] = startup[u]
            c[o_q + u * T + t] = varcost[u]

    kind = np.zeros(n, dtype=np.uint8)
    kind[o_on:o_on + non] = VarKind.BINARY
    col_ub = np.full(n, INF)
    col_ub[o_on:o_on + non] = 1.0

    names = ([f"ON_{u}_{t}" for u in range(nu) for t in range(T)]
             + [f"Q_{u}_{t}" for u in range(nu) for t in range(T)])
    return Problem(A=A, c=c,
                   row_lb=np.array(rlo), row_ub=np.array(rhi),
                   col_lb=np.zeros(n), col_ub=col_ub, kind=kind,
                   sense=ObjSense.MINIMISE,
                   name=f"sched_{nu}u_{T}t",
                   col_names=names, row_names=rn)


TEMPLATES = {
    "blending": blending,
    "production_planning": production_planning,
    "unit_scheduling": unit_scheduling,
}
