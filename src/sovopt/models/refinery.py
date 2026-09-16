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

``williams_refinery``
    Not a generator: the one refinery LP in the open literature with a
    published optimum, from Williams' textbook (problem 12.6, solution
    13.6). Two crudes, distillation, reforming, cracking, lube oil, two
    petrol grades under octane specifications, jet fuel under a vapour
    pressure specification, a fixed-recipe fuel oil. Optimal profit
    211,365.13 per day. A textbook page, but every coefficient of it is
    printed and the answer is, which makes it the only refinery model here
    that checks the solver against a number nobody in this repository
    produced.

The quality constraints in ``blending`` use the linear blending assumption
(properties combine by volume fraction), which holds for sulfur and density and
is what refinery LPs assume in practice. Octane and viscosity blend
non-linearly; those need the bilinear pooling treatment and are deliberately not
faked here.

References
----------
Williams, "Model Building in Mathematical Programming", 5th ed., Wiley
  (2013), problem 12.6 "Refinery optimisation" and its solution 13.6.
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

__all__ = ["blending", "production_planning", "unit_scheduling",
           "williams_refinery", "WILLIAMS_OPTIMUM", "TEMPLATES"]

WILLIAMS_OPTIMUM = 211365.13
"""Optimal daily profit of :func:`williams_refinery`, as printed in the
book's solution 13.6 (to the penny)."""


def williams_refinery() -> Problem:
    """Williams' refinery LP (problem 12.6), every coefficient as printed.

    Barrels per day throughout. The flows are: two crudes into distillation
    (capacity 45,000; crude 1 at most 20,000, crude 2 at most 30,000);
    the six distillates -- light, medium and heavy naphtha, light and
    heavy oil, residuum -- at the book's yields; naphthas optionally
    reformed (capacity 10,000) into reformed gasoline; oils optionally
    cracked (capacity 8,000) into cracked oil and cracked gasoline;
    residuum optionally into lube oil at a half barrel per barrel, with
    lube production between 500 and 1,000. Naphthas, reformed and cracked
    gasoline blend into premium (octane at least 94) and regular (at least
    84) petrol, premium at least 40% of regular by volume. Light oil, heavy
    oil, cracked oil and residuum blend into jet fuel (vapour pressure at
    most 1) and, in the fixed ratio 10 : 3 : 4 : 1, into fuel oil.
    Contributions per barrel: premium 7, regular 6, jet fuel 4, fuel oil
    3.5, lube oil 1.5.
    """
    cols: list[str] = []
    lb: list[float] = []
    ub: list[float] = []
    cost: dict[str, float] = {}

    def col(name, lo=0.0, hi=INF, c=0.0):
        cols.append(name); lb.append(lo); ub.append(hi)
        if c:
            cost[name] = c
        return name

    # crudes and distillates
    CR1 = col("CR1", hi=20000.0)
    CR2 = col("CR2", hi=30000.0)
    LN, MN, HN, LO, HO, R = (col(s) for s in ("LN", "MN", "HN", "LO", "HO", "R"))
    # reforming, cracking, lube
    LNRG, MNRG, HNRG, RG = (col(s) for s in ("LNRG", "MNRG", "HNRG", "RG"))
    LOCGO, HOCGO, CO, CG = (col(s) for s in ("LOCGO", "HOCGO", "CO", "CG"))
    RLBO = col("RLBO")
    LBO = col("LBO", lo=500.0, hi=1000.0, c=1.5)
    # blending streams into premium (PMF) and regular (RMF) petrol
    blend_pmf = {s: col(f"{s}PMF") for s in ("LN", "MN", "HN", "RG", "CG")}
    blend_rmf = {s: col(f"{s}RMF") for s in ("LN", "MN", "HN", "RG", "CG")}
    PMF = col("PMF", c=7.0)
    RMF = col("RMF", c=6.0)
    # jet fuel and fuel oil
    blend_jf = {s: col(f"{s}JF") for s in ("LO", "HO", "CO", "R")}
    JF = col("JF", c=4.0)
    blend_fo = {s: col(f"{s}FO") for s in ("LO", "HO", "CO", "R")}
    FO = col("FO", c=3.5)

    j = {name: k for k, name in enumerate(cols)}
    rows_i: list[int] = []
    rows_j: list[int] = []
    vals: list[float] = []
    row_lb: list[float] = []
    row_ub: list[float] = []
    row_names: list[str] = []

    def row(name, terms, lo, hi):
        r = len(row_names)
        for v, a in terms.items():
            rows_i.append(r); rows_j.append(j[v]); vals.append(float(a))
        row_lb.append(lo); row_ub.append(hi); row_names.append(name)

    def eq(name, terms):
        row(name, terms, 0.0, 0.0)

    # distillation yields per barrel of crude 1 / crude 2
    yields = {LN: (0.10, 0.15), MN: (0.20, 0.25), HN: (0.20, 0.18),
              LO: (0.12, 0.08), HO: (0.20, 0.19), R: (0.13, 0.12)}
    for d, (y1, y2) in yields.items():
        eq(f"DIST_{d}", {CR1: y1, CR2: y2, d: -1.0})
    row("DIST_CAP", {CR1: 1.0, CR2: 1.0}, -INF, 45000.0)

    # reforming: reformed gasoline per barrel of naphtha
    eq("REFORM", {LNRG: 0.60, MNRG: 0.52, HNRG: 0.45, RG: -1.0})
    row("REFORM_CAP", {LNRG: 1.0, MNRG: 1.0, HNRG: 1.0}, -INF, 10000.0)
    # cracking: cracked oil and cracked gasoline per barrel of oil
    eq("CRACK_CO", {LOCGO: 0.68, HOCGO: 0.75, CO: -1.0})
    eq("CRACK_CG", {LOCGO: 0.28, HOCGO: 0.20, CG: -1.0})
    row("CRACK_CAP", {LOCGO: 1.0, HOCGO: 1.0}, -INF, 8000.0)
    # lube oil: half a barrel per barrel of residuum
    eq("LUBE", {RLBO: 0.5, LBO: -1.0})

    # material balances on every intermediate
    eq("BAL_LN", {LN: 1.0, LNRG: -1.0, blend_pmf["LN"]: -1.0, blend_rmf["LN"]: -1.0})
    eq("BAL_MN", {MN: 1.0, MNRG: -1.0, blend_pmf["MN"]: -1.0, blend_rmf["MN"]: -1.0})
    eq("BAL_HN", {HN: 1.0, HNRG: -1.0, blend_pmf["HN"]: -1.0, blend_rmf["HN"]: -1.0})
    eq("BAL_LO", {LO: 1.0, LOCGO: -1.0, blend_jf["LO"]: -1.0, blend_fo["LO"]: -1.0})
    eq("BAL_HO", {HO: 1.0, HOCGO: -1.0, blend_jf["HO"]: -1.0, blend_fo["HO"]: -1.0})
    eq("BAL_R", {R: 1.0, RLBO: -1.0, blend_jf["R"]: -1.0, blend_fo["R"]: -1.0})
    eq("BAL_RG", {RG: 1.0, blend_pmf["RG"]: -1.0, blend_rmf["RG"]: -1.0})
    eq("BAL_CO", {CO: 1.0, blend_jf["CO"]: -1.0, blend_fo["CO"]: -1.0})
    eq("BAL_CG", {CG: 1.0, blend_pmf["CG"]: -1.0, blend_rmf["CG"]: -1.0})

    # products are the sums of their blending streams
    eq("SUM_PMF", {**{v: 1.0 for v in blend_pmf.values()}, PMF: -1.0})
    eq("SUM_RMF", {**{v: 1.0 for v in blend_rmf.values()}, RMF: -1.0})
    eq("SUM_JF", {**{v: 1.0 for v in blend_jf.values()}, JF: -1.0})
    # fuel oil is a fixed recipe: light oil : heavy oil : cracked oil :
    # residuum = 10 : 3 : 4 : 1, so each stream is its share of the product
    for s, share in (("LO", 10.0), ("HO", 3.0), ("CO", 4.0), ("R", 1.0)):
        eq(f"FO_{s}", {blend_fo[s]: 18.0, FO: -share})

    # octane: the blend's number is the volume-weighted mean of the streams'
    octane = {"LN": 90.0, "MN": 80.0, "HN": 70.0, "RG": 115.0, "CG": 105.0}
    row("OCT_PMF", {v: octane[s] - 94.0 for s, v in blend_pmf.items()}, 0.0, INF)
    row("OCT_RMF", {v: octane[s] - 84.0 for s, v in blend_rmf.items()}, 0.0, INF)
    # vapour pressure of jet fuel, same linear blending
    vp = {"LO": 1.0, "HO": 0.6, "CO": 1.5, "R": 0.05}
    row("VP_JF", {v: 1.0 - vp[s] for s, v in blend_jf.items()}, 0.0, INF)
    # premium at least 40% of regular
    row("PMF_RMF", {PMF: 1.0, RMF: -0.4}, 0.0, INF)

    n = len(cols)
    A = SparseMatrix.from_triplets(rows_i, rows_j, vals, len(row_names), n)
    c = np.zeros(n)
    for name, v in cost.items():
        c[j[name]] = v
    return Problem(A=A, c=c,
                   row_lb=np.array(row_lb), row_ub=np.array(row_ub),
                   col_lb=np.array(lb), col_ub=np.array(ub),
                   sense=ObjSense.MAXIMISE, name="williams_refinery",
                   col_names=cols, row_names=row_names)


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
