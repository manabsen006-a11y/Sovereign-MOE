"""Multi-pool pooling in the pq-formulation.

Three ways to write the same physical network, in increasing relaxation
strength:

**p-formulation** -- the natural one, used by :func:`sovopt.models.pooling.haverly`.
The unknowns are pool *qualities*, and the products are quality times flow. It
is the easiest to read and the weakest to relax.

**q-formulation** (Ben-Tal, Eiger & Gershovitz) -- replace pool qualities with
the *proportions* ``q_ij`` of each source in each pool. Quality becomes linear in
the proportions, and the only products left are

    v_ijk = q_ij * y_jk        flow of source i, through pool j, to product k

**pq-formulation** -- the q-formulation plus the reformulation-linearisation
rows obtained by multiplying the proportion identity through by each outgoing
flow:

    (sum_i q_ij) * y_jk  =  y_jk        becomes        sum_i v_ijk = y_jk

That is a *linear* constraint on variables the model already has, and it is what
makes the relaxation tight. Without it each auxiliary ``v_ijk`` floats
independently inside its own McCormick box; with it they are tied to the flow
they decompose, and the relaxation can no longer buy cheap quality from one
auxiliary while selling volume from another. The rows cost almost nothing and
are usually the difference between a search that closes and one that does not.

Both variants describe the **same feasible set and the same global optimum**;
they differ only in the bound. ``pooling_pq(..., rlt=False)`` builds the plain
q-formulation, which is what makes that difference measurable.

Why multi-pool matters here
---------------------------
A real refinery has many pools -- tanks, headers, intermediate streams -- and
the interesting non-convexity is precisely that several pools feed the same
product. A single-pool instance (Haverly) is the right unit test; it is not the
industrial case.

References
----------
Ben-Tal, Eiger & Gershovitz, "Global minimization by reducing the duality gap",
  Math. Prog. 63 (1994) 193-212 -- the q-formulation.
Quesada & Grossmann, "Global optimization of bilinear process networks with
  multicomponent flows", Computers & Chem. Eng. 19 (1995) 1219-1242.
Tawarmalani & Sahinidis, *Convexification and Global Optimization*, Kluwer 2002,
  ch. 9 -- the pq-formulation and why the RLT rows dominate.
Misener & Floudas, "Advances for the pooling problem", Applied and Computational
  Mathematics 8 (2009) 3-22 -- computational comparison of the formulations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.problem import ObjSense, Problem
from ..core.sparse import SparseMatrix
from ..core.tolerances import INF
from ..globalopt.bilinear import BilinearProblem, BilinearTerm

__all__ = ["PoolingData", "pooling_pq", "haverly_pq", "random_pooling"]


@dataclass
class PoolingData:
    """A generalised pooling network."""

    cost: np.ndarray            # (n_src,)
    avail: np.ndarray           # (n_src,)
    quality: np.ndarray         # (n_src, n_qual)
    pool_cap: np.ndarray        # (n_pool,)
    price: np.ndarray           # (n_prod,)
    demand: np.ndarray          # (n_prod,)
    spec: np.ndarray            # (n_prod, n_qual), maximum allowed
    bypass: bool = True

    feed: np.ndarray | None = None
    """(n_src, n_pool) boolean: which sources may feed which pools."""

    direct: np.ndarray | None = None
    """(n_src, n_prod) boolean: which sources may ship straight to a product.

    **Not every source may bypass every pool.** Getting this wrong does not
    produce an infeasible model, it produces an *easier* one: if every source
    could ship direct there would be no pooling decision at all, the problem
    would collapse to an LP, and the solver would confidently report an
    objective better than the true optimum. That is exactly how the first
    version of this builder was caught -- it "solved" Haverly to 500 against a
    published 400, by routing everything around the pool.
    """

    def feed_mask(self):
        if self.feed is None:
            return np.ones((self.n_src, self.n_pool), dtype=bool)
        return np.asarray(self.feed, dtype=bool)

    def direct_mask(self):
        if not self.bypass:
            return np.zeros((self.n_src, self.n_prod), dtype=bool)
        if self.direct is None:
            return np.ones((self.n_src, self.n_prod), dtype=bool)
        return np.asarray(self.direct, dtype=bool)

    @property
    def n_src(self) -> int:
        return int(self.cost.shape[0])

    @property
    def n_pool(self) -> int:
        return int(self.pool_cap.shape[0])

    @property
    def n_prod(self) -> int:
        return int(self.price.shape[0])

    @property
    def n_qual(self) -> int:
        return int(self.quality.shape[1])


class PQIndex:
    """Column layout: proportions, pool flows, bypass flows, then auxiliaries."""

    def __init__(self, d: PoolingData):
        ns, npl, nk = d.n_src, d.n_pool, d.n_prod
        self.ns, self.npl, self.nk = ns, npl, nk
        self.bypass = d.bypass
        self.q0 = 0
        self.y0 = self.q0 + ns * npl
        self.z0 = self.y0 + npl * nk
        self.v0 = self.z0 + (ns * nk if d.bypass else 0)
        self.n = self.v0 + ns * npl * nk

    def q(self, i, j):
        return self.q0 + i * self.npl + j

    def y(self, j, k):
        return self.y0 + j * self.nk + k

    def z(self, i, k):
        return self.z0 + i * self.nk + k

    def v(self, i, j, k):
        return self.v0 + (i * self.npl + j) * self.nk + k


def pooling_pq(d: PoolingData, rlt: bool = True,
               name: str = "pq") -> BilinearProblem:
    """Build a multi-pool pooling problem in the pq-formulation.

    ``rlt=False`` drops the reformulation-linearisation rows, leaving the plain
    q-formulation: same feasible set, same optimum, weaker bound.
    """
    ix = PQIndex(d)
    n = ix.n
    ns, npl, nk, nw = d.n_src, d.n_pool, d.n_prod, d.n_qual
    feed = d.feed_mask()
    direct = d.direct_mask()

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rlo: list[float] = []
    rub: list[float] = []
    rnm: list[str] = []
    r = 0

    def add(entries, lb, ub, nm):
        nonlocal r
        for col, val in entries:
            rows.append(r)
            cols.append(col)
            vals.append(val)
        rlo.append(lb)
        rub.append(ub)
        rnm.append(nm)
        r += 1

    # 1. proportions sum to one at every pool
    for j in range(npl):
        members = [i for i in range(ns) if feed[i, j]]
        if not members:
            continue
        add([(ix.q(i, j), 1.0) for i in members], 1.0, 1.0, f"prop_{j}")

    # 2. the pq rows -- the whole point of the formulation
    if rlt:
        for j in range(npl):
            for k in range(nk):
                add([(ix.v(i, j, k), 1.0) for i in range(ns)]
                    + [(ix.y(j, k), -1.0)], 0.0, 0.0, f"rlt_{j}_{k}")

    # 3. source availability, measured through the auxiliaries
    for i in range(ns):
        ent = [(ix.v(i, j, k), 1.0) for j in range(npl) for k in range(nk)]
        if d.bypass:
            ent += [(ix.z(i, k), 1.0) for k in range(nk)]
        add(ent, -INF, float(d.avail[i]), f"avail_{i}")

    # 4. pool throughput
    for j in range(npl):
        add([(ix.y(j, k), 1.0) for k in range(nk)], -INF,
            float(d.pool_cap[j]), f"poolcap_{j}")

    # 5. product demand
    for k in range(nk):
        ent = [(ix.y(j, k), 1.0) for j in range(npl)]
        if d.bypass:
            ent += [(ix.z(i, k), 1.0) for i in range(ns)]
        add(ent, -INF, float(d.demand[k]), f"demand_{k}")

    # 6. product quality, already linear in this formulation
    for k in range(nk):
        for w in range(nw):
            ent = []
            for i in range(ns):
                dev = float(d.quality[i, w] - d.spec[k, w])
                for j in range(npl):
                    ent.append((ix.v(i, j, k), dev))
                if d.bypass:
                    ent.append((ix.z(i, k), dev))
            add(ent, -INF, 0.0, f"spec_{k}_{w}")

    A = SparseMatrix.from_triplets(rows, cols, vals, r, n)

    c = np.zeros(n)
    for k in range(nk):
        for j in range(npl):
            c[ix.y(j, k)] += float(d.price[k])
        if d.bypass:
            for i in range(ns):
                c[ix.z(i, k)] += float(d.price[k] - d.cost[i])
    for i in range(ns):
        for j in range(npl):
            for k in range(nk):
                c[ix.v(i, j, k)] -= float(d.cost[i])

    lo = np.zeros(n)
    hi = np.zeros(n)
    for i in range(ns):
        for j in range(npl):
            hi[ix.q(i, j)] = 1.0 if feed[i, j] else 0.0
    for j in range(npl):
        for k in range(nk):
            hi[ix.y(j, k)] = min(float(d.pool_cap[j]), float(d.demand[k]))
    if d.bypass:
        for i in range(ns):
            for k in range(nk):
                hi[ix.z(i, k)] = (min(float(d.avail[i]), float(d.demand[k]))
                                  if direct[i, k] else 0.0)
    for i in range(ns):
        for j in range(npl):
            for k in range(nk):
                hi[ix.v(i, j, k)] = hi[ix.y(j, k)] if feed[i, j] else 0.0

    names = ([f"q_{i}_{j}" for i in range(ns) for j in range(npl)]
             + [f"y_{j}_{k}" for j in range(npl) for k in range(nk)]
             + ([f"z_{i}_{k}" for i in range(ns) for k in range(nk)]
                if d.bypass else [])
             + [f"v_{i}_{j}_{k}" for i in range(ns) for j in range(npl)
                for k in range(nk)])

    linear = Problem(A=A, c=c, row_lb=np.array(rlo), row_ub=np.array(rub),
                     col_lb=lo, col_ub=hi, sense=ObjSense.MAXIMISE,
                     name=name, col_names=names, row_names=rnm)

    terms = [BilinearTerm(w=ix.v(i, j, k), x=ix.q(i, j), y=ix.y(j, k))
             for i in range(ns) for j in range(npl) for k in range(nk)]
    return BilinearProblem(linear=linear, terms=terms, name=name)


def haverly_pq(variant: int = 1, rlt: bool = True) -> BilinearProblem:
    """Haverly's instance in the pq-formulation.

    The same physical network as the p-formulation version, so the global optima
    must agree -- a cross-check between two independent encodings of one model,
    and a way to measure the RLT rows on an instance whose answer is published.
    """
    cost_b = 13.0 if variant == 3 else 16.0
    demand_x = 600.0 if variant == 2 else 100.0
    d = PoolingData(
        cost=np.array([6.0, cost_b, 10.0]),
        avail=np.array([600.0, 600.0, 600.0]),
        quality=np.array([[3.0], [1.0], [2.0]]),
        pool_cap=np.array([600.0]),
        price=np.array([9.0, 15.0]),
        demand=np.array([demand_x, 200.0]),
        spec=np.array([[2.5], [1.5]]),
        bypass=True,
        # A and B feed the pool; C bypasses it. This restriction *is* the
        # pooling problem -- without it every source could ship direct and the
        # model would be a linear program.
        feed=np.array([[True], [True], [False]]),
        direct=np.array([[False, False], [False, False], [True, True]]),
    )
    return pooling_pq(d, rlt=rlt, name=f"haverly{variant}_pq")


def random_pooling(n_src: int = 5, n_pool: int = 2, n_prod: int = 3,
                   n_qual: int = 2, seed: int = 0,
                   rlt: bool = True) -> BilinearProblem:
    """A random multi-pool network with binding quality specifications."""
    rng = np.random.default_rng(seed)
    quality = rng.uniform(0.5, 3.5, (n_src, n_qual))
    d = PoolingData(
        cost=rng.uniform(5.0, 18.0, n_src),
        avail=rng.uniform(60.0, 200.0, n_src),
        quality=quality,
        pool_cap=rng.uniform(80.0, 250.0, n_pool),
        price=rng.uniform(14.0, 26.0, n_prod),
        demand=rng.uniform(40.0, 140.0, n_prod),
        # specs near the middle of the available qualities, so they bind and the
        # blending decision actually matters
        spec=np.tile(quality.mean(axis=0) * 1.02, (n_prod, 1)),
        bypass=True,
        # the last source bypasses, the rest are pooled -- otherwise there is
        # no blending decision to make
        feed=np.array([[i < n_src - 1 for _ in range(n_pool)]
                       for i in range(n_src)]),
        direct=np.array([[i >= n_src - 1 for _ in range(n_prod)]
                         for i in range(n_src)]),
    )
    return pooling_pq(d, rlt=rlt,
                      name=f"pool_{n_src}x{n_pool}x{n_prod}_s{seed}")
