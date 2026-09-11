"""Spatial branch-and-bound for bilinear programs.

Ordinary branch-and-bound splits *integer* variables because a fractional value
is infeasible. Spatial branch-and-bound splits *continuous* ones, because the
relaxation is wrong rather than fractional: the McCormick envelope of a product
is loosest in the middle of the box, so halving a range roughly quarters the
error that term contributes. Branching drives the relaxation towards the true
non-convex surface, and the bound and incumbent close on the global optimum.

The loop
--------
1. Relax: build the McCormick LP over the node's box and solve it. Its optimum
   is a rigorous bound, because every bilinear-feasible point of the box
   satisfies all four envelopes.
2. Check: if the relaxation's solution already satisfies ``w = x*y`` it is
   genuinely feasible, and it is an incumbent.
3. Improve: otherwise fix one factor of every product at its relaxed value and
   re-solve. **When a factor's range collapses to a point the McCormick
   envelope becomes exact**, so this is just the same relaxation on a degenerate
   box -- and the answer is bilinear-feasible by construction. It is also
   precisely *distributive recursion*, the fixed-point iteration refineries run
   inside PIMS and GRTMPS. Here it is a heuristic that supplies incumbents while
   the tree proves optimality, rather than the whole method with no guarantee.
4. Branch: a fractional integer first, then the most-violated product at its
   relaxed value. Integer variables were added for the non-convex QP route
   (:mod:`sovopt.globalopt.nonconvex_qp`) and cost the pooling models nothing;
   an integer split is the ordinary ``{<= ⌊v⌋} ∪ {>= ⌈v⌉}`` and a node is only
   finished when its point is both integral and bilinear-feasible.

The node bound is certified
---------------------------
The relaxation is an LP solved by the exact simplex, whose optimal objective is
a bound to the simplex's own tolerance. Each node is pruned instead on the
Neumaier-Shcherbina bound built from the simplex's duals -- valid for any dual
vector, exact at an optimal one -- so the tree's proof does not rest on the
node LP having converged. Where that bound is vacuous (an envelope half was
dropped for an infinite factor) the LP objective is used and the node counted
in ``info["uncertified_nodes"]``.

Optimality-based bound tightening
---------------------------------
Before the tree starts, each variable appearing in a product is minimised and
then maximised over the relaxation. Every such solve is an independent LP, and
the tightened box feeds straight back into the envelopes. It costs two LPs per
variable and routinely pays for itself many times over, because envelope
quality is quadratic in box width.

What this gives that recursion does not
---------------------------------------
A **bound**. Recursion returns a blend and no way to know whether a better one
exists; this returns a blend together with a proof of how far from optimal it
can possibly be. On the Haverly instances recursion is known to stop at local
optima worth less than the global one, which is quality give-away with a price
attached.

References
----------
Horst & Tuy, *Global Optimization: Deterministic Approaches*, Springer 1996.
Ryoo & Sahinidis, "A branch-and-reduce approach to global optimization",
  J. Global Optimization 8 (1996) 107-138 -- bound tightening inside the tree.
Tawarmalani & Sahinidis, "Global optimization of mixed-integer nonlinear
  programs: a theoretical and computational study", Math. Prog. 99 (2004).
Gleixner, Berthold, Müller & Weltge, "Three enhancements for optimization-based
  bound tightening", J. Global Optim. 67 (2017) 731-757.
Misener & Floudas, "Advances for the pooling problem", Applied and Computational
  Mathematics 8 (2009) -- formulations and benchmark results.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.problem import ObjSense, Problem, Solution, Status
from ..core.sparse import VAL
from ..core.tolerances import INF
from ..lp.simplex import SimplexParams, solve_simplex
from ..mip.safebound import safe_dual_bound
from .bilinear import BilinearProblem, build_relaxation

__all__ = ["SpatialParams", "solve_global"]


@dataclass
class SpatialParams:
    gap_rel: float = 1e-6
    gap_abs: float = 1e-9
    time_limit: float = 120.0
    node_limit: int = 200_000

    bilinear_tol: float = 1e-7
    """Largest ``|w - x*y|`` still counted as bilinear-feasible."""

    obbt_rounds: int = 2
    obbt_time_frac: float = 0.25
    branch_eps: float = 1e-4
    """Keep a split this far inside the range so children are strictly smaller."""

    integrality: float = 1e-6
    polish_steps: int = 50
    """Projected-gradient steps in the factor space from each relaxation
    point, keeping only a point that is feasible for the rows once the
    products are recomputed. An incumbent heuristic; the bound never sees it.
    Added for the non-convex QP route, where the relaxation point alone gave
    incumbents 50-75% above the optimum at 25 variables."""
    root_multistart: int = 16
    """Extra polish runs from seeded random points of the box at the root.
    A relaxation vertex is a poor place to start a descent on a quadratic;
    sixteen random ones cost a few milliseconds and set the incumbent the
    whole tree prunes against."""
    warm_start: bool = True
    """Start each node LP from its parent's basis. The McCormick rows change
    coefficients between parent and child but not shape, so the parent's
    basis is a valid, usually near-optimal, start."""
    verbose: bool = False
    log_every: float = 1.0


@dataclass
class _Node:
    bound: float
    lo: np.ndarray
    hi: np.ndarray
    depth: int = 0
    _order: int = 0
    basis: object = None

    def __lt__(self, other):
        if self.bound != other.bound:
            return self.bound < other.bound
        return self._order < other._order


def _solve_relaxation(bp, lo, hi, time_limit, warm=None):
    relax = build_relaxation(bp, lo, hi)
    relax.kind[:] = 0                          # the node LP relaxes integrality
    if warm is not None and len(warm) != relax.n + relax.m:
        warm = None                            # an envelope half came or went
    sol = solve_simplex(relax, SimplexParams(time_limit=time_limit),
                        warm_basis=warm)
    sol.info = {**(getattr(sol, "info", None) or {}), "relaxation": relax}
    return sol


def _polish(bp, x, lo, hi, steps):
    """Projected gradient on the factors, products recomputed, rows checked.

    The objective is linear in the extended vector and the products are
    functions of the factors, so its gradient in factor space is one chain
    rule: ``c_i + Σ c_w · (the other factor)``. Only a point that satisfies
    the rows with its products recomputed is returned, so this can only ever
    improve the incumbent -- never the bound, which does not see it.
    """
    if steps <= 0 or not bp.terms:
        return None
    c = bp.linear.c
    factors = sorted({t.x for t in bp.terms} | {t.y for t in bp.terms})
    prods = np.array([t.w for t in bp.terms])
    tx = np.array([t.x for t in bp.terms])
    ty = np.array([t.y for t in bp.terms])
    n = x.size

    def close(v):
        v = v.copy()
        v[prods] = v[tx] * v[ty]
        return v

    def grad(v):
        g = c.copy()
        np.add.at(g, tx, c[prods] * v[ty])
        np.add.at(g, ty, c[prods] * v[tx])
        g[prods] = 0.0
        return g

    cur = close(np.clip(x, lo, hi))
    best_x, best_f = None, bp.linear.objective(cur)
    g = grad(cur)
    step = 1.0 / max(float(np.abs(g[factors]).max(initial=0.0)), 1e-8)
    for _ in range(steps):
        g = grad(cur)
        cand = cur.copy()
        cand[factors] = np.clip(cur[factors] - step * g[factors],
                                lo[factors], hi[factors])
        cand = close(cand)
        f = bp.linear.objective(cand)
        if f < best_f - 1e-12:
            rv, bv, _ = bp.linear.violation(cand)
            if max(rv, bv) <= 1e-9:
                best_x, best_f = cand, f
            cur = cand
        else:
            step *= 0.5
            if step < 1e-12:
                break
    return best_x


def _certified(rel, fallback):
    """Neumaier-Shcherbina on the node LP's duals; ``(bound, certified?)``."""
    if rel.y is None:
        return fallback, False
    r = rel.info["relaxation"]
    b = safe_dual_bound(r.A, r.c, r.row_lb, r.row_ub, r.col_lb, r.col_ub,
                        rel.y, strict=True) + r.obj_offset
    if np.isfinite(b):
        return b, True
    return fallback, False


def _recursion_incumbent(bp, x, lo, hi, time_limit):
    """Fix one factor of every product and re-solve: a feasible blend.

    Collapsing a factor's range to a point makes its McCormick envelope exact,
    so no special machinery is needed -- the relaxation on the degenerate box
    *is* the fixed problem.
    """
    lo2 = lo.copy()
    hi2 = hi.copy()
    for t in bp.terms:
        v = float(np.clip(x[t.x], lo[t.x], hi[t.x]))
        lo2[t.x] = hi2[t.x] = v
    if all(lo2[t.y] == hi2[t.y] for t in bp.terms):
        # Every factor is fixed -- the case for a quadratic objective, where
        # each x_i is the first factor of its own square -- so the LP would
        # only be evaluating the products. Do that directly: on a 20-variable
        # QP the LP was 118 cold simplex solves and 40% of the run.
        pt = np.clip(x, lo2, hi2)
        for t in bp.terms:
            pt[t.w] = pt[t.x] * pt[t.y]
        rv, bv, _ = bp.linear.violation(pt)
        return pt if max(rv, bv) <= 1e-9 else None
    sol = _solve_relaxation(bp, lo2, hi2, time_limit)
    if sol.status != Status.OPTIMAL or sol.x is None:
        return None
    return sol.x


def _obbt(bp, lo, hi, params, deadline):
    """Tighten every variable that appears in a product, by LP in both directions."""
    touched = sorted({t.x for t in bp.terms} | {t.y for t in bp.terms})
    n = bp.n
    for _rnd in range(params.obbt_rounds):
        improved = False
        for j in touched:
            if time.perf_counter() > deadline:
                return lo, hi, improved
            if hi[j] - lo[j] <= params.branch_eps:
                continue
            for sense in (+1.0, -1.0):
                probe = build_relaxation(bp, lo, hi)
                probe.c = np.zeros(n, dtype=VAL)
                probe.c[j] = sense
                probe.sense = ObjSense.MINIMISE
                probe.obj_offset = 0.0
                s = solve_simplex(probe, SimplexParams(
                    time_limit=max(1.0, deadline - time.perf_counter())))
                if s.status != Status.OPTIMAL or s.x is None:
                    continue
                val = float(s.x[j])
                if sense > 0 and val > lo[j] + 1e-9:
                    lo[j] = min(val, hi[j])
                    improved = True
                elif sense < 0 and val < hi[j] - 1e-9:
                    hi[j] = max(val, lo[j])
                    improved = True
        if not improved:
            break
    return lo, hi, True


def _pick_branch(bp, x, lo, hi, params):
    """Most-violated product; split whichever factor has the wider live range."""
    best_t, best_v = None, params.bilinear_tol
    for t in bp.terms:
        v = t.violation(x)
        if v > best_v:
            best_v, best_t = v, t
    if best_t is None:
        return None, 0.0

    cand = []
    for j in (best_t.x, best_t.y):
        width = hi[j] - lo[j]
        if width > params.branch_eps and np.isfinite(width):
            cand.append((width, j))
    if not cand:
        return None, 0.0
    _, j = max(cand)

    val = float(x[j])
    lo_j, hi_j = float(lo[j]), float(hi[j])
    margin = params.branch_eps * max(1.0, hi_j - lo_j)
    if not np.isfinite(val) or val <= lo_j + margin or val >= hi_j - margin:
        val = 0.5 * (lo_j + hi_j)          # bisect rather than make a null child
    return j, val


def solve_global(bp: BilinearProblem,
                 params: SpatialParams | None = None) -> Solution:
    """Solve a bilinear program to proven global optimality."""
    params = params or SpatialParams()
    t0 = time.perf_counter()

    base = bp.linear
    flip = base.sense == ObjSense.MAXIMISE
    if flip:                                # work in minimisation throughout
        work = base.copy()
        work.c = -work.c
        work.obj_offset = -work.obj_offset
        work.sense = ObjSense.MINIMISE
        bp = BilinearProblem(linear=work, terms=bp.terms, name=bp.name)

    lo = bp.linear.col_lb.copy()
    hi = bp.linear.col_ub.copy()

    n_obbt = 0
    if params.obbt_rounds > 0 and bp.terms:
        lo, hi, _ = _obbt(bp, lo, hi, params,
                          t0 + params.obbt_time_frac * params.time_limit)
        n_obbt = 1

    incumbent = np.inf
    best_x = None
    int_idx = np.flatnonzero(bp.linear.integer_mask)
    uncertified = 0

    def integral(x):
        return int_idx.size == 0 or \
            np.abs(x[int_idx] - np.round(x[int_idx])).max() <= params.integrality

    def accept(x):
        """An incumbent must be bilinear-feasible *and* integral."""
        nonlocal incumbent, best_x
        if x is None or not integral(x):
            return
        if bp.max_violation(x) > params.bilinear_tol:
            return
        v = bp.linear.objective(x)
        if v < incumbent:
            incumbent, best_x = v, np.array(x, copy=True)

    root = _solve_relaxation(bp, lo, hi, params.time_limit)
    if root.status == Status.INFEASIBLE:
        return Solution(status=Status.INFEASIBLE, time=time.perf_counter() - t0,
                        method="spatial-bb")
    if root.status != Status.OPTIMAL or root.x is None:
        return Solution(status=root.status, time=time.perf_counter() - t0,
                        method="spatial-bb")

    accept(_recursion_incumbent(bp, root.x, lo, hi, params.time_limit))

    counter = 0
    frontier: list[_Node] = []
    root_bound, ok = _certified(root, root.objective)
    uncertified += not ok
    accept(_polish(bp, root.x, lo, hi, params.polish_steps))
    if params.root_multistart > 0 and bp.terms:
        rng = np.random.default_rng(0)
        finite = np.isfinite(lo) & np.isfinite(hi) & (lo > -INF) & (hi < INF)
        for _ in range(params.root_multistart):
            start = np.where(finite, rng.uniform(np.where(finite, lo, 0.0),
                                                 np.where(finite, hi, 1.0)),
                             root.x)
            accept(_polish(bp, start, lo, hi, params.polish_steps))
    heapq.heappush(frontier, _Node(root_bound, lo, hi, 0, counter,
                                   root.basis_status if params.warm_start else None))

    nodes = 0
    status = Status.NODE_LIMIT
    last_log = t0

    while frontier:
        if time.perf_counter() - t0 > params.time_limit:
            status = Status.TIME_LIMIT
            break
        if nodes >= params.node_limit:
            status = Status.NODE_LIMIT
            break

        nd = heapq.heappop(frontier)
        tol = max(params.gap_abs, params.gap_rel * abs(incumbent)) \
            if np.isfinite(incumbent) else params.gap_abs
        if np.isfinite(incumbent) and nd.bound >= incumbent - tol:
            continue

        rel = _solve_relaxation(bp, nd.lo, nd.hi, params.time_limit, nd.basis)
        nodes += 1
        if rel.status == Status.INFEASIBLE or rel.x is None:
            continue
        if rel.status != Status.OPTIMAL:
            continue
        cert, ok = _certified(rel, rel.objective)
        uncertified += not ok
        bound = max(cert, nd.bound)
        if np.isfinite(incumbent) and bound >= incumbent - tol:
            continue

        x = rel.x
        viol = bp.max_violation(x)
        if viol <= params.bilinear_tol and integral(x):
            accept(x)
            continue

        accept(_recursion_incumbent(bp, x, nd.lo, nd.hi, params.time_limit))
        accept(_polish(bp, x, nd.lo, nd.hi, params.polish_steps))

        if not integral(x):
            # a fractional integer comes first: its split is exact, where a
            # product's split only tightens an envelope
            fr = np.abs(x[int_idx] - np.round(x[int_idx]))
            j = int(int_idx[int(np.argmax(fr))])
            v = float(x[j])
            left_lo, left_hi = nd.lo.copy(), nd.hi.copy()
            left_hi[j] = np.floor(v)
            right_lo, right_hi = nd.lo.copy(), nd.hi.copy()
            right_lo[j] = np.ceil(v)
        else:
            j, val = _pick_branch(bp, x, nd.lo, nd.hi, params)
            if j is None:
                continue
            left_lo, left_hi = nd.lo.copy(), nd.hi.copy()
            left_hi[j] = val
            right_lo, right_hi = nd.lo.copy(), nd.hi.copy()
            right_lo[j] = val

        basis = rel.basis_status if params.warm_start else None
        for cl, ch in ((left_lo, left_hi), (right_lo, right_hi)):
            if (cl <= ch).all():
                counter += 1
                heapq.heappush(frontier, _Node(bound, cl, ch, nd.depth + 1,
                                               counter, basis))

        if params.verbose and time.perf_counter() - last_log > params.log_every:
            last_log = time.perf_counter()
            bb = frontier[0].bound if frontier else incumbent
            print(f"  nodes {nodes:>7d}  open {len(frontier):>6d}  "
                  f"bound {bb:< 14.8g} incumbent {incumbent:< 14.8g}  "
                  f"{time.perf_counter()-t0:6.1f}s")

    dual_bound = frontier[0].bound if frontier else incumbent
    if not frontier and np.isfinite(incumbent):
        status = Status.OPTIMAL
        dual_bound = incumbent

    if best_x is None:
        return Solution(status=Status.INFEASIBLE if status == Status.OPTIMAL
                        else status, nodes=nodes,
                        time=time.perf_counter() - t0, method="spatial-bb")

    obj = bp.linear.objective(best_x)
    if flip:
        obj = -obj
        dual_bound = -dual_bound

    sol = Solution(status=status, x=best_x, objective=obj,
                   dual_bound=dual_bound, nodes=nodes,
                   time=time.perf_counter() - t0, method="spatial-bb")
    sol.info = {
        "obbt_rounds": n_obbt,
        "terms": len(bp.terms),
        "max_bilinear_violation": bp.max_violation(best_x),
        "uncertified_nodes": uncertified,
        "bound_is_rigorous": uncertified == 0,
    }
    return sol
