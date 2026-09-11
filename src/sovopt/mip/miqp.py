"""Mixed-integer quadratic programming: branch-and-bound over convex QPs.

The engine has had a convex QP solver and a branch-and-bound tree for a while
and no way to put them together, so a model with both a quadratic objective and
an integer variable was **refused** rather than approximated -- which was the
right refusal, since solving it as a MILP silently discards the quadratic term
and returns a confident wrong number. This is the missing join: the same
search, with a QP at every node instead of an LP.

Why it is a separate tree
-------------------------
:mod:`sovopt.mip.tree` is built around an LP node solver and everything that
comes with one -- a basis to warm-start from, a dual ray to learn a conflict
clause from, cuts separated from a tableau. A first-order QP has none of those:
no basis, no ray, no tableau. Bolting a second node type into that tree would
have meant threading "is this an LP or a QP" through every one of those
mechanisms to have most of them switched off. What is here instead is the part
of branch-and-bound that does not depend on the relaxation being an LP --
propagation, best-bound selection, branching, incumbent validation -- and
nothing else.

The bound is rigorous, and how
------------------------------
On the LP path a node bound is made safe by the Neumaier-Shcherbina correction
in :mod:`sovopt.mip.safebound`, so an unconverged relaxation costs search
effort and never a wrong answer. The first version of this tree had no
equivalent: its bound was whatever the proximal QP solver reported, good to
that solver's tolerance and no better, and it said so.

:func:`~sovopt.mip.safebound.safe_qp_bound` closes that. Convexity puts the
tangent plane at the solver's iterate *below* the objective everywhere, and the
tangent plane is linear, so the Neumaier-Shcherbina bound applies to it as-is.
The result is a lower bound that holds for **any** iterate and **any** dual
vector -- converged, half-converged or stopped by a time limit -- and is exact
at an optimal pair. Every node here is pruned on that bound and never on the
solver's own objective, which is why a relaxation solved deliberately badly
still produces the brute-force optimum (see the tests).

Two things remain from before, because they are cheap and independent: every
incumbent is validated against the *original* model -- feasibility,
integrality and objective recomputed from scratch -- and the reported dual
bound is the minimum over every leaf's certified bound, so the gap the caller
sees is one the caller could recompute.

Rigour is conditional on ``Q ⪰ 0``, which is the premise of this path rather
than something it can prove in general: the QP solver checks it by estimating
the smallest eigenvalue, and a non-convex ``Q`` is refused there. Where the
scaled Gershgorin test of :mod:`sovopt.globalopt.alphabb` certifies convexity
outright -- every shift it would need is zero -- the bound is unconditional,
and ``info["convexity_certified"]`` says which case a solve was.

References
----------
Land & Doig, "An automatic method of solving discrete programming problems",
  Econometrica 28 (1960) 497-520 -- branch and bound.
Fletcher & Leyffer, "Numerical experience with lower bounds for MIQP
  branch-and-bound", SIAM J. Optimization 8 (1998) 604-616 -- QP relaxations as
  node bounds, and how much the bound quality is worth.
Bonami, Kilinç & Linderoth, "Algorithms and software for convex mixed integer
  nonlinear programs", in *Mixed Integer Nonlinear Programming*, Springer
  (2012) 1-39 -- the convex MINLP branch-and-bound this specialises.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.problem import ObjSense, Problem, Solution, Status
from ..core.sparse import VAL
from ..core.tolerances import DEFAULT, Tolerances
from ..qp import QPParams, solve_qp
from .propagate import propagate
from .safebound import safe_qp_bound

__all__ = ["MIQPParams", "solve_miqp"]


@dataclass
class MIQPParams:
    """Search settings. The QP solver's own settings live in ``qp``."""

    time_limit: float = 300.0
    node_limit: int = 100_000

    gap_rel: float = 1e-4
    gap_abs: float = 1e-9

    propagate_rounds: int = 6
    integrality: float = 1e-6

    qp: QPParams = field(default_factory=QPParams)
    tol: Tolerances = field(default_factory=lambda: DEFAULT)
    verbose: bool = False


def _relaxation(prob: Problem, lo, hi, params: MIQPParams, deadline: float,
                tighten: int = 0):
    """Solve the node's continuous relaxation over ``[lo, hi]``.

    ``tighten`` re-solves with the tolerances divided by ``100**tighten`` and
    the iteration budget multiplied by ``4**tighten``; see the loop for when
    that is needed.
    """
    node = prob.copy()
    node.col_lb = lo
    node.col_ub = hi
    node.kind = np.zeros(prob.n, dtype=node.kind.dtype)   # relax integrality
    f = 100.0 ** tighten
    qp = QPParams(**{**vars(params.qp),
                     "eps_abs": params.qp.eps_abs / f,
                     "eps_rel": params.qp.eps_rel / f,
                     "max_iter": params.qp.max_iter * 4 ** tighten,
                     "time_limit": max(0.0, deadline - time.perf_counter())})
    return solve_qp(node, qp)


def solve_miqp(prob: Problem, params: MIQPParams | None = None) -> Solution:
    """Solve a convex mixed-integer quadratic program.

    A non-convex ``Q`` is refused by the QP solver rather than approximated,
    and a maximisation is converted by negating both ``c`` and ``Q`` -- so a
    maximisation is solvable exactly when the *negated* ``Q`` is positive
    semidefinite, which is the honest condition.
    """
    params = params or MIQPParams()
    tol = params.tol
    t0 = time.perf_counter()
    deadline = t0 + params.time_limit

    if prob.Q is None:
        raise ValueError("solve_miqp needs a quadratic objective; this model "
                         "has none, so use solve_mip")
    if not prob.is_mip:
        return solve_qp(prob, params.qp)

    flip = prob.sense == ObjSense.MAXIMISE
    work = prob
    if flip:
        work = prob.copy()
        work.c = -work.c
        work.obj_offset = -work.obj_offset
        work.Q = work.Q.copy()
        work.Q.cx = -work.Q.cx
        work.Q.rx = -work.Q.rx
        work.sense = ObjSense.MINIMISE

    from ..globalopt.alphabb import gershgorin_alpha
    certified = bool((gershgorin_alpha(work.Q, work.col_lb, work.col_ub) == 0.0).all())

    int_mask = work.integer_mask
    root = propagate(work.A, work.row_lb, work.row_ub,
                     work.col_lb.astype(VAL), work.col_ub.astype(VAL),
                     int_mask, max_rounds=params.propagate_rounds,
                     feas_tol=tol.primal_feas)
    if root.infeasible:
        return Solution(status=Status.INFEASIBLE, nodes=0,
                        time=time.perf_counter() - t0, method="miqp-bb")

    incumbent = np.inf
    best_x = None
    nodes = 0
    status = Status.NODE_LIMIT
    leaf_lb = np.inf        # min certified bound over every closed node
    undecided = 0           # nodes closed without a bound that closes them

    def consider(x) -> bool:
        """Accept a candidate only if the *original* model agrees it is one."""
        nonlocal incumbent, best_x
        if x is None:
            return False
        cand = np.array(x, dtype=VAL, copy=True)
        idx = np.flatnonzero(int_mask)
        cand[idx] = np.round(cand[idx])
        np.clip(cand, prob.col_lb, prob.col_ub, out=cand)
        row_v, col_v, int_v = prob.violation(cand)
        if max(row_v, col_v) > 1e-6 or int_v > params.integrality:
            return False
        value = float(work.c @ cand) + work.obj_offset \
            + 0.5 * float(cand @ work.Q.matvec(cand))
        if value < incumbent - 1e-12:
            incumbent, best_x = value, cand
            return True
        return False

    # (bound, order, path) -- order breaks ties so the heap is deterministic
    frontier: list = []
    heapq.heappush(frontier, (-np.inf, 0, ()))
    order = 1

    while frontier:
        if time.perf_counter() > deadline:
            status = Status.TIME_LIMIT
            break
        if nodes >= params.node_limit:
            status = Status.NODE_LIMIT
            break

        bound, _, path = heapq.heappop(frontier)
        gap = max(params.gap_abs, params.gap_rel * abs(incumbent)) \
            if np.isfinite(incumbent) else 0.0
        if np.isfinite(incumbent) and bound >= incumbent - gap:
            leaf_lb = min(leaf_lb, bound)
            continue

        lo = root.lo.copy()
        hi = root.hi.copy()
        for j, is_lower, v in path:
            if is_lower:
                lo[j] = max(lo[j], v)
            else:
                hi[j] = min(hi[j], v)
        if np.any(lo > hi + tol.primal_feas):
            continue

        pr = propagate(work.A, work.row_lb, work.row_ub, lo, hi, int_mask,
                       max_rounds=2, feas_tol=tol.primal_feas)
        if pr.infeasible:
            continue
        lo, hi = pr.lo, pr.hi

        nodes += 1
        r = _relaxation(work, lo, hi, params, deadline)
        if r.status in (Status.INFEASIBLE, Status.INFEASIBLE_OR_UNBOUNDED):
            continue
        if r.x is None:
            # the proximal solver always returns its best iterate; a relaxation
            # with no point at all is a bug, not a node to drop silently
            raise RuntimeError("QP relaxation returned no iterate")
        x = np.asarray(r.x, dtype=VAL)
        # The bound is certified from the iterate and its duals, whatever the
        # solver's status -- never taken from the solver's own objective.
        y = r.y if r.y is not None else np.zeros(work.m, dtype=VAL)
        node_bound = safe_qp_bound(work.A, work.c, work.Q, work.row_lb,
                                   work.row_ub, lo, hi, x, y, strict=True) \
            + work.obj_offset
        node_bound = max(node_bound, bound)      # a child is never looser than its parent
        if np.isfinite(incumbent) and node_bound >= incumbent - gap:
            leaf_lb = min(leaf_lb, node_bound)
            continue

        idx = np.flatnonzero(int_mask)
        frac = np.abs(x[idx] - np.round(x[idx]))
        consider(x)                      # a rounding is always worth a try

        if frac.size == 0 or frac.max() <= params.integrality:
            # An integral iterate is not a finished node. The LP tree learnt
            # this the hard way (README, "A valid bound is not a valid
            # search"): a bound licenses *discarding* a node, and only a bound
            # that meets the incumbent licenses *closing* one. An unconverged
            # iterate can look integral while the node's real optimum is a
            # different integer point, so closing on it loses that point and
            # nothing downstream can notice.
            closed = np.isfinite(incumbent) and node_bound >= incumbent - gap
            if not closed:
                unfixed = idx[hi[idx] - lo[idx] > 0.5]
                if unfixed.size:
                    # split an integer domain at the iterate: {<= v} ∪ {>= v+1}
                    # covers every integer, so nothing is lost and the domain
                    # shrinks either way
                    j = int(unfixed[int(np.argmax(hi[unfixed] - lo[unfixed]))])
                    v = float(np.round(x[j]))
                    if v >= hi[j]:
                        v -= 1.0
                    for child in ((j, False, v), (j, True, v + 1.0)):
                        heapq.heappush(frontier,
                                       (node_bound, order, path + (child,)))
                        order += 1
                    continue
                # every integer is fixed: this is a continuous QP, and the
                # only way to close it is to solve it better
                for tighten in (1, 2):
                    r = _relaxation(work, lo, hi, params, deadline, tighten)
                    if r.x is None:
                        break
                    x = np.asarray(r.x, dtype=VAL)
                    y = r.y if r.y is not None else np.zeros(work.m, dtype=VAL)
                    nb = safe_qp_bound(work.A, work.c, work.Q, work.row_lb,
                                       work.row_ub, lo, hi, x, y, strict=True) \
                        + work.obj_offset
                    node_bound = max(node_bound, nb)
                    consider(x)
                    gap = max(params.gap_abs, params.gap_rel * abs(incumbent)) \
                        if np.isfinite(incumbent) else 0.0
                    if np.isfinite(incumbent) and node_bound >= incumbent - gap:
                        closed = True
                        break
                if not closed:
                    undecided += 1
            leaf_lb = min(leaf_lb, node_bound)
            continue

        j = int(idx[int(np.argmax(frac))])
        v = float(x[j])
        for child in ((j, False, np.floor(v)), (j, True, np.ceil(v))):
            heapq.heappush(frontier, (node_bound, order, path + (child,)))
            order += 1

        if params.verbose:
            print(f"  miqp nodes {nodes:>7d}  open {len(frontier):>6d}  "
                  f"bound {node_bound:< 14.8g}  incumbent "
                  f"{incumbent if np.isfinite(incumbent) else float('nan'):< 14.8g}"
                  f"  {time.perf_counter() - t0:6.1f}s")

    if not frontier and status == Status.NODE_LIMIT and not undecided:
        status = Status.OPTIMAL if best_x is not None else Status.INFEASIBLE
    # an exhausted search with an undecided node keeps NODE_LIMIT, as the LP
    # tree does: the incumbent is real, the certified bound below is real,
    # and the gap between them is exactly what was not proved

    # the global lower bound: nothing open or closed lies below it
    global_lb = min(leaf_lb, min((b for b, _, _ in frontier), default=np.inf))
    if best_x is None:
        return Solution(status=status if status != Status.NODE_LIMIT
                        else Status.NODE_LIMIT,
                        nodes=nodes, time=time.perf_counter() - t0,
                        method="miqp-bb").drop_objective_if_unsolved()

    obj = prob.objective(best_x)
    global_lb = min(global_lb, incumbent)
    sol = Solution(status=status, x=best_x, objective=obj, nodes=nodes,
                   time=time.perf_counter() - t0, method="miqp-bb")
    sol.dual_bound = -global_lb if flip else global_lb
    sol.info = {"nodes": nodes, "bound_is_rigorous": True,
                "convexity_certified": certified}
    return sol
