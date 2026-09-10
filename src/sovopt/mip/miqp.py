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

The bound is not rigorous, and that matters
-------------------------------------------
On the LP path a node bound is made safe by the Neumaier-Shcherbina correction
in :mod:`sovopt.mip.safebound`, so an unconverged relaxation costs search
effort and never a wrong answer. **There is no equivalent here.** The bound is
whatever the proximal QP solver reports, accurate to its own tolerance and no
better, so a node could in principle be pruned on a bound that is slightly too
high and the true optimum lost with it.

Two things keep that honest rather than hidden. Pruning subtracts
``bound_slack`` from every node bound before comparing, so a bound has to beat
the incumbent by more than the QP solver's accuracy before the subtree is
discarded. And every incumbent is validated against the *original* model --
feasibility, integrality and objective recomputed from scratch -- so a wrong
answer cannot be reported even if a bound was wrong; the search would return a
worse answer, not an invalid one. A certified QP bound is the piece that would
make this rigorous, and it is not written.

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
from ..core.tolerances import DEFAULT, INF, Tolerances
from ..qp import QPParams, solve_qp
from .propagate import propagate

__all__ = ["MIQPParams", "solve_miqp"]


@dataclass
class MIQPParams:
    """Search settings. The QP solver's own settings live in ``qp``."""

    time_limit: float = 300.0
    node_limit: int = 100_000

    gap_rel: float = 1e-4
    gap_abs: float = 1e-9

    bound_slack: float = 1e-6
    """How far a node bound must beat the incumbent before its subtree is cut.

    The relaxation is solved by a first-order method to a finite tolerance, so
    its objective is a lower bound only up to that tolerance. Requiring the
    bound to clear the incumbent by this margin trades a little search effort
    for not discarding a subtree on a bound that was slightly optimistic."""

    propagate_rounds: int = 6
    integrality: float = 1e-6

    qp: QPParams = field(default_factory=QPParams)
    tol: Tolerances = field(default_factory=lambda: DEFAULT)
    verbose: bool = False


def _relaxation(prob: Problem, lo, hi, params: MIQPParams, deadline: float):
    """Solve the node's continuous relaxation over ``[lo, hi]``."""
    node = prob.copy()
    node.col_lb = lo
    node.col_ub = hi
    node.kind = np.zeros(prob.n, dtype=node.kind.dtype)   # relax integrality
    qp = QPParams(**{**vars(params.qp),
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
        if np.isfinite(incumbent) and bound - params.bound_slack >= incumbent - gap:
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
        if r.x is None or r.status in (Status.INFEASIBLE,
                                       Status.INFEASIBLE_OR_UNBOUNDED):
            continue
        node_bound = float(r.objective)
        if np.isfinite(incumbent) and \
                node_bound - params.bound_slack >= incumbent - gap:
            continue

        x = np.asarray(r.x, dtype=VAL)
        frac = np.abs(x[int_mask] - np.round(x[int_mask]))
        consider(x)                      # a rounding is always worth a try

        if frac.size == 0 or frac.max() <= params.integrality:
            continue                     # relaxation was integral: nothing to branch

        idx = np.flatnonzero(int_mask)
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

    if not frontier and status == Status.NODE_LIMIT:
        status = Status.OPTIMAL if best_x is not None else Status.INFEASIBLE

    if best_x is None:
        return Solution(status=status if status != Status.NODE_LIMIT
                        else Status.NODE_LIMIT,
                        nodes=nodes, time=time.perf_counter() - t0,
                        method="miqp-bb").drop_objective_if_unsolved()

    obj = prob.objective(best_x)
    dual = -incumbent if flip else incumbent
    sol = Solution(status=status, x=best_x, objective=obj, nodes=nodes,
                   time=time.perf_counter() - t0, method="miqp-bb")
    sol.dual_bound = dual if status == Status.OPTIMAL else float("nan")
    sol.info = {"nodes": nodes, "bound_is_rigorous": False}
    return sol
