"""Non-convex (mixed-integer) quadratic programs, to proven global optimality.

    minimise  ½ xᵀQx + cᵀx     subject to   rl <= Ax <= ru,  l <= x <= u,
                                            x_j integral for j ∈ I,
                                            Q **not** positive semidefinite.

Until this module the answer to that shape was a refusal, which was right for
a convex solver and is no answer for a refinery: a blending margin with a
concave penalty, or a planning objective with cross terms between grades, is
exactly this. A local method returns *a* stationary point; the problem
statement asks for optimisation, and optimising a non-convex function means a
bound.

Two relaxations were built, and the measured one is the default
------------------------------------------------------------------
**McCormick** (default). Every product ``x_i x_j`` that ``Q`` touches becomes a
variable ``w_ij`` with the four McCormick envelope rows over the node's box,
and the objective becomes linear in ``w``. That is a bilinear program, and
:mod:`sovopt.globalopt.spatial` -- the pooling engine -- already solves those:
exact simplex node LPs, optimality-based bound tightening, and now integer
branching and a Neumaier-Shcherbina-certified node bound from the simplex's
duals. The envelope is the convex hull of a single product, which is why it is
tight where a diagonal shift is not.

**αBB** (:mod:`sovopt.globalopt.alphabb`, opt-in). Shift ``Q``'s diagonal until
Gershgorin certifies convexity and solve the convex QP at each node. It was
built first, it is rigorous, and on dense indefinite ``Q`` it loses to the
envelopes by a wide margin: at 15 variables it times out at a 10-12% gap
where McCormick proves optimality in 18-44 s. It stays because it needs no
product variables -- ``n²`` of them for a dense ``Q`` -- and its bound is
unconditional; the table in the README is what decides between them, and
``relaxation="alphabb"`` is there so that table can be regenerated.

What both need
--------------
A finite box on every variable that appears in a non-convex product: the
envelopes and the underestimator are both built from it. A model without one
is refused with the variable named, rather than given a default bound that
would silently become part of the answer.

References
----------
McCormick, "Computability of global solutions to factorable nonconvex
  programs: Part I -- Convex underestimating problems", Math. Prog. 10 (1976)
  147-175.
Sherali & Tuncbilek, "A global optimization algorithm for polynomial
  programming problems using a reformulation-linearization technique",
  J. Global Optim. 2 (1992) 101-112 -- the products-as-variables relaxation.
Burer & Letchford, "Non-convex mixed-integer nonlinear programming: a survey",
  Surveys in OR and Management Science 17 (2012) 97-106.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.problem import Problem, Solution, Status
from ..core.sparse import IDX, SparseMatrix, VAL
from ..core.tolerances import INF
from .alphabb import AlphaBBParams, solve_alphabb
from .bilinear import BilinearProblem, BilinearTerm
from .spatial import SpatialParams, solve_global

__all__ = ["NonconvexQPParams", "qp_to_bilinear", "solve_nonconvex_qp"]


@dataclass
class NonconvexQPParams:
    time_limit: float = 300.0
    gap_rel: float = 1e-4
    gap_abs: float = 1e-9
    relaxation: str = "auto"
    """``"mccormick"`` (products as variables, LP nodes), ``"alphabb"``
    (diagonal shift, convex QP nodes), or ``"auto"``: McCormick unless the
    products would exceed ``max_products``, which is the one regime where a
    relaxation with no product variables is the only one that fits."""
    max_products: int = 5000
    verbose: bool = False
    spatial: SpatialParams | None = None
    alphabb: AlphaBBParams | None = None


def _product_bounds(li, ui, lj, uj, square: bool):
    if square:
        if li >= 0.0:
            return li * li, ui * ui
        if ui <= 0.0:
            return ui * ui, li * li
        return 0.0, max(li * li, ui * ui)
    corners = (li * lj, li * uj, ui * lj, ui * uj)
    return min(corners), max(corners)


def qp_to_bilinear(prob: Problem) -> BilinearProblem:
    """Rewrite ``½xᵀQx`` as a linear function of product variables.

    Each unordered pair ``(i, j)`` with a non-zero in ``Q`` gets a column
    ``w_ij = x_i x_j`` with the objective coefficient ``½(Q_ij + Q_ji)`` --
    which is ``½Q_ii`` on the diagonal and ``Q_ij`` off it for a symmetric
    ``Q`` -- and bounds from the corners of the factors' box. Rows, integrality
    and the sense carry over unchanged; the products are continuous.
    """
    Q, n, m = prob.Q, prob.n, prob.m
    cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(Q.cp))
    rows = Q.ci.astype(np.int64)
    key_lo = np.minimum(rows, cols)
    key_hi = np.maximum(rows, cols)
    keys = key_lo * n + key_hi
    uniq, inverse = np.unique(keys, return_inverse=True)
    coef = np.zeros(uniq.size, dtype=VAL)
    np.add.at(coef, inverse, 0.5 * Q.cx)
    keep = coef != 0.0
    uniq, coef = uniq[keep], coef[keep]
    pi = (uniq // n).astype(np.int64)
    pj = (uniq % n).astype(np.int64)
    k = uniq.size

    lo = np.concatenate([prob.col_lb, np.zeros(k)])
    hi = np.concatenate([prob.col_ub, np.zeros(k)])
    for t in range(k):
        i, j = int(pi[t]), int(pj[t])
        li, ui = float(prob.col_lb[i]), float(prob.col_ub[i])
        lj, uj = float(prob.col_lb[j]), float(prob.col_ub[j])
        if li <= -INF or ui >= INF or lj <= -INF or uj >= INF:
            lo[n + t], hi[n + t] = -INF, INF
        else:
            lo[n + t], hi[n + t] = _product_bounds(li, ui, lj, uj, i == j)

    A = SparseMatrix.from_triplets(
        np.repeat(np.arange(m, dtype=IDX), np.diff(prob.A.rp)),
        prob.A.ri.astype(IDX), prob.A.rx, m, n + k)
    kind = np.concatenate([prob.kind, np.zeros(k, dtype=np.uint8)])
    names = None
    if prob.col_names:
        names = list(prob.col_names) + [
            f"w[{prob.col_names[i]}*{prob.col_names[j]}]"
            for i, j in zip(pi, pj)]
    lin = Problem(A=A, c=np.concatenate([prob.c, coef]),
                  row_lb=prob.row_lb.copy(), row_ub=prob.row_ub.copy(),
                  col_lb=lo, col_ub=hi, kind=kind, obj_offset=prob.obj_offset,
                  sense=prob.sense, name=prob.name, col_names=names,
                  row_names=list(prob.row_names) if prob.row_names else None)
    terms = [BilinearTerm(w=n + t, x=int(pi[t]), y=int(pj[t])) for t in range(k)]
    return BilinearProblem(linear=lin, terms=terms, name=prob.name)


def _implied_bounds(prob: Problem) -> Problem:
    """Tighten the box by activity-based propagation before anything else.

    QPLIB_0018 is ``x >= 0`` with ``Σx = 1`` and no upper bounds written
    down; every variable is bounded by 1 and the file does not say so. The
    same propagation the MIP tree runs at every node finds that in one pass,
    and a model refused for "no finite bounds" when its rows imply them
    would be a refusal on a technicality.
    """
    from ..mip.propagate import propagate
    r = propagate(prob.A, prob.row_lb, prob.row_ub, prob.col_lb, prob.col_ub,
                  prob.integer_mask, max_rounds=10, feas_tol=1e-9)
    if r.infeasible:
        return prob
    tightened = prob.copy()
    tightened.col_lb = r.lo
    tightened.col_ub = r.hi
    return tightened


def _refuse_unbounded(prob: Problem, bp: BilinearProblem):
    touched = sorted({t.x for t in bp.terms} | {t.y for t in bp.terms})
    for j in touched:
        if prob.col_lb[j] <= -INF or prob.col_ub[j] >= INF:
            name = prob.col_names[j] if prob.col_names else f"x[{j}]"
            raise ValueError(
                f"non-convex QP needs finite bounds on every variable in a "
                f"quadratic term; {name} has none, and the rows imply none")


def solve_nonconvex_qp(prob: Problem,
                       params: NonconvexQPParams | None = None) -> Solution:
    """Solve a non-convex (MI)QP to global optimality within ``gap_rel``."""
    params = params or NonconvexQPParams()
    if prob.Q is None:
        raise ValueError("solve_nonconvex_qp needs a quadratic objective")
    original = prob
    prob = _implied_bounds(prob)

    relaxation = params.relaxation
    if relaxation == "auto":
        Q = prob.Q
        n_products = (Q.nnz + int((Q.ci == np.repeat(np.arange(Q.n), np.diff(Q.cp))).sum())) // 2
        relaxation = "alphabb" if n_products > params.max_products else "mccormick"
    if relaxation == "alphabb":
        ap = params.alphabb or AlphaBBParams()
        ap.time_limit, ap.gap_rel, ap.gap_abs = \
            params.time_limit, params.gap_rel, params.gap_abs
        ap.verbose = params.verbose
        sol = solve_alphabb(prob, ap)
        sol.method = "nonconvex-qp[alphabb]"
        return sol
    if relaxation != "mccormick":
        raise ValueError(f"unknown relaxation {params.relaxation!r}")

    bp = qp_to_bilinear(prob)
    _refuse_unbounded(prob, bp)
    sp = params.spatial or SpatialParams()
    sp.time_limit, sp.gap_rel, sp.gap_abs = \
        params.time_limit, params.gap_rel, params.gap_abs
    sp.verbose = params.verbose
    inner = solve_global(bp, sp)

    n = prob.n
    sol = Solution(status=inner.status, nodes=inner.nodes, time=inner.time,
                   dual_bound=inner.dual_bound, method="nonconvex-qp[mccormick]")
    sol.info = {**(getattr(inner, "info", None) or {}), "relaxation": "mccormick",
                "products": len(bp.terms)}
    if inner.x is not None:
        x = np.asarray(inner.x[:n], dtype=VAL)
        # the answer is re-validated against the model that was asked, not
        # the reformulation that was solved
        row_v, col_v, int_v = original.violation(x)
        if max(row_v, col_v) > 1e-6 or int_v > sp.integrality:
            sol.status = Status.NUMERICAL
            return sol.drop_objective_if_unsolved()
        sol.x = x
        sol.objective = original.objective(x)
    return sol.drop_objective_if_unsolved()
