"""Rigorously valid dual bounds from *approximate* dual vectors.

This module is what makes GPU-computed bounds safe to prune with.

The problem it solves
---------------------
A first-order method produces an approximate primal-dual pair. Its dual iterate
is not dual feasible, so its dual objective is not a valid bound and cannot be
used to discard a subtree -- an over-optimistic bound cuts off the optimum and
the solver silently returns a wrong answer. The usual answer is "solve the node
LP to optimality with a simplex", which throws away the GPU.

The Neumaier-Shcherbina correction removes the dilemma. For **any** vector ``y``,
feasible or not, it produces a bound that is genuinely valid.

Derivation
----------
For any ``y``, write ``d = c − Aᵀy``. Then for every ``x``:

    cᵀx  =  dᵀx + yᵀ(Ax)

Every feasible ``x`` of the node satisfies ``rl <= Ax <= ru`` and ``l <= x <= u``,
so each term can be bounded below independently, coordinate by coordinate:

    yᵀ(Ax) >= Σ_i  (y_i > 0 ? y_i·rl_i : y_i·ru_i)
    dᵀx    >= Σ_j  (d_j > 0 ? d_j·l_j  : d_j·u_j)

Adding them gives a lower bound on ``cᵀx`` over the whole node. It requires
nothing of ``y`` -- no sign condition, no complementarity, no convergence. A
better ``y`` gives a tighter bound; a poor ``y`` gives a weak but still valid
one, and a term with an infinite bound on the wrong side simply gives ``-inf``,
which prunes nothing and is still correct.

That is the property that lets an unconverged GPU iterate drive an exact search.

The same idea for a convex quadratic
------------------------------------
For ``f(x) = ½xᵀQx + cᵀx`` with ``Q ⪰ 0`` and *any* point ``x̂``, convexity says
the tangent plane at ``x̂`` lies below ``f`` everywhere:

    f(x)  >=  f(x̂) + g(x̂)ᵀ(x − x̂),        g(x̂) = Qx̂ + c

(the difference is ``½(x−x̂)ᵀQ(x−x̂)``, which is what ``Q ⪰ 0`` means). The
right-hand side is *linear* in ``x``, so the bound above applies to it with
``g(x̂)`` in the place of ``c``, and

    f(x)  >=  −½ x̂ᵀQx̂  +  LP-bound(A, g(x̂), rl, ru, l, u, y)

for every feasible ``x``, every ``x̂`` and every ``y``. At an optimal pair the
reduced costs ``g − Aᵀy`` have the signs complementarity demands and the bound
is exactly ``f(x*)``; away from one it is weaker and still valid. So a
first-order QP solve stopped at any accuracy still yields a node bound that is
safe to prune on -- which is what turns branch-and-bound over QP relaxations
from "probably right" into "right".

Rigour here is conditional on ``Q ⪰ 0``, which is the caller's premise, not
this function's finding: the convex QP path checks it by estimating the
smallest eigenvalue. A caller that can *certify* convexity -- a diagonal shift
large enough for Gershgorin, say -- makes the bound unconditional.

Floating point
--------------
The bound above is exact in real arithmetic. In floating point the accumulation
itself rounds, so ``strict=True`` walks the sum with a compensated accumulator
and then subtracts a conservative error allowance, yielding a bound that holds
even against the rounding. The allowance is tiny (a few ulps of the magnitudes
involved) and costs one extra pass.

References
----------
Neumaier & Shcherbina, "Safe bounds in linear and mixed-integer linear
  programming", Math. Prog. 99 (2004) 283-296.
Cook, Koch, Steffy & Wolter, "A hybrid branch-and-bound approach for exact
  rational mixed-integer programming", Math. Prog. Computation 5 (2013).
Althaus & Dumitriu, "Certifying feasibility and objective value of linear
  programs", Oper. Res. Letters 40 (2012).
Fletcher & Leyffer, "Numerical experience with lower bounds for MIQP
  branch-and-bound", SIAM J. Optim. 8 (1998) 604-616 -- the Lagrangian bound
  for a convex QP relaxation that the quadratic form above makes safe.
"""

from __future__ import annotations

import numpy as np

from ..core.tolerances import INF

__all__ = ["safe_dual_bound", "safe_qp_bound", "safe_dual_bound_batch",
           "bound_slack"]


def _term(coef, lo, hi, xp):
    """``Σ min(coef·lo, coef·hi)`` with infinities handled as ``-inf``."""
    pos = coef > 0.0
    chosen = xp.where(pos, lo, hi)
    bad = xp.where(pos, lo <= -INF, hi >= INF) & (coef != 0.0)
    contrib = xp.where(bad, -xp.inf, coef * xp.where(bad, 0.0, chosen))
    return contrib


def safe_dual_bound(A, c, row_lb, row_ub, col_lb, col_ub, y,
                    strict: bool = False) -> float:
    """A valid lower bound on ``min cᵀx`` over the node, for arbitrary ``y``.

    ``y`` uses the textbook sign convention (``d = c − Aᵀy``). Returns ``-inf``
    when the bound is vacuous, which is always safe.
    """
    y = np.asarray(y, dtype=np.float64)
    d = c - A.rmatvec(y)

    row_terms = _term(y, row_lb, row_ub, np)
    col_terms = _term(d, col_lb, col_ub, np)

    if not np.isfinite(row_terms).all() or not np.isfinite(col_terms).all():
        return -np.inf

    if strict:
        # compensated (Neumaier) summation, then back off by a conservative
        # allowance so the reported bound survives its own rounding
        total, comp = _neumaier_sum(row_terms)
        t2, c2 = _neumaier_sum(col_terms)
        total, comp = total + t2, comp + c2
        value = total + comp
        eps = np.finfo(np.float64).eps
        mag = float(np.abs(row_terms).sum() + np.abs(col_terms).sum())
        return float(value - 4.0 * eps * mag)

    return float(row_terms.sum() + col_terms.sum())


def safe_qp_bound(A, c, Q, row_lb, row_ub, col_lb, col_ub, x_hat, y,
                  strict: bool = False) -> float:
    """A valid lower bound on ``min ½xᵀQx + cᵀx`` over the node, for any
    ``x_hat`` and any ``y``, given ``Q ⪰ 0``.

    ``x_hat`` is the point the tangent plane is taken at -- the QP solver's
    iterate, converged or not. ``Q`` may be ``None``, in which case this is
    exactly :func:`safe_dual_bound`. Returns ``-inf`` when vacuous.
    """
    if Q is None:
        return safe_dual_bound(A, c, row_lb, row_ub, col_lb, col_ub, y,
                               strict=strict)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    qx = Q.matvec(x_hat)
    g = c + qx
    linear = safe_dual_bound(A, g, row_lb, row_ub, col_lb, col_ub, y,
                             strict=strict)
    if not np.isfinite(linear):
        return -np.inf
    # −½ x̂ᵀQx̂, and in strict mode an allowance for the rounding in Qx̂ and
    # in the products that formed it
    prods = x_hat * qx
    if strict:
        s, comp = _neumaier_sum(prods)
        eps = np.finfo(np.float64).eps
        mag = float(np.abs(prods).sum())
        # each qx_i carries at most nnz(row i) roundings of its own terms;
        # bounding that by nnz(Q)·eps·|x̂|·‖Q‖_max·|x̂| is loose and still tiny
        qmag = float(np.abs(Q.cx).max(initial=0.0)) * float(np.abs(x_hat).max(initial=0.0)) ** 2
        allowance = 4.0 * eps * mag + Q.nnz * eps * qmag
        return float(linear - 0.5 * (s + comp) - allowance)
    return float(linear - 0.5 * float(prods.sum()))


def _neumaier_sum(a):
    """Compensated sum; returns ``(sum, correction)``."""
    s = 0.0
    comp = 0.0
    for v in a:
        t = s + v
        if abs(s) >= abs(v):
            comp += (s - t) + v
        else:
            comp += (v - t) + s
        s = t
    return s, comp


def safe_dual_bound_batch(bk, cp, ci, cx, c, row_lb, row_ub,
                          col_lb, col_ub, Y, n, m, out=None, aty_buf=None):
    """Valid lower bounds for a whole batch of nodes at once.

    ``Y`` is ``(m, K)`` with the node index fastest; ``col_lb``/``col_ub`` are
    ``(n, K)`` because branching changes only the variable bounds. ``row_lb`` and
    ``row_ub`` are shared ``(m,)`` vectors -- branching never touches rows, which
    is exactly why the batch shares one matrix.

    Returns ``(K,)`` lower bounds. This is one SpMM plus elementwise work, so a
    frontier of 256 nodes is bounded for roughly the cost of two single-node
    sparse products.
    """
    xp = bk.xp
    K = Y.shape[1]

    aty = bk.empty((n, K)) if aty_buf is None else aty_buf
    bk.spmm(cp, ci, cx, Y, aty, n, K)         # CSC of A == CSR of Aᵀ

    d = c[:, None] - aty

    ru_inf = (row_ub >= INF)[:, None]
    rl_inf = (row_lb <= -INF)[:, None]
    ypos = Y > 0.0
    row_bad = xp.where(ypos, rl_inf, ru_inf) & (Y != 0.0)
    row_pick = xp.where(ypos, row_lb[:, None], row_ub[:, None])
    row_contrib = xp.where(row_bad, 0.0, Y * xp.where(row_bad, 0.0, row_pick))
    row_vacuous = row_bad.any(axis=0)

    dpos = d > 0.0
    col_bad = xp.where(dpos, col_lb <= -INF, col_ub >= INF) & (d != 0.0)
    col_pick = xp.where(dpos, col_lb, col_ub)
    col_contrib = xp.where(col_bad, 0.0, d * xp.where(col_bad, 0.0, col_pick))
    col_vacuous = col_bad.any(axis=0)

    bounds = row_contrib.sum(axis=0) + col_contrib.sum(axis=0)
    bounds = xp.where(row_vacuous | col_vacuous, -xp.inf, bounds)

    if out is not None:
        out[:] = bounds
        return out
    return bounds


def bound_slack(bound: float, incumbent: float, gap_abs: float,
                gap_rel: float) -> bool:
    """Can this node be discarded against the incumbent?

    Uses the same absolute/relative test the tree reports its final gap with, so
    "pruned" and "proved optimal" mean the same thing.
    """
    if not np.isfinite(bound):
        return False
    if not np.isfinite(incumbent):
        return False
    return bound >= incumbent - max(gap_abs, gap_rel * abs(incumbent))
