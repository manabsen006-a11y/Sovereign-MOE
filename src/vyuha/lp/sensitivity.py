"""Sensitivity analysis: ranging on costs and right-hand sides.

This is what a planner actually reads off a solve. "The optimum is 4.2 crore" is
the least interesting number a refinery LP produces. The questions that change
decisions are:

* *What is one more tonne of this crude worth?* -- the shadow price.
* *How far can its price move before I should buy something else?* -- cost
  ranging.
* *Over what range of that capacity is this price still valid?* -- RHS ranging.

None of it can be answered without a **basis**, which is why this arrives with
the revised simplex and could never have come from the first-order method: an
interior point knows the optimal value but not which constraints are binding or
what the next unit of each is worth.

Cost ranging
------------
How far can ``c_j`` move before the current basis stops being optimal?

*Nonbasic* variables are easy: optimality only requires the reduced cost keep
its sign, and ``d_j`` moves one-for-one with ``c_j``. A variable at its lower
bound can have its cost fall by ``d_j`` before it becomes attractive, and can
rise without limit.

*Basic* variables are the interesting case. Changing ``c_j`` changes the duals,
hence *every* reduced cost. With ``alpha_r`` the tableau row of the basis
position holding ``j``, perturbing ``c_j`` by ``delta`` moves each nonbasic
``d_k`` by ``-delta * alpha_r[k]``, and a ratio test over the nonbasics gives
the interval on which no sign flips.

RHS ranging
-----------
How far can a row's active bound move before the basis stops being feasible?

If the row's logical variable is *basic* the row is not binding: its activity
sits strictly inside the bounds, its shadow price is zero, and the bound can be
moved right up to the current activity before anything happens.

If the logical is *nonbasic* the row is binding. Moving its bound by ``delta``
shifts the basic solution along ``beta = B^-1 e_i``, and a ratio test on the
basic variables' bounds gives the interval over which the basis -- and therefore
the shadow price -- remains valid.

Degeneracy, stated honestly
---------------------------
At a degenerate vertex several bases describe the same point, and the ranges
below are those of *the basis the solver happened to stop at*. A different
optimal basis gives different intervals, and a range of width zero is the normal
signature of degeneracy rather than a bug. Refinery LPs are massively
degenerate, so this caveat is not academic: a zero-width range means "this price
is one of several valid ones here", not "this constraint is infinitely
sensitive". Commercial solvers report the same quantity with the same caveat.

References
----------
Chvátal, *Linear Programming*, Freeman 1983, ch. 10 -- sensitivity analysis.
Bradley, Hax & Magnanti, *Applied Mathematical Programming*, Addison-Wesley
  1977, ch. 3 -- ranging and its interpretation for planners.
Maros, *Computational Techniques of the Simplex Method*, Kluwer 2003, §9.5.
Greenberg, "An analysis of degeneracy", Naval Research Logistics 33 (1986) --
  why ranges collapse at degenerate optima.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.sparse import VAL
from ..core.tolerances import INF
from .basis import AT_LOWER, AT_UPPER, BASIC, FIXED, FREE

__all__ = ["Sensitivity", "compute_sensitivity"]


@dataclass
class Sensitivity:
    """Ranging report, in the units of the *original* model."""

    # per variable
    value: np.ndarray = None
    reduced_cost: np.ndarray = None
    cost: np.ndarray = None
    cost_lo: np.ndarray = None
    cost_hi: np.ndarray = None

    # per row
    activity: np.ndarray = None
    dual: np.ndarray = None
    rhs: np.ndarray = None
    rhs_lo: np.ndarray = None
    rhs_hi: np.ndarray = None
    binding: np.ndarray = None

    col_names: list | None = None
    row_names: list | None = None
    degenerate_cols: int = 0
    degenerate_rows: int = 0

    def binding_rows(self):
        """Indices of rows whose shadow price is non-zero, most valuable first."""
        idx = np.flatnonzero(np.abs(self.dual) > 1e-9)
        return idx[np.argsort(-np.abs(self.dual[idx]))]

    def report(self, max_rows: int = 20, max_cols: int = 20) -> str:
        cn = self.col_names or [f"C{j}" for j in range(self.value.shape[0])]
        rn = self.row_names or [f"R{i}" for i in range(self.dual.shape[0])]
        out = []

        def fmt(v):
            if v <= -INF:
                return "-inf"
            if v >= INF:
                return "+inf"
            return f"{v:.6g}"

        out.append("  shadow prices (binding rows, by value)")
        out.append(f"    {'row':<20} {'dual':>14} {'activity':>14} "
                   f"{'rhs range':>30}")
        shown = 0
        for i in self.binding_rows():
            if shown >= max_rows:
                break
            out.append(f"    {rn[i][:20]:<20} {self.dual[i]:>14.6g} "
                       f"{self.activity[i]:>14.6g} "
                       f"{'[' + fmt(self.rhs_lo[i]) + ', ' + fmt(self.rhs_hi[i]) + ']':>30}")
            shown += 1
        if shown == 0:
            out.append("    (no binding rows)")

        out.append("")
        out.append("  cost ranging (variables at non-zero value)")
        out.append(f"    {'column':<20} {'value':>14} {'cost':>12} "
                   f"{'reduced':>12} {'cost range':>30}")
        nz = np.flatnonzero(np.abs(self.value) > 1e-9)
        for j in nz[:max_cols]:
            out.append(f"    {cn[j][:20]:<20} {self.value[j]:>14.6g} "
                       f"{self.cost[j]:>12.6g} {self.reduced_cost[j]:>12.6g} "
                       f"{'[' + fmt(self.cost_lo[j]) + ', ' + fmt(self.cost_hi[j]) + ']':>30}")
        if nz.size > max_cols:
            out.append(f"    ... {nz.size - max_cols} more")

        if self.degenerate_cols or self.degenerate_rows:
            out.append("")
            out.append(f"  note: {self.degenerate_cols} cost ranges and "
                       f"{self.degenerate_rows} rhs ranges have zero width, "
                       f"which means the optimum is degenerate -- these prices "
                       f"are valid but not unique.")
        return "\n".join(out)


def _basic_cost_range(S, r, d):
    """Interval of ``delta`` on ``c_j`` keeping every reduced cost feasible."""
    B = S.B
    S.rho.fill(0.0)
    S.rho[r] = 1.0
    B.btran(S.rho)
    alpha = B.pivot_row(S.rho)

    lo, hi = -np.inf, np.inf
    st = B.status
    for k in range(B.N):
        s = st[k]
        if s == BASIC or s == FIXED:
            continue
        a = alpha[k]
        if a == 0.0 or abs(a) < 1e-12:
            continue
        ratio = d[k] / a
        if s == AT_LOWER or (s == FREE and d[k] >= 0.0):
            # need d_k - delta*a >= 0
            if a > 0.0:
                hi = min(hi, ratio)
            else:
                lo = max(lo, ratio)
        else:
            # need d_k - delta*a <= 0
            if a > 0.0:
                lo = max(lo, ratio)
            else:
                hi = min(hi, ratio)
    return lo, hi


def _rhs_range(S, i, zB):
    """Interval of ``delta`` on a binding row's bound keeping the basis feasible."""
    B = S.B
    beta = np.zeros(B.m, dtype=VAL)
    beta[i] = 1.0
    B.ftran(beta)

    lo, hi = -np.inf, np.inf
    for r in range(B.m):
        a = beta[r]
        if abs(a) < 1e-12:
            continue
        j = B.basic[r]
        l, u = B.lower[j], B.upper[j]
        # z_B[r] + a*delta must stay in [l, u]
        if a > 0.0:
            if l > -INF:
                lo = max(lo, (l - zB[r]) / a)
            if u < INF:
                hi = min(hi, (u - zB[r]) / a)
        else:
            if u < INF:
                lo = max(lo, (u - zB[r]) / a)
            if l > -INF:
                hi = min(hi, (l - zB[r]) / a)
    return lo, hi


def compute_sensitivity(S, sc, prob, flip: bool) -> Sensitivity:
    """Build the ranging report from a solved simplex state.

    ``S`` is the internal solver state, ``sc`` the scaling applied to reach it,
    ``prob`` the original problem, and ``flip`` whether the objective was
    negated for a maximisation. Everything is converted back to the original
    units before it is returned.
    """
    B = S.B
    n, m = B.n, B.m
    zB = S.zB

    y = B.compute_duals()
    d_full = B.reduced_costs(y)

    z = B.nonbasic_values()
    z[B.basic] = zB

    cost_lo = np.full(n, -np.inf)
    cost_hi = np.full(n, np.inf)

    for j in range(n):
        st = B.status[j]
        if st == AT_LOWER:
            cost_lo[j] = B.cost[j] - d_full[j]
            cost_hi[j] = np.inf
        elif st == AT_UPPER:
            cost_lo[j] = -np.inf
            cost_hi[j] = B.cost[j] - d_full[j]
        elif st == FREE or st == FIXED:
            cost_lo[j] = -np.inf
            cost_hi[j] = np.inf

    for r in range(m):
        j = int(B.basic[r])
        if j >= n:
            continue                       # a logical, no objective coefficient
        dl, dh = _basic_cost_range(S, r, d_full)
        cost_lo[j] = B.cost[j] + dl
        cost_hi[j] = B.cost[j] + dh

    # ---- rows -------------------------------------------------------------
    activity = z[n:]                       # the logicals *are* the activities
    rhs = np.empty(m, dtype=VAL)
    rhs_lo = np.full(m, -np.inf)
    rhs_hi = np.full(m, np.inf)
    binding = np.zeros(m, dtype=bool)

    for i in range(m):
        jl = n + i
        st = B.status[jl]
        lo_i, hi_i = B.lower[jl], B.upper[jl]
        if st == BASIC:
            # not binding: the bound may travel to the current activity
            rhs[i] = lo_i if lo_i > -INF else hi_i
            rhs_lo[i] = -np.inf
            rhs_hi[i] = activity[i]
            if hi_i < INF and (lo_i <= -INF):
                rhs_lo[i] = activity[i]
                rhs_hi[i] = np.inf
        else:
            binding[i] = True
            b = lo_i if st == AT_LOWER or st == FIXED else hi_i
            rhs[i] = b
            dl, dh = _rhs_range(S, i, zB)
            rhs_lo[i] = b + dl
            rhs_hi[i] = b + dh

    # ---- unscale ----------------------------------------------------------
    col = sc.col
    row = sc.row
    obj = sc.obj if sc.obj else 1.0
    sgn = -1.0 if flip else 1.0

    def unscale_cost(v):
        out = v / (obj * col)
        out[v <= -INF] = -np.inf
        out[v >= INF] = np.inf
        return out * sgn

    c_lo = unscale_cost(cost_lo)
    c_hi = unscale_cost(cost_hi)
    if flip:
        c_lo, c_hi = c_hi, c_lo            # negation reverses the interval

    def unscale_rhs(v):
        out = v / row
        out[v <= -INF] = -np.inf
        out[v >= INF] = np.inf
        return out

    r_lo = unscale_rhs(rhs_lo)
    r_hi = unscale_rhs(rhs_hi)
    swap = row < 0                          # scales are positive, but be safe
    if np.any(swap):
        r_lo[swap], r_hi[swap] = r_hi[swap], r_lo[swap]

    x = np.clip(sc.unscale_primal(z[:n]), prob.col_lb, prob.col_ub)
    dual = sc.unscale_dual(y) * (-1.0 if flip else 1.0)
    red = sc.unscale_reduced(d_full[:n]) * (-1.0 if flip else 1.0)

    return Sensitivity(
        value=x,
        reduced_cost=red,
        cost=prob.c.copy(),
        cost_lo=c_lo, cost_hi=c_hi,
        activity=unscale_rhs(activity),
        dual=dual,
        rhs=unscale_rhs(rhs),
        rhs_lo=r_lo, rhs_hi=r_hi,
        binding=binding,
        col_names=prob.col_names, row_names=prob.row_names,
        degenerate_cols=int(np.sum(np.isfinite(c_lo) & np.isfinite(c_hi)
                                   & (c_hi - c_lo <= 1e-9))),
        degenerate_rows=int(np.sum(binding & np.isfinite(r_lo)
                                   & np.isfinite(r_hi)
                                   & (r_hi - r_lo <= 1e-9))),
    )
