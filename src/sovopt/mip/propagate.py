"""Domain propagation: activity-based bound tightening to a fixpoint.

Propagation is the cheapest thing a MILP solver does and one of the most
valuable. At every node it takes the branching decision that just happened and
pushes its consequences through the constraint matrix, often fixing dozens of
other variables before any LP is solved. On refinery scheduling models, where a
single tank assignment forces a cascade of timing decisions, it routinely does
more work than the LP relaxation.

The rule
--------
For a row ``rl <= Σ_j a_j x_j <= ru`` define the extreme activities over the
current box:

    minact = Σ_j  (a_j > 0 ? a_j·l_j : a_j·u_j)
    maxact = Σ_j  (a_j > 0 ? a_j·u_j : a_j·l_j)

If ``minact > ru`` or ``maxact < rl`` the node is infeasible. Otherwise, for
each variable in the row, remove its own contribution and solve the row for it:

    a_j·x_j <= ru − (minact − own_min)
    a_j·x_j >= rl − (maxact − own_max)

and divide by ``a_j``, flipping the inequality when ``a_j < 0``. Integer
variables then get the bound rounded inwards, which is where most of the
strength comes from: a bound of ``x <= 2.4`` on an integer becomes ``x <= 2``,
which can trigger another round.

Infinite bounds
---------------
The subtlety that implementations get wrong. If any contribution is infinite the
activity is infinite and nothing can be deduced -- *except* for the one variable
responsible for it. So the counts of infinite contributions are tracked
separately from the finite part of the sum, and a variable can still be
tightened when it is the *only* source of infinity in its row. Dropping this
case loses a large fraction of the available reductions on models with free or
unbounded variables, which is most industrial models before presolve.

Numerical safety
----------------
Bounds are only accepted when they improve on the current one by a relative
margin, and tightening is skipped for rows whose coefficients span too wide a
range: a bound derived by dividing by a tiny pivot is numerically meaningless
and can cut off the optimum. Correctness beats strength here.

References
----------
Savelsbergh, "Preprocessing and probing techniques for mixed integer programming
  problems", ORSA J. Computing 6 (1994) 445-454.
Brearley, Mitra & Williams, "Analysis of mathematical programming problems prior
  to applying the simplex algorithm", Math. Prog. 8 (1975) 54-83.
Achterberg, *Constraint Integer Programming*, PhD thesis, TU Berlin 2007, §7.1.
Gleixner, Berthold, Müller & Weltge, "Three enhancements for optimization-based
  bound tightening", J. Global Optim. 67 (2017).
"""

from __future__ import annotations

import numpy as np

from ..core._jit import jit_kernel
from ..core.sparse import IDX, VAL
from ..core.tolerances import INF

__all__ = ["propagate", "PropagationResult"]

INFEASIBLE = -1
UNCHANGED = 0
TIGHTENED = 1


class PropagationResult:
    __slots__ = ("status", "rounds", "n_tightened", "n_fixed", "lo", "hi")

    def __init__(self, status, rounds, n_tightened, n_fixed, lo, hi):
        self.status = status
        self.rounds = rounds
        self.n_tightened = n_tightened
        self.n_fixed = n_fixed
        self.lo = lo
        self.hi = hi

    @property
    def infeasible(self) -> bool:
        return self.status == INFEASIBLE

    def __repr__(self):
        s = "INFEASIBLE" if self.infeasible else "ok"
        return (f"PropagationResult({s}, rounds={self.rounds}, "
                f"tightened={self.n_tightened}, fixed={self.n_fixed})")


@jit_kernel()
def _row_activities(rp, ri, rx, lo, hi, minf, maxf, mininf, maxinf):
    """Finite parts and infinite-contribution counts of every row activity."""
    m = rp.shape[0] - 1
    for i in range(m):
        s_min = 0.0
        s_max = 0.0
        n_min = 0
        n_max = 0
        for p in range(rp[i], rp[i + 1]):
            a = rx[p]
            j = ri[p]
            l = lo[j]
            u = hi[j]
            if a > 0.0:
                if l <= -INF:
                    n_min += 1
                else:
                    s_min += a * l
                if u >= INF:
                    n_max += 1
                else:
                    s_max += a * u
            else:
                if u >= INF:
                    n_min += 1
                else:
                    s_min += a * u
                if l <= -INF:
                    n_max += 1
                else:
                    s_max += a * l
        minf[i] = s_min
        maxf[i] = s_max
        mininf[i] = n_min
        maxinf[i] = n_max


@jit_kernel()
def _propagate_rows(rp, ri, rx, row_lb, row_ub, lo, hi, is_int,
                    minf, maxf, mininf, maxinf,
                    feas_tol, bound_tol, max_ratio, changed_flags):
    """One sweep over every row. Returns ``(status, n_tightened)``."""
    m = rp.shape[0] - 1
    n_tight = 0

    for i in range(m):
        rlo = row_lb[i]
        rhi = row_ub[i]
        if rlo <= -INF and rhi >= INF:
            continue

        s_min = minf[i]
        s_max = maxf[i]
        n_min = mininf[i]
        n_max = maxinf[i]

        # --- infeasibility of the row itself ---
        if n_min == 0 and rhi < INF and s_min > rhi + feas_tol:
            return INFEASIBLE, n_tight
        if n_max == 0 and rlo > -INF and s_max < rlo - feas_tol:
            return INFEASIBLE, n_tight

        # skip rows whose coefficients are too spread out to divide by safely
        amax = 0.0
        amin = INF
        for p in range(rp[i], rp[i + 1]):
            v = abs(rx[p])
            if v > amax:
                amax = v
            if v < amin and v > 0.0:
                amin = v
        if amin <= 0.0 or amax / amin > max_ratio:
            continue

        for p in range(rp[i], rp[i + 1]):
            a = rx[p]
            j = ri[p]
            if a == 0.0:
                continue
            l = lo[j]
            u = hi[j]

            # --- residual activity with variable j removed ---
            if a > 0.0:
                own_min_inf = l <= -INF
                own_min = 0.0 if own_min_inf else a * l
                own_max_inf = u >= INF
                own_max = 0.0 if own_max_inf else a * u
            else:
                own_min_inf = u >= INF
                own_min = 0.0 if own_min_inf else a * u
                own_max_inf = l <= -INF
                own_max = 0.0 if own_max_inf else a * l

            r_min_inf = n_min - (1 if own_min_inf else 0)
            r_max_inf = n_max - (1 if own_max_inf else 0)
            r_min = s_min - own_min
            r_max = s_max - own_max

            new_l = -INF
            new_u = INF

            # a_j x_j <= ru - residual_min
            if rhi < INF and r_min_inf == 0:
                t = (rhi - r_min) / a
                if a > 0.0:
                    new_u = t
                else:
                    new_l = t
            # a_j x_j >= rl - residual_max
            if rlo > -INF and r_max_inf == 0:
                t = (rlo - r_max) / a
                if a > 0.0:
                    if t > new_l:
                        new_l = t
                else:
                    if t < new_u:
                        new_u = t

            if is_int[j]:
                if new_l > -INF:
                    new_l = np.ceil(new_l - feas_tol)
                if new_u < INF:
                    new_u = np.floor(new_u + feas_tol)

            improved = False
            if new_l > -INF and new_l > l + bound_tol * (1.0 + abs(l)):
                lo[j] = new_l
                l = new_l
                improved = True
            if new_u < INF and new_u < u - bound_tol * (1.0 + abs(u)):
                hi[j] = new_u
                u = new_u
                improved = True

            if improved:
                n_tight += 1
                changed_flags[j] = True
                if lo[j] > hi[j] + feas_tol:
                    return INFEASIBLE, n_tight

    return TIGHTENED if n_tight > 0 else UNCHANGED, n_tight


def propagate(A, row_lb, row_ub, col_lb, col_ub, is_int,
              max_rounds: int = 10, feas_tol: float = 1e-9,
              bound_tol: float = 1e-9, max_ratio: float = 1e8,
              inplace: bool = False) -> PropagationResult:
    """Tighten variable bounds until nothing changes.

    ``A`` is a :class:`~sovopt.core.sparse.SparseMatrix`. Returns the tightened
    bounds; the input arrays are left alone unless ``inplace``.
    """
    lo = col_lb if inplace else np.array(col_lb, dtype=VAL, copy=True)
    hi = col_ub if inplace else np.array(col_ub, dtype=VAL, copy=True)
    is_int = np.ascontiguousarray(is_int, dtype=np.bool_)
    row_lb = np.ascontiguousarray(row_lb, dtype=VAL)
    row_ub = np.ascontiguousarray(row_ub, dtype=VAL)

    m, n = A.shape
    minf = np.zeros(m, dtype=VAL)
    maxf = np.zeros(m, dtype=VAL)
    mininf = np.zeros(m, dtype=np.int32)
    maxinf = np.zeros(m, dtype=np.int32)
    changed = np.zeros(n, dtype=np.bool_)

    total = 0
    status = UNCHANGED
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        _row_activities(A.rp, A.ri, A.rx, lo, hi, minf, maxf, mininf, maxinf)
        changed[:] = False
        st, n_tight = _propagate_rows(
            A.rp, A.ri, A.rx, row_lb, row_ub, lo, hi, is_int,
            minf, maxf, mininf, maxinf,
            feas_tol, bound_tol, max_ratio, changed)
        total += n_tight
        if st == INFEASIBLE:
            status = INFEASIBLE
            break
        if st == UNCHANGED:
            status = TIGHTENED if total else UNCHANGED
            break
        status = TIGHTENED

    n_fixed = int(np.sum(hi - lo <= feas_tol)) if status != INFEASIBLE else 0
    return PropagationResult(status, rounds, total, n_fixed, lo, hi)
