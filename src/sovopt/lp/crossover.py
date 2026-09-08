"""Crossover: turn an interior point into an optimal basis.

An interior-point method and a first-order method both stop at a point that is
*not* a vertex. That costs three things a basis would give:

* **no ranging.** Cost and RHS ranging are read off a basis inverse. A planner
  asking "how far can this crude's price move before the plan changes" is
  asking a question only a basis can answer, and :mod:`sovopt.lp.sensitivity`
  needs one.
* **no warm start.** A branch-and-bound node re-solves its parent's LP with one
  bound changed. From a basis that is a handful of dual pivots; from an interior
  point it is a solve from scratch.
* **an answer that is optimal but not extreme.** On a degenerate model the
  interior point converges to the analytic centre of the optimal face -- every
  point of which is optimal, so the objective is right, but the *solution* is a
  blend where a planner expects a corner.

What this does, and what it does not
------------------------------------
The rigorous construction is Megiddo's: push the interior point to a vertex
along the optimal face, one variable at a time, provably without leaving
optimality. What is implemented here is the practical variant that Bixby and
Saltzman describe and that solvers actually ship -- **basis identification
followed by simplex cleanup**:

1. Score every variable of ``[x ; s]`` by how far it sits from its nearest
   bound, relative to its own range.
2. Starting from the all-logical basis, **pivot** candidates in worst-last, one
   FTRAN each, taking each pivot on the largest entry in a row that still holds
   its original logical.
3. Hand the result to the revised simplex as a warm start and let it finish.

Step 2 is a rank test, and it is not optional. Ranking alone -- taking the
``m`` best-scoring variables and calling them a basis -- produces a set that is
almost always singular, because at an optimum only a handful of variables sit
strictly between their bounds and the rest of the picks are ties at zero. Tried
that way first on seven MIPLIB relaxations, the simplex rejected the basis
every single time and fell back to the all-logical one: identical pivot counts
to a cold start, and dcmulti eighteen times *slower* for the wasted repair.

Step 3 is what makes the result *exact* rather than nearly so, and it is why
this is honest without a proof of its own: the simplex terminates only at an
optimal basis, so a poor identification costs pivots and never costs
correctness.

The measurement that matters is therefore not "is it right" but "did it save
anything", and that is what the tests and `bench/` assert: same objective as a
cold simplex, from far fewer pivots.

References
----------
Megiddo, "On finding primal- and dual-optimal bases", ORSA J. Computing 3
  (1991) 63-65 -- the existence result and the push.
Bixby & Saltzman, "Recovering an optimal LP basis from an interior point
  solution", Oper. Res. Letters 15 (1994) 169-178 -- identification by ranking
  the distance to the nearest bound, which is the scoring used here.
Andersen & Ye, "Combining interior-point and pivoting algorithms for linear
  programming", Management Science 42 (1996) 1719-1731 -- identification
  followed by a simplex clean-up, and why the clean-up is cheap when the
  identification is good.
Andersen, "Finding all linearly dependent rows in large-scale linear
  programming", Optim. Methods Softw. 6 (1995) 219-227 -- why a candidate set
  may be rank deficient and what to do about it.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.problem import ObjSense, Problem, Solution, Status
from ..core.sparse import VAL
from ..core.tolerances import INF
from .basis import AT_LOWER, AT_UPPER, BASIC, FIXED, FREE

__all__ = ["basis_from_point", "crossover"]


def basis_from_point(prob: Problem, x, y=None, params=None,
                     tol_fixed: float = 1e-12):
    """Identify a candidate basis from a primal point, as a status vector.

    Returns the ``n + m`` vector the simplex takes as ``warm_basis``: exactly
    ``m`` entries marked ``BASIC``, the rest resting on a bound. The variable
    order is the basis module's own -- structurals first, then one logical per
    row, bounded by that row's bounds -- which is the same ``[x ; s]`` space
    the interior point already works in, so no translation is needed.

    **The candidates are pivoted in, not merely ranked.** Scoring alone gives a
    set that is almost always rank deficient: at an optimum only a handful of
    variables sit strictly between their bounds, so the rest of the ``m`` picks
    are ties at a score of zero and arbitrary. Measured that way on seven
    MIPLIB relaxations, the simplex rejected the basis every time and fell back
    to the all-logical one -- identical pivot counts to a cold start, and
    dcmulti eighteen times *slower*, because the factorisation spent its
    singularity-repair budget swapping the whole set back out again.

    So the construction starts from the all-logical basis, which is ``-I`` and
    nonsingular by inspection, and pushes candidates in one at a time: for each
    the FTRAN gives ``B⁻¹M[:,j]``, and the pivot is taken on the largest
    entry in a row that still holds its original logical. Every pivot is on a
    nonzero, so the basis is nonsingular at every step and the rank test comes
    free with the arithmetic that was needed anyway.
    """
    from .basis import Basis
    from .simplex import SimplexParams

    params = params or SimplexParams()
    m, n = prob.m, prob.n
    N = n + m

    x = np.asarray(x, dtype=VAL)
    s = prob.A.matvec(x)
    z = np.concatenate([x, s])

    lower = np.concatenate([prob.col_lb, prob.row_lb]).astype(VAL)
    upper = np.concatenate([prob.col_ub, prob.row_ub]).astype(VAL)
    has_lo = lower > -INF
    has_hi = upper < INF
    fixed = has_lo & has_hi & (upper - lower
                               <= tol_fixed * np.maximum(1.0, np.abs(lower)))
    free = ~has_lo & ~has_hi

    # Reduced costs over the same space. The basis is over [A | -I] with zero
    # cost on the logicals, so d = c - Aᵀy on the structurals and d = +y on the
    # logicals. Used only to break ties, so an absent y is not fatal.
    d = np.zeros(N, dtype=VAL)
    if y is not None:
        y = np.asarray(y, dtype=VAL)
        d[:n] = prob.c - prob.A.rmatvec(y)
        d[n:] = y

    # How interior is each variable, relative to its own range? A variable
    # halfway between its bounds is a far better basic candidate than one a
    # rounding error away from one, and normalising by the range is what stops
    # a wide column outranking a narrow one merely for being wide.
    span = np.where(has_lo & has_hi, upper - lower, 0.0)
    scale = np.maximum(1.0, np.where(span > 0, span, np.abs(z)))
    dist_lo = np.where(has_lo, z - lower, np.inf)
    dist_hi = np.where(has_hi, upper - z, np.inf)
    interior = np.minimum(dist_lo, dist_hi) / scale
    interior[~np.isfinite(interior)] = np.inf        # free: maximally interior

    # A near-zero reduced cost is one complementarity is not pinning to a
    # bound, so it is the better candidate at equal interiority.
    score = interior - 1e-9 * np.abs(d)
    score[fixed] = -np.inf                           # pinned: never basic
    score[free] = np.inf                             # free: always basic

    # ---- push candidates into the logical basis --------------------------- #
    B = Basis(prob, lu_tol=params.lu_tol if hasattr(params, "lu_tol") else 0.01)
    B.set_logical_basis()
    B.factorize()

    # Only structurals are pushed: a logical is already basic, and a candidate
    # that is not measurably off its bound has nothing to recommend it over the
    # logical sitting in the row it would take.
    live = np.flatnonzero((score[:n] > 1e-9) & np.isfinite(score[:n]))
    order = live[np.argsort(-score[live], kind="stable")]
    forced = np.flatnonzero(free[:n])
    order = np.concatenate([forced, order[~np.isin(order, forced)]])

    taken = np.zeros(m, dtype=bool)
    alpha = np.zeros(m, dtype=VAL)
    pivot_tol = max(params.pivot_tol if hasattr(params, "pivot_tol") else 1e-7,
                    1e-7)
    placed = 0
    for j in order:
        if placed == m:
            break
        B.ftran_column(int(j), alpha)
        mag = np.where(taken, 0.0, np.abs(alpha))
        r = int(np.argmax(mag))
        if mag[r] <= pivot_tol:
            continue                                 # dependent on what is in
        B.update(r, int(j), alpha)
        taken[r] = True
        placed += 1
        if B.needs_refactorization:
            B.factorize()

    # ---- statuses: basic from the construction, the rest by nearest bound -- #
    at_lower = (np.abs(z - lower) <= np.abs(upper - z)) | ~has_hi
    status = np.where(at_lower & has_lo, AT_LOWER,
                      np.where(has_hi, AT_UPPER, FREE)).astype(np.uint8)
    status[fixed] = FIXED
    status[B.basic] = BASIC
    return status


def crossover(prob: Problem, x, y=None, params=None) -> Solution:
    """Cross over from an interior point to an optimal basic solution.

    ``x`` is a primal point in the caller's own sense and units -- whatever the
    interior point or PDLP returned. The simplex finishes the job, so the
    result is an exact optimum with a basis, duals and reduced costs, and is
    usable for ranging.
    """
    from .simplex import SimplexParams, solve_simplex

    params = params or SimplexParams()
    t0 = time.perf_counter()

    # The scored basis is combinatorial, so it is chosen in the caller's units
    # and stays valid under the scaling solve_simplex applies internally. Sign
    # is the one thing that does not survive: the simplex minimises internally,
    # and the tie-break above reads the sign of a reduced cost.
    yy = y
    if y is not None and prob.sense == ObjSense.MAXIMISE:
        yy = -np.asarray(y, dtype=VAL)
    work = prob
    if prob.sense == ObjSense.MAXIMISE:
        work = prob.copy()
        work.c = -work.c

    status = basis_from_point(work, x, yy)
    sol = solve_simplex(prob, params, warm_basis=status)
    sol.info = dict(getattr(sol, "info", {}))
    sol.info["crossover"] = True
    sol.info["crossover_time"] = time.perf_counter() - t0
    return sol
