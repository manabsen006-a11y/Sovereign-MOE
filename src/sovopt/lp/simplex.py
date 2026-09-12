"""Revised simplex: bounded-variable primal and dual.

This is the exact LP engine. Unlike the first-order method in
:mod:`sovopt.lp.pdlp` it terminates with a **basis**, which is what makes duals,
reduced costs, sensitivity ranging and warm-started branch-and-bound possible at
all. A first-order method can tell you the optimal value; only a basis tells you
which constraints are binding and what a marginal unit of each is worth.

Two algorithms over one basis
-----------------------------
**Primal** keeps every basic variable inside its bounds and drives the reduced
costs to feasibility. It is the right choice from a cold start.

**Dual** keeps the reduced costs feasible and drives the basic variables inside
their bounds. It is the right choice after a bound changes -- exactly what
branching does -- because a bound change destroys primal feasibility but leaves
dual feasibility untouched. That is why every branch-and-bound implementation
re-solves node LPs with dual simplex, and why it is the algorithm that matters
most for the MILP path.

The two share the ratio-test machinery below and differ only in which side of
the optimality conditions they repair.

Ratio tests
-----------
Both use a **Harris two-pass** test. The first pass computes the largest step
allowed when every bound is relaxed by the feasibility tolerance; the second
pass then picks, among all rows that could block within that relaxed step, the
one with the **largest pivot magnitude**. Taking the textbook minimum-ratio row
instead means routinely pivoting on a coefficient of 1e-9 because it happened to
tie, which destroys the factorisation a few iterations later. Trading a
tolerance-sized bound violation for a numerically sound pivot is the right
bargain, and it is what makes the method survive degenerate industrial models.

Degeneracy
----------
Refinery LPs are massively degenerate: huge numbers of bases give the same
objective, and a naive implementation stalls, cycling between them at zero step
length. Two defences are in place -- the Harris test above tolerates tiny
negative steps rather than jamming, and after a run of degenerate pivots the
costs are randomly perturbed, the perturbed problem solved, and the perturbation
removed with a clean-up phase.

Pricing
-------
Entering variables are chosen by **Devex** reference weights rather than by the
largest reduced cost. Dantzig's rule is scale-dependent and takes far more
iterations; Devex approximates steepest-edge pricing at a fraction of the cost by
maintaining weights against a reference framework, resetting when they drift.

The **dual** loop prices its leaving row the same way, which it did not always
do. ``_worst_infeasible`` has always scored rows as ``violation**2 / weight``,
the steepest-edge form, but nothing maintained the weights: they stayed at 1 and
the rule silently collapsed to Dantzig's. The cost of that is not subtle. On one
119-row relaxation the dual loop ran past 100,000 iterations without finishing,
with no degeneracy to blame -- not one pivot in 100,000 took a step below the
feasibility tolerance, and the mean step was 2.4e2. It was choosing badly, every
iteration. With the weights maintained the same solve takes 439 iterations.

References
----------
Dantzig, *Linear Programming and Extensions*, Princeton 1963.
Harris, "Pivot selection methods of the Devex LP code", Math. Prog. 5 (1973)
  1-28 -- the two-pass ratio test and Devex pricing.
Gill, Murray, Saunders & Wright, "A practical anti-cycling procedure for
  linearly constrained optimization", Math. Prog. 45 (1989) 437-474 -- EXPAND.
Forrest & Goldfarb, "Steepest-edge simplex algorithms for linear programming",
  Math. Prog. 57 (1992) 341-374.
Fourer, "Notes on the dual simplex method", 1994 -- the bounded-variable dual.
Koberstein, "The dual simplex method: techniques for a fast and stable
  implementation", PhD thesis, Paderborn 2005.
Maros, *Computational Techniques of the Simplex Method*, Kluwer 2003.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..core._jit import jit_kernel
from ..core.problem import ObjSense, Problem, Solution, Status
from ..core.sparse import VAL
from ..core.tolerances import INF
from ..numerics.scaling import scale_problem
from ..numerics.lu import LUSingular
from .basis import AT_LOWER, AT_UPPER, BASIC, FIXED, FREE, Basis

__all__ = ["SimplexParams", "solve_simplex"]


@dataclass
class SimplexParams:
    max_iter: int = 1_000_000
    time_limit: float = 300.0

    feas_tol: float = 1e-7
    opt_tol: float = 1e-7
    pivot_tol: float = 1e-9
    harris_relax: float = 1e-9

    node_kernel: bool = True
    """Run a node LP's dual loop as one compiled kernel
    (:mod:`sovopt.lp.nodelp`) instead of the Python loop below. Same pivots,
    five times faster per node on p0201, and no GIL held while it runs, which
    is what lets branch-and-bound use threads."""

    basis_update: str = "pfi"
    """``"pfi"`` (product form of the inverse) or ``"ft"`` (Forrest-Tomlin).

    Forrest-Tomlin is 2.4x cheaper per pivot at a budget of 1000 than the
    product form at the same budget, and no cheaper than the product form at
    150 -- and its path takes more pivots on most instances. The default is
    the measured winner; see :mod:`sovopt.numerics.ft` and
    docs/NEGATIVE-RESULTS.md."""

    refactor_freq: int = 150
    """Pivots between refactorisations of the basis.

    The product-form update appends an eta per pivot, so a longer budget
    means a longer eta chain on every FTRAN and BTRAN, traded against a
    full LU refactorisation. The README blamed the product form itself for
    10teams and named Forrest-Tomlin as the fix; measured, the budget was
    simply set too low. Sweeping it over the LP set, total solve time:

        20      40      60     100     150     250
        4.152s  3.622s  4.057s  2.576s  2.171s  2.585s

    150 is the best setting for seven of eleven instances, with 10teams 2.55x
    and qnet1 2.20x. Repeating the 60-against-150 comparison back to back gave
    5.2s against 2.8s and then 4.8s against 3.5s -- so the **direction is
    reproducible and the magnitude is not**: somewhere between 1.4x and 1.9x
    on total time, on a machine whose absolute speed drifts by about that much
    on its own. Accuracy is not traded for any of it: the verifier accepts
    11/11 at both settings across every run, and the worst relative error is
    5.10e-07 either way, to the digit.

    **The branch-and-bound tree overrides this back to 60**, because the
    two paths want opposite things and the reason is structural: a root LP
    does thousands of pivots from one basis and amortises a long chain,
    while a node LP does a handful from a warm start and pays for the
    chain without ever using it. Measured over the MIP set, 150 proves
    5/11 against 60's 6/11 and is 10% slower. See :mod:`sovopt.mip.tree`."""
    recompute_freq: int = 200
    """How often basic values are rebuilt from the factors rather than updated
    incrementally. Incremental updates drift; this bounds the drift."""

    devex_reset: float = 1e12
    perturb_after: int = 500
    """Consecutive degenerate pivots before the costs are perturbed."""

    perturb_scale: float = 1e-7
    sensitivity: bool = False
    """Compute cost and RHS ranging after solving. Costs one BTRAN per basic
    structural column and one FTRAN per binding row, so it is opt-in."""

    scaling: str = "auto"
    verbose: bool = False
    log_every: int = 5000


# --------------------------------------------------------------------------- #
# ratio tests                                                                  #
# --------------------------------------------------------------------------- #


@jit_kernel()
def _primal_ratio(zB, basic, lower, upper, alpha, dirn, range_q,
                  pivot_tol, relax, feas_tol):
    """Harris two-pass primal ratio test.

    Moving the entering variable by ``dirn * t`` moves each basic variable by
    ``-alpha[i] * dirn * t``. Returns ``(r, t, hit_upper, blocked)``:

    ``r``          basis position that blocks, or -1 for a bound flip / unbounded
    ``t``          step length in the entering variable
    ``hit_upper``  1 if the leaving variable lands on its upper bound
    ``blocked``    1 if some row blocks, 0 if the entering bound binds first
    """
    m = zB.shape[0]

    # Which bound a basic variable runs into depends on where it currently sits
    # relative to its box, and there are three cases, not two:
    #
    #   moving toward the box from outside -> breakpoint at the bound it enters
    #   moving inside the box               -> breakpoint at the bound it leaves
    #   moving away from the box from outside -> NO breakpoint at all
    #
    # The third case only arises in phase 1, where basic variables are allowed
    # to be infeasible. Computing a breakpoint for it yields a negative ratio,
    # which clamps the step to zero and jams phase 1 at a point that is not
    # phase-1 optimal -- the solver then reports a feasible LP as INFEASIBLE.
    # Such a variable simply gets worse at a constant rate, which the phase-1
    # objective already accounts for, so it must impose no limit.

    # --- pass 1: largest step with bounds relaxed by the tolerance ---------
    tmax = range_q
    for i in range(m):
        a = alpha[i] * dirn
        j = basic[i]
        lo = lower[j]
        up = upper[j]
        if a > pivot_tol:                     # zB[i] decreasing
            if zB[i] > up + feas_tol:
                target = up
            elif zB[i] < lo - feas_tol:
                continue                      # below lower, heading lower
            else:
                target = lo
            if target <= -INF:
                continue
            lim = (zB[i] - (target - relax)) / a
            if lim < tmax:
                tmax = lim
        elif a < -pivot_tol:                  # zB[i] increasing
            if zB[i] < lo - feas_tol:
                target = lo
            elif zB[i] > up + feas_tol:
                continue                      # above upper, heading higher
            else:
                target = up
            if target >= INF:
                continue
            lim = (zB[i] - (target + relax)) / a
            if lim < tmax:
                tmax = lim
    if tmax < 0.0:
        tmax = 0.0

    # --- pass 2: among rows blocking within tmax, take the biggest pivot ---
    best = -1
    best_piv = pivot_tol
    best_t = 0.0
    best_upper = 0
    for i in range(m):
        a = alpha[i] * dirn
        aa = a if a > 0.0 else -a
        if aa <= pivot_tol:
            continue
        j = basic[i]
        lo = lower[j]
        up = upper[j]
        if a > 0.0:
            if zB[i] > up + feas_tol:
                target = up
                hit_up = 1
            elif zB[i] < lo - feas_tol:
                continue
            else:
                target = lo
                hit_up = 0
            if target <= -INF:
                continue
        else:
            if zB[i] < lo - feas_tol:
                target = lo
                hit_up = 0
            elif zB[i] > up + feas_tol:
                continue
            else:
                target = up
                hit_up = 1
            if target >= INF:
                continue
        lim = (zB[i] - target) / a
        if lim <= tmax and aa > best_piv:
            best_piv = aa
            best = i
            best_t = lim if lim > 0.0 else 0.0
            best_upper = hit_up

    if best < 0:
        # nothing blocks: either a bound flip or genuinely unbounded
        return -1, range_q, 0, 0
    if range_q < best_t:
        return -1, range_q, 0, 0
    return best, best_t, best_upper, 1


@jit_kernel()
def _dual_ratio(alpha_row, d, status, sigma, pivot_tol, dual_tol, relax):
    """Harris two-pass dual ratio test.

    A candidate ``j`` must be able to move in the direction that repairs the
    leaving variable while keeping its own reduced cost feasible. Returns
    ``(q, t)`` with ``q = -1`` when no candidate exists, which certifies primal
    infeasibility.
    """
    N = d.shape[0]

    # For a leaving variable below its lower bound (sigma=+1) the primal step
    # is theta_q = -delta/alpha_r[q] with delta > 0, so a column at its lower
    # bound (which may only increase) needs alpha_r[q] < 0. Writing
    # sa = -sigma*alpha_r[j] makes the admissible set uniformly
    # "sa > 0 at lower, sa < 0 at upper" for both signs of sigma, and makes
    # every ratio d_j/sa non-negative. Dropping the minus sign selects exactly
    # the inadmissible columns and the method walks the wrong way.
    tmax = 1e30
    for j in range(N):
        st = status[j]
        if st == BASIC or st == FIXED:
            continue
        sa = -sigma * alpha_row[j]
        aa = sa if sa > 0.0 else -sa
        if aa <= pivot_tol:
            continue
        if st == AT_LOWER:
            if sa <= 0.0:
                continue
            lim = (d[j] + relax) / sa
        elif st == AT_UPPER:
            if sa >= 0.0:
                continue
            lim = (d[j] - relax) / sa
        else:                                   # FREE: any nonzero pivot works
            lim = (d[j] + relax) / sa if sa > 0.0 else (d[j] - relax) / sa
        if lim < tmax:
            tmax = lim
    if tmax >= 1e30:
        return -1, 0.0
    if tmax < 0.0:
        tmax = 0.0

    best = -1
    best_piv = pivot_tol
    best_t = 0.0
    for j in range(N):
        st = status[j]
        if st == BASIC or st == FIXED:
            continue
        sa = -sigma * alpha_row[j]
        aa = sa if sa > 0.0 else -sa
        if aa <= pivot_tol:
            continue
        if st == AT_LOWER and sa <= 0.0:
            continue
        if st == AT_UPPER and sa >= 0.0:
            continue
        lim = d[j] / sa
        if lim <= tmax and aa > best_piv:
            best_piv = aa
            best = j
            best_t = lim if lim > 0.0 else 0.0
    return best, best_t


@jit_kernel()
def _price_devex(d, status, weight, opt_tol):
    """Devex entering choice: maximise ``d_j^2 / w_j`` over eligible columns."""
    N = d.shape[0]
    best = -1
    best_score = 0.0
    best_dir = 1.0
    for j in range(N):
        st = status[j]
        if st == BASIC or st == FIXED:
            continue
        dj = d[j]
        if st == AT_LOWER:
            if dj >= -opt_tol:
                continue
            dirn = 1.0
        elif st == AT_UPPER:
            if dj <= opt_tol:
                continue
            dirn = -1.0
        else:
            if dj > -opt_tol and dj < opt_tol:
                continue
            dirn = 1.0 if dj < 0.0 else -1.0
        score = dj * dj / weight[j]
        if score > best_score:
            best_score = score
            best = j
            best_dir = dirn
    return best, best_dir


@jit_kernel()
def _worst_infeasible(zB, basic, lower, upper, weight, feas_tol):
    """Leaving row for the dual: largest weighted bound violation."""
    m = zB.shape[0]
    best = -1
    best_score = 0.0
    best_sigma = 0.0
    for i in range(m):
        j = basic[i]
        v = 0.0
        sg = 0.0
        if zB[i] < lower[j] - feas_tol:
            v = lower[j] - zB[i]
            sg = 1.0
        elif zB[i] > upper[j] + feas_tol:
            v = zB[i] - upper[j]
            sg = -1.0
        if sg != 0.0:
            score = v * v / weight[i]
            if score > best_score:
                best_score = score
                best = i
                best_sigma = sg
    return best, best_sigma


@jit_kernel()
def _dual_devex_update(weight, alpha, r, pivot, floor):
    """Dual Devex reference weights after pivoting on row ``r``.

    ``_worst_infeasible`` already scores rows as ``v**2 / weight[i]``, which is
    the dual steepest-edge form -- it just had nothing maintaining the weights,
    so every weight stayed at 1 and the rule collapsed to Dantzig's: pick the
    largest violation. That is as bad in the dual as it is in the primal, and
    the primal loop has had Devex weights all along.

    The reference-framework update, with ``alpha = B^-1 A_q`` and the pivot
    ``alpha[r]``: a row's weight can only grow, and the pivot row's is rescaled
    by the square of the pivot. Approximating steepest edge this way costs one
    pass over ``m`` per iteration and no extra solve.
    """
    m = weight.shape[0]
    wr = weight[r]
    inv = 1.0 / pivot
    for i in range(m):
        if i != r:
            a = alpha[i] * inv
            cand = a * a * wr
            if cand > weight[i]:
                weight[i] = cand
    nw = wr * inv * inv
    weight[r] = nw if nw > floor else floor


@jit_kernel()
def _phase1_costs(zB, basic, lower, upper, feas_tol, out):
    """Composite phase-1 cost: push each infeasible basic towards feasibility."""
    m = zB.shape[0]
    total = 0.0
    for i in range(m):
        j = basic[i]
        if zB[i] < lower[j] - feas_tol:
            out[i] = -1.0
            total += lower[j] - zB[i]
        elif zB[i] > upper[j] + feas_tol:
            out[i] = 1.0
            total += zB[i] - upper[j]
        else:
            out[i] = 0.0
    return total


# --------------------------------------------------------------------------- #
# the solver                                                                   #
# --------------------------------------------------------------------------- #


class _Simplex:
    """Shared state for one solve."""

    def __init__(self, prob: Problem, params: SimplexParams):
        self.p = params
        self.prob = prob
        self.B = Basis(prob, refactor_freq=params.refactor_freq,
                       update=params.basis_update)
        self.m, self.n, self.N = prob.m, prob.n, prob.n + prob.m
        self.iters = 0
        self.degenerate_run = 0
        self.perturbed = False
        self.t0 = time.perf_counter()

        self.zB = np.zeros(self.m, dtype=VAL)
        self.alpha = np.zeros(self.m, dtype=VAL)
        self.rho = np.zeros(self.m, dtype=VAL)
        self.devex = np.ones(self.N, dtype=VAL)
        self.dual_weight = np.ones(self.m, dtype=VAL)
        self._cost_backup = None
        self.farkas = None
        """Dual ray certifying primal infeasibility, when one was produced.

        Set by the dual simplex at the moment its ratio test fails, and by the
        phase-1 primal when it stalls with infeasibility remaining. It is what
        :mod:`sovopt.mip.conflict` analyses to learn why a node was empty."""

    # -- bookkeeping -------------------------------------------------------- #

    def out_of_time(self) -> bool:
        return (time.perf_counter() - self.t0) > self.p.time_limit

    def refresh(self):
        self.zB = self.B.compute_basic_values()

    def maybe_refactorize(self, force=False):
        if force or self.B.needs_refactorization:
            self.B.factorize()
            self.refresh()

    def primal_infeasibility(self) -> float:
        lo = self.B.lower[self.B.basic]
        up = self.B.upper[self.B.basic]
        return float(max(np.max(lo - self.zB, initial=0.0),
                         np.max(self.zB - up, initial=0.0)))

    # -- degeneracy --------------------------------------------------------- #

    def perturb(self):
        """Randomly nudge the costs to break a degenerate stall.

        The perturbation is proportional to the local cost magnitude so it does
        not swamp a badly scaled objective, and it is removed before the answer
        is reported -- see :meth:`unperturb`.
        """
        if self.perturbed:
            return
        rng = np.random.default_rng(0xC0FFEE)
        self._cost_backup = self.B.cost.copy()
        scale = self.p.perturb_scale * (1.0 + np.abs(self.B.cost))
        nb = self.B.status != BASIC
        self.B.cost[nb] += scale[nb] * rng.random(int(nb.sum()))
        self.perturbed = True

    def unperturb(self):
        if not self.perturbed:
            return
        self.B.cost[:] = self._cost_backup
        self.perturbed = False
        self.degenerate_run = 0

    # -- pivoting ----------------------------------------------------------- #

    def apply_primal_pivot(self, q, dirn, r, t, hit_upper):
        """Move ``t`` along the entering direction and swap the basis."""
        B = self.B
        step = dirn * t
        if step != 0.0:
            self.zB -= self.alpha * step
        zq = B.nonbasic_value(q) + step

        if r < 0:                                     # bound flip only
            B.status[q] = AT_UPPER if B.status[q] == AT_LOWER else AT_LOWER
            return True

        try:
            leaving = B.update(r, q, self.alpha)
        except LUSingular:
            self.maybe_refactorize(force=True)
            return False
        self.zB[r] = zq
        B.status[leaving] = AT_UPPER if hit_upper else AT_LOWER
        if B.upper[leaving] - B.lower[leaving] <= 0.0:
            B.status[leaving] = FIXED
        return True

    def update_devex(self, q, r, alpha_row, dirn):
        """Devex weight update against the current reference framework."""
        aq = alpha_row[q]
        if aq == 0.0 or not np.isfinite(aq):
            return
        wq = self.devex[q]
        ratio = wq / (aq * aq)
        nb = np.abs(alpha_row) > 0.0
        cand = ratio * (alpha_row * alpha_row)
        np.maximum(self.devex, cand, out=self.devex, where=nb)
        leaving = self.B.basic[r] if r >= 0 else -1
        if leaving >= 0:
            self.devex[leaving] = max(ratio, 1.0)
        self.devex[q] = 1.0
        if self.devex.max() > self.p.devex_reset:
            self.devex[:] = 1.0


def _primal_loop(S: _Simplex, phase: int):
    """Primal simplex. ``phase`` is 1 (minimise infeasibility) or 2."""
    p, B = S.p, S.B
    m, N = S.m, S.N
    ph1_cost = np.zeros(m, dtype=VAL)

    while True:
        if S.iters >= p.max_iter:
            return Status.ITERATION_LIMIT
        if S.iters % 32 == 0 and S.out_of_time():
            return Status.TIME_LIMIT
        if S.iters % p.recompute_freq == 0:
            S.refresh()

        # ---- duals and reduced costs -------------------------------------
        if phase == 1:
            infeas = _phase1_costs(S.zB, B.basic, B.lower, B.upper,
                                   p.feas_tol, ph1_cost)
            if infeas <= p.feas_tol:
                return Status.OPTIMAL                 # phase 1 succeeded
            y = ph1_cost.copy()
            B.btran(y)
            d = -np.concatenate([B.prob.A.rmatvec(y), -y])
        else:
            y = B.compute_duals()
            d = B.reduced_costs(y)

        # ---- price --------------------------------------------------------
        q, dirn = _price_devex(d, B.status, S.devex, p.opt_tol)
        if q < 0:
            return Status.OPTIMAL

        # ---- entering column and ratio test -------------------------------
        S.alpha = B.ftran_column(q, out=S.alpha)
        # A bound is infinite at the 1e30 sentinel, which np.isfinite calls
        # finite -- so the range must be tested against INF, not isfinite, or a
        # free column "bound flips" by 1e30.
        boundless = (B.upper[q] >= INF) or (B.lower[q] <= -INF)
        rng_q = np.inf if boundless else (B.upper[q] - B.lower[q])

        r, t, hit_upper, blocked = _primal_ratio(
            S.zB, B.basic, B.lower, B.upper, S.alpha, dirn, rng_q,
            p.pivot_tol, p.harris_relax, p.feas_tol)

        if r < 0 and boundless:
            if phase == 2:
                return Status.UNBOUNDED
            # in phase 1 an unbounded direction cannot happen with finite
            # infeasibility; treat as numerical trouble and refactorise
            S.maybe_refactorize(force=True)
            S.iters += 1
            continue

        # ---- Devex weights need the pivot row -----------------------------
        if r >= 0:
            S.rho.fill(0.0)
            S.rho[r] = 1.0
            B.btran(S.rho)
            alpha_row = B.pivot_row(S.rho)
            S.update_devex(q, r, alpha_row, dirn)

        if t <= p.feas_tol:
            S.degenerate_run += 1
            if S.degenerate_run > p.perturb_after and phase == 2:
                S.perturb()
                S.degenerate_run = 0
        else:
            S.degenerate_run = 0

        S.apply_primal_pivot(q, dirn, r, t, hit_upper)
        S.iters += 1
        S.maybe_refactorize()


def _dual_loop(S: _Simplex):
    """Dual simplex: repair primal infeasibility, keeping duals feasible."""
    p, B = S.p, S.B

    while True:
        if S.iters >= p.max_iter:
            return Status.ITERATION_LIMIT
        if S.iters % 32 == 0 and S.out_of_time():
            return Status.TIME_LIMIT
        if S.iters % p.recompute_freq == 0:
            S.refresh()

        r, sigma = _worst_infeasible(S.zB, B.basic, B.lower, B.upper,
                                     S.dual_weight, p.feas_tol)
        if r < 0:
            return Status.OPTIMAL

        # ---- pivot row ----------------------------------------------------
        S.rho.fill(0.0)
        S.rho[r] = 1.0
        B.btran(S.rho)
        alpha_row = B.pivot_row(S.rho)

        y = B.compute_duals()
        d = B.reduced_costs(y)

        q, _tdual = _dual_ratio(alpha_row, d, B.status, sigma,
                                p.pivot_tol, p.opt_tol, p.harris_relax)
        if q < 0:
            # No column can repair row r without breaking dual feasibility:
            # rho is row r of B^-1, and it is exactly the Farkas certificate
            # that this subproblem is empty. Keep it for conflict analysis.
            S.farkas = S.rho.copy()
            return Status.INFEASIBLE

        # ---- primal step --------------------------------------------------
        j = B.basic[r]
        target = B.lower[j] if sigma > 0 else B.upper[j]
        delta = target - S.zB[r]
        arq = alpha_row[q]
        if abs(arq) <= p.pivot_tol:
            S.maybe_refactorize(force=True)
            S.iters += 1
            continue
        theta = -delta / arq

        S.alpha = B.ftran_column(q, out=S.alpha)
        # alpha_row[q] and alpha[r] are the same number computed two ways -- one
        # from the pivot row, one from the FTRAN'd entering column. When the
        # factorisation has drifted they disagree, and the ratio test can pass
        # on a pivot the update then finds to be zero. Refactorise and retry
        # rather than raising out of the solver.
        if abs(S.alpha[r]) <= p.pivot_tol:
            S.maybe_refactorize(force=True)
            S.iters += 1
            continue

        # Price the next leaving row properly. Without this the weights stay
        # at 1 and _worst_infeasible degenerates to "largest violation", which
        # on gt2's cut-augmented relaxation took the dual loop past 100,000
        # iterations on 119 rows -- with no degeneracy at all to blame: not one
        # pivot in 100,000 had a step below the feasibility tolerance, and the
        # mean step was 2.4e2. It was simply choosing badly, every time.
        _dual_devex_update(S.dual_weight, S.alpha, r, S.alpha[r], 1.0)
        if S.dual_weight[r] > p.devex_reset:
            S.dual_weight[:] = 1.0

        S.zB -= S.alpha * theta
        zq = B.nonbasic_value(q) + theta

        try:
            leaving = B.update(r, q, S.alpha)
        except LUSingular:
            S.maybe_refactorize(force=True)
            S.iters += 1
            continue
        S.zB[r] = zq
        B.status[leaving] = AT_LOWER if sigma > 0 else AT_UPPER
        if B.upper[leaving] - B.lower[leaving] <= 0.0:
            B.status[leaving] = FIXED

        S.iters += 1
        S.maybe_refactorize()


def _dual_feasible(B: Basis, tol: float) -> bool:
    y = B.compute_duals()
    d = B.reduced_costs(y)
    st = B.status
    bad_lo = (st == AT_LOWER) & (d < -tol)
    bad_up = (st == AT_UPPER) & (d > tol)
    bad_fr = (st == FREE) & (np.abs(d) > tol)
    return not (bad_lo.any() or bad_up.any() or bad_fr.any())


@dataclass
class NodeResult:
    status: Status
    objective: float
    x: np.ndarray | None
    basis: np.ndarray | None
    iterations: int = 0
    farkas: np.ndarray | None = None
    """Dual ray certifying infeasibility, when the solve produced one."""


class NodeSolver:
    """Re-solve one LP repeatedly under changed *variable bounds*.

    This is the branch-and-bound workhorse. The matrix, the objective and the
    row bounds never change between nodes -- only ``l`` and ``u`` -- so almost
    everything can be reused:

    * The scaling and the augmented-form bookkeeping are built once.
    * The parent's basis is still **dual feasible** at the child, exactly.
      Reduced costs are ``d = c - Aᵀ B^-T c_B``, which depends on the basis and
      the objective but not on the bounds at all. Tightening a bound therefore
      cannot disturb dual feasibility -- it only pushes basic variables outside
      their box. That is precisely the situation the dual simplex is for, and it
      is why node re-solves take a handful of pivots instead of a fresh solve.

    The caller supplies bounds in the **scaled** space. Integer columns are
    pinned to a scale factor of one, so branching bounds are identical in both
    spaces.
    """

    def __init__(self, scaled_prob: Problem, params: SimplexParams | None = None):
        self.params = params or SimplexParams()
        self.prob = scaled_prob
        self.n = scaled_prob.n
        self.m = scaled_prob.m
        self.S = _Simplex(scaled_prob, self.params)
        self.S.B.set_logical_basis()
        self.S.B.set_status_from_costs()
        self.total_iterations = 0
        self.n_solves = 0
        self.n_warm = 0

    def _repair_status(self):
        """Put nonbasic variables back on a bound that still exists."""
        B = self.S.B
        for j in range(B.N):
            st = B.status[j]
            if st == BASIC:
                continue
            lo, up = B.lower[j], B.upper[j]
            if up - lo <= 0.0:
                B.status[j] = FIXED
            elif st == FIXED:
                B.status[j] = AT_LOWER if lo > -INF else AT_UPPER
            elif st == AT_LOWER and lo <= -INF:
                B.status[j] = AT_UPPER if up < INF else FREE
            elif st == AT_UPPER and up >= INF:
                B.status[j] = AT_LOWER if lo > -INF else FREE

    def solve(self, col_lb, col_ub, warm_basis=None,
              cutoff: float | None = None) -> NodeResult:
        p = self.params
        B = self.S.B
        S = self.S

        if np.any(col_lb > col_ub + 1e-12):
            return NodeResult(Status.INFEASIBLE, np.inf, None, None, 0)

        B.lower[:self.n] = col_lb
        B.upper[:self.n] = col_ub

        used_warm = False
        if warm_basis is not None and warm_basis.shape[0] == B.N:
            pos = np.flatnonzero(warm_basis == BASIC)
            if pos.size == B.m:
                B.status[:] = warm_basis
                B.basic[:] = pos.astype(B.basic.dtype)
                B.factorize()
                used_warm = True
        if not used_warm:
            B.set_logical_basis()
            B.set_status_from_costs()

        self._repair_status()
        S.iters = 0
        S.devex[:] = 1.0
        S.dual_weight[:] = 1.0
        S.farkas = None
        S.refresh()

        try:
            if S.primal_infeasibility() <= p.feas_tol:
                status = _primal_loop(S, phase=2)
            elif _dual_feasible(B, p.opt_tol):
                if p.node_kernel and B.ft is None:
                    status = self._dual_loop_kernel()
                else:
                    status = _dual_loop(S)
                if status == Status.OPTIMAL:
                    status = _primal_loop(S, phase=2)
            else:
                st1 = _primal_loop(S, phase=1)
                if st1 != Status.OPTIMAL:
                    status = st1
                else:
                    S.refresh()
                    if S.primal_infeasibility() > _infeasible_threshold(
                            self.prob, p.feas_tol):
                        # Phase 1 stalled with infeasibility left. Its own duals
                        # are the certificate: y1 = B^-T c1 prices the rows in a
                        # combination no feasible point can satisfy.
                        ph1 = np.zeros(S.m, dtype=VAL)
                        _phase1_costs(S.zB, B.basic, B.lower, B.upper,
                                      p.feas_tol, ph1)
                        B.btran(ph1)
                        S.farkas = ph1
                        status = Status.INFEASIBLE
                    else:
                        status = _primal_loop(S, phase=2)
        except LUSingular:
            status = Status.NUMERICAL

        if S.perturbed:
            S.unperturb()
            S.maybe_refactorize(force=True)
            if status == Status.OPTIMAL:
                status = _primal_loop(S, phase=2)

        self.total_iterations += S.iters
        self.n_solves += 1
        self.n_warm += int(used_warm)

        if status != Status.OPTIMAL:
            return NodeResult(status, np.inf if status == Status.INFEASIBLE
                              else np.nan, None, None, S.iters,
                              farkas=(None if S.farkas is None
                                      else np.asarray(S.farkas).copy()))

        z = B.nonbasic_values()
        z[B.basic] = S.zB
        obj = float(self.prob.c @ z[:self.n]) + self.prob.obj_offset
        return NodeResult(Status.OPTIMAL, obj, z[:self.n].copy(),
                          B.status.copy(), S.iters)

    def _dual_loop_kernel(self):
        """The dual loop as one compiled call; the basis is rebuilt after."""
        from .nodelp import INFEASIBLE, ITERATION_LIMIT, NUMERICAL, OPTIMAL,             dual_simplex_kernel
        p, S, B = self.params, self.S, self.S.B
        A = self.prob.A
        tol = B.lu_tol
        farkas = np.zeros(self.m, dtype=VAL)
        code, iters, n_fact, factors = dual_simplex_kernel(
            A.cp, A.ci, A.cx, A.rp, A.ri, A.rx, self.n, self.m,
            B.cost, B.lower, B.upper, B.status, B.basic, S.zB, S.dual_weight,
            farkas, p.feas_tol, p.opt_tol, p.pivot_tol, p.harris_relax,
            p.refactor_freq, p.max_iter, p.recompute_freq, p.devex_reset,
            tol, 1e-14)
        S.iters += iters
        B.n_factorizations += n_fact
        B.n_updates += iters
        # carry on from the exact factors the kernel ended with: its LU and
        # eta file become the basis's, so nothing is refactorised here
        (Lp, Li, Lx, Up, Ui, Ux, pinv, q,
         estart, eidx, evals, epiv, erow, n_eta) = factors
        from ..numerics.lu import LUFactor
        B.lu = LUFactor(self.m, Lp, Li, Lx, Up, Ui, Ux, pinv, q)
        B._eta_start, B._eta_idx, B._eta_val = estart, eidx, evals
        B._eta_piv, B._eta_row, B._n_eta = epiv, erow, int(n_eta)
        B._sync_positions()
        S.refresh()
        if code == OPTIMAL:
            return Status.OPTIMAL
        if code == INFEASIBLE:
            S.farkas = farkas
            return Status.INFEASIBLE
        if code == ITERATION_LIMIT:
            return Status.ITERATION_LIMIT
        return Status.NUMERICAL

    def stats(self) -> dict:
        return {
            "node_lp_solves": self.n_solves,
            "node_lp_iterations": self.total_iterations,
            "warm_started": self.n_warm,
            "avg_iterations": self.total_iterations / max(self.n_solves, 1),
            **self.S.B.stats(),
        }


def _infeasible_threshold(prob, feas_tol: float) -> float:
    """How much phase-1 infeasibility is too much to call it feasible.

    Phase 1 drives the sum of bound violations down and stops when it can
    do no better; whatever is left decides between OPTIMAL and INFEASIBLE.
    Comparing that residual against an **absolute** ``100 * feas_tol`` is
    wrong on any model whose rows are large, because the residual is in the
    units of the rows and the threshold is not.

    Netlib bore3d is the case that found it. It is feasible, with a
    published optimum of 1373.080394, and the simplex returns exactly that
    at the default tolerance -- then calls it **INFEASIBLE** at
    ``feas_tol=1e-8``, which is the tolerance the benchmark harness uses.
    Tightening a tolerance made the solver reject a feasible model rather
    than solve it more carefully, which is backwards, and an INFEASIBLE
    verdict is the one answer a caller cannot check for themselves.

    So the threshold is scaled by the size of the finite row bounds. The
    infinite ones are excluded: they are stored as a 1e30 sentinel, and
    letting that into a maximum makes the threshold meaningless.
    """
    lo, hi = prob.row_lb, prob.row_ub
    finite = np.concatenate([lo[lo > -INF], hi[hi < INF]])
    scale = float(np.abs(finite).max()) if finite.size else 0.0
    return feas_tol * 100.0 * max(1.0, scale)


def solve_simplex(prob: Problem, params: SimplexParams | None = None,
                  warm_basis=None) -> Solution:
    """Solve an LP exactly with the revised simplex.

    Returns a :class:`~sovopt.core.problem.Solution` carrying the primal point,
    row duals, reduced costs and the final basis status vector, so the result
    can be used to warm-start a later solve.
    """
    params = params or SimplexParams()
    t0 = time.perf_counter()

    if prob.Q is not None:
        raise NotImplementedError(
            "solve_simplex optimises a linear objective; this model has a "
            "quadratic term, which would be silently discarded")

    flip = prob.sense == ObjSense.MAXIMISE
    work = prob
    if flip:
        work = prob.copy()
        work.c = -work.c
        work.obj_offset = -work.obj_offset
        work.sense = ObjSense.MINIMISE

    scaled, sc = scale_problem(work, method=params.scaling)
    S = _Simplex(scaled, params)
    B = S.B

    # ---- starting basis --------------------------------------------------- #
    if warm_basis is not None and len(warm_basis) == B.N:
        B.status[:] = np.asarray(warm_basis, dtype=np.uint8)
        pos = np.flatnonzero(B.status == BASIC)
        if pos.size == B.m:
            B.basic[:] = pos.astype(B.basic.dtype)
            B.factorize()
        else:
            B.set_logical_basis()
    else:
        B.set_logical_basis()
        B.set_status_from_costs()
    S.refresh()

    # ---- choose the algorithm -------------------------------------------- #
    # Any numerical failure below is reported as a status, never raised: a
    # caller deep inside a spatial branch-and-bound cannot do anything useful
    # with an exception, and one bad node must not abort the search.
    method = "dual"
    status = Status.NOT_SOLVED
    try:
        if S.primal_infeasibility() <= params.feas_tol:
            method = "primal"
            status = _primal_loop(S, phase=2)
        elif _dual_feasible(B, params.opt_tol):
            status = _dual_loop(S)
            if status == Status.OPTIMAL:
                # dual simplex ends primal feasible; polish residual dual error
                status = _primal_loop(S, phase=2)
            elif status == Status.INFEASIBLE:
                # The dual ratio test finding no entering column is a Farkas
                # certificate *if* it is not numerical, and at a tight tolerance
                # it can be numerical: fewer columns qualify as dual feasible,
                # so the test runs out of candidates on a problem that has an
                # answer. Netlib bore3d is exactly that. It is feasible, with a
                # published optimum of 1373.080394, and this branch returned
                # INFEASIBLE for it at `feas_tol=1e-8` while returning the
                # published value at 1e-7 -- tightening a tolerance made the
                # solver reject a feasible model instead of solving it more
                # carefully.
                #
                # INFEASIBLE is the one answer a caller cannot check for
                # themselves: there is no point to hand a verifier. So it is
                # confirmed here by an independent primal phase 1 before it is
                # reported, and the cost is paid only on the rare path that
                # claims it.
                method = "primal(2-phase, confirming dual infeasible)"
                B.set_logical_basis()
                B.set_status_from_costs()
                S.refresh()
                st1 = _primal_loop(S, phase=1)
                if st1 == Status.OPTIMAL:
                    S.refresh()
                    if S.primal_infeasibility() > _infeasible_threshold(
                            scaled, params.feas_tol):
                        status = Status.INFEASIBLE
                    else:
                        S.farkas = None      # the dual verdict was numerical
                        status = _primal_loop(S, phase=2)
                else:
                    status = st1
        else:
            method = "primal(2-phase)"
            st1 = _primal_loop(S, phase=1)
            if st1 == Status.OPTIMAL:
                S.refresh()
                if S.primal_infeasibility() > _infeasible_threshold(
                        scaled, params.feas_tol):
                    status = Status.INFEASIBLE
                else:
                    status = _primal_loop(S, phase=2)
            else:
                status = st1
    except LUSingular:
        status = Status.NUMERICAL

    # ---- remove any perturbation and re-optimise -------------------------- #
    if S.perturbed:
        S.unperturb()
        S.maybe_refactorize(force=True)
        if status == Status.OPTIMAL:
            status = _primal_loop(S, phase=2)

    S.maybe_refactorize(force=True)

    # ---- assemble the answer --------------------------------------------- #
    z = B.nonbasic_values()
    z[B.basic] = S.zB
    x_scaled = z[:B.n]
    y_scaled = B.compute_duals()
    d_scaled = B.reduced_costs(y_scaled)[:B.n]

    x = np.clip(sc.unscale_primal(x_scaled), work.col_lb, work.col_ub)
    y = sc.unscale_dual(y_scaled)
    d = sc.unscale_reduced(d_scaled)

    obj = float(prob.c @ x) + prob.obj_offset
    if flip:
        y = -y
        d = -d

    # A phase-1 iterate that never reached feasibility is not a solution, and
    # both TIME_LIMIT and ITERATION_LIMIT report ``has_solution`` -- so handing
    # the point back invites a caller to use it. Measured on a 15,360-row
    # planning model at a 120 s limit: the run returned TIME_LIMIT with an
    # objective of 0 and a point the independent verifier rejected outright.
    # The test is on the *unscaled* model, which is the yardstick the verifier
    # uses; a phase-2 timeout is feasible and keeps its point, which is the
    # case worth reporting.
    # Only the statuses that *claim* a solution: INFEASIBLE and UNBOUNDED
    # already report has_solution False, and their last iterate is worth
    # keeping for diagnostics and for the JSON report.
    if x is not None and Status(status).has_solution and status != Status.OPTIMAL:
        row_v, col_v, _ = prob.violation(x)
        if max(row_v, col_v) > max(params.feas_tol, 1e-6):
            x = None
            obj = float("nan")

    sol = Solution(status=status, x=x, objective=obj, y=y, reduced_costs=d,
                   basis_status=B.status.copy(), iterations=S.iters,
                   time=time.perf_counter() - t0,
                   method=f"simplex[{method}]")
    if params.sensitivity and status == Status.OPTIMAL:
        from .sensitivity import compute_sensitivity
        sol.sensitivity = compute_sensitivity(S, sc, prob, flip)
    sol.dual_bound = obj
    sol.info = {**B.stats(), "algorithm": method,
                "perturbed": S.perturbed,
                "primal_infeasibility": S.primal_infeasibility()}
    return sol.drop_objective_if_unsolved()
