"""Primal heuristics: finding integer-feasible points.

On the instances this solver failed, it did not return a *poor* solution -- it
returned none at all. That is a primal failure, and no amount of bounding fixes
it. Proving optimality needs a good bound; producing an answer at all needs a
good incumbent, and industrial users care about the second far more: a plan in
sixty seconds beats a proof in six hours.

Three heuristics, in increasing cost:

``feasibility_jump``
    Pure local search on the *weighted constraint violation*. No LP, no
    factorisation -- it walks integer assignments directly, moving one variable
    at a time to the value that most reduces violation, and raising the weight
    of constraints it cannot satisfy so the search stops ignoring them. This is
    the cheapest way to get a first feasible point and it works on models where
    rounding the relaxation never will. Weight-bumping is what escapes local
    minima: it is the same device WalkSAT uses, applied to a MIP.

``fix_and_propagate``
    Round integer variables one at a time in order of confidence, propagating
    after each fixing so the consequences cascade, and backtracking once on the
    other rounding when propagation proves infeasibility. Cheap, and it exploits
    the propagation engine already present.

``feasibility_pump``
    Alternate between an integer point (round the LP solution) and an LP point
    (project by minimising distance to the rounding). Escapes cycles by flipping
    the variables whose rounding is least certain. Costs one LP per round, so it
    runs only where the cheaper two have failed.

References
----------
Luteberget & Sandvik, "Feasibility jump: an LP-free Lagrangian MIP heuristic",
  Math. Prog. Computation 15 (2023) 365-388 -- winner of the MIP 2022
  computational competition.
Fischetti, Glover & Lodi, "The feasibility pump", Math. Prog. 104 (2005)
  91-104.
Achterberg & Berthold, "Improving the feasibility pump", Discrete Optimization
  4 (2007) 77-86 -- the objective feasibility pump.
Berthold, "RENS -- the optimal rounding", Math. Prog. Computation 6 (2014)
  33-54.
Danna, Rothberg & Le Pape, "Exploring relaxation induced neighborhoods to
  improve MIP solutions", Math. Prog. 102 (2005) 71-90 -- RINS.
"""

from __future__ import annotations

import numpy as np

from ..core._jit import jit_kernel
from ..core.sparse import VAL
from ..core.tolerances import INF
from .propagate import propagate

__all__ = ["feasibility_jump", "fix_and_propagate", "feasibility_pump",
           "HeuristicStats"]


class HeuristicStats:
    __slots__ = ("calls", "successes", "by_name")

    def __init__(self):
        self.calls = 0
        self.successes = 0
        self.by_name: dict[str, list[int]] = {}

    def record(self, name, ok):
        self.calls += 1
        self.successes += int(ok)
        c = self.by_name.setdefault(name, [0, 0])
        c[0] += 1
        c[1] += int(ok)

    def summary(self) -> dict:
        return {k: f"{v[1]}/{v[0]}" for k, v in self.by_name.items()}


# --------------------------------------------------------------------------- #
# Feasibility Jump                                                             #
# --------------------------------------------------------------------------- #


@jit_kernel()
def _row_violation(act, rl, ru, i):
    a = act[i]
    if a < rl[i]:
        return rl[i] - a
    if a > ru[i]:
        return a - ru[i]
    return 0.0


@jit_kernel()
def _fj_kernel(cp, ci, cx, rp, ri, rx, rl, ru, lo, hi, is_int,
               x, act, w, max_iter, feas_tol, seed):
    """Weighted-violation local search.

    Returns ``(solved, iterations)``. ``x`` and ``act`` are updated in place.
    """
    m = rp.shape[0] - 1
    n = cp.shape[0] - 1
    rng = seed
    it = 0

    while it < max_iter:
        it += 1

        # ---- collect the violated rows ----------------------------------
        nviol = 0
        first_viol = -1
        for i in range(m):
            if _row_violation(act, rl, ru, i) > feas_tol:
                nviol += 1
                if first_viol < 0:
                    first_viol = i
        if nviol == 0:
            return 1, it

        # pick one violated row pseudo-randomly (xorshift, deterministic)
        rng ^= (rng << 13) & 0xFFFFFFFFFFFFFFFF
        rng ^= (rng >> 7)
        rng ^= (rng << 17) & 0xFFFFFFFFFFFFFFFF
        pick = rng % nviol
        row = first_viol
        seen = 0
        for i in range(m):
            if _row_violation(act, rl, ru, i) > feas_tol:
                if seen == pick:
                    row = i
                    break
                seen += 1

        # ---- best single-variable move among that row's columns ----------
        best_j = -1
        best_delta = 0.0
        best_score = -1e-9              # must strictly improve

        for p in range(rp[row], rp[row + 1]):
            j = ri[p]
            aij = rx[p]
            if aij == 0.0:
                continue

            # candidate deltas: whatever makes this row exactly satisfied
            for side in range(2):
                target = rl[row] if side == 0 else ru[row]
                if target <= -INF or target >= INF:
                    continue
                delta = (target - act[row]) / aij
                if is_int[j]:
                    delta = np.floor(delta) if delta < 0.0 else np.ceil(delta)
                nv = x[j] + delta
                if nv < lo[j]:
                    nv = lo[j]
                elif nv > hi[j]:
                    nv = hi[j]
                if is_int[j]:
                    nv = np.round(nv)
                delta = nv - x[j]
                if delta == 0.0:
                    continue

                # score the move over every row this column touches
                score = 0.0
                for q in range(cp[j], cp[j + 1]):
                    i2 = ci[q]
                    old = _row_violation(act, rl, ru, i2)
                    a2 = act[i2] + cx[q] * delta
                    new = 0.0
                    if a2 < rl[i2]:
                        new = rl[i2] - a2
                    elif a2 > ru[i2]:
                        new = a2 - ru[i2]
                    score += w[i2] * (old - new)

                if score > best_score:
                    best_score = score
                    best_j = j
                    best_delta = delta

        if best_j >= 0:
            j = best_j
            for q in range(cp[j], cp[j + 1]):
                act[ci[q]] += cx[q] * best_delta
            x[j] += best_delta
        else:
            # local minimum: make the unsatisfied rows matter more
            for i in range(m):
                if _row_violation(act, rl, ru, i) > feas_tol:
                    w[i] += 1.0

    return 0, it


def feasibility_jump(prob, int_mask, lo, hi, x0=None, max_iter: int = 20000,
                     seed: int = 12345, feas_tol: float = 1e-7):
    """Search for an integer-feasible point by weighted-violation local search.

    Returns a point satisfying the rows and bounds, or None. Continuous
    variables are moved too, so the result is feasible for the whole model, not
    only its integer part.
    """
    A = prob.A
    n = prob.n

    lo = np.ascontiguousarray(lo, dtype=VAL)
    hi = np.ascontiguousarray(hi, dtype=VAL)
    is_int = np.ascontiguousarray(int_mask, dtype=np.bool_)

    if x0 is None:
        x = np.clip(np.zeros(n, dtype=VAL), lo, hi)
    else:
        x = np.clip(np.asarray(x0, dtype=VAL).copy(), lo, hi)
    x[is_int] = np.round(x[is_int])
    x = np.clip(x, lo, hi)
    x[is_int] = np.round(x[is_int])

    act = A.matvec(x)
    w = np.ones(prob.m, dtype=VAL)

    solved, _iters = _fj_kernel(A.cp, A.ci, A.cx, A.rp, A.ri, A.rx,
                                prob.row_lb, prob.row_ub, lo, hi, is_int,
                                x, act, w, int(max_iter), feas_tol, int(seed))
    if not solved:
        return None
    rv, bv, iv = prob.violation(x)
    if max(rv, bv) > 1e-6 or iv > 1e-6:
        return None
    return x


# --------------------------------------------------------------------------- #
# fix and propagate                                                            #
# --------------------------------------------------------------------------- #


def fix_and_propagate(prob, x_lp, int_mask, lo, hi, feas_tol: float = 1e-9,
                      lp_solve=None, max_backtracks: int = 20):
    """Round integers one at a time, propagating after each fixing.

    Variables are taken in order of *confidence* -- least fractional first --
    because an early wrong guess is the expensive one. When propagation proves
    the fixing infeasible, the opposite rounding is tried once before giving up.
    """
    lo2 = np.array(lo, dtype=VAL, copy=True)
    hi2 = np.array(hi, dtype=VAL, copy=True)
    idx = np.flatnonzero(int_mask)
    if idx.size == 0:
        return None

    frac = np.abs(x_lp[idx] - np.round(x_lp[idx]))
    order = idx[np.argsort(frac)]
    backtracks = 0

    for j in order:
        if hi2[j] - lo2[j] <= feas_tol:
            continue
        v = float(np.clip(np.round(x_lp[j]), lo2[j], hi2[j]))
        alt = float(np.clip(np.floor(x_lp[j]) if v > x_lp[j]
                            else np.ceil(x_lp[j]), lo2[j], hi2[j]))

        for attempt, val in enumerate((v, alt)):
            save_lo, save_hi = lo2.copy(), hi2.copy()
            lo2[j] = hi2[j] = val
            res = propagate(prob.A, prob.row_lb, prob.row_ub, lo2, hi2,
                            int_mask, max_rounds=3, feas_tol=feas_tol,
                            inplace=True)
            if not res.infeasible:
                break
            lo2, hi2 = save_lo, save_hi
            if attempt == 1 or val == v:
                backtracks += 1
                if backtracks > max_backtracks:
                    return None
        else:
            return None

    # every integer is fixed; let an LP place the continuous part
    if lp_solve is not None:
        x = lp_solve(lo2, hi2)
        if x is None:
            return None
    else:
        x = np.clip(x_lp.copy(), lo2, hi2)

    x[int_mask] = np.round(x[int_mask])
    rv, bv, iv = prob.violation(x)
    if max(rv, bv) > 1e-6 or iv > 1e-6:
        return None
    return x


# --------------------------------------------------------------------------- #
# feasibility pump                                                             #
# --------------------------------------------------------------------------- #


def feasibility_pump(prob, int_mask, lo, hi, lp_solve, x_lp=None,
                     max_rounds: int = 40, seed: int = 7,
                     alpha: float = 0.9, decay: float = 0.9):
    """Alternate rounding and LP projection until the two agree.

    ``lp_solve(lo, hi, obj)`` must minimise ``obj`` over the relaxation. The
    objective mixes the distance to the current rounding with the true
    objective, the weight decaying each round -- the "objective feasibility
    pump", which finds points that are feasible *and* not terrible.

    Cycles are broken by flipping the integers whose rounding was least
    decisive, which is the standard restart and the reason the method
    terminates at all.
    """
    idx = np.flatnonzero(int_mask)
    if idx.size == 0:
        return None

    rng = np.random.default_rng(seed)
    x = x_lp if x_lp is not None else lp_solve(lo, hi, prob.c)
    if x is None:
        return None

    cnorm = float(np.linalg.norm(prob.c)) or 1.0
    seen: set[bytes] = set()
    a = alpha

    for _ in range(max_rounds):
        xr = x.copy()
        xr[idx] = np.clip(np.round(xr[idx]), lo[idx], hi[idx])

        rv, bv, iv = prob.violation(xr)
        if max(rv, bv) <= 1e-6 and iv <= 1e-6:
            return xr

        key = xr[idx].tobytes()
        if key in seen:
            # cycle: flip the least decisive roundings
            d = np.abs(x[idx] - xr[idx])
            k = max(1, int(0.1 * idx.size))
            flip = idx[np.argsort(-d)[:k]]
            for j in flip:
                xr[j] = hi[j] if xr[j] <= lo[j] + 0.5 else lo[j]
            key = xr[idx].tobytes()
        seen.add(key)

        # distance objective: +1 where the rounding sits at the lower bound,
        # -1 where it sits at the upper, which is the linearisation of |x - xr|
        dist = np.zeros(prob.n, dtype=VAL)
        at_lo = xr[idx] <= lo[idx] + 0.5
        dist[idx] = np.where(at_lo, 1.0, -1.0)
        dnorm = float(np.linalg.norm(dist)) or 1.0

        obj = (1.0 - a) * dist / dnorm + a * prob.c / cnorm
        a *= decay

        x = lp_solve(lo, hi, obj)
        if x is None:
            return None

    return None
