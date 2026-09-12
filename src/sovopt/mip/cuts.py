"""Cutting planes: Gomory mixed-integer and knapsack cover.

A cut tightens the relaxation without removing a single integer-feasible point.
That is the whole game on the instances this solver currently fails: the tree
does not lose because it searches badly, it loses because the root bound is far
from the integer optimum and no amount of branching closes that gap in a minute.

Gomory mixed-integer cuts
-------------------------
Derived from a row of the simplex tableau, so they need a basis -- which is why
they arrive with the revised simplex and not before.

Write the tableau row for a basic integer variable whose value came out
fractional. In the augmented form ``M z = 0`` with ``alpha_j = B^-1 M_j``:

    z_B[i] + sum_{j in N} alpha_ij z_j = 0

Each nonbasic sits at a bound, so shift it to a non-negative deviation:

    at lower:  z_j = l_j + t_j,    gamma_ij =  alpha_ij
    at upper:  z_j = u_j - t_j,    gamma_ij = -alpha_ij

giving ``z_B[i] + sum_j gamma_ij t_j = b_i`` with every ``t_j >= 0`` and ``b_i``
the current (fractional) value. With ``f0 = b_i - floor(b_i)`` and
``f_j = gamma_ij - floor(gamma_ij)``, the GMI cut is

    sum_j  psi_j t_j  >=  1

    integer j:     psi_j = f_j/f0            if f_j <= f0
                          (1-f_j)/(1-f0)     otherwise
    continuous j:  psi_j = gamma_ij/f0       if gamma_ij > 0
                          -gamma_ij/(1-f0)   otherwise

Every ``t_j`` is zero at the current point, so the cut is violated by exactly 1
in ``t``-space: it always cuts off the fractional vertex it was derived from.

Translating back, with ``g_j = +psi_j`` at a lower bound and ``-psi_j`` at an
upper bound, the cut is ``sum_j g_j z_j >= 1 + sum_j g_j z_j*``. Logical
variables are then expanded through ``z_{n+i} = (A x)_i``, which is what turns a
tableau row back into a constraint on the original variables.

Knapsack cover cuts
-------------------
For a row ``sum a_j x_j <= b`` over binaries with positive coefficients, any
*cover* ``C`` with ``sum_{j in C} a_j > b`` cannot be fully selected:

    sum_{j in C} x_j <= |C| - 1

The separation problem -- find the cover most violated by ``x*`` -- is itself a
knapsack, solved here greedily by taking variables in increasing
``(1 - x*_j)/a_j``, then shrinking the cover to a minimal one so the inequality
is as strong as possible.

Numerical safeguards
--------------------
Cuts are where a MILP solver most easily poisons itself: a cut with a
coefficient range of 1e12 is valid in exact arithmetic and catastrophic in
floating point. Every candidate is rejected unless its fractionality is well
away from 0 and 1, its dynamism (max/min nonzero magnitude) is bounded, and its
efficacy -- violation divided by the coefficient norm -- is worth the row it
will occupy.

References
----------
Gomory, "An algorithm for the mixed integer problem", RAND RM-2597, 1960.
Balas, Ceria, Cornuéjols & Natraj, "Gomory cuts revisited", Oper. Res. Letters
  19 (1996) 1-9 -- why GMI cuts work in practice and how to keep them clean.
Crowder, Johnson & Padberg, "Solving large-scale zero-one linear programming
  problems", Oper. Res. 31 (1983) 803-834 -- cover cuts and lifting.
Marchand & Wolsey, "Aggregation and mixed integer rounding to solve MIPs",
  Oper. Res. 49 (2001) 363-371.
Wesselmann & Suhl, "Implementing cutting plane management and selection
  techniques", technical report, Paderborn 2012 -- efficacy, orthogonality,
  dynamism filtering.
Achterberg, *Constraint Integer Programming*, PhD thesis, TU Berlin 2007, §8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.sparse import IDX, SparseMatrix, VAL
from ..core.tolerances import INF
from ..lp.basis import AT_LOWER, AT_UPPER, BASIC, FIXED, FREE

__all__ = ["Cut", "CutPool", "generate_gomory", "generate_cover",
           "append_cuts"]


@dataclass
class Cut:
    """``coef . x >= rhs``, expressed on the structural variables."""

    idx: np.ndarray
    val: np.ndarray
    rhs: float
    kind: str = "gmi"
    efficacy: float = 0.0

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.val))

    def violation(self, x) -> float:
        return float(self.rhs - self.val @ x[self.idx])

    def dynamism(self) -> float:
        a = np.abs(self.val)
        a = a[a > 0]
        return float(a.max() / a.min()) if a.size else 1.0


class CutPool:
    """Selects a diverse, well-conditioned subset of candidate cuts.

    Adding every violated cut makes the LP bigger and slower without making the
    bound much better, because near-parallel cuts carry the same information.
    Selection is greedy by efficacy subject to an orthogonality floor against
    everything already chosen -- the standard compromise.

    **The orthogonality floor is 0.001, not the 0.05 that reads as reasonable.**
    "Near-parallel cuts carry the same information" is true of the direction and
    false of the *depth*: two cuts 2% apart in angle can sit at very different
    distances from the relaxation optimum, and rejecting the second throws away
    the deeper one whenever the shallower one was found first. Models whose
    bound is closed by a fan of similar cuts are exactly the ones this hurts.
    Measured on gt2 (m=29, optimum 21166), root bound after cuts:

        floor 0.05    21155.00   gap 11.00   80 cuts   never proved in 60 s
        floor 0.001   21166.00   gap  0.00   86 cuts   OPTIMAL in 5.6 s

    Across the MIPLIB set the looser floor costs nothing and closes gt2 at the
    root. The efficacy floor was checked at the same time and is not implicated:
    1e-4 through 1e-8 all close gt2 once the orthogonality floor is right.
    """

    def __init__(self, min_efficacy: float = 1e-4, max_dynamism: float = 1e8,
                 min_orthogonality: float = 0.001, max_cuts: int = 200,
                 max_density_frac: float = 0.4, density_multiple: float = 5.0,
                 min_support: int = 20, small_basis_rows: int = 150):
        self.min_efficacy = min_efficacy
        self.max_dynamism = max_dynamism
        self.min_orthogonality = min_orthogonality
        self.max_cuts = max_cuts
        self.max_density_frac = max_density_frac
        self.density_multiple = density_multiple
        self.min_support = min_support
        self.small_basis_rows = small_basis_rows
        self.rejected_dense = 0
        self.accepted: list[Cut] = []

    def support_limit(self, n: int, avg_row_nnz: float, m: int | None = None) -> int:
        """Largest cut support worth adding to a matrix of this shape.

        **The most important filter here, and the one whose scaling is easy to
        get backwards.** A GMI cut derived from a sparse model is typically
        *dense*: on MIPLIB p0201, whose rows average 14.5 nonzeros, the median
        GMI cut touches 168 of 201 columns. Near-dense rows make the basis
        factor dense, so FTRAN and BTRAN stop being hypersparse -- the exact
        property the LU is built to exploit -- and each node LP slows by orders
        of magnitude. Measured on p0201: 1619 nodes in 9s without cuts, 63
        nodes in 60s with them, for a bound gain of 290 on an objective of 7615.

        The cost scales with the **basis size m**, not the column count. Below
        a few hundred rows a dense triangular solve is a few hundred flops and
        there was no hypersparsity to lose in the first place, so a density cap
        keyed to ``n`` only throws away good cuts. Measured on gt2 (m=29):
        dense cuts lift the root bound from 13941 to 20725 and close the model
        in 0.9s, while capping them leaves it unsolved after 60s.

        So: no density restriction on a small basis; above that, keep cuts
        comparable to the model's own rows.
        """
        if m is not None and m <= self.small_basis_rows:
            return n
        return int(max(self.min_support,
                       min(self.max_density_frac * n,
                           self.density_multiple * max(avg_row_nnz, 1.0))))

    def _dense(self, cut: Cut, n: int):
        v = np.zeros(n, dtype=VAL)
        v[cut.idx] = cut.val
        nrm = np.linalg.norm(v)
        return (v / nrm) if nrm > 0 else v

    def select(self, candidates: list[Cut], x, n: int, limit: int | None = None,
               avg_row_nnz: float | None = None, m: int | None = None):
        """Filter and rank candidates; returns the cuts to add."""
        limit = limit if limit is not None else self.max_cuts
        cap = (self.support_limit(n, avg_row_nnz, m)
               if avg_row_nnz is not None else n)
        scored = []
        for c in candidates:
            if c.idx.size == 0:
                continue
            if c.idx.size > cap:
                self.rejected_dense += 1
                continue
            nrm = c.norm
            if nrm <= 0 or not np.isfinite(nrm):
                continue
            if c.dynamism() > self.max_dynamism:
                continue
            eff = c.violation(x) / nrm
            if eff < self.min_efficacy or not np.isfinite(eff):
                continue
            c.efficacy = eff
            scored.append(c)

        scored.sort(key=lambda c: -c.efficacy)

        chosen: list[Cut] = []
        dirs: list[np.ndarray] = []
        for c in scored:
            if len(chosen) >= limit:
                break
            d = self._dense(c, n)
            if any(abs(float(d @ e)) > 1.0 - self.min_orthogonality for e in dirs):
                continue
            chosen.append(c)
            dirs.append(d)
        self.accepted.extend(chosen)
        return chosen


# --------------------------------------------------------------------------- #
# Gomory mixed-integer                                                         #
# --------------------------------------------------------------------------- #


def generate_gomory(basis, zB, int_mask_full, max_cuts: int = 50,
                    min_frac: float = 0.01, away: float = 0.01,
                    max_coef: float = 1e7) -> list[Cut]:
    """GMI cuts from the current basis, one per fractional basic integer.

    ``int_mask_full`` is over the augmented variable set (structural + logical);
    logicals are integral exactly when their row has all-integer coefficients on
    integer columns, which the caller decides.
    """
    n, m = basis.n, basis.m
    A = basis.prob.A
    cuts: list[Cut] = []

    # Most fractional first, on the fractionality *rounded to nine digits*
    # with the row index as tiebreak. Sorting the raw values let 1e-14 of
    # factorisation noise decide the order among rows with equal
    # fractionality, so two solvers holding the same basis -- one with a
    # fresh LU, one with an eta file -- produced different cuts, and on gt2
    # one root bound closed the gap and the other never did.
    frac = np.round(np.abs(zB - np.round(zB)), 9)
    order = np.lexsort((np.arange(m), -frac))
    rho = np.zeros(m, dtype=VAL)

    for r in order:
        if len(cuts) >= max_cuts:
            break
        j_basic = int(basis.basic[r])
        if not int_mask_full[j_basic]:
            continue
        b = float(zB[r])
        f0 = b - np.floor(b)
        if f0 < away or f0 > 1.0 - away:
            continue

        rho.fill(0.0)
        rho[r] = 1.0
        basis.btran(rho)
        alpha = basis.pivot_row(rho)              # row r of B^-1 M, all columns

        # gamma: shift each nonbasic to a non-negative deviation from its bound
        g = np.zeros(basis.N, dtype=VAL)
        ok = True
        for j in range(basis.N):
            st = basis.status[j]
            if st == BASIC or st == FIXED:
                continue
            a = alpha[j]
            if a == 0.0:
                continue
            if st == AT_LOWER:
                gamma = a
                sgn = 1.0
            elif st == AT_UPPER:
                gamma = -a
                sgn = -1.0
            else:
                ok = False                        # a free nonbasic has no bound
                break

            if int_mask_full[j]:
                fj = gamma - np.floor(gamma)
                psi = (fj / f0) if fj <= f0 else ((1.0 - fj) / (1.0 - f0))
            else:
                psi = (gamma / f0) if gamma > 0.0 else (-gamma / (1.0 - f0))
            if not np.isfinite(psi) or abs(psi) > max_coef:
                ok = False
                break
            if psi != 0.0:
                g[j] = sgn * psi
        if not ok:
            continue

        # rhs: 1 + sum_j g_j * (current bound value of z_j)
        zhat = basis.nonbasic_values()
        rhs = 1.0 + float(g @ zhat)

        # Expand the logicals. In the augmented form ``A x - s = 0``, so
        # ``z_{n+i} = (A x)_i`` and the logical part of the cut contributes
        # ``sum_i g_{n+i} (A x)_i = (Aᵀ g_logical) . x``.
        coef = g[:n].copy()
        gl = np.ascontiguousarray(g[n:])
        if np.any(gl):
            coef = coef + A.rmatvec(gl)

        idx = np.flatnonzero(np.abs(coef) > 1e-11)
        if idx.size == 0:
            continue
        cuts.append(Cut(idx.astype(IDX), coef[idx].copy(), rhs, kind="gmi"))
    return cuts


# --------------------------------------------------------------------------- #
# knapsack cover                                                               #
# --------------------------------------------------------------------------- #


def generate_cover(prob, x, int_mask, lo, hi, max_cuts: int = 50,
                   min_violation: float = 1e-4) -> list[Cut]:
    """Cover cuts from ``<=`` rows over binaries with positive coefficients."""
    cuts: list[Cut] = []
    binary = int_mask & (lo >= -1e-9) & (hi <= 1.0 + 1e-9)
    if not binary.any():
        return cuts

    for i in range(prob.m):
        if len(cuts) >= max_cuts:
            break
        ru = prob.row_ub[i]
        if ru >= INF:
            continue
        cols, vals = prob.A.row(i)
        if cols.size == 0 or cols.size > 500:
            continue
        keep = binary[cols] & (vals > 1e-9)
        if keep.sum() < 2:
            continue
        c = cols[keep]
        a = vals[keep]
        # coefficients on non-binary or negative terms are pushed into the rhs
        rest = ~keep
        b = ru
        if rest.any():
            other = cols[rest]
            ov = vals[rest]
            b -= float(np.sum(np.where(ov > 0, ov * lo[other], ov * hi[other])))
        if b <= 0 or a.sum() <= b + 1e-9:
            continue                              # no cover exists

        xs = np.clip(x[c], 0.0, 1.0)
        # greedy: prefer items that are nearly 1 and have large weight
        order = np.argsort((1.0 - xs) / np.maximum(a, 1e-12))
        total = 0.0
        cover = []
        for t in order:
            cover.append(t)
            total += a[t]
            if total > b + 1e-9:
                break
        if total <= b + 1e-9:
            continue

        # shrink to a minimal cover: drop any item still leaving a cover
        cover.sort(key=lambda t: xs[t])
        k = 0
        while k < len(cover):
            t = cover[k]
            if total - a[t] > b + 1e-9 and len(cover) > 2:
                total -= a[t]
                cover.pop(k)
            else:
                k += 1

        cov = np.array(cover, dtype=int)
        lhs = float(xs[cov].sum())
        if lhs <= len(cov) - 1 + min_violation:
            continue
        # sum_{C} x_j <= |C| - 1   ->   -sum x_j >= 1 - |C|
        cuts.append(Cut(c[cov].astype(IDX), -np.ones(cov.size, dtype=VAL),
                        float(1 - len(cov)), kind="cover"))
    return cuts


# --------------------------------------------------------------------------- #
# adding cuts to a model                                                       #
# --------------------------------------------------------------------------- #


def append_cuts(prob, cuts: list[Cut]):
    """Return a copy of ``prob`` with the cuts appended as ``>=`` rows."""
    if not cuts:
        return prob
    A = prob.A
    m, n = A.shape
    k = len(cuts)

    rows, cols, vals = [], [], []
    rp = A.rp
    for i in range(m):
        s, e = rp[i], rp[i + 1]
        rows.append(np.full(e - s, i, dtype=IDX))
        cols.append(A.ri[s:e])
        vals.append(A.rx[s:e])
    for t, c in enumerate(cuts):
        rows.append(np.full(c.idx.size, m + t, dtype=IDX))
        cols.append(c.idx)
        vals.append(c.val)

    newA = SparseMatrix.from_triplets(np.concatenate(rows),
                                      np.concatenate(cols),
                                      np.concatenate(vals), m + k, n)
    out = prob.copy()
    out.A = newA
    out.row_lb = np.concatenate([prob.row_lb,
                                 np.array([c.rhs for c in cuts], dtype=VAL)])
    out.row_ub = np.concatenate([prob.row_ub, np.full(k, INF, dtype=VAL)])
    if prob.row_names:
        out.row_names = list(prob.row_names) + [f"cut_{c.kind}_{t}"
                                                for t, c in enumerate(cuts)]
    return out


# --------------------------------------------------------------------------- #
# mixed-integer rounding                                                       #
# --------------------------------------------------------------------------- #


def _mir_function(a, f0):
    """MIR coefficient for an integer column: ``floor(a) + max(0, f-f0)/(1-f0)``."""
    fl = np.floor(a)
    f = a - fl
    return fl + np.maximum(f - f0, 0.0) / (1.0 - f0)


def generate_mir(prob, x, int_mask, lo, hi, max_cuts: int = 50,
                 min_violation: float = 1e-5, away: float = 0.01,
                 max_coef: float = 1e7) -> list[Cut]:
    """Mixed-integer rounding cuts from single original rows.

    GMI cuts are MIR applied to a *tableau* row; these are MIR applied to the
    model's own rows, which reaches inequalities the tableau does not expose --
    and, unlike GMI, they need no basis, so they can be separated anywhere.

    For a row ``sum a_j x_j <= b`` with every variable shifted to be
    non-negative, and any scale ``d > 0``, writing ``a'_j = a_j/d``,
    ``b' = b/d`` and ``f0 = b' - floor(b')``, the inequality

        sum_{j integer} MIR(a'_j) x_j + sum_{j continuous, a'_j<0} a'_j/(1-f0) x_j
            <= floor(b')

    is valid whenever ``f0`` is strictly between 0 and 1. Several scales are
    tried per row -- the coefficients of the integer columns are the classic
    choices, because they are what make ``f0`` land away from the ends.

    Rows containing a free variable are skipped: the shift needs a finite bound
    to complement against, and a cut derived without one is not valid.
    """
    cuts: list[Cut] = []
    A = prob.A

    for i in range(prob.m):
        if len(cuts) >= max_cuts:
            break
        for side in (1.0, -1.0):
            bound = prob.row_ub[i] if side > 0 else prob.row_lb[i]
            if side > 0 and bound >= INF:
                continue
            if side < 0 and bound <= -INF:
                continue

            cols, vals = A.row(i)
            if cols.size == 0 or cols.size > 400:
                continue
            a = side * vals
            b = side * bound

            # shift every variable to a non-negative deviation from a bound
            shift_up = np.zeros(cols.size, dtype=bool)
            ok = True
            for t, j in enumerate(cols):
                if lo[j] > -INF:
                    b -= a[t] * lo[j]
                elif hi[j] < INF:
                    shift_up[t] = True
                    b -= a[t] * hi[j]
                    a[t] = -a[t]
                else:
                    ok = False
                    break
            if not ok:
                continue

            isint = int_mask[cols]
            if not isint.any():
                continue

            scales = {1.0}
            for t in np.flatnonzero(isint):
                v = abs(a[t])
                if v > 1e-9:
                    scales.add(v)
            for d in sorted(scales)[:6]:
                if len(cuts) >= max_cuts:
                    break
                ad = a / d
                bd = b / d
                f0 = bd - np.floor(bd)
                if f0 < away or f0 > 1.0 - away:
                    continue

                coef = np.zeros(cols.size, dtype=VAL)
                coef[isint] = _mir_function(ad[isint], f0)
                cont = ~isint
                neg = cont & (ad < 0.0)
                coef[neg] = ad[neg] / (1.0 - f0)
                rhs = np.floor(bd)

                if not np.isfinite(coef).all() or np.abs(coef).max() > max_coef:
                    continue

                # Undo the shift. The cut holds in the shifted space as
                #     sum_t coef_t * t_t <= rhs,
                # with t_t = x_j - lo_j, or hi_j - x_j where shift_up. Both
                # substitutions move a constant to the right-hand side:
                #     x_j - lo_j :  coef*x_j - coef*lo_j   ->  r += coef*lo_j
                #     hi_j - x_j : -coef*x_j + coef*hi_j   ->  r -= coef*hi_j
                # Getting either sign wrong is invisible on any column whose
                # lower bound is zero, which is most columns in most models.
                # It stayed invisible through 133 brute-forced cuts over 55
                # instances, until flugpl -- five columns with lower bound 57
                # -- where the cut removed the integer optimum and the solver
                # reported INFEASIBLE on a provably feasible model.
                g = np.where(shift_up, -coef, coef)
                r = rhs
                for t, j in enumerate(cols):
                    if shift_up[t]:
                        r -= coef[t] * hi[j]
                    else:
                        r += coef[t] * lo[j]
                # express as  -g.x >= -r   (the >= convention used here)
                idx = np.flatnonzero(np.abs(g) > 1e-11)
                if idx.size == 0:
                    continue
                cut = Cut(cols[idx].astype(IDX), -g[idx].copy(), float(-r),
                          kind="mir")
                if cut.violation(x) > min_violation:
                    cuts.append(cut)
    return cuts
