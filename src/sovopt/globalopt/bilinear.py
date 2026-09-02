"""Bilinear problems and their McCormick relaxation.

Why this matters for a refinery
-------------------------------
Blending is not linear. A pool's sulfur content multiplied by the flow leaving
the pool is a product of two decision variables, and every quality specification
downstream inherits that product. The feasible set is non-convex, so a local
optimum is not a global one -- and the industry's standard workaround
(*distributive recursion* inside PIMS and GRTMPS: guess the pool qualities,
solve the resulting LP, update the guess, repeat) is a fixed-point iteration
with no global guarantee. It can and does converge to a blend worth less than
the true optimum, and the difference is quality give-away that shows up as
margin.

Representation
--------------
A bilinear problem is an ordinary :class:`~sovopt.core.problem.Problem` over an
*extended* variable vector, together with a list of product definitions

    w_t = x_t * y_t

The linear part references ``w_t`` like any other column, so every existing
piece of machinery -- scaling, presolve, the simplex, propagation -- works
unchanged. Only the relaxation and the branching know that ``w`` is special.

McCormick envelopes
-------------------
Over a box ``x in [xL, xU]``, ``y in [yL, yU]``, the four inequalities

    w >= xL*y + x*yL - xL*yL          w <= xU*y + x*yL - xU*yL
    w >= xU*y + x*yU - xU*yU          w <= xL*y + x*yU - xL*yU

are exactly the convex hull of ``{(x, y, w) : w = x*y}`` over that box. They are
tight at the four corners and loosest in the middle, which is what drives the
search: **the envelope tightens quadratically as the box shrinks**, so splitting
a range in half roughly quarters the relaxation gap contributed by that term.
That is the entire mechanism of spatial branch-and-bound.

A product of a variable with itself (``x*x``) gets the same treatment; the
envelope is still valid, merely weaker than the secant/tangent pair a dedicated
convex-quadratic handler would use.

References
----------
McCormick, "Computability of global solutions to factorable non-convex
  programs: Part I -- convex underestimating problems", Math. Prog. 10 (1976)
  147-175.
Al-Khayyal & Falk, "Jointly constrained biconvex programming", Math. Oper. Res.
  8 (1983) 273-286 -- the envelopes are the convex hull.
Haverly, "Studies of the behaviour of recursion for the pooling problem", ACM
  SIGMAP Bulletin 25 (1978) 19-28 -- why recursion is not enough.
Tawarmalani & Sahinidis, "Convexification and global optimization in continuous
  and mixed-integer nonlinear programming", Kluwer 2002.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.problem import ObjSense, Problem
from ..core.sparse import IDX, SparseMatrix, VAL
from ..core.tolerances import INF

__all__ = ["BilinearTerm", "BilinearProblem", "mccormick_rows",
           "build_relaxation"]


@dataclass(frozen=True)
class BilinearTerm:
    """``w = x * y``, all three referring to columns of the extended vector."""

    w: int
    x: int
    y: int

    def value(self, vec) -> float:
        return float(vec[self.x] * vec[self.y])

    def violation(self, vec) -> float:
        return abs(float(vec[self.w]) - self.value(vec))


@dataclass
class BilinearProblem:
    """Linear model plus product definitions."""

    linear: Problem
    terms: list = field(default_factory=list)
    name: str = "bilinear"

    @property
    def n(self) -> int:
        return self.linear.n

    def violations(self, x) -> np.ndarray:
        return np.array([t.violation(x) for t in self.terms], dtype=VAL) \
            if self.terms else np.zeros(0, dtype=VAL)

    def max_violation(self, x) -> float:
        v = self.violations(x)
        return float(v.max()) if v.size else 0.0

    def objective(self, x) -> float:
        return self.linear.objective(x)

    def feasible(self, x, tol: float = 1e-6):
        """``(row/bound violation, product violation)`` at ``x``."""
        rv, bv, _ = self.linear.violation(x)
        return max(rv, bv), self.max_violation(x)


def mccormick_rows(terms, lo, hi):
    """The four envelope inequalities per term, as ``(idx, val, lb, ub)`` rows.

    Rows are emitted in the canonical ``rl <= a.x <= ru`` form the rest of the
    solver uses. Infinite bounds are handled by dropping the envelope halves
    that would need them: a product with an unbounded factor has no finite
    convex hull, and emitting a row with a 1e30 coefficient would poison the
    factorisation for nothing.
    """
    rows = []
    for t in terms:
        xl, xu = float(lo[t.x]), float(hi[t.x])
        yl, yu = float(lo[t.y]), float(hi[t.y])
        finite_l = xl > -INF and yl > -INF
        finite_u = xu < INF and yu < INF

        if finite_l:
            # w - yl*x - xl*y >= -xl*yl
            rows.append((np.array([t.w, t.x, t.y], dtype=IDX),
                         np.array([1.0, -yl, -xl], dtype=VAL),
                         -xl * yl, INF))
        if finite_u:
            # w - yu*x - xu*y >= -xu*yu
            rows.append((np.array([t.w, t.x, t.y], dtype=IDX),
                         np.array([1.0, -yu, -xu], dtype=VAL),
                         -xu * yu, INF))
        if xu < INF and yl > -INF:
            # w - yl*x - xu*y <= -xu*yl
            rows.append((np.array([t.w, t.x, t.y], dtype=IDX),
                         np.array([1.0, -yl, -xu], dtype=VAL),
                         -INF, -xu * yl))
        if xl > -INF and yu < INF:
            # w - yu*x - xl*y <= -xl*yu
            rows.append((np.array([t.w, t.x, t.y], dtype=IDX),
                         np.array([1.0, -yu, -xl], dtype=VAL),
                         -INF, -xl * yu))
    return rows


def build_relaxation(bp: BilinearProblem, lo, hi) -> Problem:
    """The convex LP relaxation of ``bp`` over the box ``[lo, hi]``.

    Its optimum is a rigorous bound on the global optimum, because every
    bilinear-feasible point of the box satisfies all four envelopes.
    """
    base = bp.linear
    extra = mccormick_rows(bp.terms, lo, hi)
    m0, n = base.m, base.n

    rows, cols, vals = [], [], []
    rp = base.A.rp
    for i in range(m0):
        s, e = rp[i], rp[i + 1]
        rows.append(np.full(e - s, i, dtype=IDX))
        cols.append(base.A.ri[s:e])
        vals.append(base.A.rx[s:e])
    for t, (idx, val, _lb, _ub) in enumerate(extra):
        rows.append(np.full(idx.size, m0 + t, dtype=IDX))
        cols.append(idx)
        vals.append(val)

    A = SparseMatrix.from_triplets(
        np.concatenate(rows) if rows else np.zeros(0, dtype=IDX),
        np.concatenate(cols) if cols else np.zeros(0, dtype=IDX),
        np.concatenate(vals) if vals else np.zeros(0, dtype=VAL),
        m0 + len(extra), n)

    row_lb = np.concatenate([base.row_lb,
                             np.array([r[2] for r in extra], dtype=VAL)]) \
        if extra else base.row_lb.copy()
    row_ub = np.concatenate([base.row_ub,
                             np.array([r[3] for r in extra], dtype=VAL)]) \
        if extra else base.row_ub.copy()

    return Problem(A=A, c=base.c.copy(), row_lb=row_lb, row_ub=row_ub,
                   col_lb=np.asarray(lo, dtype=VAL).copy(),
                   col_ub=np.asarray(hi, dtype=VAL).copy(),
                   kind=base.kind.copy(), obj_offset=base.obj_offset,
                   sense=base.sense, name=f"{bp.name}_relax",
                   col_names=base.col_names)
