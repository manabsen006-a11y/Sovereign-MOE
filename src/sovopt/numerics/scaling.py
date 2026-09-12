"""Matrix scaling.

Refinery models are the canonical badly-scaled matrices: kilotonne flows sit in
the same constraint as sulfur in parts per million, so ``|a_ij|`` routinely
spans ten orders of magnitude. An unscaled factorisation of such a matrix loses
most of its significant digits, and the solver reports "numerical trouble" or,
worse, a confident wrong dual price.

Three scalings are provided:

``geometric``
    Cheap, iterative, ``r_i = 1/sqrt(max_j |a_ij| · min_j |a_ij|)``. A good
    default and what most solvers do by default.

``ruiz``
    Iterative ∞-norm equilibration. Drives every row and column max towards 1.
    Converges linearly and is the standard preconditioner for first-order
    methods, so the PDLP path uses it.

``curtis_reid``
    Least-squares scaling: choose ``r``, ``c`` minimising
    ``Σ (log₂|a_ij| + r_i + c_j)²`` over the stored non-zeros. Unlike
    equilibration this looks at the *whole distribution* of magnitudes rather
    than just the extremes, which is why it does better on matrices with a few
    outlying coefficients -- exactly the refinery case. Solved here by conjugate
    gradients on the normal equations.

Every scale factor is finally rounded to a **power of two**. That makes the
scaling exact in binary floating point: it moves exponents and touches no
mantissa bit, so scaling itself contributes zero rounding error. Skipping this
step is a common and subtle source of irreproducibility.

Sign conventions
----------------
With ``A' = R A C`` (``R = diag(r)``, ``C = diag(c)``):

    x  = C x'          primal
    y  = R y'          row duals
    d  = C⁻¹ d'        reduced costs
    c' = C c           objective
    rl' = R rl, ru' = R ru,  l' = C⁻¹ l, u' = C⁻¹ u

References
----------
Curtis & Reid, "On the automatic scaling of matrices for Gaussian elimination",
  IMA J. Appl. Math. 10 (1972) 118-124.
Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in
  matrices", RAL-TR-2001-034, Rutherford Appleton Laboratory, 2001.
Tomlin, "On scaling linear programming problems", Math. Prog. Study 4 (1975).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.problem import Problem
from ..core.sparse import SparseMatrix, VAL
from ..core.tolerances import INF

__all__ = ["Scaling", "compute_scaling", "scale_problem"]


@dataclass
class Scaling:
    """Diagonal row and column scale factors, plus objective and rhs scalars."""

    row: np.ndarray
    col: np.ndarray
    obj: float = 1.0
    rhs: float = 1.0
    method: str = "none"
    before: tuple[float, float] = (0.0, 0.0)
    after: tuple[float, float] = (0.0, 0.0)

    @property
    def ratio_before(self) -> float:
        lo, hi = self.before
        return hi / lo if lo > 0 else float("inf")

    @property
    def ratio_after(self) -> float:
        lo, hi = self.after
        return hi / lo if lo > 0 else float("inf")

    def unscale_primal(self, x_scaled):
        return np.asarray(x_scaled, dtype=VAL) * self.col

    def scale_primal(self, x):
        return np.asarray(x, dtype=VAL) / self.col

    def unscale_dual(self, y_scaled):
        """Row duals back to original units.

        With ``A' = R A C`` and ``c' = s·C c`` (``s`` the objective scale), the
        scaled reduced cost is ``d' = c' + A'ᵀy' = C (s·c + Aᵀ R y')``. Matching
        that against ``d = c + Aᵀy`` in original units forces

            y = R y' / s

        Note the **division** by ``s``: multiplying instead leaves the duals
        wrong by a factor of ``s²``, which still satisfies the sign conditions
        (so dual feasibility looks fine) but pins the duality gap at a constant
        and reports wrong marginal prices.
        """
        return np.asarray(y_scaled, dtype=VAL) * self.row / (self.obj / self.rhs)

    def unscale_reduced(self, d_scaled):
        """Reduced costs back to original units: ``d = C⁻¹ d' / s``."""
        return np.asarray(d_scaled, dtype=VAL) / self.col / self.obj

    def report(self) -> str:
        return (f"scaling[{self.method}] coefficient ratio "
                f"{self.ratio_before:.3e} -> {self.ratio_after:.3e}")


def _pow2_round(s: np.ndarray) -> np.ndarray:
    """Round each factor to the nearest power of two.

    ``frexp`` gives ``s = mantissa · 2**e`` with mantissa in [0.5, 1). Rounding
    the exponent this way keeps the factor within a factor of sqrt(2) of the
    ideal while making every multiplication exact.
    """
    s = np.asarray(s, dtype=VAL)
    out = np.ones_like(s)
    good = np.isfinite(s) & (s > 0.0)
    if good.any():
        out[good] = np.exp2(np.rint(np.log2(s[good])))
    return out


def _extremes(A: SparseMatrix) -> tuple[float, float]:
    if A.nnz == 0:
        return (1.0, 1.0)
    a = np.abs(A.cx)
    nz = a[a > 0.0]
    if nz.size == 0:
        return (1.0, 1.0)
    return (float(nz.min()), float(nz.max()))


# --------------------------------------------------------------------------- #
# algorithms                                                                   #
# --------------------------------------------------------------------------- #


def _ruiz(A: SparseMatrix, iters: int = 20, tol: float = 1e-2):
    """∞-norm equilibration. Returns unrounded (r, c)."""
    m, n = A.shape
    r = np.ones(m, dtype=VAL)
    c = np.ones(n, dtype=VAL)
    W = A.copy()

    for _ in range(iters):
        rmax = W.row_absmax()
        cmax = W.col_absmax()
        rmax[rmax <= 0.0] = 1.0
        cmax[cmax <= 0.0] = 1.0
        dev = max(float(np.abs(1.0 - rmax).max(initial=0.0)),
                  float(np.abs(1.0 - cmax).max(initial=0.0)))
        dr = 1.0 / np.sqrt(rmax)
        dc = 1.0 / np.sqrt(cmax)
        W.scale(dr, dc)
        r *= dr
        c *= dc
        if dev < tol:
            break
    return r, c


def _geometric(A: SparseMatrix, iters: int = 10, tol: float = 1e-2):
    """Geometric-mean scaling: drive sqrt(max·min) of each row and column to 1."""
    m, n = A.shape
    r = np.ones(m, dtype=VAL)
    c = np.ones(n, dtype=VAL)
    W = A.copy()

    prev = float("inf")
    for _ in range(iters):
        rmax, rmin = W.row_absmax(), W.row_absmin()
        g = np.sqrt(rmax * rmin)
        g[g <= 0.0] = 1.0
        dr = 1.0 / g
        W.scale(dr, np.ones(n, dtype=VAL))
        r *= dr

        cmax, cmin = W.col_absmax(), W.col_absmin()
        g = np.sqrt(cmax * cmin)
        g[g <= 0.0] = 1.0
        dc = 1.0 / g
        W.scale(np.ones(m, dtype=VAL), dc)
        c *= dc

        lo, hi = _extremes(W)
        ratio = hi / lo if lo > 0 else float("inf")
        if ratio > 0.9 * prev:
            break
        prev = ratio
    return r, c


def _curtis_reid(A: SparseMatrix, iters: int = 40, tol: float = 1e-8):
    """Least-squares scaling in log2 space, by conjugate gradients.

    Minimise ``Σ_{(i,j)∈nz} (log₂|a_ij| + r_i + c_j)²``. The normal equations are

        diag(R) r  +  P   c = -σ
        Pᵀ      r  +  diag(C) c = -τ

    with ``P`` the 0/1 sparsity pattern, ``R``/``C`` the row/column counts and
    ``σ``/``τ`` the row/column sums of ``log₂|a_ij|``. The block system is
    symmetric positive semi-definite, so CG applies directly; we never form it.
    """
    m, n = A.shape
    if A.nnz == 0:
        return np.ones(m, dtype=VAL), np.ones(n, dtype=VAL)

    logs = np.log2(np.abs(A.cx))
    P = SparseMatrix(m, n, A.cp, A.ci, np.ones(A.nnz, dtype=VAL))

    Rcnt = np.diff(A.rp).astype(VAL)
    Ccnt = np.diff(A.cp).astype(VAL)
    Rsafe = np.where(Rcnt > 0, Rcnt, 1.0)
    Csafe = np.where(Ccnt > 0, Ccnt, 1.0)

    # sigma_i = sum over row i of log2|a_ij|;  tau_j = same over column j
    L = SparseMatrix(m, n, A.cp, A.ci, logs)
    sigma = L.matvec(np.ones(n, dtype=VAL))
    tau = L.rmatvec(np.ones(m, dtype=VAL))

    def apply(r, c):
        """The block operator applied to (r, c)."""
        return Rcnt * r + P.matvec(c), P.rmatvec(r) + Ccnt * c

    r = np.zeros(m, dtype=VAL)
    c = np.zeros(n, dtype=VAL)
    br, bc = -sigma, -tau

    Ar, Ac = apply(r, c)
    resr, resc = br - Ar, bc - Ac
    # Jacobi preconditioner: the diagonal is exactly the row/column counts
    zr, zc = resr / Rsafe, resc / Csafe
    pr, pc = zr.copy(), zc.copy()
    rz = float(resr @ zr + resc @ zc)
    rz0 = rz

    for _ in range(iters):
        if rz <= tol * max(rz0, 1.0):
            break
        Apr, Apc = apply(pr, pc)
        denom = float(pr @ Apr + pc @ Apc)
        if abs(denom) < 1e-300:
            break
        alpha = rz / denom
        r += alpha * pr
        c += alpha * pc
        resr -= alpha * Apr
        resc -= alpha * Apc
        zr, zc = resr / Rsafe, resc / Csafe
        rz_new = float(resr @ zr + resc @ zc)
        beta = rz_new / rz
        pr = zr + beta * pr
        pc = zc + beta * pc
        rz = rz_new

    # r, c are exponents; empty rows/columns get no scaling
    r[Rcnt == 0] = 0.0
    c[Ccnt == 0] = 0.0
    return np.exp2(r), np.exp2(c)


def _pock_chambolle(A: SparseMatrix, alpha: float = 1.0):
    """Pock-Chambolle diagonal preconditioner.

    ``r_i = 1/sqrt(Σ_j |a_ij|^(2-α))``, ``c_j = 1/sqrt(Σ_i |a_ij|^α)``. At
    ``α = 1`` this is the choice that makes the PDHG step-size condition
    ``τσ‖A‖² < 1`` hold with ``τ = σ = 1`` for the preconditioned matrix, which
    is why the first-order method uses it rather than plain equilibration.
    """
    absx = np.abs(A.cx)
    if alpha == 1.0:
        w = absx
        wt = absx
    else:
        w = absx ** (2.0 - alpha)
        wt = absx ** alpha
    Rw = SparseMatrix(A.m, A.n, A.cp, A.ci, w)
    Cw = SparseMatrix(A.m, A.n, A.cp, A.ci, wt)
    rsum = Rw.matvec(np.ones(A.n, dtype=VAL))
    csum = Cw.rmatvec(np.ones(A.m, dtype=VAL))
    rsum[rsum <= 0.0] = 1.0
    csum[csum <= 0.0] = 1.0
    return 1.0 / np.sqrt(rsum), 1.0 / np.sqrt(csum)


def _pdlp(A: SparseMatrix, ruiz_iters: int = 10):
    """The preconditioner the first-order method wants: Ruiz, then Pock-Chambolle.

    Ruiz flattens the extremes; Pock-Chambolle then balances the *row and column
    mass*, which is what governs the PDHG step size. Applying only one of the two
    leaves a measurable amount of convergence on the table.
    """
    r, c = _ruiz(A, iters=ruiz_iters)
    W = A.copy()
    W.scale(r, c)
    r2, c2 = _pock_chambolle(W)
    return r * r2, c * c2


_METHODS = {
    "ruiz": _ruiz,
    "geometric": _geometric,
    "curtis_reid": _curtis_reid,
    "pock_chambolle": _pock_chambolle,
    "pdlp": _pdlp,
}


def compute_scaling(A: SparseMatrix, method: str = "auto",
                    pow2: bool = True, unscaled_cols: np.ndarray | None = None) -> Scaling:
    """Compute row and column scale factors for ``A``.

    ``method='auto'`` picks Curtis-Reid when the coefficient spread is wide
    enough to matter (ratio > 1e4) and geometric scaling otherwise -- Curtis-Reid
    costs a handful of sparse products, which is not worth paying on an
    already-well-scaled matrix.

    ``unscaled_cols`` is a boolean mask of columns that must be left at a scale
    factor of exactly 1. **Integer variables belong in it**: the scaled variable
    is ``x' = x / c_j``, and unless ``c_j = 1`` the requirement "x is an integer"
    does not survive the change of variables. Forgetting this produces a solver
    that returns fractional values it believes are integral.
    """
    before = _extremes(A)
    ratio = before[1] / before[0] if before[0] > 0 else float("inf")

    auto_selected = method == "auto"
    if auto_selected:
        method = "curtis_reid" if ratio > 1e4 else "geometric"
    if method in ("none", None):
        return Scaling(np.ones(A.m, dtype=VAL), np.ones(A.n, dtype=VAL),
                       method="none", before=before, after=before)
    if method not in _METHODS:
        raise ValueError(f"unknown scaling method {method!r}; "
                         f"choose from {sorted(_METHODS)} or 'none'/'auto'")

    r, c = _METHODS[method](A)
    if pow2:
        r = _pow2_round(r)
        c = _pow2_round(c)
    if unscaled_cols is not None and np.any(unscaled_cols):
        c = c.copy()
        c[np.asarray(unscaled_cols, dtype=bool)] = 1.0

    W = A.copy()
    W.scale(r, c)
    after = _extremes(W)

    # Refuse a scaling that made things worse -- it happens on already-balanced
    # matrices and silently degrading the model is not acceptable. This guard
    # applies only when we chose the method; an explicitly requested
    # preconditioner is honoured, because the first-order method optimises the
    # PDHG step size rather than the coefficient ratio.
    if auto_selected and after[1] / max(after[0], 1e-300) > ratio:
        return Scaling(np.ones(A.m, dtype=VAL), np.ones(A.n, dtype=VAL),
                       method="none (rejected: no improvement)",
                       before=before, after=before)

    return Scaling(r, c, method=method, before=before, after=after)


def scale_problem(prob: Problem, method: str = "auto",
                  scale_obj: bool = True) -> tuple[Problem, Scaling]:
    """Return a scaled copy of ``prob`` and the :class:`Scaling` used.

    Infinite bounds stay infinite; only finite ones are transformed. Integer,
    binary and semi-continuous columns are pinned to a scale factor of 1 so
    integrality is preserved exactly.
    """
    from ..core.problem import VarKind
    pinned = prob.kind != VarKind.CONTINUOUS
    sc = compute_scaling(prob.A, method=method, unscaled_cols=pinned)

    A = prob.A.copy()
    A.scale(sc.row, sc.col)

    c = prob.c * sc.col
    obj_scale = 1.0
    if scale_obj:
        # The objective's scale is the scale of its gradient, and for a
        # quadratic that is ``c + Qx``, not ``c``. QPLIB_8559 has ``c = 0``
        # and a ``Q`` whose diagonal runs to 95,000: with the gradient left
        # at that size the interior point started with a dual residual of
        # 1.7e5, drove the iterate to its bounds before the residual was
        # gone, and then crawled -- the primal step length pinned near
        # zero, 87 iterations in 600 s without converging. With the
        # diagonal of the column-scaled ``Q`` in the geometric mean the
        # same instance takes 13 iterations. (The diagonal is the right
        # summary: it is the curvature along each scaled unit direction.)
        nz = np.abs(c[c != 0.0])
        if prob.Q is not None:
            qd = np.abs(prob.Q.diagonal()) * sc.col * sc.col
            nz = np.concatenate([nz, qd[qd != 0.0]])
        if nz.size:
            g = float(np.exp2(np.rint(np.log2(np.sqrt(nz.max() * nz.min())))))
            if g > 0 and np.isfinite(g):
                obj_scale = 1.0 / g
    c = c * obj_scale
    sc.obj = obj_scale

    def _row_bound(b, s):
        out = b * s
        out[b <= -INF] = -INF
        out[b >= INF] = INF
        return out

    def _col_bound(b, s):
        out = b / s
        out[b <= -INF] = -INF
        out[b >= INF] = INF
        return out

    Q = None
    if prob.Q is not None:
        Q = prob.Q.copy()
        Q.scale(sc.col, sc.col)
        Q.cx *= obj_scale
        Q.rx *= obj_scale

    scaled = Problem(
        A=A, c=c,
        row_lb=_row_bound(prob.row_lb, sc.row),
        row_ub=_row_bound(prob.row_ub, sc.row),
        col_lb=_col_bound(prob.col_lb, sc.col),
        col_ub=_col_bound(prob.col_ub, sc.col),
        kind=prob.kind.copy(), Q=Q,
        obj_offset=prob.obj_offset * obj_scale,
        sense=prob.sense, name=prob.name,
        col_names=prob.col_names, row_names=prob.row_names,
        meta=dict(prob.meta),
    )
    scaled.meta["scaling"] = sc
    return scaled, sc
