"""Iterative refinement, compensated residuals, and condition estimation.

A factorisation of an ill-conditioned matrix loses roughly ``log10(cond)``
digits. Refinery bases routinely reach ``cond ~ 1e12``, which leaves about four
correct digits out of sixteen -- not enough to decide whether a reduced cost is
negative, which is the only question the simplex ever asks. Iterative refinement
buys those digits back, but **only if the residual is computed more accurately
than the solve**. Refining with a residual computed in the same precision
converges to the same wrong answer.

There is no reliable quad precision on this platform (NumPy's ``float128`` is an
alias for ``float64`` under MSVC), so the extra precision is manufactured with
**error-free transformations**: ``two_product`` and ``two_sum`` recover the exact
rounding error of a single multiplication or addition as a second float. Summing
those errors alongside the main accumulator gives an effectively double-double
residual at roughly 4x the cost of a plain one -- far cheaper than a second
factorisation, and it is what turns a 1e-5 solve into a 1e-15 one.

References
----------
Dekker, "A floating-point technique for extending the available precision",
  Numer. Math. 18 (1971) 224-242 -- the splitting used by ``two_product``.
Knuth, *TAOCP* vol. 2, §4.2.2 -- ``two_sum``.
Ogita, Rump & Oishi, "Accurate sum and dot product", SIAM J. Sci. Comput. 26
  (2005) 1955-1988 -- compensated dot products, the scheme used here.
Wilkinson, *Rounding Errors in Algebraic Processes*, HMSO 1963 -- iterative
  refinement and its convergence condition.
Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002,
  Ch. 12 and 15 -- refinement in fixed precision, condition estimation.
Hager, "Condition estimates", SIAM J. Sci. Stat. Comput. 5 (1984) 311-316.
Higham & Tisseur, "A block algorithm for matrix 1-norm estimation", SIAM J.
  Matrix Anal. Appl. 21 (2000) 1185-1201.
"""

from __future__ import annotations

import numpy as np

from ..core._jit import jit_kernel, prange
from ..core.sparse import VAL

__all__ = ["compensated_residual", "refine_solve", "estimate_condition",
           "RefineResult"]

_SPLIT = 134217729.0     # 2**27 + 1, the Dekker splitting constant


@jit_kernel(parallel=True)
def _residual_csr(rp, ri, rx, x, b, out):
    """``out = b - A x`` with a compensated dot product per row.

    The main accumulator ``s`` carries the double-precision sum; ``comp``
    carries the accumulated rounding errors of every product and every addition.
    Their sum is the residual to roughly twice working precision.
    """
    m = rp.shape[0] - 1
    for i in prange(m):
        s = 0.0
        comp = 0.0
        for p in range(rp[i], rp[i + 1]):
            a = rx[p]
            xv = x[ri[p]]

            # --- two_product(a, xv) -> (prod, err) exactly ---
            prod = a * xv
            ca = _SPLIT * a
            ah = ca - (ca - a)
            al = a - ah
            cb = _SPLIT * xv
            bh = cb - (cb - xv)
            bl = xv - bh
            err = ((ah * bh - prod) + ah * bl + al * bh) + al * bl

            # --- two_sum(s, prod) -> (s, e) exactly ---
            t = s + prod
            bb = t - s
            e = (s - (t - bb)) + (prod - bb)
            s = t
            comp += e + err

        # r = b - (s + comp), keeping the low-order part of b - s
        bi = b[i]
        t = bi - s
        bb = t - bi
        e = (bi - (t - bb)) + (-s - bb)
        out[i] = t + (e - comp)


class RefineResult:
    """Outcome of a refined solve."""

    __slots__ = ("x", "iterations", "residual", "residual0", "converged",
                 "step_norm")

    def __init__(self, x, iterations, residual, residual0, converged, step_norm):
        self.x = x
        self.iterations = iterations
        self.residual = residual
        self.residual0 = residual0
        self.converged = converged
        self.step_norm = step_norm

    @property
    def improvement(self) -> float:
        return self.residual0 / max(self.residual, 1e-300)

    def __repr__(self):
        return (f"RefineResult(iters={self.iterations}, "
                f"res {self.residual0:.2e} -> {self.residual:.2e}, "
                f"{'converged' if self.converged else 'stalled'})")


def compensated_residual(rp, ri, rx, x, b, out=None):
    """``b - A x`` to roughly double-double accuracy. ``A`` in CSR."""
    m = rp.shape[0] - 1
    if out is None:
        out = np.empty(m, dtype=VAL)
    _residual_csr(rp, ri, rx, np.ascontiguousarray(x, dtype=VAL),
                  np.ascontiguousarray(b, dtype=VAL), out)
    return out


def refine_solve(factor, rp, ri, rx, b, x0=None, max_iter: int = 5,
                 tol: float = 1e-14, transpose: bool = False) -> RefineResult:
    """Solve ``B x = b`` (or ``Bᵀ x = b``) with iterative refinement.

    ``factor`` is an :class:`~sovopt.numerics.lu.LUFactor`; ``rp/ri/rx`` are the
    CSR arrays of the *same* matrix the factor was built from -- for the
    transposed solve, pass the CSC arrays instead, which are the CSR arrays of
    ``Bᵀ``.

    Stops when the residual stops improving: refinement in fixed precision
    converges only while the correction is larger than the noise floor, and
    iterating past that point wastes solves and can drift.
    """
    b = np.ascontiguousarray(b, dtype=VAL)
    solve = factor.btran if transpose else factor.ftran

    x = solve(b.copy()) if x0 is None else np.array(x0, dtype=VAL, copy=True)

    r = compensated_residual(rp, ri, rx, x, b)
    res0 = float(np.abs(r).max(initial=0.0))
    res = res0
    bnorm = max(float(np.abs(b).max(initial=0.0)), 1.0)
    step = 0.0
    it = 0

    for it in range(1, max_iter + 1):
        if res <= tol * bnorm:
            break
        dx = solve(r.copy())
        step = float(np.abs(dx).max(initial=0.0))
        x_new = x + dx
        r_new = compensated_residual(rp, ri, rx, x_new, b)
        res_new = float(np.abs(r_new).max(initial=0.0))
        if res_new >= 0.9 * res:      # no longer paying for itself
            if res_new < res:
                x, r, res = x_new, r_new, res_new
            break
        x, r, res = x_new, r_new, res_new

    return RefineResult(x, it, res, res0, res <= tol * bnorm, step)


def estimate_condition(factor, anorm: float, n: int, iters: int = 5) -> float:
    """Estimate ``cond_1(B) = ||B||_1 · ||B⁻¹||_1`` without forming ``B⁻¹``.

    Hager's algorithm: maximise ``||B⁻¹ v||_1`` over the unit 1-norm ball by
    alternating a solve with ``B`` and a solve with ``Bᵀ`` applied to the sign
    vector. Each iteration costs one FTRAN and one BTRAN, so a few iterations
    are affordable at every refactorisation -- which is what makes it practical
    to report a numerical-quality warning on every solve rather than never.

    The result is a lower bound on the true condition number, typically within
    a factor of 3. That is ample for deciding "this model is going to hurt".
    """
    if n == 0 or anorm <= 0.0:
        return 1.0

    v = np.full(n, 1.0 / n, dtype=VAL)
    est = 0.0
    prev_j = -1

    for _ in range(max(1, iters)):
        x = factor.ftran(v.copy())
        est_new = float(np.abs(x).sum())
        xi = np.sign(x)
        xi[xi == 0.0] = 1.0
        z = factor.btran(xi)
        j = int(np.argmax(np.abs(z)))
        if est_new <= est or j == prev_j:
            est = max(est, est_new)
            break
        est = est_new
        prev_j = j
        v = np.zeros(n, dtype=VAL)
        v[j] = 1.0

    return float(anorm * est)
