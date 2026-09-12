"""Sparse LDLᵀ for quasi-definite matrices, with the pivots the ordering names.

Why the LU was not enough
-------------------------
:mod:`sovopt.numerics.lu` is an unsymmetric factorisation with threshold
pivoting: given a column order it still chooses each pivot *row* for size,
and it follows a symmetric ordering only while the diagonal passes the
threshold. On the interior point's KKT that is exactly the case that fails.
The (1,1) block is ``-(Q + Θx⁻¹)``, and ``Θx⁻¹`` runs from 1e8 down to 1e-8
across the variables as the iterates approach their bounds, so a diagonal of
1e-8 sits in a column with entries of ``A`` at 1 -- the LU pivots off it, the
ordering's structure is gone, and the fill that minimum degree predicted at
87x on the QPLIB_8559 KKT is not what gets factorised. A 900 s solve
completed two iterations.

Why no pivoting is legitimate here
----------------------------------
The regularised KKT is *quasi-definite*: ``[[-E, Aᵀ], [A, F]]`` with ``E``
and ``F`` symmetric positive definite (the regularisation makes them so
even where ``Θ`` is zero). Vanderbei showed that for such a matrix ``LDLᵀ``
with 1×1 pivots exists for **every** symmetric permutation, with ``D``
carrying exactly ``n`` negative and ``m`` positive entries. So the pivot
order can be chosen for sparsity alone, which is what an ordering is for --
and the count of negative pivots is a check the factorisation makes on
itself. Stability is that of the regularised system, which is why the
interior point refines every solve against the unregularised matrix.

The factorisation
-----------------
Up-looking, one row of ``L`` per step. Row ``k`` of ``L`` has the nonzero
pattern of the *reach* of column ``k``'s upper entries in the elimination
tree, and that set comes out of the tree walk in an order that is already
topological, so the triangular solve that yields the row needs no sorting:

    y = L[:k, :k]⁻¹ a          a = the upper part of column k
    L[k, j] = y_j / d_j        d_k = a_kk − Σ_j L[k, j] y_j

The tree and the column counts are computed once from the pattern -- the
KKT's pattern never changes between iterations -- and every later
factorisation is the numeric pass alone.

References
----------
Vanderbei, "Symmetric quasi-definite matrices", SIAM J. Optim. 5 (1995)
  100-113 -- existence of LDLᵀ under any symmetric permutation, and the
  inertia.
Davis, *Direct Methods for Sparse Linear Systems*, SIAM 2006, ch. 4 -- the
  elimination tree, the reach of a column in it, and the up-looking
  factorisation built on both.
Liu, "The role of elimination trees in sparse factorization", SIAM J. Matrix
  Anal. Appl. 11 (1990) 134-172.
Altman & Gondzio, "Regularized symmetric indefinite systems in interior point
  methods for linear and quadratic optimization", Optim. Methods Softw. 11
  (1999) 275-302 -- the regularisation that makes the KKT quasi-definite.
"""

from __future__ import annotations

import numpy as np

from ..core._jit import jit_kernel
from ..core.sparse import IDX, VAL, coo_to_csc

__all__ = ["LDLSymbolic", "LDLFactor", "LDLSingular"]


class LDLSingular(Exception):
    """A pivot too small to trust: the matrix is not quasi-definite enough."""

    def __init__(self, k: int, msg: str = ""):
        super().__init__(msg or f"LDLᵀ pivot {k} negligible")
        self.k = k


# --------------------------------------------------------------------------- #
# symbolic                                                                     #
# --------------------------------------------------------------------------- #


@jit_kernel()
def _etree(n, Cp, Ci, parent, ancestor):
    """Elimination tree of the upper-triangular pattern ``C`` (Liu)."""
    for k in range(n):
        parent[k] = -1
        ancestor[k] = -1
        for p in range(Cp[k], Cp[k + 1]):
            i = Ci[p]
            while i != -1 and i < k:
                inext = ancestor[i]
                ancestor[i] = k
                if inext == -1:
                    parent[i] = k
                i = inext


@jit_kernel()
def _ereach(k, Cp, Ci, parent, s, w, n):
    """Pattern of row ``k`` of ``L``: nodes reached in the tree from column
    ``k``'s upper entries. Returned in ``s[top:n]``, topologically ordered;
    ``w`` is the visit mark, restored on exit."""
    top = n
    w[k] = k
    for p in range(Cp[k], Cp[k + 1]):
        i = Ci[p]
        if i > k:
            continue
        ln = 0
        while w[i] != k:
            s[ln] = i
            ln += 1
            w[i] = k
            i = parent[i]
        while ln > 0:
            ln -= 1
            top -= 1
            s[top] = s[ln]
    for p in range(top, n):
        w[s[p]] = -1
    w[k] = -1
    return top


@jit_kernel()
def _counts(n, Cp, Ci, parent, s, w, cnt):
    """Column counts of ``L`` by walking every row's reach once."""
    for k in range(n):
        cnt[k] = 0
        w[k] = -1
    total = 0
    for k in range(n):
        top = _ereach(k, Cp, Ci, parent, s, w, n)
        for p in range(top, n):
            cnt[s[p]] += 1
        total += n - top
    return total


# --------------------------------------------------------------------------- #
# numeric                                                                      #
# --------------------------------------------------------------------------- #


@jit_kernel()
def _ldl_numeric(n, Cp, Ci, Cx, parent, Lp, Li, Lx, D, s, w, x, c,
                 sign, delta):
    """Up-looking LDLᵀ with dynamic regularisation.

    ``sign[k]`` is the sign the pivot in permuted position ``k`` must have
    (``-1`` in the negative block, ``+1`` in the positive one, ``0`` when the
    caller does not know), and ``delta`` the magnitude it is replaced by
    when rounding has left it on the wrong side or too small to divide by.
    Returns ``(k_failed, n_negative, n_corrected)``; ``k_failed`` is ``-1``
    on success and a position only when no sign is known there and the
    pivot is unusable.
    """
    for k in range(n):
        c[k] = Lp[k]
        w[k] = -1
        x[k] = 0.0
    n_neg = 0
    n_fix = 0
    for k in range(n):
        top = _ereach(k, Cp, Ci, parent, s, w, n)
        # scatter the upper part of column k
        for p in range(Cp[k], Cp[k + 1]):
            i = Ci[p]
            if i <= k:
                x[i] += Cx[p]
        d = x[k]
        x[k] = 0.0
        for p in range(top, n):
            i = s[p]
            yi = x[i]
            x[i] = 0.0
            # existing entries of column i: rows strictly between i and k
            for q in range(Lp[i], c[i]):
                x[Li[q]] -= Lx[q] * yi
            lki = yi / D[i]
            d -= lki * yi
            Li[c[i]] = k
            Lx[c[i]] = lki
            c[i] += 1
        sg = sign[k]
        if sg != 0:
            if not np.isfinite(d) or d * sg < delta:
                d = sg * delta
                n_fix += 1
        elif not (abs(d) > 0.0) or not np.isfinite(d):
            return k, n_neg, n_fix
        D[k] = d
        if d < 0.0:
            n_neg += 1
    return -1, n_neg, n_fix


@jit_kernel()
def _ldl_solve(n, Lp, Li, Lx, D, x):
    """``x <- (L D Lᵀ)⁻¹ x`` in the permuted space."""
    for j in range(n):                      # L y = x
        xj = x[j]
        if xj != 0.0:
            for p in range(Lp[j], Lp[j + 1]):
                x[Li[p]] -= Lx[p] * xj
    for j in range(n):                      # D z = y
        x[j] /= D[j]
    for j in range(n - 1, -1, -1):          # Lᵀ w = z
        acc = x[j]
        for p in range(Lp[j], Lp[j + 1]):
            acc -= Lx[p] * x[Li[p]]
        x[j] = acc


# --------------------------------------------------------------------------- #
# objects                                                                      #
# --------------------------------------------------------------------------- #


class LDLSymbolic:
    """The permuted upper pattern, its elimination tree and column counts.

    Built once per pattern; :meth:`factor` is then the numeric pass alone.
    ``pos[p]`` maps the caller's CSC entries to the permuted upper CSC, so a
    revalued matrix with the same pattern is scattered without rebuilding.
    """

    def __init__(self, cp, ci, n, perm):
        n = int(n)
        self.n = n
        self.perm = np.ascontiguousarray(perm, dtype=IDX)
        pinv = np.empty(n, dtype=np.int64)
        pinv[self.perm] = np.arange(n, dtype=np.int64)
        self.pinv = pinv
        cp = np.asarray(cp, dtype=np.int64)
        ci = np.asarray(ci, dtype=np.int64)
        cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(cp))
        r = pinv[ci]
        c = pinv[cols]
        upper = r <= c
        self._src = np.flatnonzero(upper)              # entries kept
        rr, cc = r[upper], c[upper]
        # CSC of the permuted upper triangle; remember where each source lands
        order = np.lexsort((rr, cc))
        Cp = np.zeros(n + 1, dtype=np.int64)
        np.add.at(Cp[1:], cc, 1)
        np.cumsum(Cp, out=Cp)
        self.Cp = Cp
        self.Ci = np.ascontiguousarray(rr[order], dtype=np.int64)
        self._dest = np.empty(self._src.size, dtype=np.int64)
        self._dest[order] = np.arange(self._src.size, dtype=np.int64)
        self.parent = np.empty(n, dtype=np.int64)
        anc = np.empty(n, dtype=np.int64)
        _etree(n, self.Cp, self.Ci, self.parent, anc)
        self._s = np.empty(n, dtype=np.int64)
        self._w = np.empty(n, dtype=np.int64)
        cnt = np.empty(n, dtype=np.int64)
        self.lnz = int(_counts(n, self.Cp, self.Ci, self.parent, self._s,
                               self._w, cnt))
        self.Lp = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(cnt, out=self.Lp[1:])

    @property
    def nnz(self) -> int:
        """Entries of ``L`` below the diagonal, plus ``D``."""
        return self.lnz + self.n

    def factor(self, cx, sign=None, delta: float = 1e-8) -> "LDLFactor":
        """Numeric factorisation of the matrix whose values are ``cx``.

        ``sign`` gives the sign each pivot must have, in the *caller's*
        numbering (``-1`` for the negative block, ``+1`` for the positive
        one); a pivot rounding has put on the wrong side, or within ``delta``
        of zero, is replaced by ``sign * delta``. That is dynamic
        regularisation (Altman & Gondzio): the factorisation is then of a
        matrix perturbed by at most ``delta`` on a few diagonals, which the
        interior point's refinement against the unperturbed matrix removes.
        Without a ``sign`` only an exactly zero pivot is refused.
        """
        n = self.n
        Cx = np.zeros(self.Ci.size, dtype=VAL)
        Cx[self._dest] = np.asarray(cx, dtype=VAL)[self._src]
        Li = np.empty(max(self.lnz, 1), dtype=IDX)
        Lx = np.empty(max(self.lnz, 1), dtype=VAL)
        D = np.empty(n, dtype=VAL)
        x = np.empty(n, dtype=VAL)
        c = np.empty(n, dtype=np.int64)
        if sign is None:
            sg = np.zeros(n, dtype=np.int64)
        else:
            sg = np.asarray(sign, dtype=np.int64)[self.perm]
        k, n_neg, n_fix = _ldl_numeric(n, self.Cp, self.Ci, Cx, self.parent,
                                       self.Lp, Li, Lx, D, self._s, self._w,
                                       x, c, sg, float(delta))
        if k >= 0:
            raise LDLSingular(int(k))
        return LDLFactor(self, Li, Lx, D, int(n_neg), int(n_fix))


class LDLFactor:
    """``P A Pᵀ = L D Lᵀ``; solves with the same interface as an LU factor."""

    def __init__(self, sym: LDLSymbolic, Li, Lx, D, n_neg: int, n_fix: int = 0):
        self.sym = sym
        self.n = sym.n
        self.Li, self.Lx, self.D = Li, Lx, D
        self.n_neg = n_neg
        self.n_corrected = n_fix
        self._w = np.empty(self.n, dtype=VAL)

    @property
    def nnz(self) -> int:
        return self.sym.nnz

    def ftran(self, b, out=None):
        """Solve ``A x = b``."""
        sym = self.sym
        w = self._w
        np.take(b, sym.perm, out=w)                    # w = Pᵀ b
        _ldl_solve(self.n, sym.Lp, self.Li, self.Lx, self.D, w)
        if out is None:
            out = np.empty(self.n, dtype=VAL)
        out[sym.perm] = w                              # x = P w
        return out

    btran = ftran                                      # symmetric
