"""Sparse matrix core.

Written from scratch. No ``scipy.sparse`` anywhere in the solve path -- see
``CLAUDE.md``.

Storage
-------
Every constraint matrix is held in **both** compressed-column (CSC) and
compressed-row (CSR) form. This doubles memory for the matrix but removes
atomics from every product we need:

    A @ x    walks CSR, parallel over rows    -- each output entry is one row dot
    Aᵀ @ y   walks CSC, parallel over columns -- each output entry is one col dot

The batched forms ``A @ X`` and ``Aᵀ @ Y`` for a block of ``k`` right-hand sides
turn the same sweep into a sparse-times-dense product (SpMM). That is the
primitive the batched node relaxation in ``sovopt.mip.bnr`` is built on: every
node of a branch-and-bound tree shares the same ``A`` and differs only in its
variable bounds, so an entire frontier of nodes can be advanced in one SpMM
instead of ``k`` separate SpMVs.

References
----------
Duff, Erisman & Reid, *Direct Methods for Sparse Matrices*, 2nd ed., OUP 2017 --
  storage schemes, Ch. 2.
Davis, *Direct Methods for Sparse Linear Systems*, SIAM 2006 -- transpose by
  counting sort, Ch. 2.
Merrill & Garland, "Merge-based parallel sparse matrix-vector multiplication",
  SC'16 -- row-split vs merge-based load balancing (we use row-split; merge-based
  is the GPU path in ``sovopt.lp.pdlp_gpu``).
"""

from __future__ import annotations

import numpy as np

from ._jit import HAVE_NUMBA, jit_kernel, prange

__all__ = ["SparseMatrix", "coo_to_csc", "csc_to_csr"]

IDX = np.int32
VAL = np.float64


# --------------------------------------------------------------------------- #
# kernels                                                                      #
# --------------------------------------------------------------------------- #


@jit_kernel(parallel=True)
def _csr_spmv(indptr, indices, values, x, out):
    """out[i] = sum_j A[i, j] * x[j], A in CSR."""
    m = indptr.shape[0] - 1
    for i in prange(m):
        acc = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            acc += values[p] * x[indices[p]]
        out[i] = acc


@jit_kernel(parallel=True)
def _csc_spmv_t(indptr, indices, values, y, out):
    """out[j] = sum_i A[i, j] * y[i], A in CSC (i.e. Aᵀ @ y)."""
    n = indptr.shape[0] - 1
    for j in prange(n):
        acc = 0.0
        for p in range(indptr[j], indptr[j + 1]):
            acc += values[p] * y[indices[p]]
        out[j] = acc


@jit_kernel(parallel=True)
def _csr_spmm(indptr, indices, values, X, out):
    """out[i, :] = sum_j A[i, j] * X[j, :], A in CSR, X dense (n, k)."""
    m = indptr.shape[0] - 1
    k = X.shape[1]
    for i in prange(m):
        for t in range(k):
            out[i, t] = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            a = values[p]
            j = indices[p]
            for t in range(k):
                out[i, t] += a * X[j, t]


@jit_kernel(parallel=True)
def _csc_spmm_t(indptr, indices, values, Y, out):
    """out[j, :] = sum_i A[i, j] * Y[i, :], A in CSC, Y dense (m, k)."""
    n = indptr.shape[0] - 1
    k = Y.shape[1]
    for j in prange(n):
        for t in range(k):
            out[j, t] = 0.0
        for p in range(indptr[j], indptr[j + 1]):
            a = values[p]
            i = indices[p]
            for t in range(k):
                out[j, t] += a * Y[i, t]


@jit_kernel(parallel=True)
def _abs_max_by_slice(indptr, values, out):
    """out[s] = max |values[p]| over slice s; 0.0 for an empty slice."""
    n = indptr.shape[0] - 1
    for s in prange(n):
        hi = 0.0
        for p in range(indptr[s], indptr[s + 1]):
            v = abs(values[p])
            if v > hi:
                hi = v
        out[s] = hi


@jit_kernel(parallel=True)
def _abs_min_by_slice(indptr, values, out):
    """out[s] = min non-zero |values[p]| over slice s; 0.0 for an empty slice."""
    n = indptr.shape[0] - 1
    for s in prange(n):
        lo = np.inf
        for p in range(indptr[s], indptr[s + 1]):
            v = abs(values[p])
            if v > 0.0 and v < lo:
                lo = v
        out[s] = 0.0 if lo == np.inf else lo


@jit_kernel(parallel=True)
def _scale_csc(indptr, indices, values, rowscale, colscale):
    """values[p] *= rowscale[i] * colscale[j] in place, A in CSC."""
    n = indptr.shape[0] - 1
    for j in prange(n):
        cs = colscale[j]
        for p in range(indptr[j], indptr[j + 1]):
            values[p] *= rowscale[indices[p]] * cs


@jit_kernel(parallel=True)
def _scale_csr(indptr, indices, values, rowscale, colscale):
    """values[p] *= rowscale[i] * colscale[j] in place, A in CSR."""
    m = indptr.shape[0] - 1
    for i in prange(m):
        rs = rowscale[i]
        for p in range(indptr[i], indptr[i + 1]):
            values[p] *= rs * colscale[indices[p]]


@jit_kernel()
def _transpose(indptr, indices, values, n_other, t_indptr, t_indices, t_values):
    """CSC -> CSR (or CSR -> CSC) by counting sort. ``n_other`` = rows of A."""
    nnz = indices.shape[0]
    n = indptr.shape[0] - 1

    for i in range(n_other + 1):
        t_indptr[i] = 0
    for p in range(nnz):
        t_indptr[indices[p] + 1] += 1
    for i in range(n_other):
        t_indptr[i + 1] += t_indptr[i]

    # cursor per target slice, walked forward as entries are placed
    cursor = np.empty(n_other, dtype=np.int64)
    for i in range(n_other):
        cursor[i] = t_indptr[i]

    for j in range(n):
        for p in range(indptr[j], indptr[j + 1]):
            i = indices[p]
            q = cursor[i]
            t_indices[q] = j
            t_values[q] = values[p]
            cursor[i] = q + 1


# --------------------------------------------------------------------------- #
# NumPy fallbacks (used only when Numba is absent)                             #
# --------------------------------------------------------------------------- #


def _np_spmv(indptr, indices, values, x, out, expanded):
    np.add.reduceat(values * x[indices], indptr[:-1], out=out) if False else None
    np.copyto(out, np.bincount(expanded, weights=values * x[indices],
                               minlength=out.shape[0]))


def _expand_slices(indptr, nnz):
    """Row (or column) index for each stored entry."""
    out = np.zeros(nnz, dtype=IDX)
    counts = np.diff(indptr)
    return np.repeat(np.arange(indptr.shape[0] - 1, dtype=IDX), counts) if nnz else out


# --------------------------------------------------------------------------- #
# conversion helpers                                                           #
# --------------------------------------------------------------------------- #


def coo_to_csc(rows, cols, vals, m: int, n: int, sum_duplicates: bool = True):
    """Build CSC arrays from triplets.

    Entries are sorted by (column, row). Exact duplicates are summed, which is
    what MPS files with a repeated COLUMNS entry mean.
    """
    rows = np.asarray(rows, dtype=IDX)
    cols = np.asarray(cols, dtype=IDX)
    vals = np.asarray(vals, dtype=VAL)
    if rows.shape != cols.shape or rows.shape != vals.shape:
        raise ValueError("triplet arrays must have equal length")

    order = np.lexsort((rows, cols))
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]

    if sum_duplicates and rows.size:
        keyed = cols.astype(np.int64) * np.int64(m) + rows.astype(np.int64)
        first = np.empty(keyed.shape[0], dtype=bool)
        first[0] = True
        np.not_equal(keyed[1:], keyed[:-1], out=first[1:])
        if not first.all():
            group = np.cumsum(first) - 1
            summed = np.bincount(group, weights=vals)
            rows = rows[first]
            cols = cols[first]
            vals = summed

    # drop structural zeros: they cost work in every sweep and mean nothing
    keep = vals != 0.0
    if not keep.all():
        rows, cols, vals = rows[keep], cols[keep], vals[keep]

    indptr = np.zeros(n + 1, dtype=np.int64)
    if cols.size:
        np.add.at(indptr, cols.astype(np.int64) + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, np.ascontiguousarray(rows), np.ascontiguousarray(vals)


def csc_to_csr(indptr, indices, values, m: int):
    """Transpose the storage order: CSC arrays -> CSR arrays."""
    nnz = indices.shape[0]
    t_indptr = np.zeros(m + 1, dtype=np.int64)
    t_indices = np.empty(nnz, dtype=IDX)
    t_values = np.empty(nnz, dtype=VAL)
    if HAVE_NUMBA:
        _transpose(indptr, indices, values, m, t_indptr, t_indices, t_values)
    else:
        order = np.lexsort((indices,)) if nnz else np.empty(0, dtype=np.int64)
        cols = _expand_slices(indptr, nnz)
        order = np.lexsort((cols, indices))
        t_indices[:] = cols[order]
        t_values[:] = values[order]
        counts = np.bincount(indices, minlength=m)
        t_indptr[1:] = np.cumsum(counts)
    return t_indptr, t_indices, t_values


# --------------------------------------------------------------------------- #
# matrix                                                                       #
# --------------------------------------------------------------------------- #


class SparseMatrix:
    """An ``m x n`` sparse matrix held simultaneously in CSC and CSR.

    Immutable in structure. Values may be rescaled in place via :meth:`scale`,
    which keeps both representations consistent.
    """

    __slots__ = (
        "m", "n",
        "cp", "ci", "cx",     # CSC: col pointers, row indices, values
        "rp", "ri", "rx",     # CSR: row pointers, col indices, values
        "_row_of", "_col_of",
    )

    def __init__(self, m: int, n: int, cp, ci, cx, rp=None, ri=None, rx=None):
        self.m = int(m)
        self.n = int(n)
        self.cp = np.ascontiguousarray(cp, dtype=np.int64)
        self.ci = np.ascontiguousarray(ci, dtype=IDX)
        self.cx = np.ascontiguousarray(cx, dtype=VAL)
        if rp is None:
            rp, ri, rx = csc_to_csr(self.cp, self.ci, self.cx, self.m)
        self.rp = np.ascontiguousarray(rp, dtype=np.int64)
        self.ri = np.ascontiguousarray(ri, dtype=IDX)
        self.rx = np.ascontiguousarray(rx, dtype=VAL)
        self._row_of = None
        self._col_of = None

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_triplets(cls, rows, cols, vals, m: int, n: int) -> "SparseMatrix":
        cp, ci, cx = coo_to_csc(rows, cols, vals, m, n)
        return cls(m, n, cp, ci, cx)

    @classmethod
    def from_dense(cls, A) -> "SparseMatrix":
        A = np.asarray(A, dtype=VAL)
        rows, cols = np.nonzero(A)
        return cls.from_triplets(rows, cols, A[rows, cols], A.shape[0], A.shape[1])

    # -- shape -------------------------------------------------------------- #

    @property
    def shape(self):
        return (self.m, self.n)

    @property
    def nnz(self) -> int:
        return int(self.ci.shape[0])

    @property
    def density(self) -> float:
        d = self.m * self.n
        return self.nnz / d if d else 0.0

    def __repr__(self):
        return (f"SparseMatrix({self.m}x{self.n}, nnz={self.nnz}, "
                f"density={self.density:.2e})")

    # -- products ----------------------------------------------------------- #

    def matvec(self, x, out=None):
        """``A @ x``. Walks CSR, parallel over rows."""
        x = np.ascontiguousarray(x, dtype=VAL)
        if out is None:
            out = np.empty(self.m, dtype=VAL)
        if HAVE_NUMBA:
            _csr_spmv(self.rp, self.ri, self.rx, x, out)
        else:
            if self._row_of is None:
                self._row_of = _expand_slices(self.rp, self.nnz)
            np.copyto(out, np.bincount(self._row_of,
                                       weights=self.rx * x[self.ri],
                                       minlength=self.m))
        return out

    def rmatvec(self, y, out=None):
        """``Aᵀ @ y``. Walks CSC, parallel over columns."""
        y = np.ascontiguousarray(y, dtype=VAL)
        if out is None:
            out = np.empty(self.n, dtype=VAL)
        if HAVE_NUMBA:
            _csc_spmv_t(self.cp, self.ci, self.cx, y, out)
        else:
            if self._col_of is None:
                self._col_of = _expand_slices(self.cp, self.nnz)
            np.copyto(out, np.bincount(self._col_of,
                                       weights=self.cx * y[self.ci],
                                       minlength=self.n))
        return out

    def matmat(self, X, out=None):
        """``A @ X`` for dense ``X`` of shape ``(n, k)``. The batched primitive."""
        X = np.ascontiguousarray(X, dtype=VAL)
        if X.ndim != 2 or X.shape[0] != self.n:
            raise ValueError(f"X must be ({self.n}, k), got {X.shape}")
        if out is None:
            out = np.empty((self.m, X.shape[1]), dtype=VAL)
        if HAVE_NUMBA:
            _csr_spmm(self.rp, self.ri, self.rx, X, out)
        else:
            if self._row_of is None:
                self._row_of = _expand_slices(self.rp, self.nnz)
            contrib = self.rx[:, None] * X[self.ri]
            out.fill(0.0)
            np.add.at(out, self._row_of, contrib)
        return out

    def rmatmat(self, Y, out=None):
        """``Aᵀ @ Y`` for dense ``Y`` of shape ``(m, k)``."""
        Y = np.ascontiguousarray(Y, dtype=VAL)
        if Y.ndim != 2 or Y.shape[0] != self.m:
            raise ValueError(f"Y must be ({self.m}, k), got {Y.shape}")
        if out is None:
            out = np.empty((self.n, Y.shape[1]), dtype=VAL)
        if HAVE_NUMBA:
            _csc_spmm_t(self.cp, self.ci, self.cx, Y, out)
        else:
            if self._col_of is None:
                self._col_of = _expand_slices(self.cp, self.nnz)
            contrib = self.cx[:, None] * Y[self.ci]
            out.fill(0.0)
            np.add.at(out, self._col_of, contrib)
        return out

    # -- slices ------------------------------------------------------------- #

    def col(self, j: int):
        """(row indices, values) of column ``j``."""
        s, e = self.cp[j], self.cp[j + 1]
        return self.ci[s:e], self.cx[s:e]

    def row(self, i: int):
        """(column indices, values) of row ``i``."""
        s, e = self.rp[i], self.rp[i + 1]
        return self.ri[s:e], self.rx[s:e]

    def col_counts(self):
        return np.diff(self.cp).astype(IDX)

    def row_counts(self):
        return np.diff(self.rp).astype(IDX)

    # -- norms and scaling -------------------------------------------------- #

    def row_absmax(self, out=None):
        if out is None:
            out = np.empty(self.m, dtype=VAL)
        if HAVE_NUMBA:
            _abs_max_by_slice(self.rp, self.rx, out)
        else:
            if self._row_of is None:
                self._row_of = _expand_slices(self.rp, self.nnz)
            out.fill(0.0)
            np.maximum.at(out, self._row_of, np.abs(self.rx))
        return out

    def col_absmax(self, out=None):
        if out is None:
            out = np.empty(self.n, dtype=VAL)
        if HAVE_NUMBA:
            _abs_max_by_slice(self.cp, self.cx, out)
        else:
            if self._col_of is None:
                self._col_of = _expand_slices(self.cp, self.nnz)
            out.fill(0.0)
            np.maximum.at(out, self._col_of, np.abs(self.cx))
        return out

    def row_absmin(self, out=None):
        if out is None:
            out = np.empty(self.m, dtype=VAL)
        if HAVE_NUMBA:
            _abs_min_by_slice(self.rp, self.rx, out)
        else:
            out.fill(0.0)
            for i in range(self.m):
                s, e = self.rp[i], self.rp[i + 1]
                if e > s:
                    a = np.abs(self.rx[s:e])
                    a = a[a > 0.0]
                    out[i] = a.min() if a.size else 0.0
        return out

    def col_absmin(self, out=None):
        if out is None:
            out = np.empty(self.n, dtype=VAL)
        if HAVE_NUMBA:
            _abs_min_by_slice(self.cp, self.cx, out)
        else:
            out.fill(0.0)
            for j in range(self.n):
                s, e = self.cp[j], self.cp[j + 1]
                if e > s:
                    a = np.abs(self.cx[s:e])
                    a = a[a > 0.0]
                    out[j] = a.min() if a.size else 0.0
        return out

    def scale(self, rowscale, colscale) -> None:
        """In place ``A <- diag(r) A diag(c)``, keeping CSC and CSR consistent."""
        r = np.ascontiguousarray(rowscale, dtype=VAL)
        c = np.ascontiguousarray(colscale, dtype=VAL)
        if HAVE_NUMBA:
            _scale_csc(self.cp, self.ci, self.cx, r, c)
            _scale_csr(self.rp, self.ri, self.rx, r, c)
        else:
            if self._row_of is None:
                self._row_of = _expand_slices(self.rp, self.nnz)
            if self._col_of is None:
                self._col_of = _expand_slices(self.cp, self.nnz)
            self.cx *= r[self.ci] * c[self._col_of]
            self.rx *= r[self._row_of] * c[self.ri]

    def coeff_range(self):
        """(min |a_ij|, max |a_ij|) over stored non-zeros -- the number that
        tells you whether a model is going to give the factorisation trouble."""
        if self.nnz == 0:
            return (0.0, 0.0)
        a = np.abs(self.cx)
        return (float(a.min()), float(a.max()))

    def copy(self) -> "SparseMatrix":
        return SparseMatrix(self.m, self.n,
                            self.cp.copy(), self.ci.copy(), self.cx.copy(),
                            self.rp.copy(), self.ri.copy(), self.rx.copy())

    def to_dense(self):
        """Only for tests and tiny models."""
        A = np.zeros((self.m, self.n), dtype=VAL)
        for j in range(self.n):
            idx, val = self.col(j)
            A[idx, j] = val
        return A
