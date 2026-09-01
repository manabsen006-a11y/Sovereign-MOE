"""Sparse LU factorisation and hypersparse triangular solves.

This is the innermost loop of the whole solver. Every simplex iteration does one
FTRAN and one BTRAN through these factors, and a branch-and-cut run does
millions of them. Everything above this file inherits its accuracy and its speed.

Method
------
Left-looking Gilbert-Peierls LU with **threshold partial pivoting**, preceded by
a Suhl-Suhl style singleton-peeling pass that chooses the pivot column order.

Why this combination:

* A simplex basis is mostly triangular -- typically 80-95% of its columns are
  singletons, or become singletons once earlier ones are removed. Peeling those
  first removes almost all of the work before any arithmetic happens.
* On the remaining *nucleus*, columns are ordered by ascending live count, which
  approximates the Markowitz criterion at a fraction of the bookkeeping cost.
* Threshold pivoting -- accept a pivot at ``|a| >= tau * max|column|`` rather
  than insisting on the maximum -- buys much sparser factors for a bounded loss
  of stability. ``tau = 0.01`` is the usual compromise. We additionally *prefer
  the diagonal* when it passes the threshold, which keeps the factors close to
  the natural ordering and helps the Forrest-Tomlin update stay sparse.

**Hypersparsity** is why this file is shaped the way it is. Solving ``B x = b``
with a sparse right-hand side -- which is what the simplex always does -- has a
solution whose non-zero pattern is exactly the set of nodes reachable from
``nnz(b)`` in the graph of ``L``. Finding that set by depth-first search and
touching only those columns turns an ``O(m)`` sweep into an ``O(|reach|)`` one.
On network-structured refinery models ``|reach|`` is routinely under 1% of ``m``.
A solver without this is not slower by a constant; it is slower by orders of
magnitude on exactly the models this project targets.

Factorisation identity
----------------------
``L U = P B Q``, with ``P`` the row permutation encoded by ``pinv``
(``pinv[i] = k`` means original row ``i`` was the pivot at step ``k``) and ``Q``
the column permutation ``q`` (``(B Q)[:, k] = B[:, q[k]]``). Hence
``B = Pt L U Qt`` and

    FTRAN  ``B x = b``  :  z = P b;  L z' = z;  U w = z';  x[q[k]] = w[k]
    BTRAN  ``Bt y = c`` :  v[k] = c[q[k]];  Ut u = v;  Lt w = u;  y[i] = w[pinv[i]]

``L`` stores an explicit unit diagonal as the **first** entry of each column and
``U`` stores its diagonal as the **last** entry of each column, so a triangular
solve reads the pivot without searching.

References
----------
Gilbert & Peierls, "Sparse partial pivoting in time proportional to arithmetic
  operations", SIAM J. Sci. Stat. Comput. 9 (1988) 862-874.
Markowitz, "The elimination form of the inverse and its application to linear
  programming", Management Science 3 (1957) 255-269.
Suhl & Suhl, "Computing sparse LU factorizations for large-scale linear
  programming bases", ORSA J. Computing 2 (1990) 325-335.
Duff, Erisman & Reid, *Direct Methods for Sparse Matrices*, 2nd ed., OUP 2017,
  Ch. 5, 7 -- threshold pivoting and stability.
Davis, *Direct Methods for Sparse Linear Systems*, SIAM 2006, Ch. 6 -- the
  depth-first reachability formulation.
"""

from __future__ import annotations

import numpy as np

from ..core._jit import jit_kernel
from ..core.sparse import IDX, VAL

__all__ = ["LUFactor", "lu_factor", "LUSingular"]


class LUSingular(RuntimeError):
    """No acceptable pivot existed at factorisation step ``column``."""

    def __init__(self, column: int, message: str = ""):
        self.column = column
        super().__init__(message or f"basis singular at pivot step {column}")


# --------------------------------------------------------------------------- #
# reachability                                                                 #
# --------------------------------------------------------------------------- #


@jit_kernel()
def _dfs_pinv(j0, Gp, Gi, pinv, top, xi, pstack, marker, n):
    """DFS from original-row node ``j0``, mapping node -> L column via ``pinv``.

    Used during factorisation, where ``L``'s row indices are still in original
    row numbering and only pivoted rows have a column in ``L``.
    """
    head = 0
    xi[0] = j0
    while head >= 0:
        j = xi[head]
        jnew = pinv[j]
        if not marker[j]:
            marker[j] = True
            pstack[head] = Gp[jnew] if jnew >= 0 else 0
        done = True
        pend = Gp[jnew + 1] if jnew >= 0 else 0
        for p in range(pstack[head], pend):
            i = Gi[p]
            if marker[i]:
                continue
            pstack[head] = p + 1
            head += 1
            xi[head] = i
            done = False
            break
        if done:
            head -= 1
            top -= 1
            xi[top] = j
    return top


@jit_kernel()
def _dfs(j0, Gp, Gi, top, xi, pstack, marker, n):
    """DFS on a factor whose indices are already in permuted space."""
    head = 0
    xi[0] = j0
    while head >= 0:
        j = xi[head]
        if not marker[j]:
            marker[j] = True
            pstack[head] = Gp[j]
        done = True
        pend = Gp[j + 1]
        for p in range(pstack[head], pend):
            i = Gi[p]
            if marker[i]:
                continue
            pstack[head] = p + 1
            head += 1
            xi[head] = i
            done = False
            break
        if done:
            head -= 1
            top -= 1
            xi[top] = j
    return top


@jit_kernel()
def _reach(Gp, Gi, b_idx, nb, xi, pstack, marker, n):
    """Nodes reachable from pattern ``b_idx[:nb]``; result is ``xi[top:n]``."""
    top = n
    for k in range(nb):
        j = b_idx[k]
        if not marker[j]:
            top = _dfs(j, Gp, Gi, top, xi, pstack, marker, n)
    for p in range(top, n):
        marker[xi[p]] = False
    return top


# --------------------------------------------------------------------------- #
# sparse triangular solves (hypersparse path)                                  #
# --------------------------------------------------------------------------- #


@jit_kernel()
def _sp_lsolve(Lp, Li, Lx, b_idx, b_val, nb, x, xi, pstack, marker, n):
    """``L z = b`` for sparse ``b``. Unit diagonal first. Pattern is xi[top:n]."""
    top = _reach(Lp, Li, b_idx, nb, xi, pstack, marker, n)
    for p in range(top, n):
        x[xi[p]] = 0.0
    for k in range(nb):
        x[b_idx[k]] = b_val[k]
    for px in range(top, n):
        j = xi[px]
        xj = x[j]
        if xj == 0.0:
            continue
        for p in range(Lp[j] + 1, Lp[j + 1]):
            x[Li[p]] -= Lx[p] * xj
    return top


@jit_kernel()
def _sp_usolve(Up, Ui, Ux, b_idx, b_val, nb, x, xi, pstack, marker, n):
    """``U w = b`` for sparse ``b``. Diagonal last.

    The DFS writes finished nodes from ``xi[n-1]`` downwards, so ``xi[top:n]``
    is reverse-postorder -- already a topological order of "``i`` depends on
    ``j``" for *either* triangle. Both this and :func:`_sp_lsolve` therefore
    sweep forwards; the only difference is where the diagonal sits in a column.
    """
    top = _reach(Up, Ui, b_idx, nb, xi, pstack, marker, n)
    for p in range(top, n):
        x[xi[p]] = 0.0
    for k in range(nb):
        x[b_idx[k]] = b_val[k]
    for px in range(top, n):
        j = xi[px]
        xj = x[j]
        if xj == 0.0:
            continue
        pend = Up[j + 1] - 1
        xj = xj / Ux[pend]
        x[j] = xj
        for p in range(Up[j], pend):
            x[Ui[p]] -= Ux[p] * xj
    return top


# --------------------------------------------------------------------------- #
# dense triangular solves                                                      #
# --------------------------------------------------------------------------- #


@jit_kernel()
def _dense_lsolve(Lp, Li, Lx, x, n):
    for j in range(n):
        xj = x[j]
        if xj == 0.0:
            continue
        for p in range(Lp[j] + 1, Lp[j + 1]):
            x[Li[p]] -= Lx[p] * xj


@jit_kernel()
def _dense_usolve(Up, Ui, Ux, x, n):
    for j in range(n - 1, -1, -1):
        pend = Up[j + 1] - 1
        xj = x[j] / Ux[pend]
        x[j] = xj
        if xj == 0.0:
            continue
        for p in range(Up[j], pend):
            x[Ui[p]] -= Ux[p] * xj


@jit_kernel()
def _dense_ltsolve(Lp, Li, Lx, x, n):
    for j in range(n - 1, -1, -1):
        acc = x[j]
        for p in range(Lp[j] + 1, Lp[j + 1]):
            acc -= Lx[p] * x[Li[p]]
        x[j] = acc


@jit_kernel()
def _dense_utsolve(Up, Ui, Ux, x, n):
    for j in range(n):
        pend = Up[j + 1] - 1
        acc = x[j]
        for p in range(Up[j], pend):
            acc -= Ux[p] * x[Ui[p]]
        x[j] = acc / Ux[pend]


# --------------------------------------------------------------------------- #
# factorisation                                                                #
# --------------------------------------------------------------------------- #


@jit_kernel()
def _gp_lu(cp, ci, cx, n, q, tol, drop,
           Lp, Li, Lx, Up, Ui, Ux, pinv,
           x, xi, pstack, marker, cap):
    """Gilbert-Peierls LU. Returns ``(status, lnz, unz)``.

    status 0 = success, -1 = ran out of buffer, k+1 = singular at step k.
    """
    lnz = 0
    unz = 0
    for i in range(n):
        pinv[i] = -1
        marker[i] = False
        x[i] = 0.0

    for k in range(n):
        Lp[k] = lnz
        Up[k] = unz
        if lnz + n + 1 > cap or unz + n + 1 > cap:
            return -1, lnz, unz

        col = q[k]

        # ---- x = L \ B(:, col), sparse, over reachable nodes only ----------
        top = n
        for p in range(cp[col], cp[col + 1]):
            j = ci[p]
            if not marker[j]:
                top = _dfs_pinv(j, Lp, Li, pinv, top, xi, pstack, marker, n)
        for p in range(top, n):
            marker[xi[p]] = False

        for p in range(top, n):
            x[xi[p]] = 0.0
        for p in range(cp[col], cp[col + 1]):
            x[ci[p]] = cx[p]

        for px in range(top, n):
            j = xi[px]
            jnew = pinv[j]
            if jnew < 0:
                continue                       # row not pivotal yet: stays in L
            xj = x[j]
            if xj == 0.0:
                continue
            for p in range(Lp[jnew] + 1, Lp[jnew + 1]):
                x[Li[p]] -= Lx[p] * xj

        # ---- pivot selection: threshold partial, diagonal preferred --------
        ipiv = -1
        big = -1.0
        for p in range(top, n):
            i = xi[p]
            if pinv[i] < 0:
                t = abs(x[i])
                if t > big:
                    big = t
                    ipiv = i
            else:
                v = x[i]
                if v != 0.0 and abs(v) > drop:
                    Ui[unz] = pinv[i]
                    Ux[unz] = v
                    unz += 1
        if ipiv < 0 or big <= 0.0:
            return k + 1, lnz, unz
        # prefer the natural diagonal when it is stable enough: keeps factors
        # close to the identity, which the Forrest-Tomlin update likes
        if col < n and pinv[col] < 0 and abs(x[col]) >= big * tol:
            ipiv = col

        pivot = x[ipiv]
        Ui[unz] = k
        Ux[unz] = pivot
        unz += 1
        pinv[ipiv] = k
        Li[lnz] = ipiv
        Lx[lnz] = 1.0
        lnz += 1

        for p in range(top, n):
            i = xi[p]
            if pinv[i] < 0:
                v = x[i] / pivot
                if v != 0.0 and abs(v) > drop:
                    Li[lnz] = i
                    Lx[lnz] = v
                    lnz += 1
            x[i] = 0.0

    Lp[n] = lnz
    Up[n] = unz
    # map L's row indices into permuted space so L is genuinely triangular
    for p in range(lnz):
        Li[p] = pinv[Li[p]]
    return 0, lnz, unz


def _column_order(cp, ci, m, n):
    """Pivot column order: peeled singletons first, then ascending live count.

    Runs once per factorisation on the pattern only; not on the hot path.
    """
    rowdead = np.zeros(m, dtype=bool)
    coldead = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=IDX)
    placed = 0

    live = np.diff(cp).astype(np.int64)
    changed = True
    while changed and placed < n:
        changed = False
        singles = np.flatnonzero((~coldead) & (live == 1))
        for j in singles:
            if coldead[j]:
                continue
            rows = ci[cp[j]:cp[j + 1]]
            alive = rows[~rowdead[rows]]
            if alive.size != 1:
                continue
            order[placed] = j
            placed += 1
            coldead[j] = True
            rowdead[alive[0]] = True
            changed = True
        if changed:
            rest = np.flatnonzero(~coldead)
            for j in rest:
                rows = ci[cp[j]:cp[j + 1]]
                live[j] = int((~rowdead[rows]).sum())

    rest = np.flatnonzero(~coldead)
    if rest.size:
        rest = rest[np.argsort(live[rest], kind="stable")]
        order[placed:placed + rest.size] = rest
        placed += rest.size
    if placed != n:
        raise LUSingular(placed, "column ordering did not cover every column")
    return order


class LUFactor:
    """An LU factorisation of a square basis matrix, with FTRAN and BTRAN.

    Buffers for the sparse solves are owned by the factor and reused, so a
    simplex iteration allocates nothing.
    """

    __slots__ = ("n", "Lp", "Li", "Lx", "Up", "Ui", "Ux", "pinv", "q", "pinv_q",
                 "_x", "_xi", "_pstack", "_marker", "_b", "lnz", "unz",
                 "n_updates", "growth")

    def __init__(self, n, Lp, Li, Lx, Up, Ui, Ux, pinv, q, growth=1.0):
        self.n = int(n)
        self.Lp, self.Li, self.Lx = Lp, Li, Lx
        self.Up, self.Ui, self.Ux = Up, Ui, Ux
        self.pinv = pinv
        self.q = q
        self.lnz = int(Lp[n])
        self.unz = int(Up[n])
        self.n_updates = 0
        self.growth = growth
        self._x = np.zeros(n, dtype=VAL)
        self._b = np.zeros(n, dtype=VAL)
        self._xi = np.zeros(n, dtype=IDX)
        self._pstack = np.zeros(n, dtype=np.int64)
        self._marker = np.zeros(n, dtype=np.bool_)

    @property
    def nnz(self) -> int:
        return self.lnz + self.unz

    def fill_ratio(self, base_nnz: int) -> float:
        return self.nnz / max(base_nnz, 1)

    def __repr__(self):
        return (f"LUFactor(n={self.n}, |L|={self.lnz}, |U|={self.unz}, "
                f"growth={self.growth:.2e})")

    # -- solves ------------------------------------------------------------- #

    def ftran(self, b, out=None):
        """Solve ``B x = b``. ``b`` is consumed as a dense vector."""
        n = self.n
        z = self._b
        # z = P b
        z[self.pinv] = b
        _dense_lsolve(self.Lp, self.Li, self.Lx, z, n)
        _dense_usolve(self.Up, self.Ui, self.Ux, z, n)
        if out is None:
            out = np.empty(n, dtype=VAL)
        out[self.q] = z          # x = Q w
        return out

    def btran(self, c, out=None):
        """Solve ``Bᵀ y = c``."""
        n = self.n
        v = self._b
        np.take(c, self.q, out=v)             # v = Qᵀ c
        _dense_utsolve(self.Up, self.Ui, self.Ux, v, n)
        _dense_ltsolve(self.Lp, self.Li, self.Lx, v, n)
        if out is None:
            out = np.empty(n, dtype=VAL)
        np.take(v, self.pinv, out=out)        # y = Pᵀ w
        return out

    def ftran_sparse(self, idx, val):
        """Solve ``B x = b`` for sparse ``b``, returning ``(indices, values)``.

        Touches only the reachable part of the factors. This is the hypersparse
        path the simplex uses for pricing and for the entering column.
        """
        n = self.n
        x = self._x
        # permute the sparse rhs into L's space
        pidx = np.ascontiguousarray(self.pinv[np.asarray(idx, dtype=IDX)], dtype=IDX)
        top = _sp_lsolve(self.Lp, self.Li, self.Lx,
                         pidx, np.ascontiguousarray(val, dtype=VAL),
                         pidx.shape[0], x, self._xi, self._pstack,
                         self._marker, n)
        # _xi is reused by the next solve, so the intermediate pattern and its
        # values must be copied out before calling it.
        pat = self._xi[top:n].copy()
        pval = x[pat].copy()
        top2 = _sp_usolve(self.Up, self.Ui, self.Ux,
                          pat, pval, pat.shape[0],
                          x, self._xi, self._pstack, self._marker, n)
        pat2 = self._xi[top2:n].copy()
        vals = x[pat2].copy()
        keep = vals != 0.0
        return self.q[pat2[keep]], vals[keep]

    def residual(self, cp, ci, cx, x, b):
        """``||B x - b||_inf`` -- the honest accuracy check after a solve."""
        n = self.n
        r = -np.asarray(b, dtype=VAL).copy()
        for j in range(n):
            xj = x[j]
            if xj != 0.0:
                for p in range(cp[j], cp[j + 1]):
                    r[ci[p]] += cx[p] * xj
        return float(np.abs(r).max(initial=0.0))


def lu_factor(cp, ci, cx, n, tol: float = 0.01, drop: float = 1e-14,
              order=None, cap_growth: float = 6.0) -> LUFactor:
    """Factorise the square sparse matrix given in CSC form.

    ``tol`` is the relative threshold-pivoting tolerance; ``drop`` discards
    factor entries smaller than this as noise. Buffer space starts at
    ``cap_growth * nnz`` and doubles on overflow.
    """
    cp = np.ascontiguousarray(cp, dtype=np.int64)
    ci = np.ascontiguousarray(ci, dtype=IDX)
    cx = np.ascontiguousarray(cx, dtype=VAL)
    nnz = int(ci.shape[0])

    q = _column_order(cp, ci, n, n) if order is None else \
        np.ascontiguousarray(order, dtype=IDX)

    cap = max(int(cap_growth * nnz) + 4 * n + 16, 64)
    hard_cap = max(64, min(n * n + 4 * n, 1 << 31))

    x = np.zeros(n, dtype=VAL)
    xi = np.zeros(n, dtype=IDX)
    pstack = np.zeros(n, dtype=np.int64)
    marker = np.zeros(n, dtype=np.bool_)
    pinv = np.full(n, -1, dtype=IDX)

    while True:
        Lp = np.zeros(n + 1, dtype=np.int64)
        Up = np.zeros(n + 1, dtype=np.int64)
        Li = np.zeros(cap, dtype=IDX)
        Lx = np.zeros(cap, dtype=VAL)
        Ui = np.zeros(cap, dtype=IDX)
        Ux = np.zeros(cap, dtype=VAL)

        status, lnz, unz = _gp_lu(cp, ci, cx, n, q, tol, drop,
                                  Lp, Li, Lx, Up, Ui, Ux, pinv,
                                  x, xi, pstack, marker, cap)
        if status == 0:
            break
        if status == -1:
            if cap >= hard_cap:
                raise LUSingular(-1, "LU fill exceeded the hard buffer cap")
            cap = min(cap * 2, hard_cap)
            continue
        raise LUSingular(status - 1)

    Li = Li[:lnz].copy()
    Lx = Lx[:lnz].copy()
    Ui = Ui[:unz].copy()
    Ux = Ux[:unz].copy()

    # growth factor: max |U| / max |B|, the standard stability indicator
    amax = float(np.abs(cx).max(initial=1.0))
    umax = float(np.abs(Ux).max(initial=0.0))
    growth = umax / amax if amax > 0 else 1.0

    return LUFactor(n, Lp, Li, Lx, Up, Ui, Ux, pinv, q, growth=growth)
