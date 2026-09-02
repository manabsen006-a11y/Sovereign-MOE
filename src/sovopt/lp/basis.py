"""Basis representation for the revised simplex.

Computational form
------------------
The solver works over the augmented system

    M z = 0,    M = [ A | -I ],    z = [x ; s]

where ``s`` are the *logical* (slack) variables, one per row, and the row bounds
become bounds on the logicals:

    l <= x <= u        (structural, from the column bounds)
    rl <= s <= ru      (logical, from the row bounds)

This is the standard device that lets ``<=``, ``>=``, ``=`` and ranged rows all
be handled by one code path: a row's type is entirely encoded in its logical's
bounds, and an equality row is just a logical fixed at a point. Nothing in the
simplex ever branches on row type.

Variable ``j`` is structural when ``j < n`` and the logical of row ``j - n``
otherwise. The logical's column is ``-e_i``, so no storage is needed for it.

Basis inverse
-------------
``B`` is the ``m x m`` submatrix of ``M`` on the basic columns. Rather than
refactorising after every pivot, the inverse is maintained in **product form**:
after a pivot the new basis differs from the old in one column, so

    B_new = B_old E,     E = I except column r, which holds alpha = B_old^-1 M[:,q]

and therefore ``B_new^-1 = E^-1 B_old^-1``. Each pivot appends one *eta vector*;
FTRAN applies them in order after the LU solve, BTRAN applies them in reverse
before it.

Product form is used here rather than Forrest-Tomlin because it is far simpler
to get right and its weakness -- fill grows linearly in the number of etas -- is
bounded by refactorising every few dozen pivots. Forrest-Tomlin would update the
LU factors themselves and keep them sparser for longer; it is the natural next
step and is deliberately not attempted yet.

Numerical safety
----------------
A basis can be singular, and a warm-started one frequently is. When the
factorisation fails, the offending column is replaced by a logical for a row not
already represented and the factorisation retried. That repair always
terminates, because the all-logical basis is ``-I``.

References
----------
Dantzig & Orchard-Hays, "The product form for the inverse in the simplex
  method", Math. Tables and Other Aids to Computation 8 (1954) 64-67.
Chvátal, *Linear Programming*, Freeman 1983, ch. 7 and 24 -- bounded variables
  and the revised method.
Maros, *Computational Techniques of the Simplex Method*, Kluwer 2003, ch. 5, 9 --
  the computational form with logical variables, basis maintenance.
Forrest & Tomlin, "Updated triangular factors of the basis to maintain sparsity
  in the product form simplex method", Math. Prog. 2 (1972) 263-278 -- the
  sparser update this module does not yet implement.
"""

from __future__ import annotations

import numpy as np

from ..core._jit import jit_kernel
from ..core.sparse import IDX, VAL
from ..core.tolerances import INF
from ..numerics.lu import LUSingular, lu_factor

__all__ = ["Basis", "BASIC", "AT_LOWER", "AT_UPPER", "FREE", "FIXED"]

BASIC = np.uint8(0)
AT_LOWER = np.uint8(1)
AT_UPPER = np.uint8(2)
FREE = np.uint8(3)
FIXED = np.uint8(4)


@jit_kernel()
def _build_basis_csc(Acp, Aci, Acx, basic, n, m, cp, ci, cx):
    """Assemble the CSC of ``B`` from the basic column list."""
    p = 0
    for k in range(m):
        j = basic[k]
        cp[k] = p
        if j < n:
            for q in range(Acp[j], Acp[j + 1]):
                ci[p] = Aci[q]
                cx[p] = Acx[q]
                p += 1
        else:
            ci[p] = j - n
            cx[p] = -1.0
            p += 1
    cp[m] = p
    return p


@jit_kernel()
def _basis_nnz(Acp, basic, n, m):
    t = 0
    for k in range(m):
        j = basic[k]
        t += (Acp[j + 1] - Acp[j]) if j < n else 1
    return t


@jit_kernel()
def _eta_ftran(start, idx, val, piv, rows, n_eta, v):
    """Apply ``E_1^-1 ... E_k^-1`` to ``v`` in place, forward order."""
    for e in range(n_eta):
        r = rows[e]
        t = v[r] / piv[e]
        if t != 0.0:
            for p in range(start[e], start[e + 1]):
                v[idx[p]] -= val[p] * t
        v[r] = t


@jit_kernel()
def _eta_btran(start, idx, val, piv, rows, n_eta, v):
    """Apply ``E_k^-T ... E_1^-T`` to ``v`` in place, reverse order."""
    for e in range(n_eta - 1, -1, -1):
        r = rows[e]
        acc = v[r]
        for p in range(start[e], start[e + 1]):
            acc -= val[p] * v[idx[p]]
        v[r] = acc / piv[e]


class Basis:
    """A basis over ``[A | -I]`` with FTRAN, BTRAN and a product-form update."""

    def __init__(self, prob, refactor_freq: int = 60, lu_tol: float = 0.01):
        self.prob = prob
        self.m = prob.m
        self.n = prob.n
        self.N = prob.n + prob.m
        self.refactor_freq = refactor_freq
        self.lu_tol = lu_tol

        m, n, N = self.m, self.n, self.N

        # combined bounds and costs over [x ; s]
        self.lower = np.empty(N, dtype=VAL)
        self.upper = np.empty(N, dtype=VAL)
        self.lower[:n] = prob.col_lb
        self.upper[:n] = prob.col_ub
        self.lower[n:] = prob.row_lb
        self.upper[n:] = prob.row_ub

        self.cost = np.zeros(N, dtype=VAL)
        self.cost[:n] = prob.c

        self.status = np.empty(N, dtype=np.uint8)
        self.basic = np.empty(m, dtype=IDX)
        self.pos_in_basis = np.full(N, -1, dtype=np.int32)

        self.lu = None
        self.n_factorizations = 0
        self.n_updates = 0
        self.n_repairs = 0

        # product-form etas
        cap = max(16 * m, 1024)
        self._eta_start = np.zeros(refactor_freq + 2, dtype=np.int64)
        self._eta_idx = np.empty(cap, dtype=IDX)
        self._eta_val = np.empty(cap, dtype=VAL)
        self._eta_piv = np.empty(refactor_freq + 2, dtype=VAL)
        self._eta_row = np.empty(refactor_freq + 2, dtype=IDX)
        self._n_eta = 0

        self._work = np.zeros(m, dtype=VAL)

    # -- helpers ------------------------------------------------------------ #

    def is_logical(self, j) -> bool:
        return j >= self.n

    def column_into(self, j, out):
        """Scatter ``M[:, j]`` into a zeroed dense vector."""
        if j < self.n:
            idx, val = self.prob.A.col(j)
            out[idx] = val
        else:
            out[j - self.n] = -1.0
        return out

    def bound_range(self, j) -> float:
        return self.upper[j] - self.lower[j]

    # -- start bases -------------------------------------------------------- #

    def set_logical_basis(self):
        """The all-logical (slack) basis. ``B = -I``, always nonsingular."""
        n, m = self.n, self.m
        self.basic[:] = np.arange(n, n + m, dtype=IDX)
        self.pos_in_basis[:] = -1
        self.pos_in_basis[n:] = np.arange(m, dtype=np.int32)
        self.status[:] = AT_LOWER
        self.status[n:] = BASIC

        for j in range(n):
            self.status[j] = self._nonbasic_status_for(j)
        self.factorize()

    def _nonbasic_status_for(self, j) -> np.uint8:
        lo, up = self.lower[j], self.upper[j]
        if lo <= -INF and up >= INF:
            return FREE
        if up - lo <= 0.0:
            return FIXED
        if lo > -INF:
            return AT_LOWER
        return AT_UPPER

    def set_status_from_costs(self):
        """Place each nonbasic variable at the bound that makes it dual feasible.

        With the logical basis the duals are zero, so ``d = c``: a variable with
        positive cost belongs at its lower bound and one with negative cost at
        its upper bound. Where the required bound is infinite the variable is
        left dual infeasible, which the caller detects and handles by running
        the primal algorithm instead.
        """
        for j in range(self.n):
            if self.status[j] == BASIC:
                continue
            lo, up = self.lower[j], self.upper[j]
            if lo <= -INF and up >= INF:
                self.status[j] = FREE
            elif up - lo <= 0.0:
                self.status[j] = FIXED
            elif self.cost[j] >= 0.0:
                self.status[j] = AT_LOWER if lo > -INF else AT_UPPER
            else:
                self.status[j] = AT_UPPER if up < INF else AT_LOWER

    def nonbasic_value(self, j) -> float:
        st = self.status[j]
        if st == AT_LOWER or st == FIXED:
            return self.lower[j]
        if st == AT_UPPER:
            return self.upper[j]
        return 0.0                       # FREE nonbasic sits at zero

    # -- factorisation ------------------------------------------------------ #

    def factorize(self):
        """Factorise ``B``, repairing singularity by swapping in logicals."""
        m, n = self.m, self.n
        for _attempt in range(m + 2):
            nnz = int(_basis_nnz(self.prob.A.cp, self.basic, n, m))
            cp = np.zeros(m + 1, dtype=np.int64)
            ci = np.empty(max(nnz, 1), dtype=IDX)
            cx = np.empty(max(nnz, 1), dtype=VAL)
            _build_basis_csc(self.prob.A.cp, self.prob.A.ci, self.prob.A.cx,
                             self.basic, n, m, cp, ci, cx)
            try:
                self.lu = lu_factor(cp, ci, cx, m, tol=self.lu_tol)
                self._n_eta = 0
                self.n_factorizations += 1
                self._sync_positions()
                return
            except LUSingular as e:
                k = e.column
                if k < 0 or k >= m:
                    self.set_logical_basis()
                    return
                self._repair(k)
        self.set_logical_basis()

    def _repair(self, k: int):
        """Replace basis position ``k`` with a logical for an unrepresented row."""
        n, m = self.n, self.m
        used = np.zeros(m, dtype=bool)
        for t in range(m):
            j = self.basic[t]
            if j >= n:
                used[j - n] = True
        free_rows = np.flatnonzero(~used)
        row = int(free_rows[0]) if free_rows.size else int(k)

        leaving = int(self.basic[k])
        self.basic[k] = n + row
        if leaving < self.N:
            self.status[leaving] = self._nonbasic_status_for(leaving)
        self.status[n + row] = BASIC
        self.n_repairs += 1

    def _sync_positions(self):
        self.pos_in_basis[:] = -1
        self.pos_in_basis[self.basic] = np.arange(self.m, dtype=np.int32)
        self.status[self.basic] = BASIC

    @property
    def needs_refactorization(self) -> bool:
        return self._n_eta >= self.refactor_freq

    # -- solves ------------------------------------------------------------- #

    def ftran(self, v):
        """``v <- B^-1 v``, in place."""
        self.lu.ftran(v, out=v)
        if self._n_eta:
            _eta_ftran(self._eta_start, self._eta_idx, self._eta_val,
                       self._eta_piv, self._eta_row, self._n_eta, v)
        return v

    def btran(self, v):
        """``v <- B^-T v``, in place."""
        if self._n_eta:
            _eta_btran(self._eta_start, self._eta_idx, self._eta_val,
                       self._eta_piv, self._eta_row, self._n_eta, v)
        self.lu.btran(v, out=v)
        return v

    def ftran_column(self, j, out=None):
        """``B^-1 M[:, j]`` for a single column of the augmented matrix."""
        if out is None:
            out = np.zeros(self.m, dtype=VAL)
        else:
            out.fill(0.0)
        self.column_into(j, out)
        return self.ftran(out)

    # -- update ------------------------------------------------------------- #

    def update(self, r: int, entering: int, alpha):
        """Pivot: ``entering`` replaces ``basic[r]``; ``alpha = B^-1 M[:,q]``."""
        piv = alpha[r]
        if piv == 0.0 or not np.isfinite(piv):
            raise LUSingular(r, "zero pivot in basis update")

        nz = np.flatnonzero(alpha)
        nz = nz[nz != r]
        need = self._eta_start[self._n_eta] + nz.size
        if need > self._eta_idx.shape[0]:
            newcap = max(int(need * 2), self._eta_idx.shape[0] * 2)
            self._eta_idx = np.resize(self._eta_idx, newcap)
            self._eta_val = np.resize(self._eta_val, newcap)

        s = self._eta_start[self._n_eta]
        self._eta_idx[s:s + nz.size] = nz.astype(IDX)
        self._eta_val[s:s + nz.size] = alpha[nz]
        self._eta_piv[self._n_eta] = piv
        self._eta_row[self._n_eta] = r
        self._n_eta += 1
        self._eta_start[self._n_eta] = s + nz.size

        leaving = int(self.basic[r])
        self.basic[r] = entering
        self.pos_in_basis[leaving] = -1
        self.pos_in_basis[entering] = r
        self.status[entering] = BASIC
        self.n_updates += 1
        return leaving

    # -- derived quantities ------------------------------------------------- #

    def nonbasic_values(self):
        """Dense vector of every variable's value, basics left at zero."""
        z = np.zeros(self.N, dtype=VAL)
        st = self.status
        at_lo = (st == AT_LOWER) | (st == FIXED)
        at_up = st == AT_UPPER
        z[at_lo] = self.lower[at_lo]
        z[at_up] = self.upper[at_up]
        return z

    def compute_basic_values(self):
        """``z_B = B^-1 (-M_N z_N)``."""
        z = self.nonbasic_values()
        z[self.basic] = 0.0
        rhs = -self.prob.A.matvec(z[:self.n])
        rhs += z[self.n:]                     # -( -I ) z_s  =  + z_s
        return self.ftran(rhs)

    def compute_duals(self):
        """``y = B^-T c_B``."""
        cb = self.cost[self.basic].copy()
        return self.btran(cb)

    def reduced_costs(self, y):
        """``d_j = c_j - M[:,j]^T y`` for every variable.

        For a logical, ``M[:, n+i] = -e_i``, so ``d_{n+i} = y_i``.
        """
        d = np.empty(self.N, dtype=VAL)
        d[:self.n] = self.cost[:self.n] - self.prob.A.rmatvec(y)
        d[self.n:] = y
        d[self.basic] = 0.0
        return d

    def pivot_row(self, rho):
        """``alpha_r[j] = rho^T M[:,j]`` for every ``j``, given ``rho = B^-T e_r``."""
        a = np.empty(self.N, dtype=VAL)
        a[:self.n] = self.prob.A.rmatvec(rho)
        a[self.n:] = -rho
        return a

    def stats(self) -> dict:
        return {
            "factorizations": self.n_factorizations,
            "updates": self.n_updates,
            "repairs": self.n_repairs,
            "etas": self._n_eta,
            "lu_nnz": self.lu.nnz if self.lu is not None else 0,
        }
