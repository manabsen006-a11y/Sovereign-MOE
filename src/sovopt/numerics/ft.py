"""Forrest-Tomlin update of the basis LU factors.

The product-form update (:mod:`sovopt.lp.basis`) never touches the LU: every
pivot appends an eta column, and every FTRAN and BTRAN pays for the whole eta
file. Forrest-Tomlin updates ``U`` itself, so the factors stay factors -- an
FTRAN after a hundred pivots costs about what it cost after none, plus a
hundred *row* etas that are as short as the rows they came from.

The update
----------
``B = Pᵀ L U Qᵀ`` after a factorisation. Replacing basis column ``r`` by ``a``
replaces the column of ``U`` that ``r`` maps to by the *spike*
``ũ = R L⁻¹ P a`` -- the entering column FTRAN'd through ``L`` and every
earlier row eta, but not through ``U``. That breaks triangularity in exactly
one row: the row ``ρ`` whose diagonal sat in the replaced column. Move that
row and that column to the end, and the trailing row is the only thing below
the diagonal. Eliminate it against the rows beneath it -- ``ρ ← ρ − Σ m_s·s``
-- and what remains is upper triangular again with the spike as its last
column. The multipliers ``m`` are one row eta ``R_k = I − e_ρ mᵀ``, and

    B_new = Pᵀ L R_1⁻¹ ⋯ R_k⁻¹ Ũ Q̃ᵀ

so FTRAN is ``L`` solve, the row etas in order, ``Ũ`` solve; BTRAN is the
reverse. The multipliers themselves are a *transpose* triangular solve
against the trailing part of ``Ũ``, which is why ``U`` is stored **row-wise**
here: the elimination reads rows, and both triangular solves work on rows
too -- back substitution as a gather, its transpose as a scatter.

Rows live in a *row file*: each row owns a contiguous range with spare
capacity, a row that outgrows its range is moved to the end of the file, and
the file is compacted by the next refactorisation. The first version of this
module kept rows as linked lists, and the solves ran three times slower than
the column-oriented LU's on the same ``U`` -- 88 µs against 31 µs on woodw's
1,098-row basis -- purely from chasing pointers. Contiguous rows are what make
the comparison with the product form a comparison of updates rather than of
memory layouts.

Permutations are logical. Each column of ``U`` is a *slot* with a position
in the triangular order; each row likewise. Moving ``(ρ, s_r)`` to the end
is a shift of the positions between, ``O(m)`` bookkeeping and no data moved.
A replaced slot is marked dead and its entries are skipped wherever a row is
walked.

Stability
---------
The new diagonal ``d = ũ_ρ − Σ m_s ũ_s`` is the pivot the update is betting
on, and the multipliers are divisions by whatever diagonals the trailing rows
happen to have. Two guards: a pivot negligible against the spike, or a
multiplier beyond ``max_mult``, refuses the update with :class:`LUSingular`
and the basis refactorises. Neither guard applies to a *fresh* factor's first
update, because the caller answers a refusal with a refactorisation and the
same pivot again, and a refusal that repeats is a livelock -- measured on
woodw before that rule: 27,167 refusals in 27,360 pivots.

References
----------
Forrest & Tomlin, "Updated triangular factors of the basis to maintain
  sparsity in the product form simplex method", Math. Prog. 2 (1972) 263-278.
Chvátal, *Linear Programming*, Freeman 1983, ch. 24 -- the update as row
  operations on the trailing submatrix.
Suhl & Suhl, "A fast LU update for linear programming", Annals of OR 43
  (1993) 33-47 -- row-wise ``U`` in a row file, and logical permutations.
Maros, *Computational Techniques of the Simplex Method*, Kluwer 2003, ch. 8.
"""

from __future__ import annotations

import numpy as np

from ..core._jit import jit_kernel
from ..core.sparse import IDX, VAL
from .lu import LUFactor, LUSingular, _dense_lsolve, _dense_ltsolve

__all__ = ["FTFactor"]


# --------------------------------------------------------------------------- #
# kernels                                                                      #
# --------------------------------------------------------------------------- #


@jit_kernel()
def _rows_from_columns(Up, Ui, Ux, m, rstart, rlen, rcap, rslot, rval, slack):
    """Lay the strictly-upper part of ``U`` out row by row, with spare room.

    Returns the file length used. The diagonals are the caller's.
    """
    for i in range(m):
        rlen[i] = 0
    for k in range(m):
        for p in range(Up[k], Up[k + 1] - 1):
            rlen[Ui[p]] += 1
    used = 0
    for i in range(m):
        rstart[i] = used
        rcap[i] = rlen[i] + slack + rlen[i] // 4
        used += rcap[i]
        rlen[i] = 0
    for k in range(m):
        for p in range(Up[k], Up[k + 1] - 1):
            i = Ui[p]
            q = rstart[i] + rlen[i]
            rslot[q] = k
            rval[q] = Ux[p]
            rlen[i] += 1
    return used


@jit_kernel()
def _apply_R(estart, erow, emult, erho, n_eta, z):
    """``z ← R_k ⋯ R_1 z``: each eta subtracts its row combination from ρ."""
    for e in range(n_eta):
        acc = z[erho[e]]
        for p in range(estart[e], estart[e + 1]):
            acc -= emult[p] * z[erow[p]]
        z[erho[e]] = acc


@jit_kernel()
def _apply_Rt(estart, erow, emult, erho, n_eta, u):
    """``u ← R_1ᵀ ⋯ R_kᵀ u``: the transposes, in reverse order."""
    for e in range(n_eta - 1, -1, -1):
        t = u[erho[e]]
        if t != 0.0:
            for p in range(estart[e], estart[e + 1]):
                u[erow[p]] -= emult[p] * t


@jit_kernel()
def _usolve_rows(m, row_at, col_at, udiag, rstart, rlen, rslot, rval, alive,
                 z, w):
    """``Ũ w = z`` by back substitution over positions; ``w`` is by slot."""
    for pos in range(m - 1, -1, -1):
        rid = row_at[pos]
        acc = z[rid]
        for p in range(rstart[rid], rstart[rid] + rlen[rid]):
            t = rslot[p]
            if alive[t]:
                acc -= rval[p] * w[t]
        w[col_at[pos]] = acc / udiag[col_at[pos]]


@jit_kernel()
def _utsolve_rows(m, row_at, col_at, udiag, rstart, rlen, rslot, rval, alive,
                  v, u):
    """``Ũᵀ u = v`` by forward substitution; ``v`` by slot, ``u`` by row."""
    for pos in range(m):
        rid = row_at[pos]
        t = v[col_at[pos]] / udiag[col_at[pos]]
        u[rid] = t
        if t != 0.0:
            for p in range(rstart[rid], rstart[rid] + rlen[rid]):
                q = rslot[p]
                if alive[q]:
                    v[q] -= rval[p] * t


@jit_kernel()
def _ft_update(m, p_r, s_r, s_new, spike, rho,
               row_at, col_at, rowpos, colpos, udiag, urow,
               rstart, rlen, rcap, rslot, rval, alive, used, file_cap,
               wv, erow, emult, estart_e, pivot_rel, max_mult, fresh):
    """Insert the spike, eliminate row ``ρ``, shift the positions.

    Returns ``(used, n_mult, d)``: ``d`` is the new diagonal, ``n_mult`` the
    number of multipliers written at ``erow[estart_e:]``. A negative
    ``n_mult`` reports a refusal -- ``-1`` a pivot too small against the
    spike, ``-2`` a multiplier beyond ``max_mult``, ``-3`` no room left in
    the row file (the caller grows it and retries; nothing was changed).
    """
    # -- room check first, so a refusal never leaves a half-moved row --------- #
    need = 0
    for rid in range(m):
        if spike[rid] != 0.0 and rid != rho and rlen[rid] >= rcap[rid]:
            need += 2 * rcap[rid] + 4
    if used + need > file_cap:
        return used, -3, 0.0

    # -- the spike joins every row but ρ ------------------------------------ #
    smax = 0.0
    for rid in range(m):
        v = spike[rid]
        if v == 0.0:
            continue
        av = abs(v)
        if av > smax:
            smax = av
        if rid == rho:
            continue
        if rlen[rid] >= rcap[rid]:
            # relocate the row to the end of the file with room to grow
            newcap = 2 * rcap[rid] + 4
            src = rstart[rid]
            for q in range(rlen[rid]):
                rslot[used + q] = rslot[src + q]
                rval[used + q] = rval[src + q]
            rstart[rid] = used
            rcap[rid] = newcap
            used += newcap
        q = rstart[rid] + rlen[rid]
        rslot[q] = s_new
        rval[q] = v
        rlen[rid] += 1

    # -- row ρ into the dense work vector over slots -------------------------- #
    alive[s_r] = False
    alive[s_new] = True          # the spike takes part in the elimination
    wv[s_new] = spike[rho]
    for p in range(rstart[rho], rstart[rho] + rlen[rho]):
        t = rslot[p]
        if alive[t]:
            wv[t] += rval[p]

    # -- eliminate: a transpose solve against the trailing submatrix ---------- #
    n_mult = 0
    too_big = False
    for pos in range(p_r + 1, m):
        s = col_at[pos]
        t = wv[s]
        if t == 0.0:
            continue
        wv[s] = 0.0
        mult = t / udiag[s]
        if abs(mult) > max_mult:
            too_big = True
        rid_s = row_at[pos]
        erow[estart_e + n_mult] = rid_s
        emult[estart_e + n_mult] = mult
        n_mult += 1
        for p in range(rstart[rid_s], rstart[rid_s] + rlen[rid_s]):
            q = rslot[p]
            if alive[q]:
                wv[q] -= mult * rval[p]
    d = wv[s_new]
    wv[s_new] = 0.0
    bad_pivot = (d == 0.0) or not np.isfinite(d) or \
        (not fresh and not (abs(d) > pivot_rel * smax))
    if bad_pivot or (too_big and not fresh):
        # undo the spike's entries: each was appended last to its row
        for rid in range(m):
            if spike[rid] != 0.0 and rid != rho:
                rlen[rid] -= 1
        alive[s_new] = False
        alive[s_r] = True
        return used, (-1 if bad_pivot else -2), d

    # -- commit: ρ keeps only its new diagonal, positions shift --------------- #
    rlen[rho] = 0
    udiag[s_new] = d
    urow[s_new] = rho
    for pos in range(p_r, m - 1):
        row_at[pos] = row_at[pos + 1]
        col_at[pos] = col_at[pos + 1]
        rowpos[row_at[pos]] = pos
        colpos[col_at[pos]] = pos
    row_at[m - 1] = rho
    col_at[m - 1] = s_new
    rowpos[rho] = m - 1
    colpos[s_new] = m - 1
    return used, n_mult, d


# --------------------------------------------------------------------------- #
# the factor                                                                   #
# --------------------------------------------------------------------------- #


class FTFactor:
    """An :class:`LUFactor` whose ``U`` can be updated column by column.

    ``bpos_of_slot[s]`` is the basis position a slot stands for, so a caller
    can address updates by basis position exactly as with the product form.
    """

    def __init__(self, lu: LUFactor, max_updates: int = 512,
                 pivot_rel: float = 1e-8, max_mult: float = 1e8):
        m = lu.n
        self.lu = lu
        self.m = m
        self.pivot_rel = pivot_rel
        self.max_mult = max_mult
        self.max_updates = max_updates
        self.n_refused = 0
        unz = int(lu.Up[m])

        # slots: the m factor columns, then one per update
        cap_slots = m + max_updates + 1
        self.udiag = np.zeros(cap_slots, dtype=VAL)
        self.urow = np.full(cap_slots, -1, dtype=IDX)
        self.alive = np.zeros(cap_slots, dtype=np.bool_)
        self.colpos = np.full(cap_slots, -1, dtype=IDX)
        self.bpos_of_slot = np.full(cap_slots, -1, dtype=IDX)
        self.udiag[:m] = lu.Ux[lu.Up[1:m + 1] - 1]
        self.urow[:m] = np.arange(m, dtype=IDX)
        self.alive[:m] = True
        self.colpos[:m] = np.arange(m, dtype=IDX)
        self.bpos_of_slot[:m] = lu.q
        self.slot_of_bpos = np.empty(m, dtype=IDX)
        self.slot_of_bpos[lu.q] = np.arange(m, dtype=IDX)
        self.n_slots = m

        self.row_at = np.arange(m, dtype=IDX)
        self.col_at = np.arange(m, dtype=IDX)
        self.rowpos = np.arange(m, dtype=IDX)

        # the row file
        self.rstart = np.zeros(m, dtype=np.int64)
        self.rlen = np.zeros(m, dtype=np.int64)
        self.rcap = np.zeros(m, dtype=np.int64)
        cap = max(3 * unz + 8 * m + 64, 256)
        self.rslot = np.empty(cap, dtype=IDX)
        self.rval = np.empty(cap, dtype=VAL)
        self.used = int(_rows_from_columns(lu.Up, lu.Ui, lu.Ux, m, self.rstart,
                                           self.rlen, self.rcap, self.rslot,
                                           self.rval, 4))

        # row etas
        self.estart = np.zeros(max_updates + 2, dtype=np.int64)
        self.erho = np.zeros(max_updates + 1, dtype=IDX)
        ecap = max(4 * m, 256)
        self.erow = np.empty(ecap, dtype=IDX)
        self.emult = np.empty(ecap, dtype=VAL)
        self.n_eta = 0

        self._z = np.empty(m, dtype=VAL)
        self._w = np.empty(cap_slots, dtype=VAL)
        self._wv = np.zeros(cap_slots, dtype=VAL)
        self.n_updates = 0

    @property
    def nnz(self) -> int:
        return int(self.lu.Lp[self.m]) + int(self.rlen.sum()) + self.n_slots \
            + int(self.estart[self.n_eta])

    # -- solves ------------------------------------------------------------- #

    def ftran(self, b, out=None):
        """``B x = b`` -- ``L``, the row etas, then ``Ũ``."""
        lu, m = self.lu, self.m
        z = self._z
        z[lu.pinv] = b
        _dense_lsolve(lu.Lp, lu.Li, lu.Lx, z, m)
        if self.n_eta:
            _apply_R(self.estart, self.erow, self.emult, self.erho, self.n_eta, z)
        w = self._w
        _usolve_rows(m, self.row_at, self.col_at, self.udiag, self.rstart,
                     self.rlen, self.rslot, self.rval, self.alive, z, w)
        if out is None:
            out = np.empty(m, dtype=VAL)
        np.take(w, self.slot_of_bpos, out=out)     # one live slot per position
        return out

    def btran(self, c, out=None):
        """``Bᵀ y = c`` -- ``Ũᵀ``, the row etas transposed, then ``Lᵀ``."""
        lu, m = self.lu, self.m
        v = self._w
        v[self.slot_of_bpos] = c
        u = self._z
        _utsolve_rows(m, self.row_at, self.col_at, self.udiag, self.rstart,
                      self.rlen, self.rslot, self.rval, self.alive, v, u)
        if self.n_eta:
            _apply_Rt(self.estart, self.erow, self.emult, self.erho, self.n_eta, u)
        _dense_ltsolve(lu.Lp, lu.Li, lu.Lx, u, m)
        if out is None:
            out = np.empty(m, dtype=VAL)
        np.take(u, lu.pinv, out=out)
        return out

    # -- update ------------------------------------------------------------- #

    def update(self, r: int, col_idx, col_val):
        """Replace basis position ``r`` by the column ``(col_idx, col_val)``.

        A refusal is answered by the caller with a refactorisation and the
        same pivot again, so a fresh factor accepts anything but an exactly
        zero or non-finite pivot -- see the module header.
        """
        lu, m = self.lu, self.m
        if self.n_updates >= self.max_updates:
            raise LUSingular(r, "Forrest-Tomlin update budget exhausted")

        # the spike: R L^-1 P a
        z = self._z
        z[:] = 0.0
        z[lu.pinv[np.asarray(col_idx, dtype=IDX)]] = col_val
        _dense_lsolve(lu.Lp, lu.Li, lu.Lx, z, m)
        if self.n_eta:
            _apply_R(self.estart, self.erow, self.emult, self.erho, self.n_eta, z)

        s_r = int(self.slot_of_bpos[r])
        p_r = int(self.colpos[s_r])
        rho = int(self.row_at[p_r])
        s_new = self.n_slots
        e0 = int(self.estart[self.n_eta])
        if e0 + m + 1 > self.erow.shape[0]:
            self._grow_etas()

        while True:
            used, n_mult, d = _ft_update(
                m, p_r, s_r, s_new, z, rho,
                self.row_at, self.col_at, self.rowpos, self.colpos,
                self.udiag, self.urow,
                self.rstart, self.rlen, self.rcap, self.rslot, self.rval,
                self.alive, self.used, self.rslot.shape[0],
                self._wv, self.erow, self.emult, e0, self.pivot_rel,
                self.max_mult, self.n_updates == 0)
            if n_mult == -3:
                self._grow_rows()
                continue
            break
        self.used = used
        if n_mult < 0:
            self.n_refused += 1
            if n_mult == -1:
                raise LUSingular(r, f"Forrest-Tomlin pivot {d:.3e} too small")
            raise LUSingular(r, "Forrest-Tomlin multiplier beyond the growth cap")
        self.bpos_of_slot[s_new] = r
        self.slot_of_bpos[r] = s_new
        self.n_slots += 1
        self.erho[self.n_eta] = rho
        self.n_eta += 1
        self.estart[self.n_eta] = e0 + n_mult
        self.n_updates += 1

    def _grow_rows(self):
        cap = self.rslot.shape[0] * 2
        for name in ("rslot", "rval"):
            arr = getattr(self, name)
            new = np.empty(cap, dtype=arr.dtype)
            new[:arr.shape[0]] = arr
            setattr(self, name, new)

    def _grow_etas(self):
        cap = self.erow.shape[0] * 2
        for name in ("erow", "emult"):
            arr = getattr(self, name)
            new = np.empty(cap, dtype=arr.dtype)
            new[:arr.shape[0]] = arr
            setattr(self, name, new)
