"""The dual simplex as one compiled kernel, for the branch-and-bound node LP.

Why a second dual simplex
-------------------------
:mod:`sovopt.lp.simplex` is a Python loop around compiled kernels: pricing,
ratio test, FTRAN, BTRAN and the update are each a kernel, and the loop that
strings them together is Python. On a full LP that is fine -- a pivot on a
6,071-row model costs 11 ms and the loop is noise. On a branch-and-bound node
it is not: a node LP re-solves a 133-row model from a warm basis in a handful
of pivots, and measured on p0201 those pivots cost **176 µs each** of which
the kernels account for perhaps 30. The rest is forty numpy calls' worth of
dispatch. And the loop holds the GIL between kernels, which is why a tree
that solved sixteen independent node LPs on eight threads ran at 0.64x
(docs/NEGATIVE-RESULTS.md, "A parallel branch-and-bound tree").

This file is that loop as a single ``nogil`` kernel: one call per node
solve, no Python between pivots, no GIL while it runs. It is the same
algorithm -- dual simplex over the computational form ``[A | -I]``, Harris
two-pass ratio test, dual Devex pricing, product-form updates with periodic
refactorisation by the same Gilbert-Peierls LU -- reusing the same kernels
where they exist, so that it can be pinned to the Python loop pivot for pivot
in the tests. What it does not do: the primal polish after OPTIMAL, the
phase-1 fallback when the warm basis is not dual feasible, perturbation.
Those stay in :class:`~sovopt.lp.simplex.NodeSolver`, which calls this for
the dual loop and finishes the rare cases itself.

The refactorisation's singular repair replaces the column at the *pivot
step* that failed -- the position ``q[k]``, not ``k`` -- with the logical of
a row no logical in the basis represents; the all-logical basis is ``-I``,
so the repair terminates.

References
----------
Same as :mod:`sovopt.lp.simplex` and :mod:`sovopt.numerics.lu`; this module
adds no algorithm, only a compilation boundary.
Koberstein, "The dual simplex method, techniques for a fast and stable
  implementation", PhD thesis, Paderborn 2005 -- the bounded dual simplex as
  implemented here, in one place.
"""

from __future__ import annotations

import numpy as np

from ..core._jit import jit_kernel
from ..core.sparse import IDX, VAL
from ..core.tolerances import INF
from ..numerics.lu import _dense_lsolve, _dense_ltsolve, _dense_usolve, \
    _dense_utsolve, _gp_lu, _peel_singletons
from .basis import AT_LOWER, AT_UPPER, BASIC, FIXED, FREE, _basis_nnz, \
    _build_basis_csc, _eta_btran, _eta_ftran
from .simplex import _dual_devex_update, _dual_ratio, _worst_infeasible

__all__ = ["dual_simplex_kernel", "OPTIMAL", "INFEASIBLE", "ITERATION_LIMIT",
           "NUMERICAL"]

OPTIMAL = 0
INFEASIBLE = 1
ITERATION_LIMIT = 4
NUMERICAL = 8


@jit_kernel()
def _column_order_k(cp, ci, m, n):
    """The LU's column order, in nopython: peeled singletons then live count."""
    order = np.empty(n, dtype=IDX)
    live = np.empty(n, dtype=np.int64)
    coldead = np.empty(n, dtype=np.bool_)
    placed = _peel_singletons(cp, ci, m, n, order, live, coldead)
    rest = 0
    for j in range(n):
        if not coldead[j]:
            rest += 1
    if rest > 0:
        idx = np.empty(rest, dtype=np.int64)
        keys = np.empty(rest, dtype=np.int64)
        t = 0
        for j in range(n):
            if not coldead[j]:
                idx[t] = j
                keys[t] = live[j]
                t += 1
        perm = np.argsort(keys, kind="mergesort")
        for t in range(rest):
            order[placed + t] = idx[perm[t]]
        placed += rest
    return order, placed


@jit_kernel()
def _factorize_k(Acp, Aci, Acx, basic, status, lower, upper, n, m,
                 lu_tol, lu_drop):
    """Factorise the basis, repairing singularity with logicals.

    Returns ``(ok, Lp, Li, Lx, Up, Ui, Ux, pinv, q, n_repairs)``; ``ok`` is
    false only if the all-logical fallback itself failed, which cannot happen
    for ``-I`` and is reported rather than assumed.
    """
    n_repairs = 0
    x = np.zeros(m, dtype=VAL)
    xi = np.zeros(m, dtype=IDX)
    pstack = np.zeros(m, dtype=np.int64)
    marker = np.zeros(m, dtype=np.bool_)
    pinv = np.full(m, -1, dtype=IDX)
    Lp = np.zeros(m + 1, dtype=np.int64)
    Up = np.zeros(m + 1, dtype=np.int64)
    for _attempt in range(m + 2):
        nnz = _basis_nnz(Acp, basic, n, m)
        cp = np.zeros(m + 1, dtype=np.int64)
        ci = np.empty(max(nnz, 1), dtype=IDX)
        cx = np.empty(max(nnz, 1), dtype=VAL)
        _build_basis_csc(Acp, Aci, Acx, basic, n, m, cp, ci, cx)
        q, placed = _column_order_k(cp, ci, m, m)
        if placed != m:
            # the ordering could not cover the columns: treat as singular at
            # the first uncovered step
            k = placed
        else:
            cap = max(6 * nnz + 4 * m + 16, 64)
            hard_cap = max(64, min(m * m + 4 * m, 1 << 31))
            k = -2
            while True:
                Li = np.zeros(cap, dtype=IDX)
                Lx = np.zeros(cap, dtype=VAL)
                Ui = np.zeros(cap, dtype=IDX)
                Ux = np.zeros(cap, dtype=VAL)
                st, lnz, unz = _gp_lu(cp, ci, cx, m, q, lu_tol, lu_drop,
                                      Lp, Li, Lx, Up, Ui, Ux, pinv,
                                      x, xi, pstack, marker, cap)
                if st == 0:
                    return (True, Lp, Li[:lnz].copy(), Lx[:lnz].copy(),
                            Up, Ui[:unz].copy(), Ux[:unz].copy(),
                            pinv, q, n_repairs)
                if st == -1:
                    if cap >= hard_cap:
                        k = 0
                        break
                    cap = min(cap * 2, hard_cap)
                    continue
                k = st - 1
                break
        # ---- repair: replace the column at pivot step k by a logical ------- #
        pos = q[k] if k < m else 0
        used = np.zeros(m, dtype=np.bool_)
        for t in range(m):
            j = basic[t]
            if j >= n:
                used[j - n] = True
        row = -1
        for i in range(m):
            if not used[i]:
                row = i
                break
        if row < 0:
            row = 0
        leaving = basic[pos]
        basic[pos] = n + row
        lo = lower[leaving]
        up = upper[leaving]
        if lo <= -INF and up >= INF:
            status[leaving] = FREE
        elif up - lo <= 0.0:
            status[leaving] = FIXED
        elif lo > -INF:
            status[leaving] = AT_LOWER
        else:
            status[leaving] = AT_UPPER
        status[n + row] = BASIC
        n_repairs += 1
    # unreachable in practice: -I always factorises
    return (False, Lp, np.zeros(1, dtype=IDX), np.zeros(1, dtype=VAL),
            Up, np.zeros(1, dtype=IDX), np.zeros(1, dtype=VAL), pinv,
            np.arange(m).astype(IDX), n_repairs)


@jit_kernel()
def _ftran_k(Lp, Li, Lx, Up, Ui, Ux, pinv, q, m, v, work,
             estart, eidx, eval_, epiv, erow, n_eta):
    """``v <- B^-1 v`` through the LU and the eta file."""
    for i in range(m):
        work[pinv[i]] = v[i]
    _dense_lsolve(Lp, Li, Lx, work, m)
    _dense_usolve(Up, Ui, Ux, work, m)
    for k in range(m):
        v[q[k]] = work[k]
    if n_eta > 0:
        _eta_ftran(estart, eidx, eval_, epiv, erow, n_eta, v)


@jit_kernel()
def _btran_k(Lp, Li, Lx, Up, Ui, Ux, pinv, q, m, v, work,
             estart, eidx, eval_, epiv, erow, n_eta):
    """``v <- B^-T v`` through the eta file and the LU."""
    if n_eta > 0:
        _eta_btran(estart, eidx, eval_, epiv, erow, n_eta, v)
    for k in range(m):
        work[k] = v[q[k]]
    _dense_utsolve(Up, Ui, Ux, work, m)
    _dense_ltsolve(Lp, Li, Lx, work, m)
    for i in range(m):
        v[i] = work[pinv[i]]


@jit_kernel()
def _basic_values_k(Acp, Aci, Acx, n, m, status, lower, upper, basic,
                    z, rhs):
    """``rhs <- -M_N z_N`` for the current nonbasic placement."""
    for j in range(n + m):
        st = status[j]
        if st == AT_LOWER or st == FIXED:
            z[j] = lower[j]
        elif st == AT_UPPER:
            z[j] = upper[j]
        else:
            z[j] = 0.0
    for i in range(m):
        z[basic[i]] = 0.0
    for i in range(m):
        rhs[i] = z[n + i]                    # -(-I) z_s = +z_s
    for j in range(n):
        zj = z[j]
        if zj != 0.0:
            for p in range(Acp[j], Acp[j + 1]):
                rhs[Aci[p]] -= Acx[p] * zj


@jit_kernel()
def dual_simplex_kernel(Acp, Aci, Acx, Arp, Ari, Arx, n, m,
                        cost, lower, upper, status, basic, zB, dual_weight,
                        farkas,
                        feas_tol, opt_tol, pivot_tol, harris_relax,
                        refactor_freq, max_iter, recompute_freq, devex_reset,
                        lu_tol, lu_drop):
    """Run the dual simplex to completion on the given warm basis.

    Modifies ``status``, ``basic``, ``zB``, ``dual_weight`` in place; on
    INFEASIBLE writes the Farkas row into ``farkas``. Returns
    ``(code, iterations, factorizations, factors)`` where ``factors`` is the
    final LU and eta file -- ``(Lp, Li, Lx, Up, Ui, Ux, pinv, q, estart,
    eidx, evals, epiv, erow, n_eta)`` -- so the caller can carry on from the
    exact basis representation the kernel ended with, without refactorising.
    """
    N = n + m
    ok, Lp, Li, Lx, Up, Ui, Ux, pinv, q, _ = _factorize_k(
        Acp, Aci, Acx, basic, status, lower, upper, n, m, lu_tol, lu_drop)
    n_fact = 1

    # eta file
    ecap = max(16 * m, 1024)
    estart = np.zeros(refactor_freq + 2, dtype=np.int64)
    eidx = np.empty(ecap, dtype=IDX)
    eval_ = np.empty(ecap, dtype=VAL)
    epiv = np.empty(refactor_freq + 2, dtype=VAL)
    erow = np.empty(refactor_freq + 2, dtype=IDX)
    n_eta = 0
    if not ok:
        return NUMERICAL, 0, n_fact, (Lp, Li, Lx, Up, Ui, Ux, pinv, q,
                                      estart, eidx, eval_, epiv, erow, n_eta)

    work = np.empty(m, dtype=VAL)
    z = np.empty(N, dtype=VAL)
    rho = np.empty(m, dtype=VAL)
    alpha = np.empty(m, dtype=VAL)
    alpha_row = np.empty(N, dtype=VAL)
    y = np.empty(m, dtype=VAL)
    d = np.empty(N, dtype=VAL)

    _basic_values_k(Acp, Aci, Acx, n, m, status, lower, upper, basic, z, zB)
    _ftran_k(Lp, Li, Lx, Up, Ui, Ux, pinv, q, m, zB, work,
             estart, eidx, eval_, epiv, erow, n_eta)

    iters = 0
    while True:
        if iters >= max_iter:
            return ITERATION_LIMIT, iters, n_fact, (Lp, Li, Lx, Up, Ui, Ux, pinv, q, estart, eidx, eval_, epiv, erow, n_eta)
        if iters > 0 and iters % recompute_freq == 0:
            _basic_values_k(Acp, Aci, Acx, n, m, status, lower, upper, basic, z, zB)
            _ftran_k(Lp, Li, Lx, Up, Ui, Ux, pinv, q, m, zB, work,
                     estart, eidx, eval_, epiv, erow, n_eta)

        r, sigma = _worst_infeasible(zB, basic, lower, upper, dual_weight, feas_tol)
        if r < 0:
            return OPTIMAL, iters, n_fact, (Lp, Li, Lx, Up, Ui, Ux, pinv, q, estart, eidx, eval_, epiv, erow, n_eta)

        # ---- pivot row: rho = B^-T e_r, alpha_row = rho^T M ---------------- #
        for i in range(m):
            rho[i] = 0.0
        rho[r] = 1.0
        _btran_k(Lp, Li, Lx, Up, Ui, Ux, pinv, q, m, rho, work,
                 estart, eidx, eval_, epiv, erow, n_eta)
        for j in range(n):
            acc = 0.0
            for p in range(Acp[j], Acp[j + 1]):
                acc += Acx[p] * rho[Aci[p]]
            alpha_row[j] = acc
        for i in range(m):
            alpha_row[n + i] = -rho[i]

        # ---- duals and reduced costs --------------------------------------- #
        for i in range(m):
            y[i] = cost[basic[i]]
        _btran_k(Lp, Li, Lx, Up, Ui, Ux, pinv, q, m, y, work,
                 estart, eidx, eval_, epiv, erow, n_eta)
        for j in range(n):
            acc = cost[j]
            for p in range(Acp[j], Acp[j + 1]):
                acc -= Acx[p] * y[Aci[p]]
            d[j] = acc
        for i in range(m):
            d[n + i] = y[i]
        for i in range(m):
            d[basic[i]] = 0.0

        qcol, _t = _dual_ratio(alpha_row, d, status, sigma, pivot_tol,
                               opt_tol, harris_relax)
        if qcol < 0:
            for i in range(m):
                farkas[i] = rho[i]
            return INFEASIBLE, iters, n_fact, (Lp, Li, Lx, Up, Ui, Ux, pinv, q, estart, eidx, eval_, epiv, erow, n_eta)

        # ---- primal step --------------------------------------------------- #
        jl = basic[r]
        target = lower[jl] if sigma > 0.0 else upper[jl]
        delta = target - zB[r]
        arq = alpha_row[qcol]
        refactor_now = False
        skipped = False
        if abs(arq) <= pivot_tol:
            refactor_now = True
            skipped = True
        else:
            theta = -delta / arq
            # alpha = B^-1 M[:, q]
            for i in range(m):
                alpha[i] = 0.0
            if qcol < n:
                for p in range(Acp[qcol], Acp[qcol + 1]):
                    alpha[Aci[p]] = Acx[p]
            else:
                alpha[qcol - n] = -1.0
            _ftran_k(Lp, Li, Lx, Up, Ui, Ux, pinv, q, m, alpha, work,
                     estart, eidx, eval_, epiv, erow, n_eta)
            piv = alpha[r]
            if abs(piv) <= pivot_tol or not np.isfinite(piv):
                # alpha_row[q] and alpha[r] are the same number computed two
                # ways; when the factorisation has drifted they disagree, and
                # the answer is a fresh factorisation, not an exception
                refactor_now = True
                skipped = True
            else:
                _dual_devex_update(dual_weight, alpha, r, piv, 1.0)
                if dual_weight[r] > devex_reset:
                    for i in range(m):
                        dual_weight[i] = 1.0
                for i in range(m):
                    zB[i] -= alpha[i] * theta
                st_q = status[qcol]
                zq = theta
                if st_q == AT_LOWER or st_q == FIXED:
                    zq += lower[qcol]
                elif st_q == AT_UPPER:
                    zq += upper[qcol]

                # ---- product-form update ----------------------------------- #
                nz = 0
                for i in range(m):
                    if alpha[i] != 0.0 and i != r:
                        nz += 1
                s0 = estart[n_eta]
                if s0 + nz > eidx.shape[0]:
                    newcap = max(2 * (s0 + nz), 2 * eidx.shape[0])
                    ni = np.empty(newcap, dtype=IDX)
                    nv = np.empty(newcap, dtype=VAL)
                    ni[:s0] = eidx[:s0]
                    nv[:s0] = eval_[:s0]
                    eidx = ni
                    eval_ = nv
                t = s0
                for i in range(m):
                    if alpha[i] != 0.0 and i != r:
                        eidx[t] = i
                        eval_[t] = alpha[i]
                        t += 1
                epiv[n_eta] = piv
                erow[n_eta] = r
                n_eta += 1
                estart[n_eta] = t

                basic[r] = qcol
                status[qcol] = BASIC
                zB[r] = zq
                if upper[jl] - lower[jl] <= 0.0:
                    status[jl] = FIXED
                else:
                    status[jl] = AT_LOWER if sigma > 0.0 else AT_UPPER
                iters += 1
                if n_eta < refactor_freq:
                    continue
                refactor_now = True
        if refactor_now:
            ok, Lp, Li, Lx, Up, Ui, Ux, pinv, q, _ = _factorize_k(
                Acp, Aci, Acx, basic, status, lower, upper, n, m, lu_tol, lu_drop)
            n_fact += 1
            if not ok:
                return NUMERICAL, iters, n_fact, (Lp, Li, Lx, Up, Ui, Ux, pinv, q, estart, eidx, eval_, epiv, erow, n_eta)
            n_eta = 0
            _basic_values_k(Acp, Aci, Acx, n, m, status, lower, upper, basic, z, zB)
            _ftran_k(Lp, Li, Lx, Up, Ui, Ux, pinv, q, m, zB, work,
                     estart, eidx, eval_, epiv, erow, n_eta)
            if skipped:
                iters += 1              # a skipped pivot still counts, as in the loop
