"""Fill-reducing orderings for sparse factorisation.

An LU or Cholesky factor is only as sparse as the elimination order allows.
Permute a matrix badly and the factors fill in; permute it well and they stay
close to the original sparsity. The permutation is chosen from the *pattern*
alone, before a single arithmetic operation, and it is often worth more than
every numerical trick applied afterwards.

Why this file exists
--------------------
:mod:`sovopt.numerics.lu` chooses its column order by peeling singletons and
then sorting by live column count -- Suhl-Suhl, and exactly right for a simplex
basis, which is 80-95% triangular already. It is the wrong order for a
saddle-point matrix. Measured on the interior-point KKT of a 3,840-row refinery
planning model, over the eight iterations of a real solve:

    fill (nnz(L+U) / nnz(K))    10.2x  10.1x  15.2x  14.7x  13.2x  13.1x  13.2x  9.9x
    factorisation share of the total solve                                    97%

Ninety-seven per cent of an interior-point solve is one factorisation repeated,
and that factorisation is carrying a 10-15x fill penalty. The interior point
never reached a second iteration on a 61,440-row model.

The saving grace of an interior-point method is that **the KKT pattern never
changes**: only the two diagonal blocks move from iteration to iteration. So the
ordering is computed once and reused for every factorisation of the solve, and
its cost is amortised over the whole run rather than paid per iteration.

What is here
------------
:func:`rcm_order` -- reverse Cuthill-McKee. Bandwidth reduction by
breadth-first search over ``A + Aᵀ``, returning a permutation applied
symmetrically. A multi-period model couples period ``t`` only to ``t±1``, so
its graph is a long thin strip and this exploits it directly.

:func:`amd_order` -- approximate minimum degree, the general fill-reducing
order: eliminate the node of least degree, replace it by the clique of its
neighbours, repeat. The quotient graph keeps that clique as an *element*
rather than as edges, so the graph never grows past the size of the matrix;
the degree is *approximated* by the bound ``|A_i| + |L_p \ i| + Σ_e |L_e \ L_p|``,
which costs one pass over the elements touching the pivot's clique instead of
a set union per node; elements absorbed by the pivot and variables whose only
neighbour is the pivot are removed in the same pass. Rows with more than
``10√n`` neighbours are set aside and ordered last, which is what keeps one
dense row from sitting in every clique. On the KKTs it was written for it
reproduces the fill of an exact minimum degree to the entry, and it runs in
seconds on a 122,880-node pattern.

:func:`symbolic_fill` -- ``nnz(L)`` a symmetric elimination would have in a
given order, by merging each column's pattern into its elimination-tree
parent. Tens of milliseconds, and the reason the interior point no longer
has to factorise a candidate to learn that it lost.

**Neither ordering wins everywhere, which is why the caller races them.**
Measured on interior-point KKTs, nnz(L+U) over nnz(K):

    mod010   (wide, 2,801 nodes)          amd 1.5x    natural 1.8x    rcm 268x
    blend k=8 (wide, 26,640 nodes)        amd 2.1x    natural 2.9x    rcm 4.1x
    plan k=4  (banded, 7,680 nodes)       amd 5.3x    natural 9.1x    rcm 3.1x
    plan k=16 (banded, 122,880 nodes)     amd 8.6x                    rcm 3.4x
    QPLIB_8559 (15,000 nodes)             amd 87x     natural 505x    rcm 334x

Minimum degree is greedy and a banded strip is the shape that punishes greed:
on plan k=4 an *exact* minimum degree gives the same 5.3x, so it is the
method and not the approximation. And on a *simplex basis* both lose to
:mod:`sovopt.numerics.lu`'s own singleton-peeling order (qnet1: own 3.02x,
rcm 2.86x, amd 3.23x; woodw: 2.56x, 3.63x, 3.98x), which is the result that
order was designed for. So :mod:`sovopt.lp.ipm` ranks the three by symbolic
fill, factorises the predicted winner in full, and tries a challenger only
when it was predicted within 1.5x -- under the winner's nnz as a cap.

References
----------
Amestoy, Davis & Duff, "An approximate minimum degree ordering algorithm", SIAM
  J. Matrix Anal. Appl. 17 (1996) 886-905 -- the quotient graph, the
  approximate external degree bound, supervariable detection by hashing, and
  element absorption.
George & Liu, *Computer Solution of Large Sparse Positive Definite Systems*,
  Prentice-Hall (1981), ch. 5 -- minimum degree and the quotient-graph
  representation it rests on.
Cuthill & McKee, "Reducing the bandwidth of sparse symmetric matrices", Proc.
  24th ACM National Conference (1969) 157-172.
George, "Computer implementation of the finite element method", Stanford
  STAN-CS-71-208 (1971) -- reversing the Cuthill-McKee order never increases
  the profile and usually decreases it.
Davis, *Direct Methods for Sparse Linear Systems*, SIAM (2006), ch. 7.
"""

from __future__ import annotations

import numpy as np

from ..core._jit import jit_kernel
from ..core.sparse import IDX

__all__ = ["amd_order", "rcm_order", "symbolic_fill", "symmetric_pattern"]


def symmetric_pattern(cp, ci, n):
    """Pattern of ``A + Aᵀ`` with the diagonal removed, as ``(ap, ai)``.

    Both orderings work on an undirected graph, so each edge must appear in
    both endpoints' lists exactly once. The diagonal is dropped because a
    self-loop is not an edge and would corrupt every degree count.
    """
    cp = np.asarray(cp, dtype=np.int64)
    ci = np.asarray(ci, dtype=np.int64)

    rows = ci
    cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(cp))
    keep = rows != cols
    r, c = rows[keep], cols[keep]

    # both directions, then drop duplicates
    ur = np.concatenate([r, c])
    uc = np.concatenate([c, r])
    key = ur * np.int64(n) + uc
    order = np.argsort(key, kind="stable")
    key = key[order]
    uniq = np.empty(key.size, dtype=bool)
    uniq[:1] = True
    if key.size > 1:
        np.not_equal(key[1:], key[:-1], out=uniq[1:])
    ur, uc = ur[order][uniq], uc[order][uniq]

    ap = np.zeros(n + 1, dtype=np.int64)
    np.add.at(ap[1:], ur, 1)
    np.cumsum(ap, out=ap)
    return ap, uc.astype(np.int64)


# --------------------------------------------------------------------------- #
# reverse Cuthill-McKee                                                        #
# --------------------------------------------------------------------------- #


@jit_kernel()
def _rcm(n, ap, ai, deg, order, mask, queue):
    """Cuthill-McKee level sets, one connected component at a time.

    Each component is swept twice: once to find it and pick a low-degree root,
    once to lay out its level sets in ascending degree. The two passes are the
    fix for a real bug -- choosing the root by scanning *all* remaining nodes
    picks a minimum-degree node that may sit in a different component, so the
    node the outer loop was standing on is never reached and the permutation
    comes back short. Measured on a mod010 simplex basis: 85 nodes of 146.

    The caller reverses the result, which never increases the profile and
    usually decreases it (George 1971).
    """
    placed = 0
    for seed in range(n):
        if mask[seed]:
            continue

        # pass 1: collect this component and choose its lowest-degree node
        mask[seed] = True
        queue[0] = seed
        comp = 1
        h = 0
        while h < comp:
            v = queue[h]
            h += 1
            for p in range(ap[v], ap[v + 1]):
                u = ai[p]
                if not mask[u]:
                    mask[u] = True
                    queue[comp] = u
                    comp += 1
        root = queue[0]
        best = deg[root]
        for a in range(comp):
            if deg[queue[a]] < best:
                best = deg[queue[a]]
                root = queue[a]
        for a in range(comp):
            mask[queue[a]] = False

        # pass 2: level sets from that root, each sorted by ascending degree
        head = placed
        mask[root] = True
        order[placed] = root
        placed += 1
        while head < placed:
            v = order[head]
            head += 1
            lo = placed
            for p in range(ap[v], ap[v + 1]):
                u = ai[p]
                if not mask[u]:
                    mask[u] = True
                    order[placed] = u
                    placed += 1
            for a in range(lo + 1, placed):
                x = order[a]
                dx = deg[x]
                b = a - 1
                while b >= lo and deg[order[b]] > dx:
                    order[b + 1] = order[b]
                    b -= 1
                order[b + 1] = x
    return placed


def rcm_order(cp, ci, n):
    """Reverse Cuthill-McKee permutation of a square pattern.

    Returns ``q`` such that permuting rows and columns by ``q`` reduces the
    bandwidth. Cheap, deterministic, and strong on matrices that are banded to
    begin with -- a multi-period model couples period ``t`` only to ``t±1``,
    so its graph is a long thin strip and this exploits it directly.
    """
    ap, ai = symmetric_pattern(cp, ci, n)
    deg = np.diff(ap).astype(np.int64)
    order = np.zeros(n, dtype=np.int64)
    mask = np.zeros(n, dtype=np.bool_)
    queue = np.zeros(n, dtype=np.int64)
    placed = _rcm(n, ap, ai, deg, order, mask, queue)
    if placed != n:                      # pragma: no cover - defensive
        raise RuntimeError(f"RCM visited {placed} of {n} nodes")
    return order[::-1].copy().astype(IDX)


# --------------------------------------------------------------------------- #
# approximate minimum degree                                                   #
# --------------------------------------------------------------------------- #

# Node states in the quotient graph.
_VAR = 0        # an uneliminated variable
_ELEM = 1       # an eliminated pivot, kept as an element (a clique)
_GONE = 2       # absorbed element or mass-eliminated variable
_DENSE = 3      # set aside at the start, ordered last


@jit_kernel()
def _cell_push(head, cnt, node, val, cnext, cval, free):
    """Prepend ``val`` to node's list. Returns the new free-list head, or -1."""
    c = free[0]
    if c < 0:
        return -1
    free[0] = cnext[c]
    cval[c] = val
    cnext[c] = head[node]
    head[node] = c
    cnt[node] += 1
    return c


@jit_kernel()
def _cell_free_list(head, cnt, node, cnext, free):
    """Return every cell of node's list to the free list."""
    c = head[node]
    while c >= 0:
        nxt = cnext[c]
        cnext[c] = free[0]
        free[0] = c
        c = nxt
    head[node] = -1
    cnt[node] = 0


@jit_kernel()
def _deg_remove(i, deg, dhead, dnext, dprev):
    d = deg[i]
    if dprev[i] >= 0:
        dnext[dprev[i]] = dnext[i]
    else:
        dhead[d] = dnext[i]
    if dnext[i] >= 0:
        dprev[dnext[i]] = dprev[i]
    dnext[i] = -1
    dprev[i] = -1


@jit_kernel()
def _deg_insert(i, d, deg, dhead, dnext, dprev):
    deg[i] = d
    dnext[i] = dhead[d]
    dprev[i] = -1
    if dhead[d] >= 0:
        dprev[dhead[d]] = i
    dhead[d] = i


@jit_kernel()
def _amd(n, ap, ai, dense_limit, cap, order):
    """Approximate minimum degree on the quotient graph. Returns ``(placed,
    status)``; status -1 means the cell store ran out and the caller retries
    with a larger one.

    Lists are singly linked cells drawn from one store: a variable owns its
    list of adjacent variables (A) and of adjacent elements (E); an element
    owns the list of variables it contains (L), kept in the A slot since a
    node is one or the other. Degree buckets are doubly linked so the minimum
    is found by advancing a pointer.
    """
    cnext = np.empty(cap, dtype=np.int64)
    cval = np.empty(cap, dtype=np.int64)
    for c in range(cap - 1):
        cnext[c] = c + 1
    cnext[cap - 1] = -1
    free = np.zeros(1, dtype=np.int64)

    ahead = np.full(n, -1, dtype=np.int64)
    alen = np.zeros(n, dtype=np.int64)
    ehead = np.full(n, -1, dtype=np.int64)
    elen = np.zeros(n, dtype=np.int64)
    state = np.zeros(n, dtype=np.int64)
    deg = np.zeros(n, dtype=np.int64)
    dhead = np.full(n + 1, -1, dtype=np.int64)
    dnext = np.full(n, -1, dtype=np.int64)
    dprev = np.full(n, -1, dtype=np.int64)
    mark = np.zeros(n, dtype=np.int64)
    w = np.zeros(n, dtype=np.int64)
    wstamp = np.zeros(n, dtype=np.int64)
    lp_buf = np.empty(n, dtype=np.int64)

    # ---- dense rows aside, then the adjacency lists ------------------------ #
    n_dense = 0
    for i in range(n):
        if ap[i + 1] - ap[i] > dense_limit:
            state[i] = _DENSE
            n_dense += 1
    for i in range(n):
        if state[i] == _DENSE:
            continue
        for p in range(ap[i], ap[i + 1]):
            j = ai[p]
            if state[j] == _DENSE:
                continue
            if _cell_push(ahead, alen, i, j, cnext, cval, free) < 0:
                return 0, -1
        _deg_insert(i, alen[i], deg, dhead, dnext, dprev)

    n_active = n - n_dense
    placed = 0
    mindeg = 0
    tag = 0
    while placed < n_active:
        # ---- pivot: the variable of minimum approximate degree ------------- #
        while mindeg <= n and dhead[mindeg] < 0:
            mindeg += 1
        if mindeg > n:
            break
        pv = dhead[mindeg]
        _deg_remove(pv, deg, dhead, dnext, dprev)
        order[placed] = pv
        placed += 1
        state[pv] = _ELEM
        tag += 1

        # ---- L_p = A_p ∪ (∪ L_e for e in E_p) minus eliminated nodes ------ #
        nlp = 0
        c = ahead[pv]
        while c >= 0:
            j = cval[c]
            if state[j] == _VAR and mark[j] != tag:
                mark[j] = tag
                lp_buf[nlp] = j
                nlp += 1
            c = cnext[c]
        c = ehead[pv]
        while c >= 0:
            e = cval[c]
            if state[e] == _ELEM:
                d = ahead[e]
                while d >= 0:
                    j = cval[d]
                    if state[j] == _VAR and mark[j] != tag:
                        mark[j] = tag
                        lp_buf[nlp] = j
                        nlp += 1
                    d = cnext[d]
                _cell_free_list(ahead, alen, e, cnext, free)   # absorbed into p
                state[e] = _GONE
            c = cnext[c]
        _cell_free_list(ahead, alen, pv, cnext, free)
        _cell_free_list(ehead, elen, pv, cnext, free)

        # ---- each i in L_p: prune A_i and E_i, then E_i gains p ------------ #
        for t in range(nlp):
            i = lp_buf[t]
            # A_i: drop p, absorbed nodes and anything now reached through p
            prev = -1
            c = ahead[i]
            while c >= 0:
                j = cval[c]
                nxt = cnext[c]
                if state[j] != _VAR or mark[j] == tag:
                    if prev < 0:
                        ahead[i] = nxt
                    else:
                        cnext[prev] = nxt
                    cnext[c] = free[0]
                    free[0] = c
                    alen[i] -= 1
                else:
                    prev = c
                c = nxt
            # E_i: drop absorbed elements
            prev = -1
            c = ehead[i]
            while c >= 0:
                e = cval[c]
                nxt = cnext[c]
                if state[e] != _ELEM:
                    if prev < 0:
                        ehead[i] = nxt
                    else:
                        cnext[prev] = nxt
                    cnext[c] = free[0]
                    free[0] = c
                    elen[i] -= 1
                else:
                    prev = c
                c = nxt
            if _cell_push(ehead, elen, i, pv, cnext, cval, free) < 0:
                return placed, -1

        # ---- |L_e \ L_p| for every element touching L_p ------------------- #
        for t in range(nlp):
            i = lp_buf[t]
            c = ehead[i]
            while c >= 0:
                e = cval[c]
                if e != pv:
                    if wstamp[e] != tag:
                        wstamp[e] = tag
                        w[e] = alen[e]
                    w[e] -= 1
                c = cnext[c]

        # ---- mass elimination: i with A_i empty and E_i = {p} -------------- #
        keep = 0
        for t in range(nlp):
            i = lp_buf[t]
            if alen[i] == 0 and elen[i] == 1:
                _deg_remove(i, deg, dhead, dnext, dprev)
                order[placed] = i
                placed += 1
                state[i] = _GONE
                _cell_free_list(ehead, elen, i, cnext, free)
            else:
                lp_buf[keep] = i
                keep += 1
        nlp = keep

        # ---- element p's list, and the approximate degrees ----------------- #
        for t in range(nlp):
            if _cell_push(ahead, alen, pv, lp_buf[t], cnext, cval, free) < 0:
                return placed, -1
        remaining = n_active - placed
        for t in range(nlp):
            i = lp_buf[t]
            d = alen[i] + nlp - 1
            prev = -1
            c = ehead[i]
            while c >= 0:
                e = cval[c]
                nxt = cnext[c]
                if e != pv:
                    if w[e] == 0:
                        # L_e is inside L_p: e adds nothing and is absorbed
                        if prev < 0:
                            ehead[i] = nxt
                        else:
                            cnext[prev] = nxt
                        cnext[c] = free[0]
                        free[0] = c
                        elen[i] -= 1
                        if state[e] == _ELEM:
                            _cell_free_list(ahead, alen, e, cnext, free)
                            state[e] = _GONE
                        c = nxt
                        continue
                    d += w[e]
                prev = c
                c = nxt
            bound = deg[i] + nlp - 1
            if bound < d:
                d = bound
            if remaining - 1 < d:
                d = remaining - 1
            if d < 0:
                d = 0
            _deg_remove(i, deg, dhead, dnext, dprev)
            _deg_insert(i, d, deg, dhead, dnext, dprev)
            if d < mindeg:
                mindeg = d

    if placed != n_active:
        return placed, -2
    for i in range(n):
        if state[i] == _DENSE:
            order[placed] = i
            placed += 1
    return placed, 0


def amd_order(cp, ci, n, dense: float = 10.0):
    """Approximate minimum degree permutation of a square pattern.

    The general-purpose fill-reducing order: eliminate the node of least
    (approximately counted) degree, form its clique, repeat. Rows with more
    than ``max(16, dense * sqrt(n))`` neighbours are set aside and ordered
    last, which is what keeps an objective row or a dense column from
    dominating every degree. Deterministic.
    """
    ap, ai = symmetric_pattern(cp, ci, n)
    n = int(n)
    if n == 0:
        return np.zeros(0, dtype=IDX)
    dense_limit = int(max(16.0, dense * np.sqrt(n)))
    nnz = int(ap[-1])
    cap = max(2 * nnz + 8 * n + 1024, 4096)
    for _ in range(12):
        order = np.zeros(n, dtype=np.int64)
        placed, status = _amd(n, ap, ai, dense_limit, cap, order)
        if status == 0:
            return order.astype(IDX)
        if status == -1:
            cap *= 2
            continue
        raise RuntimeError(f"AMD placed {placed} of {n} nodes")   # pragma: no cover
    raise RuntimeError("AMD cell store kept overflowing")          # pragma: no cover


# --------------------------------------------------------------------------- #
# symbolic fill: how much a permutation would cost, before factorising         #
# --------------------------------------------------------------------------- #


@jit_kernel()
def _symbolic(n, ap, ai, pos, cap, arena_cap):
    """``nnz(L)`` of a symmetric elimination in the order ``pos`` describes.

    Column ``k``'s pattern is its later neighbours in ``A`` plus the patterns
    of the columns whose parent it is (the elimination tree), and each
    pattern is merged into its parent once. Patterns live in an arena and a
    node whose pattern outgrows its slot is moved to the end. Returns ``-1``
    when the count passes ``cap`` -- the caller only wants to know that it
    lost -- and ``-2`` when the arena is exhausted.
    """
    start = np.zeros(n, dtype=np.int64)
    length = np.zeros(n, dtype=np.int64)
    room = np.zeros(n, dtype=np.int64)
    arena = np.empty(arena_cap, dtype=np.int64)
    mark = np.full(n, -1, dtype=np.int64)
    # initial patterns: later neighbours, with slack for merges
    used = 0
    for i in range(n):
        cnt = 0
        for p in range(ap[i], ap[i + 1]):
            if pos[ai[p]] > pos[i]:
                cnt += 1
        start[i] = used
        room[i] = cnt + 4 + cnt // 2
        used += room[i]
        if used > arena_cap:
            return -2
        for p in range(ap[i], ap[i + 1]):
            j = ai[p]
            if pos[j] > pos[i]:
                arena[start[i] + length[i]] = j
                length[i] += 1
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        order[pos[i]] = i
    total = n
    for t in range(n):
        k = order[t]
        ln = length[k]
        total += ln
        if total > cap:
            return -1
        if ln == 0:
            continue
        # parent: the earliest-ordered member of the pattern
        s = start[k]
        par = arena[s]
        for q in range(s + 1, s + ln):
            if pos[arena[q]] < pos[par]:
                par = arena[q]
        # merge pattern \ {parent} into the parent's pattern
        ps = start[par]
        for q in range(ps, ps + length[par]):
            mark[arena[q]] = par
        mark[par] = par
        add = 0
        for q in range(s, s + ln):
            if mark[arena[q]] != par:
                add += 1
        if add > 0:
            if length[par] + add > room[par]:
                newroom = 2 * (length[par] + add) + 4
                if used + newroom > arena_cap:
                    return -2
                for q in range(length[par]):
                    arena[used + q] = arena[ps + q]
                start[par] = used
                room[par] = newroom
                used += newroom
                ps = start[par]
            for q in range(s, s + ln):
                v = arena[q]
                if mark[v] != par:
                    mark[v] = par
                    arena[ps + length[par]] = v
                    length[par] += 1
    return total


def symbolic_fill(cp, ci, n, order, cap=None):
    """``nnz(L)`` a symmetric factorisation would have in ``order``.

    The prediction assumes diagonal pivots. The LU below pivots for
    stability and can come in under it -- on the QPLIB_8559 KKT the natural
    order predicts 505x and threshold pivoting delivers 64x -- so this ranks
    candidates rather than promising a count. ``None`` if the count passes
    ``cap``.
    """
    ap, ai = symmetric_pattern(cp, ci, n)
    n = int(n)
    if n == 0:
        return 0
    pos = np.empty(n, dtype=np.int64)
    pos[np.asarray(order, dtype=np.int64)] = np.arange(n, dtype=np.int64)
    limit = int(cap) if cap is not None else (1 << 60)
    arena_cap = max(4 * int(ap[-1]) + 8 * n + 1024, 4096)
    for _ in range(8):
        got = _symbolic(n, ap, ai, pos, limit, arena_cap)
        if got == -2:
            arena_cap *= 2
            continue
        return None if got < 0 else int(got)
    return None                                                     # pragma: no cover
