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

What is here, and what is not
-----------------------------
:func:`rcm_order` -- reverse Cuthill-McKee. Bandwidth reduction by
breadth-first search over ``A + Aᵀ``, returning a permutation applied
symmetrically. A multi-period model couples period ``t`` only to ``t±1``, so
its graph is a long thin strip and this exploits it directly.

**Approximate minimum degree is the textbook answer and is deliberately not
here.** AMD is the general ordering, and it was the intended implementation
until RCM was measured first: on the KKT above it cut fill from 10-15x to
2.3-3.4x and the factorisation from 1.5-3.4 s to 0.04-0.08 s, which is a 20-45x
speedup from ~60 lines of breadth-first search. Writing 350 lines of intricate
quotient-graph code to try to beat that, before knowing whether anything needed
beating, would have been the wrong order to work in. AMD is still the right next
step for the case RCM does not reach -- see the ceiling recorded under
`Known limits` -- and it is a project rather than a patch.

**RCM is not a universal win, which is why nothing here is a default.** On a
wide shallow matrix there is no band to find and its level sets only get in the
way: on the mod010 KKT it reaches 277x fill in 24 s against the incumbent
ordering's 1.79x in 0.004 s. And on a *simplex basis* it loses outright --
measured over the factorisations of four real solves, RCM produced 4-45% more
fill than :mod:`sovopt.numerics.lu`'s own singleton-peeling order every time,
which is the result that order was designed for. So the caller chooses, and
:mod:`sovopt.lp.ipm` chooses by racing the candidates once and keeping the
sparser factors.

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

__all__ = ["rcm_order", "symmetric_pattern"]


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
