"""Symmetry detection and symmetry breaking.

Why it matters
--------------
Refineries have identical parallel units; schedules have interchangeable time
slots; blending has identical tanks. When a model has such a group, a
branch-and-bound tree re-derives the *same* plan under every relabelling of the
identical objects. With ``k`` identical units the search can waste a factor of
``k!`` -- which is why a solver without symmetry handling grinds on models a
planner calls obvious.

How a symmetry is found
-----------------------
Detection is two-stage, and only the second stage is trusted.

**1. Colour refinement** (1-dimensional Weisfeiler-Leman) on the bipartite
variable-constraint graph. Variables start coloured by objective coefficient,
bounds and kind; constraints by their row bounds. Each round recolours a node by
the multiset of ``(neighbour colour, coefficient)`` pairs around it, until the
colouring stops changing. Refinement can never separate two variables that lie
in the same orbit, so different final colours prove non-equivalence -- but equal
colours prove nothing.

**2. Individualisation, then exact verification.** Refinement alone almost never
produces a symmetry. Give one variable ``v`` a colour of its own, re-refine, and
the consequences propagate through the whole graph; do the same for a candidate
image ``w``. If both colourings come out *discrete*, the permutation can be read
straight off the colours. It is then checked against the model in full --
objective, bounds, kinds, row bounds and every matrix entry.

Individualisation is what makes this work on real models. **A transposition of
two columns is essentially never a symmetry by itself**: swapping two identical
process units means swapping all of their variables across every time period at
once, a product of many transpositions. Refinement propagates that whole
permutation from a single individualised variable.

Because verification is exact, the search may be incomplete without ever being
unsound. It can miss symmetries; it cannot invent one.

How a symmetry is used
----------------------
For a verified automorphism ``pi``, take ``j`` = the smallest index that ``pi``
moves and add

    x_j >= x_{pi(j)}

This is valid. Every orbit of feasible solutions under the group contains a
lexicographically greatest element, and that element satisfies ``x >=_lex
pi(x)`` for every ``pi`` in the group. Since ``pi`` fixes every index below
``j``, the first coordinate where the two can differ is ``j`` itself, so
``x_j >= x_{pi(j)}`` holds there. At least one optimum therefore survives, and
the constraint is linear.

When a whole set turns out to be *fully* interchangeable -- every pairwise
transposition verifies, which does happen for genuinely identical items sharing
a single row -- the stronger chain ``x_{s1} >= x_{s2} >= ... >= x_{sk}`` is
emitted instead.

What is deliberately not done
-----------------------------
No orbital branching or orbital fixing. Both are stronger, and both need orbits
of the *stabiliser* of the branching decisions made so far -- root orbits are
not valid deeper in the tree, and using them there prunes real solutions. Static
constraints are valid everywhere and need no such care.

References
----------
Margot, "Symmetry in integer linear programming", in *50 Years of Integer
  Programming*, Springer 2010 -- survey; validity of static symmetry breaking.
Ostrowski, Linderoth, Rossi & Smriglio, "Orbital branching", Math. Prog. 126
  (2011) 147-178.
Kaibel, Peinhardt & Pfetsch, "Orbitopal fixing", Discrete Optimization 8 (2011).
Weisfeiler & Leman, "The reduction of a graph to canonical form", 1968.
McKay & Piperno, "Practical graph isomorphism II", J. Symbolic Computation 60
  (2014) 94-112 -- individualisation-refinement, of which this is a bounded,
  verification-backed special case.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core.sparse import IDX, VAL
from ..core.tolerances import INF

__all__ = ["SymmetryInfo", "detect_symmetry", "breaking_constraints",
           "refine_colours", "find_generators", "verify_permutation",
           "orbits_from"]


@dataclass
class SymmetryInfo:
    generators: list = field(default_factory=list)
    """Verified automorphisms, as permutation arrays over the variables."""

    orbits: list[list[int]] = field(default_factory=list)
    interchangeable: list[list[int]] = field(default_factory=list)
    """Sets where *every* pairwise transposition verified."""

    colour_classes: int = 0
    candidates_tested: int = 0
    seconds: float = 0.0
    timed_out: bool = False

    @property
    def largest_orbit(self) -> int:
        return max((len(o) for o in self.orbits), default=0)

    def summary(self) -> dict:
        return {
            "generators": len(self.generators),
            "orbits": len(self.orbits),
            "largest_orbit": self.largest_orbit,
            "interchangeable_sets": len(self.interchangeable),
            "colour_classes": self.colour_classes,
            "tested": self.candidates_tested,
            "seconds": round(self.seconds, 3),
        }

    def __repr__(self):
        return (f"SymmetryInfo({len(self.generators)} generators, "
                f"{len(self.orbits)} orbits, largest {self.largest_orbit})")


# --------------------------------------------------------------------------- #
# colour refinement                                                            #
# --------------------------------------------------------------------------- #


def _quantise(a, tol=1e-9):
    """Round to a stable key so 1.0 and 1.0+1e-16 share a colour."""
    return np.round(np.asarray(a, dtype=VAL) / tol).astype(np.int64)


def _relabel(keys):
    """Map hashable keys to dense integer colours **canonically**.

    Colours are assigned by sorting the distinct keys, not by order of first
    appearance. This matters far more than it looks: the search refines two
    colourings independently (one with ``v`` individualised, one with ``w``) and
    then compares them cell by cell. With first-appearance numbering the same
    structural colour receives different integers in the two runs, so every
    cross-colouring comparison -- both the cell matching and reading the
    bijection off the colours -- silently compares unrelated things and the
    search finds nothing at all.
    """
    uniq = sorted(set(keys))
    table = {k: i for i, k in enumerate(uniq)}
    return np.fromiter((table[k] for k in keys), dtype=np.int64,
                       count=len(keys))


def _clip(a):
    return np.clip(np.asarray(a, dtype=VAL), -1e30, 1e30)


def _initial_colours(prob):
    var = list(zip(_quantise(prob.c).tolist(),
                   _quantise(_clip(prob.col_lb)).tolist(),
                   _quantise(_clip(prob.col_ub)).tolist(),
                   prob.kind.tolist()))
    con = list(zip(_quantise(_clip(prob.row_lb)).tolist(),
                   _quantise(_clip(prob.row_ub)).tolist()))
    return _relabel(var), _relabel(con)


def refine_colours(prob, max_rounds: int = 20, start=None):
    """Equitable partition of the bipartite variable-constraint graph.

    ``start`` supplies an initial colouring, which is how individualisation
    works: give one variable a colour of its own and re-refine.
    """
    A = prob.A
    if start is None:
        vcol, ccol = _initial_colours(prob)
    else:
        vcol = np.array(start[0], copy=True)
        ccol = np.array(start[1], copy=True)
    qc = _quantise(A.cx)
    qr = _quantise(A.rx)

    for _ in range(max_rounds):
        csig = []
        for i in range(A.m):
            s, e = A.rp[i], A.rp[i + 1]
            csig.append((int(ccol[i]),
                         tuple(sorted(zip(vcol[A.ri[s:e]].tolist(),
                                          qr[s:e].tolist())))))
        ccol_new = _relabel(csig)

        vsig = []
        for j in range(A.n):
            s, e = A.cp[j], A.cp[j + 1]
            vsig.append((int(vcol[j]),
                         tuple(sorted(zip(ccol_new[A.ci[s:e]].tolist(),
                                          qc[s:e].tolist())))))
        vcol_new = _relabel(vsig)

        stable = (np.unique(vcol_new).size == np.unique(vcol).size
                  and np.unique(ccol_new).size == np.unique(ccol).size)
        vcol, ccol = vcol_new, ccol_new
        if stable:
            break
    return vcol, ccol


# --------------------------------------------------------------------------- #
# exact verification                                                           #
# --------------------------------------------------------------------------- #


def _row_body(prob, i, perm_var=None):
    A = prob.A
    s, e = A.rp[i], A.rp[i + 1]
    cols = A.ri[s:e]
    if perm_var is not None:
        cols = perm_var[cols]
    order = np.argsort(cols, kind="stable")
    return tuple(zip(np.asarray(cols)[order].tolist(),
                     _quantise(A.rx[s:e][order]).tolist()))


def verify_permutation(prob, perm_var, perm_con) -> bool:
    """Is ``(perm_var, perm_con)`` an automorphism of the model? Exact test.

    Everything the optimisation problem consists of must be preserved: the
    objective, variable bounds and kinds, row bounds, and every matrix entry.
    Nothing here trusts the search that proposed the permutation, which is what
    makes an incomplete or heuristic search safe to use.
    """
    n, m = prob.n, prob.m
    perm_var = np.asarray(perm_var, dtype=np.int64)
    perm_con = np.asarray(perm_con, dtype=np.int64)
    if perm_var.shape[0] != n or perm_con.shape[0] != m:
        return False
    if not np.array_equal(np.sort(perm_var), np.arange(n)):
        return False
    if not np.array_equal(np.sort(perm_con), np.arange(m)):
        return False

    if not np.array_equal(_quantise(prob.c[perm_var]), _quantise(prob.c)):
        return False
    if not np.array_equal(prob.kind[perm_var], prob.kind):
        return False
    for arr in (prob.col_lb, prob.col_ub):
        if not np.array_equal(_quantise(_clip(arr)[perm_var]),
                              _quantise(_clip(arr))):
            return False
    for arr in (prob.row_lb, prob.row_ub):
        if not np.array_equal(_quantise(_clip(arr)[perm_con]),
                              _quantise(_clip(arr))):
            return False

    for i in range(m):
        if _row_body(prob, i, perm_var) != _row_body(prob, int(perm_con[i])):
            return False
    return True


# --------------------------------------------------------------------------- #
# generator search                                                             #
# --------------------------------------------------------------------------- #


def _bijection_from(colA, colB):
    """Read a permutation off two discrete colourings, or return None."""
    k = colA.shape[0]
    if np.unique(colA).size != k or np.unique(colB).size != k:
        return None
    hi = int(max(colA.max(initial=0), colB.max(initial=0))) + 1
    pos = np.full(hi, -1, dtype=np.int64)
    pos[colB] = np.arange(k)
    out = pos[colA]
    if (out < 0).any():
        return None
    return out


def _first_nonsingleton(col):
    """Colour and members of the first cell with more than one element."""
    order = np.argsort(col, kind="stable")
    i = 0
    n = col.shape[0]
    while i < n:
        j = i
        c = col[order[i]]
        while j < n and col[order[j]] == c:
            j += 1
        if j - i > 1:
            return int(c), [int(t) for t in order[i:j]]
        i = j
    return None, None


def _search_automorphism(prob, vA, cA, vB, cB, depth, branch, budget,
                         deadline=None):
    """Drive two individualised colourings towards a common discrete refinement.

    One round of individualisation is not enough on the cases that matter. With
    ``k`` identical objects, individualising one leaves the other ``k-1``
    mutually indistinguishable and the colouring never becomes discrete, so a
    depth-1 search finds nothing precisely where the symmetry is largest. The
    search therefore recurses, and branches on the right-hand side because the
    image of the next individualised variable is not determined.
    """
    if budget[0] <= 0:
        return None
    if deadline is not None and time.perf_counter() > deadline:
        budget[0] = 0
        return None
    budget[0] -= 1

    vA, cA = refine_colours(prob, start=(vA, cA))
    vB, cB = refine_colours(prob, start=(vB, cB))

    pv = _bijection_from(vA, vB)
    if pv is not None:
        pc = _bijection_from(cA, cB)
        if pc is not None and verify_permutation(prob, pv, pc):
            return pv, pc
        return None
    if depth <= 0:
        return None

    colour, cell = _first_nonsingleton(vA)
    if cell is None:
        return None
    a = cell[0]
    newc = int(max(vA.max(initial=0), vB.max(initial=0))) + 1

    vA2 = vA.copy()
    vA2[a] = newc
    images = [int(t) for t in np.flatnonzero(vB == colour)][:branch]
    for b in images:
        vB2 = vB.copy()
        vB2[b] = newc
        got = _search_automorphism(prob, vA2, cA, vB2, cB,
                                   depth - 1, branch, budget, deadline)
        if got is not None:
            return got
    return None


def find_generators(prob, max_generators: int = 40, time_limit: float = 2.0,
                    max_pairs_per_class: int = 40, depth: int = 6,
                    branch: int = 4, node_budget: int = 200):
    """Search for verified automorphisms by individualisation-refinement.

    Returns ``(generators, tested, timed_out)``.
    """
    t0 = time.perf_counter()
    gens: list[np.ndarray] = []
    vcol, ccol = refine_colours(prob)
    n = prob.n
    tested = 0

    order = np.argsort(vcol, kind="stable")
    start = 0
    while start < n and len(gens) < max_generators:
        end = start
        c = vcol[order[start]]
        while end < n and vcol[order[end]] == c:
            end += 1
        members = [int(j) for j in order[start:end]]
        start = end
        if len(members) < 2:
            continue
        if time.perf_counter() - t0 > time_limit:
            return gens, tested, True

        v = members[0]
        newcol = int(vcol.max(initial=0)) + 1
        vA = vcol.copy()
        vA[v] = newcol

        for w in members[1:max_pairs_per_class + 1]:
            if time.perf_counter() - t0 > time_limit:
                return gens, tested, True
            if len(gens) >= max_generators:
                break
            tested += 1
            vB = vcol.copy()
            vB[w] = newcol
            got = _search_automorphism(prob, vA, ccol, vB, ccol,
                                       depth, branch, [node_budget],
                                       deadline=t0 + time_limit)
            if got is not None and int(got[0][v]) == w:
                gens.append(got[0])
    return gens, tested, False


def orbits_from(generators, n):
    """Union-find orbits of the group generated by ``generators``."""
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for g in generators:
        for j in range(n):
            a, b = find(j), find(int(g[j]))
            if a != b:
                parent[a] = b
    buckets: dict = {}
    for j in range(n):
        buckets.setdefault(int(find(j)), []).append(j)
    return sorted((v for v in buckets.values() if len(v) > 1),
                  key=len, reverse=True)


# --------------------------------------------------------------------------- #
# fully interchangeable sets (the stronger, rarer case)                        #
# --------------------------------------------------------------------------- #


def _transposition(n, m, j, k):
    pv = np.arange(n)
    pv[j], pv[k] = k, j
    return pv, np.arange(m)


def _interchangeable_sets(prob, orbits, time_limit, t0):
    """Orbits on which *every* pairwise transposition is itself a symmetry.

    Rare but real: identical items sharing a single row. When it holds, a full
    ordering chain can be imposed rather than one inequality per generator.
    """
    out = []
    for orb in orbits:
        if len(orb) < 2 or len(orb) > 200:
            continue
        if time.perf_counter() - t0 > time_limit:
            break
        a = orb[0]
        grp = [a]
        for b in orb[1:]:
            pv, pc = _transposition(prob.n, prob.m, a, b)
            if verify_permutation(prob, pv, pc):
                grp.append(b)
        if len(grp) >= 2:
            out.append(sorted(grp))
    return out


# --------------------------------------------------------------------------- #
# detection and breaking                                                       #
# --------------------------------------------------------------------------- #


def detect_symmetry(prob, time_limit: float = 2.0,
                    max_generators: int = 40) -> SymmetryInfo:
    """Find verified automorphisms, their orbits, and interchangeable sets."""
    t0 = time.perf_counter()
    info = SymmetryInfo()
    if prob.n < 2 or prob.A.nnz == 0:
        return info

    vcol, _ = refine_colours(prob)
    info.colour_classes = int(np.unique(vcol).size)

    gens, tested, to = find_generators(prob, max_generators=max_generators,
                                       time_limit=time_limit)
    info.generators = gens
    info.candidates_tested = tested
    info.timed_out = to
    info.orbits = orbits_from(gens, prob.n)
    info.interchangeable = _interchangeable_sets(
        prob, info.orbits, time_limit, t0)
    info.seconds = time.perf_counter() - t0
    return info


def breaking_constraints(info: SymmetryInfo, integer_mask=None,
                         max_rows: int = 5000):
    """Valid static symmetry-breaking rows, as ``(idx, val, rhs)`` triples.

    One inequality per generator, ``x_j >= x_{pi(j)}`` at the smallest moved
    index, plus a full ordering chain for any fully interchangeable set.
    """
    out = []
    seen = set()

    for grp in info.interchangeable:
        members = [j for j in grp
                   if integer_mask is None or integer_mask[j]]
        for a, b in zip(members, members[1:]):
            if len(out) >= max_rows:
                return out
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((np.array([a, b], dtype=IDX),
                        np.array([1.0, -1.0], dtype=VAL), 0.0))

    for g in info.generators:
        moved = np.flatnonzero(g != np.arange(g.shape[0]))
        if moved.size == 0:
            continue
        j = int(moved[0])
        k = int(g[j])
        if integer_mask is not None and not (integer_mask[j] and integer_mask[k]):
            continue
        if (j, k) in seen or (k, j) in seen:
            continue
        seen.add((j, k))
        if len(out) >= max_rows:
            break
        out.append((np.array([j, k], dtype=IDX),
                    np.array([1.0, -1.0], dtype=VAL), 0.0))
    return out
