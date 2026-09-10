"""Conflict analysis: learning from infeasible subproblems.

When a node turns out infeasible, an ordinary branch-and-bound throws the fact
away and rediscovers it in every sibling that repeats the same mistake. Conflict
analysis instead asks *which* of the branching decisions were actually to blame,
and records a constraint forbidding that combination anywhere in the tree. It is
the idea that made SAT solvers work, carried across to MIP.

The certificate
---------------
A node is infeasible exactly when some vector ``y`` certifies it. Taking the
Neumaier-Shcherbina bound with a **zero objective** gives, for every feasible
``x`` of the node,

    0 >= L(y) = sum_i (y_i>0 ? y_i*rl_i : y_i*ru_i)
              + sum_j (d_j>0 ? d_j*l_j : d_j*u_j),      d = -Aᵀy

so ``L(y) > 0`` is a contradiction: the node is empty. This is the same
arithmetic as :mod:`sovopt.mip.safebound`, reused with ``c = 0``, and it holds
for *any* ``y`` -- the dual ray out of an infeasible LP is simply a good one.

Why the reason set is cheap to shrink
-------------------------------------
``L(y)`` is a **sum of independent per-variable terms**. Relaxing variable ``j``
from its node bound back to its root bound changes exactly one term, so testing
whether a branching decision was needed costs one subtraction rather than
another LP solve:

    L_without_j = L - term_j(node bounds) + term_j(root bounds)

If that is still positive, the decision on ``j`` played no part in the
contradiction and can be dropped. Sweeping every branched variable this way
reduces the reason to a small set, which is what makes the learned constraint
strong: a clause naming three decisions prunes vastly more of the tree than one
naming thirty.

The learned constraint
----------------------
For binary decisions, "these bounds cannot hold together" is a clause. With
``U`` the variables branched up (fixed to 1) and ``D`` those branched down
(fixed to 0) in the reason set:

    sum_{j in U} (1 - x_j)  +  sum_{j in D} x_j  >=  1

which rearranges to ``-sum_U x_j + sum_D x_j >= 1 - |U|``. It is globally valid
because the certificate is evaluated with **root** bounds everywhere except the
decisions it keeps: no feasible point of the original model can satisfy all of
them at once.

General-integer decisions are excluded. Their conflict is a bound *disjunction*
rather than a clause, which is not a linear constraint, and faking one would be
unsound.

Every learned clause is checked before use: the certificate must actually be
positive, and the clause must not be violated by the incumbent if one exists.

How often this fires, measured
------------------------------
Rarely, and for a structural reason worth stating. The tree propagates bounds
*before* solving a node's LP, so a node whose emptiness propagation can see is
discarded without the LP ever running -- and it is the LP that produces the dual
ray this module needs. Only infeasibilities propagation misses reach here.

On the MIPLIB subset: misc07 analysed 91 nodes, certified 46 and learned 42
clauses, cutting the tree from 3168 nodes to 2848. p0201, gr4x6 and gt2 analysed
**zero** nodes -- propagation had already caught everything.

The larger prize is therefore **propagation-based conflict analysis**: recording
which bound changes each propagation used, so the conflicts propagation finds
can be analysed too. That needs the propagator instrumented to carry reasons and
is not built.

Reason lengths also degrade on real models: a median of 2 on small synthetic
instances against an average of 14.6 on misc07. Long clauses prune little, so
the minimisation loop is doing less work than the synthetic numbers suggest.

References
----------
Achterberg, "Conflict analysis in mixed integer programming", Discrete
  Optimization 4 (2007) 4-20 -- the MIP formulation of the technique.
Achterberg, *Constraint Integer Programming*, PhD thesis, TU Berlin 2007, §11.
Marques-Silva & Sakallah, "GRASP: a search algorithm for propositional
  satisfiability", IEEE Trans. Computers 48 (1999) -- clause learning and the
  unique implication point.
Neumaier & Shcherbina, "Safe bounds in linear and mixed-integer linear
  programming", Math. Prog. 99 (2004) 283-296 -- the certificate arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.sparse import IDX, VAL
from ..core.tolerances import INF

__all__ = ["ConflictAnalyzer", "farkas_value", "certificate_terms"]


def certificate_terms(A, row_lb, row_ub, lo, hi, y):
    """``(row_total, per_variable_terms, d)`` of the infeasibility certificate.

    Returns ``None`` when any term is unbounded, in which case ``y`` certifies
    nothing and the caller must not use it.
    """
    y = np.asarray(y, dtype=VAL)
    d = -A.rmatvec(y)

    ypos = y > 0.0
    row_pick = np.where(ypos, row_lb, row_ub)
    row_bad = np.where(ypos, row_lb <= -INF, row_ub >= INF) & (y != 0.0)
    if row_bad.any():
        return None
    row_total = float(np.sum(row_pick * y))

    dpos = d > 0.0
    col_pick = np.where(dpos, lo, hi)
    col_bad = np.where(dpos, lo <= -INF, hi >= INF) & (d != 0.0)
    if col_bad.any():
        return None
    terms = col_pick * d
    return row_total, terms, d


def farkas_value(A, row_lb, row_ub, lo, hi, y) -> float:
    """``L(y)``. Strictly positive certifies that the node is empty."""
    got = certificate_terms(A, row_lb, row_ub, lo, hi, y)
    if got is None:
        return -np.inf
    row_total, terms, _ = got
    return float(row_total + terms.sum())


@dataclass
class ConflictAnalyzer:
    """Turns infeasible nodes into globally valid clauses."""

    root_lo: np.ndarray
    root_hi: np.ndarray
    binary: np.ndarray
    """Mask of variables a clause may mention -- binaries only."""

    min_positive: float = 1e-9
    max_clause: int = 30
    """Clauses longer than this prune too little to be worth a row."""

    max_propagation_vars: int = 40
    """Decisions on the path above which a propagation conflict is not
    analysed. The deletion filter costs one propagation per decision, so this
    bounds the work per conflict; a very deep node is also the least likely to
    yield a short clause."""

    learned: list = field(default_factory=list)
    analysed: int = 0
    certified: int = 0
    emitted: int = 0
    prop_analysed: int = 0
    prop_emitted: int = 0
    reason_lengths: list = field(default_factory=list)

    def analyse(self, A, row_lb, row_ub, node_lo, node_hi, y):
        """Analyse one infeasible node; returns a clause or ``None``.

        ``y`` is a candidate certificate -- typically the dual ray from the
        infeasible LP. Its sign convention does not matter: both are tried and
        whichever certifies is used, and if neither does the node is simply not
        learned from.
        """
        self.analysed += 1
        y = np.asarray(y, dtype=VAL)

        best = None
        for cand in (y, -y):
            got = certificate_terms(A, row_lb, row_ub, node_lo, node_hi, cand)
            if got is None:
                continue
            row_total, terms, d = got
            total = row_total + terms.sum()
            if total > self.min_positive and (best is None or total > best[0]):
                best = (total, terms, d)
        if best is None:
            return None
        self.certified += 1
        total, terms, d = best

        # Which variables were actually branched on at this node?
        branched = np.flatnonzero(
            (node_lo > self.root_lo + 1e-12) | (node_hi < self.root_hi - 1e-12))
        if branched.size == 0:
            return None

        # Root-bound value of each per-variable term. Relaxing j back to the
        # root changes only term j, so "was j needed?" is one subtraction.
        dpos = d > 0.0
        root_pick = np.where(dpos, self.root_lo, self.root_hi)
        root_bad = np.where(dpos, self.root_lo <= -INF,
                            self.root_hi >= INF) & (d != 0.0)
        root_terms = np.where(root_bad, -np.inf, root_pick * d)

        # Relax the least useful decisions first: the ones whose removal costs
        # the certificate least are the ones most likely to be droppable.
        delta = terms[branched] - root_terms[branched]
        order = branched[np.argsort(delta)]

        running = float(total)
        needed = []
        for j in order:
            j = int(j)
            if not np.isfinite(root_terms[j]):
                needed.append(j)
                continue
            trial = running - terms[j] + root_terms[j]
            if trial > self.min_positive:
                running = trial            # decision was not required
            else:
                needed.append(j)

        if not needed:
            return None
        self.reason_lengths.append(len(needed))
        clause = self._emit(needed, node_lo)
        if clause is not None:
            self.emitted += 1
        return clause

    def _emit(self, needed, node_lo):
        """Turn a set of required decisions into a globally valid clause.

        Shared by both analyses: what differs between them is *how* the
        required set is found, not what a conflict clause looks like once it
        is. A clause can only speak about binaries -- a general integer's
        conflict is a bound disjunction, which is not linear.
        """
        if not all(self.binary[j] for j in needed):
            return None
        if len(needed) > self.max_clause:
            return None

        up, down = [], []
        for j in needed:
            if node_lo[j] > self.root_lo[j] + 1e-12:
                up.append(j)               # forced to 1
            else:
                down.append(j)             # forced to 0
        if not up and not down:
            return None

        idx = np.array(up + down, dtype=IDX)
        val = np.array([-1.0] * len(up) + [1.0] * len(down), dtype=VAL)
        rhs = 1.0 - len(up)
        clause = (idx, val, float(rhs))
        self.learned.append(clause)
        return clause

    def analyse_propagation(self, A, row_lb, row_ub, node_lo, node_hi,
                            is_int, max_rounds: int = 2,
                            feas_tol: float = 1e-9):
        """Analyse a node that *propagation* refuted, before any LP sees it.

        This is the larger half of conflict analysis and the half that was
        missing. The tree propagates every node before solving it, and a node
        propagation can refute never reaches the LP that would produce a dual
        ray -- so it was discarded silently, learning nothing. Counted over the
        MIPLIB set at a 60 s limit, nodes killed by propagation against nodes
        killed by the LP:

            misc07    2,795  vs  94        gt2       1,886  vs  0
            flugpl    3,232  vs  655       p0201       904  vs  0

        On gt2 and p0201 the analyser was seeing *none* of the refutations.

        There is no dual ray here to build a certificate from, so the required
        set is found by **deletion filtering** instead: relax one decision back
        to its root bound and re-propagate; if the node is still infeasible the
        decision was not needed, and if it becomes feasible it was. What
        survives is an irreducible subset of the path's decisions that is
        infeasible on its own -- which is exactly what a conflict clause
        asserts. Soundness does not rest on the filter finding a *minimum* set,
        only an infeasible one, and every candidate is verified by an actual
        propagation rather than by an algebraic argument.

        Costs one propagation per decision on the path, which is why
        ``max_propagation_vars`` exists.
        """
        from .propagate import propagate

        branched = np.flatnonzero(
            (node_lo > self.root_lo + 1e-12) | (node_hi < self.root_hi - 1e-12))
        if branched.size == 0 or branched.size > self.max_propagation_vars:
            return None
        self.prop_analysed += 1

        def dead(lo, hi):
            return propagate(A, row_lb, row_ub, lo, hi, is_int,
                             max_rounds=max_rounds,
                             feas_tol=feas_tol).infeasible

        cur_lo = np.array(node_lo, dtype=VAL, copy=True)
        cur_hi = np.array(node_hi, dtype=VAL, copy=True)
        if not dead(cur_lo, cur_hi):
            return None                     # caller was wrong; learn nothing
        self.certified += 1

        # Relax the deepest decisions first. A clause over the decisions made
        # near the root prunes far more of the tree than one that only fires
        # after the same twenty branchings have been repeated.
        needed = []
        for j in reversed(branched.tolist()):
            j = int(j)
            trial_lo, trial_hi = cur_lo.copy(), cur_hi.copy()
            trial_lo[j] = self.root_lo[j]
            trial_hi[j] = self.root_hi[j]
            if dead(trial_lo, trial_hi):
                cur_lo, cur_hi = trial_lo, trial_hi     # not required
            else:
                needed.append(j)

        if not needed:
            return None
        self.reason_lengths.append(len(needed))
        clause = self._emit(needed, node_lo)
        if clause is not None:
            self.prop_emitted += 1
            self.emitted += 1
        return clause

    def excludes(self, clause, x) -> bool:
        """Would this clause reject a known-feasible point? A safety check."""
        idx, val, rhs = clause
        return float(val @ x[idx]) < rhs - 1e-6

    def stats(self) -> dict:
        return {
            "analysed": self.analysed,
            "certified": self.certified,
            "clauses": self.emitted,
            "propagation_analysed": self.prop_analysed,
            "propagation_clauses": self.prop_emitted,
            "avg_reason": (round(float(np.mean(self.reason_lengths)), 2)
                           if self.reason_lengths else 0.0),
        }
