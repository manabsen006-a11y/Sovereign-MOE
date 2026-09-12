"""Mixed-integer quadratic programming: branch-and-bound over convex QPs.

The engine has had a convex QP solver and a branch-and-bound tree for a while
and no way to put them together, so a model with both a quadratic objective and
an integer variable was **refused** rather than approximated -- which was the
right refusal, since solving it as a MILP silently discards the quadratic term
and returns a confident wrong number. This is the missing join: the same
search, with a QP at every node instead of an LP.

Why it is a separate tree
-------------------------
:mod:`sovopt.mip.tree` is built around an LP node solver and everything that
comes with one -- a basis to warm-start from, a dual ray to learn a conflict
clause from, cuts separated from a tableau. A first-order QP has none of those:
no basis, no ray, no tableau. Bolting a second node type into that tree would
have meant threading "is this an LP or a QP" through every one of those
mechanisms to have most of them switched off. What is here instead is the part
of branch-and-bound that does not depend on the relaxation being an LP --
propagation, best-bound selection, branching, incumbent validation -- and
nothing else.

The bound is rigorous, and how
------------------------------
On the LP path a node bound is made safe by the Neumaier-Shcherbina correction
in :mod:`sovopt.mip.safebound`, so an unconverged relaxation costs search
effort and never a wrong answer. The first version of this tree had no
equivalent: its bound was whatever the proximal QP solver reported, good to
that solver's tolerance and no better, and it said so.

:func:`~sovopt.mip.safebound.safe_qp_bound` closes that. Convexity puts the
tangent plane at the solver's iterate *below* the objective everywhere, and the
tangent plane is linear, so the Neumaier-Shcherbina bound applies to it as-is.
The result is a lower bound that holds for **any** iterate and **any** dual
vector -- converged, half-converged or stopped by a time limit -- and is exact
at an optimal pair. Every node here is pruned on that bound and never on the
solver's own objective, which is why a relaxation solved deliberately badly
still produces the brute-force optimum (see the tests).

Two things remain from before, because they are cheap and independent: every
incumbent is validated against the *original* model -- feasibility,
integrality and objective recomputed from scratch -- and the reported dual
bound is the minimum over every leaf's certified bound, so the gap the caller
sees is one the caller could recompute.

Rigour is conditional on ``Q ⪰ 0``, which is the premise of this path rather
than something it can prove in general: the QP solver checks it by estimating
the smallest eigenvalue, and a non-convex ``Q`` is refused there. Where the
matrix the tree actually relaxes factorises as ``LDLᵀ`` with every pivot
positive after ``εI`` is taken off it -- Sylvester's criterion, exact up to
rounding of order ``ε`` -- the bound is unconditional, and
``info["convexity_certified"]`` says which case a solve was.

The bound is tighter than the relaxation's, and how
---------------------------------------------------
Three things happen at a node beyond solving its QP, all of them cheap next
to the solve and all of them rigorous by the same certificate.

*The binary diagonal is shifted into the linear term.* For a binary
``x_i``, ``x_i² = x_i``, so ``½ Q_ii x_i²`` and ``½ Q_ii x_i`` agree on every
integer point -- and on the relaxation, where ``0 ≤ x_i ≤ 1``, the
difference ``½ d (x_i − x_i²)`` is non-negative. Moving a diagonal amount
``d`` from ``Q`` to ``c`` therefore leaves every integer point's objective
alone and raises the relaxation's objective everywhere else, so the
relaxation's minimum -- the node bound -- can only go up. The one
constraint is that ``Q − D`` stay positive semidefinite, or the relaxation
stops being convex and the bound stops being a bound. This is the
smallest-eigenvalue shift of Hammer & Rubin, and the uniform case of
Billionnet & Elloumi's *quadratic convex reformulation*, whose optimal
non-uniform ``D`` needs a semidefinite program this repository does not
have. The largest uniform shift is found by bisection with the ``LDLᵀ``
as the oracle: ``Q − dP_B − εI`` factorises with every pivot positive or
it does not. Measured on binary QPs whose ``Q`` has a smallest eigenvalue
of 0.1 (``tests/test_miqp.py``'s shape at 30 variables): the node count
halves -- 212 to 109, 1001 to 699, 538 to 293, 189 to 97 -- for the same
optimum. Measured on QPLIB's binary instances: nothing, and provably so.
10050's Hessian has rank 36 of 148 and 10056's 31 of 172, and every null
vector touches every column, so ``zᵀ(Q − D)z = −Σ d_i z_i² ≥ 0`` forces
every ``d_i ≤ 0``: no positive diagonal shift of any kind, uniform or
not, keeps a rank-deficient ``Q`` of that shape semidefinite. The shift
is a device for definite Hessians; those instances' bounds are their
relaxation's own.

*A child is bounded before it is solved.* The certified bound is a
formula in the node's box and the solver's pair ``(x̂, y)``; tightening a
box changes one column term of that formula, so the parent's pair gives
each child a valid bound in ``O(1)`` from the parent's reduced costs
``d = c + Qx̂ − Aᵀy``. A child whose free bound already meets the
incumbent is never solved, and never pushed. Measured: it never fires.
The tree branches on the most fractional variable, whose reduced cost at
the relaxation's optimum is zero by complementarity, so the free bound is
the parent's. It would pay under a branching rule that picks variables
with a reduced cost, and it costs one sparse product per node.

*Reduced costs fix binaries.* The same arithmetic applied to a binary at
its bound says what forcing it to the *other* value would cost: at least
``|d_j|`` on top of the node's bound. When that meets the incumbent the
other value cannot improve on what is known, and the variable is fixed
for the whole subtree -- the QP equivalent of reduced-cost fixing, valid
because the certified bound is valid for any pair. Measured: hundreds of
fixings per solve and a few percent fewer nodes (117 to 109, 701 to
699); the variables it fixes were at their bounds already.

Both the tree's heuristics run here too: fix-and-propagate and the
feasibility jump need only the rows, so they apply to a QP model
unchanged, and on the four QPLIB instances with constraints (3980, 3913,
3871, 4270) they find the first incumbent where nothing had -- at 38%,
27%, 500% and 12% above the published values, which is where the
relaxation's weakness shows next.

References
----------
Land & Doig, "An automatic method of solving discrete programming problems",
  Econometrica 28 (1960) 497-520 -- branch and bound.
Fletcher & Leyffer, "Numerical experience with lower bounds for MIQP
  branch-and-bound", SIAM J. Optimization 8 (1998) 604-616 -- QP relaxations as
  node bounds, and how much the bound quality is worth.
Bonami, Kilinç & Linderoth, "Algorithms and software for convex mixed integer
  nonlinear programs", in *Mixed Integer Nonlinear Programming*, Springer
  (2012) 1-39 -- the convex MINLP branch-and-bound this specialises.
Hammer & Rubin, "Some remarks on quadratic programming with 0-1 variables",
  RAIRO 4 (1970) 67-79 -- the smallest-eigenvalue diagonal shift for binary
  quadratics.
Billionnet & Elloumi, "Using a mixed integer quadratic programming solver for
  the unconstrained quadratic 0-1 problem", Math. Prog. 109 (2007) 55-68 --
  quadratic convex reformulation: the shift as a bound-strengthening device,
  and the SDP that would pick it per variable.
Nemhauser & Wolsey, *Integer and Combinatorial Optimization*, Wiley (1988),
  II.4 -- reduced-cost fixing.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.problem import ObjSense, Problem, Solution, Status
from ..core.sparse import IDX, VAL, SparseMatrix, coo_to_csc
from ..core.tolerances import DEFAULT, INF, Tolerances
from ..numerics.ldl import LDLSingular, LDLSymbolic
from ..numerics.ordering import amd_order
from ..qp import NotConvexError, QPParams, solve_qp
from .heuristics import feasibility_jump, fix_and_propagate
from .propagate import propagate
from .safebound import safe_qp_bound
from .tree import _Pseudocost

__all__ = ["MIQPParams", "solve_miqp"]


@dataclass
class MIQPParams:
    """Search settings. The QP solver's own settings live in ``qp``."""

    time_limit: float = 300.0
    node_limit: int = 100_000

    gap_rel: float = 1e-4
    gap_abs: float = 1e-9

    propagate_rounds: int = 6
    integrality: float = 1e-6

    binary_shift: bool = True
    """Move the largest uniform diagonal amount that keeps ``Q`` positive
    semidefinite from the binaries' quadratic term to their linear term
    (module header). Off, the relaxation is the model's own."""

    prebound: bool = True
    """Bound children from the parent's pair before solving them, and fix
    binaries by reduced cost (module header)."""

    heuristics: bool = True
    """Fix-and-propagate and the feasibility jump at the root, and again
    every ``heuristic_restart`` nodes while there is no incumbent."""
    heuristic_restart: int = 50

    branching: str = "pseudocost"
    """``"pseudocost"``: each variable's bound gain per unit of fractional
    move, learnt from the children actually solved, scored as Achterberg's
    product; unlearnt variables take the running average. ``"fractional"``:
    the most fractional variable, which is what the tree did before."""

    qp: QPParams = field(default_factory=QPParams)
    tol: Tolerances = field(default_factory=lambda: DEFAULT)
    verbose: bool = False


def _psd_with_margin(Q: SparseMatrix, shift, eps: float) -> bool:
    """Does ``Q − diag(shift) − εI`` factorise as ``LDLᵀ`` with every pivot
    positive, on the columns ``Q`` touches? Sylvester's criterion, exact up
    to rounding of order ``ε``: a yes certifies ``Q − diag(shift) ⪰ εI`` on
    that support, and the columns outside it are zero rows and columns of
    ``Q``, which add nothing either way."""
    n = Q.n
    support = np.flatnonzero(np.diff(Q.cp) > 0)
    if support.size == 0:
        return bool((np.asarray(shift) <= 0.0).all())
    pos = np.full(n, -1, dtype=np.int64)
    pos[support] = np.arange(support.size)
    k = support.size
    cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(Q.cp))
    r = np.concatenate([pos[Q.ci.astype(np.int64)], np.arange(k)])
    c = np.concatenate([pos[cols], np.arange(k)])
    v = np.concatenate([Q.cx, -np.asarray(shift, dtype=VAL)[support] - eps])
    cp, ci, cx = coo_to_csc(r, c, v, k, k)
    # a diagonal that summed to exactly zero vanishes from the pattern and
    # is a zero pivot the factorisation refuses, which is the right answer
    try:
        f = LDLSymbolic(cp, ci, k, amd_order(cp, ci, k)).factor(cx)
    except LDLSingular:
        return False
    return f.n_neg == 0 and bool((f.D > 0.0).all())


def _binary_shift(Q: SparseMatrix, binary, eps: float, rounds: int = 40):
    """The largest uniform ``d`` with ``Q − d·diag(binary) ⪰ εI``, by
    bisection on the factorisation. Returns ``(d, certified)`` where
    ``certified`` says ``Q`` itself passed the test."""
    diag = Q.diagonal()
    w = (np.asarray(binary, dtype=bool) & (diag > 0.0)).astype(VAL)
    certified = _psd_with_margin(Q, np.zeros(Q.n, dtype=VAL), eps)
    if not certified or not w.any():
        return 0.0, certified
    cap = float(diag[w > 0.0].min(initial=np.inf))   # d ≤ Q_ii is necessary
    if not np.isfinite(cap) or cap <= 0.0:
        return 0.0, certified
    lo, hi = 0.0, cap
    if _psd_with_margin(Q, hi * w, eps):
        return hi, certified
    for _ in range(rounds):
        mid = 0.5 * (lo + hi)
        if _psd_with_margin(Q, mid * w, eps):
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-6 * cap:
            break
    return lo, certified


def _shifted(prob: Problem, d: float, binary) -> Problem:
    """``½xᵀ(Q − dP)x + (c + ½d·1_B)ᵀx``: the same value at every integer
    point, a larger one everywhere else on the relaxation."""
    out = prob.copy()
    if d <= 0.0:
        return out
    Q = prob.Q.copy()
    n = prob.n
    cols = np.repeat(np.arange(n, dtype=IDX), np.diff(Q.cp))
    r = np.concatenate([Q.ci.astype(np.int64), np.arange(n)])
    c = np.concatenate([cols.astype(np.int64), np.arange(n)])
    v = np.concatenate([Q.cx, -d * np.asarray(binary, dtype=VAL)])
    cp, ci, cx = coo_to_csc(r, c, v, n, n)
    out.Q = SparseMatrix(n, n, cp, ci, cx)
    out.c = prob.c + 0.5 * d * np.asarray(binary, dtype=VAL)
    return out


def _relaxation(prob: Problem, lo, hi, params: MIQPParams, deadline: float,
                tighten: int = 0, fallback: Problem | None = None):
    """Solve the node's continuous relaxation over ``[lo, hi]``.

    ``tighten`` re-solves with the tolerances divided by ``100**tighten`` and
    the iteration budget multiplied by ``4**tighten``; see the loop for when
    that is needed.
    """
    node = prob.copy()
    node.col_lb = lo
    node.col_ub = hi
    node.kind = np.zeros(prob.n, dtype=node.kind.dtype)   # relax integrality
    f = 100.0 ** tighten
    qp = QPParams(**{**vars(params.qp),
                     "eps_abs": params.qp.eps_abs / f,
                     "eps_rel": params.qp.eps_rel / f,
                     "max_iter": params.qp.max_iter * 4 ** tighten,
                     "time_limit": max(0.0, deadline - time.perf_counter())})
    try:
        return solve_qp(node, qp)
    except NotConvexError:
        if fallback is None:
            raise
        # The shifted model sits at the edge of convexity by construction
        # (its smallest eigenvalue is the margin), and the dispatcher's
        # power-iteration estimate is an estimate. The unshifted model was
        # accepted at the root; its bound is the weaker one and still valid.
        node = fallback.copy()
        node.col_lb, node.col_ub = lo, hi
        node.kind = np.zeros(prob.n, dtype=node.kind.dtype)
        return solve_qp(node, qp)


def solve_miqp(prob: Problem, params: MIQPParams | None = None) -> Solution:
    """Solve a convex mixed-integer quadratic program.

    A non-convex ``Q`` is refused by the QP solver rather than approximated,
    and a maximisation is converted by negating both ``c`` and ``Q`` -- so a
    maximisation is solvable exactly when the *negated* ``Q`` is positive
    semidefinite, which is the honest condition.
    """
    params = params or MIQPParams()
    if params.branching not in ("pseudocost", "fractional"):
        raise ValueError(f"unknown branching rule {params.branching!r}")
    tol = params.tol
    t0 = time.perf_counter()
    deadline = t0 + params.time_limit

    if prob.Q is None:
        raise ValueError("solve_miqp needs a quadratic objective; this model "
                         "has none, so use solve_mip")
    if not prob.is_mip:
        return solve_qp(prob, params.qp)

    flip = prob.sense == ObjSense.MAXIMISE
    work = prob
    if flip:
        work = prob.copy()
        work.c = -work.c
        work.obj_offset = -work.obj_offset
        work.Q = work.Q.copy()
        work.Q.cx = -work.Q.cx
        work.Q.rx = -work.Q.rx
        work.sense = ObjSense.MINIMISE

    int_mask = work.integer_mask
    binary = int_mask & (work.col_lb >= -1e-9) & (work.col_ub <= 1.0 + 1e-9)
    q_scale = float(np.abs(work.Q.cx).max(initial=0.0))
    eps_psd = 1e-9 * max(1.0, q_scale)
    shift, certified = _binary_shift(work.Q, binary if params.binary_shift
                                     else np.zeros(work.n, dtype=bool), eps_psd)
    if not certified:
        from ..globalopt.alphabb import gershgorin_alpha
        certified = bool((gershgorin_alpha(work.Q, work.col_lb,
                                           work.col_ub) == 0.0).all())
    shifted_cols = binary & (work.Q.diagonal() > 0.0)
    relax = _shifted(work, shift, shifted_cols)
    root = propagate(work.A, work.row_lb, work.row_ub,
                     work.col_lb.astype(VAL), work.col_ub.astype(VAL),
                     int_mask, max_rounds=params.propagate_rounds,
                     feas_tol=tol.primal_feas)
    if root.infeasible:
        return Solution(status=Status.INFEASIBLE, nodes=0,
                        time=time.perf_counter() - t0, method="miqp-bb")

    incumbent = np.inf
    best_x = None
    nodes = 0
    status = Status.NODE_LIMIT
    leaf_lb = np.inf        # min certified bound over every closed node
    undecided = 0           # nodes closed without a bound that closes them
    prebound_pruned = 0     # children never solved: their free bound met the incumbent
    rc_fixed = 0            # binaries fixed by reduced cost
    heur_found = 0
    last_heur = 0

    def consider(x) -> bool:
        """Accept a candidate only if the *original* model agrees it is one."""
        nonlocal incumbent, best_x
        if x is None:
            return False
        cand = np.array(x, dtype=VAL, copy=True)
        idx = np.flatnonzero(int_mask)
        cand[idx] = np.round(cand[idx])
        np.clip(cand, prob.col_lb, prob.col_ub, out=cand)
        row_v, col_v, int_v = prob.violation(cand)
        if max(row_v, col_v) > 1e-6 or int_v > params.integrality:
            return False
        value = float(work.c @ cand) + work.obj_offset \
            + 0.5 * float(cand @ work.Q.matvec(cand))
        if value < incumbent - 1e-12:
            incumbent, best_x = value, cand
            return True
        return False

    def _heuristics(x0, lo, hi):
        """The row-only heuristics from one point; ``consider`` validates."""
        nonlocal heur_found
        if not params.heuristics:
            return
        for fn in (lambda: fix_and_propagate(work, x0, int_mask, lo, hi,
                                             tol.primal_feas),
                   lambda: feasibility_jump(work, int_mask, lo, hi, x0=x0)):
            if np.isfinite(incumbent) or time.perf_counter() > deadline:
                return
            try:
                cand = fn()
            except Exception:                            # noqa: BLE001
                cand = None
            if cand is not None and consider(cand):
                heur_found += 1

    def _column_terms(dvec, lo, hi):
        """Each column's share of the certified bound, ``min(d·lo, d·hi)``;
        ``-inf`` where a bound is missing on the side the sign needs."""
        pos = dvec > 0.0
        chosen = np.where(pos, lo, hi)
        bad = np.where(pos, lo <= -INF, hi >= INF) & (dvec != 0.0)
        return np.where(bad, -np.inf, dvec * np.where(bad, 0.0, chosen))

    # (bound, order, path, branch) -- order breaks ties so the heap is
    # deterministic; branch = (j, is_down, fraction moved) or None, so the
    # bound the child achieves can be credited to the decision that made it
    frontier: list = []
    heapq.heappush(frontier, (-np.inf, 0, (), None))
    order = 1
    pc = _Pseudocost(work.n)

    while frontier:
        if time.perf_counter() > deadline:
            status = Status.TIME_LIMIT
            break
        if nodes >= params.node_limit:
            status = Status.NODE_LIMIT
            break

        bound, _, path, branch = heapq.heappop(frontier)
        gap = max(params.gap_abs, params.gap_rel * abs(incumbent)) \
            if np.isfinite(incumbent) else 0.0
        if np.isfinite(incumbent) and bound >= incumbent - gap:
            leaf_lb = min(leaf_lb, bound)
            continue

        lo = root.lo.copy()
        hi = root.hi.copy()
        for j, is_lower, v in path:
            if is_lower:
                lo[j] = max(lo[j], v)
            else:
                hi[j] = min(hi[j], v)
        if np.any(lo > hi + tol.primal_feas):
            continue

        pr = propagate(work.A, work.row_lb, work.row_ub, lo, hi, int_mask,
                       max_rounds=2, feas_tol=tol.primal_feas)
        if pr.infeasible:
            continue
        lo, hi = pr.lo, pr.hi

        nodes += 1
        r = _relaxation(relax, lo, hi, params, deadline, fallback=work)
        if r.status in (Status.INFEASIBLE, Status.INFEASIBLE_OR_UNBOUNDED):
            continue
        if r.x is None:
            # The interior point withholds a point that fails its feasibility
            # cap at a time or iteration limit. Without a point there is no
            # certified bound and nothing to branch on: at the deadline the
            # search stops, otherwise the node is closed as undecided at its
            # parent's bound, which keeps the reported gap honest.
            if time.perf_counter() > deadline:
                status = Status.TIME_LIMIT
                break
            undecided += 1
            leaf_lb = min(leaf_lb, bound)
            continue
        x = np.asarray(r.x, dtype=VAL)
        # The bound is certified from the iterate and its duals, whatever the
        # solver's status -- never taken from the solver's own objective. It
        # is the *shifted* model's bound, which is a bound on the original
        # at every integer point (module header).
        y = r.y if r.y is not None else np.zeros(work.m, dtype=VAL)
        node_bound = safe_qp_bound(relax.A, relax.c, relax.Q, relax.row_lb,
                                   relax.row_ub, lo, hi, x, y, strict=True) \
            + relax.obj_offset
        if branch is not None and np.isfinite(bound):
            pc.update(branch[0], branch[2], node_bound - bound, branch[1])
        node_bound = max(node_bound, bound)      # a child is never looser than its parent
        if np.isfinite(incumbent) and node_bound >= incumbent - gap:
            leaf_lb = min(leaf_lb, node_bound)
            continue

        idx = np.flatnonzero(int_mask)
        frac = np.abs(x[idx] - np.round(x[idx]))
        consider(x)                      # a rounding is always worth a try
        if params.heuristics and not np.isfinite(incumbent) and (
                nodes == 1 or nodes - last_heur >= params.heuristic_restart):
            last_heur = nodes
            _heuristics(x, lo, hi)
            gap = max(params.gap_abs, params.gap_rel * abs(incumbent)) \
                if np.isfinite(incumbent) else 0.0
            if np.isfinite(incumbent) and node_bound >= incumbent - gap:
                leaf_lb = min(leaf_lb, node_bound)
                continue

        # The parent's pair prices every child: the certified bound is a sum
        # of column terms, and a child changes one of them.
        dvec = None
        terms = None
        if params.prebound:
            yy = np.where((y > 0.0) & (relax.row_lb <= -INF), 0.0, y)
            yy = np.where((yy < 0.0) & (relax.row_ub >= INF), 0.0, yy)
            dvec = relax.c + relax.Q.matvec(x) - relax.A.rmatvec(yy)
            terms = _column_terms(dvec, lo, hi)
            if np.isfinite(incumbent) and np.isfinite(terms).all():
                # reduced-cost fixing on the binaries still free here: the
                # other value costs at least |d_j| more than this node's
                # bound, and when that meets the incumbent it cannot win
                free_b = np.flatnonzero(binary & (hi - lo > 0.5))
                if free_b.size:
                    other = np.where(dvec[free_b] > 0.0, dvec[free_b],
                                     -dvec[free_b])       # cost of the flip
                    fix = free_b[node_bound + other >= incumbent - gap]
                    if fix.size:
                        up = dvec[fix] < 0.0            # cheap side is 1
                        lo[fix] = np.where(up, 1.0, 0.0)
                        hi[fix] = np.where(up, 1.0, 0.0)
                        rc_fixed += int(fix.size)
                        # the fixings hold for the whole subtree, so they
                        # travel with the path the children are built from
                        path = path + tuple((int(j), bool(u), 1.0 if u else 0.0)
                                            for j, u in zip(fix, up))

        def child_bound(j, is_lower, v):
            """The child's certified bound from the parent's pair, free."""
            if terms is None or not np.isfinite(terms[j]):
                return node_bound
            l2, h2 = (max(lo[j], v), hi[j]) if is_lower else (lo[j], min(hi[j], v))
            if l2 > h2:
                return np.inf
            t2 = _column_terms(dvec[j:j + 1], np.array([l2]), np.array([h2]))[0]
            if not np.isfinite(t2):
                return node_bound
            return max(node_bound, node_bound - terms[j] + t2)

        def push(child, moved=0.0):
            nonlocal order, prebound_pruned, leaf_lb
            b = child_bound(*child)
            if np.isfinite(incumbent) and b >= incumbent - gap:
                prebound_pruned += 1
                leaf_lb = min(leaf_lb, b)
                return
            heapq.heappush(frontier, (b, order, path + (child,),
                                      (child[0], not child[1], moved)))
            order += 1

        if frac.size == 0 or frac.max() <= params.integrality:
            # An integral iterate is not a finished node. The LP tree learnt
            # this the hard way (README, "A valid bound is not a valid
            # search"): a bound licenses *discarding* a node, and only a bound
            # that meets the incumbent licenses *closing* one. An unconverged
            # iterate can look integral while the node's real optimum is a
            # different integer point, so closing on it loses that point and
            # nothing downstream can notice.
            closed = np.isfinite(incumbent) and node_bound >= incumbent - gap
            if not closed:
                unfixed = idx[hi[idx] - lo[idx] > 0.5]
                if unfixed.size:
                    # split an integer domain at the iterate: {<= v} ∪ {>= v+1}
                    # covers every integer, so nothing is lost and the domain
                    # shrinks either way
                    j = int(unfixed[int(np.argmax(hi[unfixed] - lo[unfixed]))])
                    v = float(np.round(x[j]))
                    if v >= hi[j]:
                        v -= 1.0
                    for child in ((j, False, v), (j, True, v + 1.0)):
                        push(child)
                    continue
                # every integer is fixed: this is a continuous QP, and the
                # only way to close it is to solve it better
                for tighten in (1, 2):
                    r = _relaxation(relax, lo, hi, params, deadline, tighten,
                                    fallback=work)
                    if r.x is None:
                        break
                    x = np.asarray(r.x, dtype=VAL)
                    y = r.y if r.y is not None else np.zeros(work.m, dtype=VAL)
                    nb = safe_qp_bound(relax.A, relax.c, relax.Q, relax.row_lb,
                                       relax.row_ub, lo, hi, x, y, strict=True) \
                        + relax.obj_offset
                    node_bound = max(node_bound, nb)
                    consider(x)
                    gap = max(params.gap_abs, params.gap_rel * abs(incumbent)) \
                        if np.isfinite(incumbent) else 0.0
                    if np.isfinite(incumbent) and node_bound >= incumbent - gap:
                        closed = True
                        break
                if not closed:
                    undecided += 1
            leaf_lb = min(leaf_lb, node_bound)
            continue

        cands = idx[frac > params.integrality]
        if params.branching == "pseudocost":
            fd = x[cands] - np.floor(x[cands])
            scores = np.array([pc.score(int(jj), float(a), float(1.0 - a))
                               for jj, a in zip(cands, fd)])
            j = int(cands[int(np.argmax(scores))])
        else:
            j = int(idx[int(np.argmax(frac))])
        v = float(x[j])
        f_down = v - np.floor(v)
        push((j, False, np.floor(v)), moved=f_down)
        push((j, True, np.ceil(v)), moved=1.0 - f_down)

        if params.verbose:
            print(f"  miqp nodes {nodes:>7d}  open {len(frontier):>6d}  "
                  f"bound {node_bound:< 14.8g}  incumbent "
                  f"{incumbent if np.isfinite(incumbent) else float('nan'):< 14.8g}"
                  f"  {time.perf_counter() - t0:6.1f}s")

    if not frontier and status == Status.NODE_LIMIT and not undecided:
        status = Status.OPTIMAL if best_x is not None else Status.INFEASIBLE
    # an exhausted search with an undecided node keeps NODE_LIMIT, as the LP
    # tree does: the incumbent is real, the certified bound below is real,
    # and the gap between them is exactly what was not proved

    # the global lower bound: nothing open or closed lies below it
    global_lb = min(leaf_lb, min((b for b, _, _, _ in frontier), default=np.inf))
    if best_x is None:
        return Solution(status=status if status != Status.NODE_LIMIT
                        else Status.NODE_LIMIT,
                        nodes=nodes, time=time.perf_counter() - t0,
                        method="miqp-bb").drop_objective_if_unsolved()

    obj = prob.objective(best_x)
    global_lb = min(global_lb, incumbent)
    sol = Solution(status=status, x=best_x, objective=obj, nodes=nodes,
                   time=time.perf_counter() - t0, method="miqp-bb")
    sol.dual_bound = -global_lb if flip else global_lb
    sol.info = {"nodes": nodes, "bound_is_rigorous": True,
                "convexity_certified": certified, "binary_shift": shift,
                "prebound_pruned": prebound_pruned, "rc_fixed": rc_fixed,
                "heuristic_incumbents": heur_found}
    return sol
