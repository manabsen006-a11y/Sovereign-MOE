"""Non-convex quadratic programs to proven global optimality, by αBB.

    minimise  ½ xᵀQx + cᵀx     subject to   rl <= Ax <= ru,  l <= x <= u,
                                            x_j integral for j ∈ I,
                                            and Q **not** positive semidefinite.

Until this file the answer to that shape was a refusal, which was the right
answer for a convex solver and is no answer for a refinery: a blending margin
with a concave penalty, or a pooling objective with cross terms, is exactly
this. A local method returns *a* stationary point; the problem statement asks
for optimisation, and optimisation of a non-convex function means a bound.

The idea
--------
Every non-convex quadratic can be made convex by adding enough curvature to
its diagonal. αBB does that with a term that is *non-positive on the box*:

    f_α(x)  =  f(x)  +  Σ_j α_j (x_j − l_j)(x_j − u_j),        α_j >= 0

Each factor ``(x_j − l_j)(x_j − u_j)`` is at most zero for ``l_j <= x_j <= u_j``,
so ``f_α <= f`` on the box and its constrained minimum is a valid lower bound
on ``f``'s. The Hessian of ``f_α`` is ``Q + 2 diag(α)``, and α is chosen so
that this is positive semidefinite -- at which point the minimum of ``f_α``
over the node is a convex QP, which :mod:`sovopt.qp` solves. The gap between
``f`` and ``f_α`` is at most ``Σ α_j (u_j − l_j)² / 4``, quadratic in the box
width, so splitting a box and re-relaxing closes it, and the search is an
ordinary spatial branch-and-bound.

Certified, not estimated
------------------------
Two things could quietly break a global optimiser built this way, and both are
handled by certificates rather than by tolerances.

* **Is ``Q + 2 diag(α)`` really PSD?** α comes from the scaled Gershgorin
  circle theorem: with ``d_j = u_j − l_j``,

      α_i  =  max( 0,  −½ ( Q_ii − Σ_{j≠i} |Q_ij| d_j / d_i ) )

  makes every Gershgorin disc of ``D⁻¹(Q + 2 diag α)D`` sit in the right
  half-plane, and that matrix has the same eigenvalues as ``Q + 2 diag α``.
  This is a proof, not an eigenvalue estimate, and it costs one pass over
  ``Q``. The scaling by ``d`` is what makes it useful rather than merely valid:
  it lets a wide variable absorb curvature from a narrow one, where the gap
  it costs is smallest.
* **Is the node bound really a bound?** The convex relaxation is solved by a
  first-order method to a finite tolerance, so its objective is not one. The
  node is bounded instead by :func:`~sovopt.mip.safebound.safe_qp_bound`,
  which is valid for any iterate and any dual vector -- and because Gershgorin
  has certified convexity of the relaxation, that bound is unconditional.

Fixed variables are substituted out before a node is relaxed rather than
carried with a zero-width box. That is not tidiness: a fixed variable's
Gershgorin row would need an infinite shift to certify, and shifting it makes
the relaxation ill-conditioned for a first-order solver, while substituting it
turns its couplings into linear terms and the problem stays exactly as
well-conditioned as its free part.

What this is not
----------------
Not the default. It was built first and then measured against the McCormick
reformulation in :mod:`sovopt.globalopt.nonconvex_qp`, and on dense
indefinite ``Q`` it loses badly: a diagonal shift has to pay for every cross
term through the diagonal, where an envelope treats each product exactly. It
stays as ``relaxation="alphabb"`` because it needs no product variables --
``n²`` of them for a dense ``Q`` -- its bound is unconditional, and the table
that decided against it has to stay reproducible.

Not an active-set or interior-point local solver: incumbents come from the
relaxation's point and a projected-gradient polish, and the *bound* is what
does the proving. Not a method for unbounded variables: every variable that
appears in a non-convex coupling needs a finite box, because the underestimator
is built from the box, and a model that lacks one is refused with the variable
named rather than given a default.

References
----------
Androulakis, Maranas & Floudas, "αBB: A global optimization method for general
  constrained nonconvex problems", J. Global Optim. 7 (1995) 337-363.
Adjiman, Dallwig, Floudas & Neumaier, "A global optimization method, αBB, for
  general twice-differentiable constrained NLPs -- I. Theoretical advances",
  Computers & Chemical Engineering 22 (1998) 1137-1158 -- the scaled Gershgorin
  choice of α.
Neumaier & Shcherbina, "Safe bounds in linear and mixed-integer linear
  programming", Math. Prog. 99 (2004) 283-296 -- the certified node bound.
Horst & Tuy, *Global Optimization: Deterministic Approaches*, Springer 1996 --
  spatial branch-and-bound.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.problem import ObjSense, Problem, Solution, Status
from ..core.sparse import SparseMatrix, VAL
from ..core.tolerances import DEFAULT, INF, Tolerances
from ..mip.propagate import propagate
from ..mip.safebound import safe_qp_bound
from ..qp import QPParams, solve_qp

__all__ = ["AlphaBBParams", "gershgorin_alpha", "solve_alphabb"]


@dataclass
class AlphaBBParams:
    time_limit: float = 300.0
    node_limit: int = 100_000

    gap_rel: float = 1e-4
    gap_abs: float = 1e-9

    integrality: float = 1e-6
    branch_eps: float = 1e-7
    """A box narrower than this is not split any further."""

    propagate_rounds: int = 6
    marginal_reduction: bool = True
    """Tighten each node's box from the certified bound's reduced costs
    (Ryoo & Sahinidis) before branching. Off only to measure it."""
    polish_steps: int = 50
    """Projected-gradient steps from the relaxation point, looking for a
    better incumbent. Zero disables it; the bound does not depend on it."""

    qp: QPParams = field(default_factory=QPParams)
    tol: Tolerances = field(default_factory=lambda: DEFAULT)
    verbose: bool = False


# --------------------------------------------------------------------------- #
# the certificate of convexity                                                 #
# --------------------------------------------------------------------------- #


def gershgorin_alpha(Q: SparseMatrix, lo, hi, safety: float = 1e-12):
    """Per-variable shifts ``α >= 0`` with ``Q + 2 diag(α)`` certified PSD on
    the free variables, by the scaled Gershgorin theorem with ``d = hi − lo``.

    Fixed variables (``d_j = 0``) get ``α_j = 0`` and drop out of every other
    row's sum: their couplings are linear once they are substituted, which is
    what :func:`_reduce` does before a node is relaxed. A variable coupled to
    an unbounded one gets ``α = inf``, which the caller reports as a refusal.
    """
    n = Q.n
    lo = np.asarray(lo, dtype=VAL)
    hi = np.asarray(hi, dtype=VAL)
    d = hi - lo
    d = np.where((hi >= INF) | (lo <= -INF) | ~np.isfinite(d), np.inf, d)
    free = d > 0.0
    alpha = np.zeros(n, dtype=VAL)
    for i in range(n):
        if not free[i]:
            continue
        diag = 0.0
        off = 0.0
        for k in range(Q.cp[i], Q.cp[i + 1]):
            j = int(Q.ci[k])
            v = float(Q.cx[k])
            if j == i:
                diag += v
            elif free[j]:
                off += abs(v) * d[j]
        if not np.isfinite(off):
            alpha[i] = np.inf           # coupled to an unbounded variable
            continue
        if not np.isfinite(d[i]):
            # An unbounded variable cannot carry an underestimator term at
            # all, so it is admissible only if its row needs none: no coupling
            # to another free variable and non-negative curvature of its own.
            alpha[i] = 0.0 if (off == 0.0 and diag >= 0.0) else np.inf
            continue
        radius = off / d[i]
        # −½(diag − radius), padded so rounding in the two sums above cannot
        # leave the certificate a few ulps short
        lam = diag - radius
        pad = safety * (abs(diag) + radius) + safety
        alpha[i] = max(0.0, -0.5 * (lam - pad))
    return alpha


# --------------------------------------------------------------------------- #
# the node relaxation                                                          #
# --------------------------------------------------------------------------- #


class _Reduced:
    """A node's convex relaxation over its free variables only."""

    __slots__ = ("prob", "free", "fixed", "xfix", "alpha", "offset")

    def __init__(self, prob, free, fixed, xfix, alpha, offset):
        self.prob = prob
        self.free = free
        self.fixed = fixed
        self.xfix = xfix
        self.alpha = alpha
        self.offset = offset

    def expand(self, xf):
        """Scatter a free-variable vector back into the full space."""
        x = np.empty(self.free.size + self.fixed.size, dtype=VAL)
        x[self.free] = xf
        x[self.fixed] = self.xfix
        return x


def _take_cols(M: SparseMatrix, cols):
    """The CSC slices of ``cols``, concatenated: ``(cp, ci, cx)``."""
    starts = M.cp[cols]
    lens = M.cp[cols + 1] - starts
    total = int(lens.sum())
    cp = np.zeros(cols.size + 1, dtype=np.int64)
    np.cumsum(lens, out=cp[1:])
    if total == 0:
        return cp, np.zeros(0, dtype=M.ci.dtype), np.zeros(0, dtype=VAL)
    idx = np.repeat(starts - cp[:-1], lens) + np.arange(total)
    return cp, M.ci[idx], M.cx[idx]


def _reduce(work: Problem, lo, hi, alpha) -> _Reduced | None:
    """Substitute fixed variables out and add the αBB term on the free ones.

    Returns ``None`` when nothing is free -- the node is a single point.
    """
    n = work.n
    d = hi - lo
    free = np.flatnonzero(d > 0.0)
    fixed = np.flatnonzero(d <= 0.0)
    if free.size == 0:
        return None
    xfix = lo[fixed]
    pos = np.full(n, -1, dtype=np.int64)          # full index -> free index
    pos[free] = np.arange(free.size)
    xz = np.zeros(n, dtype=VAL)                   # the fixed part, in full space
    xz[fixed] = xfix

    # objective on the free part: ½ x_Fᵀ Q_FF x_F + (c_F + Q_FX x_X)ᵀ x_F + const
    cp, ci, cx = _take_cols(work.Q, free)
    keep = pos[ci] >= 0
    col_of = np.repeat(np.arange(free.size), np.diff(cp))
    # Q_FX x_X: entries of the free columns whose row is fixed
    c_F = work.c[free].copy()
    np.add.at(c_F, col_of[~keep], cx[~keep] * xz[ci[~keep]])
    # Q_FF: the kept entries, rows remapped
    counts = np.bincount(col_of[keep], minlength=free.size)
    cp_ff = np.zeros(free.size + 1, dtype=np.int64)
    np.cumsum(counts, out=cp_ff[1:])
    Q_FF = SparseMatrix(free.size, free.size, cp_ff, pos[ci[keep]], cx[keep].copy())
    qx = work.Q.matvec(xz)                         # Q x_X in full space
    const = float(work.obj_offset + work.c[fixed] @ xfix + 0.5 * xfix @ qx[fixed])

    # αBB: Σ α_j (x_j − l_j)(x_j − u_j) = Σ α_j x_j² − α_j (l_j + u_j) x_j + α_j l_j u_j
    a = alpha[free]
    if (a > 0.0).any():
        diag = SparseMatrix.from_dense(np.diag(2.0 * a)) if free.size <= 64 else None
        if diag is None:
            k = np.arange(free.size, dtype=np.int64)
            diag = SparseMatrix.from_triplets(k, k, 2.0 * a, free.size, free.size)
        Q_FF = SparseMatrix.from_triplets(
            np.concatenate([Q_FF.ci, diag.ci]),
            np.concatenate([np.repeat(np.arange(free.size), np.diff(Q_FF.cp)),
                            np.repeat(np.arange(free.size), np.diff(diag.cp))]),
            np.concatenate([Q_FF.cx, diag.cx]), free.size, free.size)
        c_F = c_F - a * (lo[free] + hi[free])
        const += float(np.sum(a * lo[free] * hi[free]))

    cp_a, ci_a, cx_a = _take_cols(work.A, free)
    shift = work.A.matvec(xz)
    red = Problem(A=SparseMatrix(work.m, free.size, cp_a, ci_a, cx_a), c=c_F,
                  Q=Q_FF,
                  row_lb=work.row_lb - shift, row_ub=work.row_ub - shift,
                  col_lb=lo[free].copy(), col_ub=hi[free].copy(),
                  obj_offset=const, name=work.name)
    return _Reduced(red, free, fixed, xfix, a, const)


def _polish(work: Problem, x, lo, hi, steps: int):
    """Projected gradient on the box from ``x``; keep it only if the rows
    still hold. A local improvement, never a bound."""
    if steps <= 0:
        return None
    g = work.Q.matvec(x) + work.c
    step = 1.0 / max(float(np.abs(g).max(initial=0.0)), 1e-8)
    best_x, best_f = None, work.objective(x)
    cur = x.copy()
    for _ in range(steps):
        g = work.Q.matvec(cur) + work.c
        cand = np.clip(cur - step * g, lo, hi)
        f = work.objective(cand)
        if f < best_f - 1e-12:
            row_v, _, _ = work.violation(cand)
            if row_v <= 1e-9:
                best_x, best_f = cand, f
            cur = cand
        else:
            step *= 0.5
            if step < 1e-12:
                break
    return best_x


def _marginal_reduce(red: _Reduced, x_hat, y, lo, hi, int_mask, cutoff: float,
                     out_lo, out_hi) -> int:
    """Tighten the node's box from the certified bound's own pieces.

    The bound is ``B = const + Σ_i rowmin_i + Σ_j colmin_j`` with
    ``colmin_j = min(d_j l_j, d_j u_j)``, ``d = Qx̂ + c − Aᵀy`` on the reduced
    convexified problem. Every feasible ``x`` that could still improve on
    ``cutoff`` satisfies ``B − colmin_j + d_j x_j <= cutoff``, which is a new
    bound on ``x_j`` in the direction ``d_j`` points. This is the
    marginals-based reduction of Ryoo & Sahinidis, derived from the certified
    bound rather than from the solver's reported duals, so it is exactly as
    valid as the bound is. Returns the number of bounds tightened.
    """
    rp = red.prob
    x_hat = np.asarray(x_hat, dtype=VAL)
    qx = rp.Q.matvec(x_hat)
    g = rp.c + qx
    d = g - rp.A.rmatvec(y)
    # row terms and column terms exactly as safe_dual_bound forms them
    rpos = y > 0.0
    rbad = np.where(rpos, rp.row_lb <= -INF, rp.row_ub >= INF) & (y != 0.0)
    if rbad.any():
        return 0
    rowmin = y * np.where(rpos, rp.row_lb, rp.row_ub)
    cpos = d > 0.0
    cbad = np.where(cpos, rp.col_lb <= -INF, rp.col_ub >= INF) & (d != 0.0)
    if cbad.any():
        return 0
    colmin = d * np.where(cpos, rp.col_lb, rp.col_ub)
    B = rp.obj_offset - 0.5 * float(x_hat @ qx) + float(rowmin.sum()) + float(colmin.sum())
    slack = cutoff - B
    # pad against the rounding in B: a reduction is a *cut*, and a cut that is
    # a few ulps too deep removes the optimum
    slack += 1e-9 * (abs(cutoff) + abs(B) + np.abs(rowmin).sum() + np.abs(colmin).sum()) + 1e-12
    if slack < 0.0:
        return 0            # the node is prunable; the caller will see that
    n_tight = 0
    free = red.free
    for k in np.flatnonzero(np.abs(d) > 1e-12):
        j = int(free[k])
        dk = float(d[k])
        if dk > 0.0:
            new_hi = float(rp.col_lb[k]) + slack / dk
            if int_mask[j]:
                new_hi = np.floor(new_hi + 1e-9)
            if new_hi < out_hi[j] - 1e-9 * max(1.0, abs(out_hi[j])):
                out_hi[j] = max(new_hi, out_lo[j])
                n_tight += 1
        else:
            new_lo = float(rp.col_ub[k]) + slack / dk       # dk < 0
            if int_mask[j]:
                new_lo = np.ceil(new_lo - 1e-9)
            if new_lo > out_lo[j] + 1e-9 * max(1.0, abs(out_lo[j])):
                out_lo[j] = min(new_lo, out_hi[j])
                n_tight += 1
    return n_tight


# --------------------------------------------------------------------------- #
# the search                                                                   #
# --------------------------------------------------------------------------- #


class _Node:
    __slots__ = ("bound", "order", "lo", "hi")

    def __init__(self, bound, order, lo, hi):
        self.bound, self.order, self.lo, self.hi = bound, order, lo, hi

    def __lt__(self, other):
        return (self.bound, self.order) < (other.bound, other.order)


def solve_alphabb(prob: Problem,
                       params: AlphaBBParams | None = None) -> Solution:
    """Solve a (possibly non-convex, possibly mixed-integer) QP to global
    optimality within ``gap_rel``.

    Refuses a model in which a variable coupled through ``Q`` to a non-convex
    term has no finite bounds, naming it.
    """
    params = params or AlphaBBParams()
    tol = params.tol
    t0 = time.perf_counter()
    deadline = t0 + params.time_limit
    if prob.Q is None:
        raise ValueError("solve_alphabb needs a quadratic objective")

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
    root = propagate(work.A, work.row_lb, work.row_ub,
                     work.col_lb.astype(VAL), work.col_ub.astype(VAL),
                     int_mask, max_rounds=params.propagate_rounds,
                     feas_tol=tol.primal_feas)
    if root.infeasible:
        return Solution(status=Status.INFEASIBLE, nodes=0,
                        time=time.perf_counter() - t0, method="alphabb")

    alpha0 = gershgorin_alpha(work.Q, root.lo, root.hi)
    if not np.isfinite(alpha0).all():
        # name the unbounded variable, not the bounded neighbour it poisons
        bad = np.flatnonzero(~np.isfinite(alpha0))
        unb = bad[(root.hi[bad] >= INF) | (root.lo[bad] <= -INF)]
        j = int(unb[0]) if unb.size else int(bad[0])
        name = prob.col_names[j] if prob.col_names else f"x[{j}]"
        raise ValueError(
            f"non-convex QP needs finite bounds on every variable coupled to a "
            f"non-convex term; {name} has none"
            + (f" (and {unb.size - 1} more)" if unb.size > 1 else ""))
    root_convex = bool((alpha0 == 0.0).all())

    incumbent = np.inf
    best_x = None
    nodes = 0
    status = Status.NODE_LIMIT
    leaf_lb = np.inf
    undecided = 0

    def consider(x):
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
        value = work.objective(cand)
        if value < incumbent - 1e-12:
            incumbent, best_x = value, cand
            return True
        return False

    frontier = [_Node(-np.inf, 0, root.lo.copy(), root.hi.copy())]
    order = 1

    def relax(lo, hi, tighten=0):
        alpha = alpha0 if root_convex else gershgorin_alpha(work.Q, lo, hi)
        red = _reduce(work, lo, hi, alpha)
        if red is None:
            return None, None, None, None
        f = 100.0 ** tighten
        qp = QPParams(**{**vars(params.qp),
                         "eps_abs": params.qp.eps_abs / f,
                         "eps_rel": params.qp.eps_rel / f,
                         "max_iter": params.qp.max_iter * 4 ** tighten,
                         "time_limit": max(0.0, deadline - time.perf_counter())})
        r = solve_qp(red.prob, qp)
        if r.x is None:
            raise RuntimeError("QP relaxation returned no iterate")
        y = r.y if r.y is not None else np.zeros(work.m, dtype=VAL)
        rp = red.prob
        b = safe_qp_bound(rp.A, rp.c, rp.Q, rp.row_lb, rp.row_ub,
                          rp.col_lb, rp.col_ub, r.x, y, strict=True) + rp.obj_offset
        last["red"], last["xf"], last["y"] = red, np.asarray(r.x, dtype=VAL), y
        return red, red.expand(last["xf"]), b, alpha

    last = {}
    reductions = 0

    while frontier:
        if time.perf_counter() > deadline:
            status = Status.TIME_LIMIT
            break
        if nodes >= params.node_limit:
            status = Status.NODE_LIMIT
            break

        node = heapq.heappop(frontier)
        gap = max(params.gap_abs, params.gap_rel * abs(incumbent)) \
            if np.isfinite(incumbent) else 0.0
        if np.isfinite(incumbent) and node.bound >= incumbent - gap:
            leaf_lb = min(leaf_lb, node.bound)
            continue

        pr = propagate(work.A, work.row_lb, work.row_ub, node.lo, node.hi,
                       int_mask, max_rounds=2, feas_tol=tol.primal_feas)
        if pr.infeasible:
            continue
        lo, hi = pr.lo, pr.hi

        nodes += 1
        red, x, node_bound, alpha = relax(lo, hi)
        if red is None:
            # every variable fixed: the node is one point, and either it is
            # an incumbent or it is nothing
            row_v, col_v, int_v = work.violation(lo)
            if max(row_v, col_v) <= 1e-6 and int_v <= params.integrality:
                consider(lo)
                leaf_lb = min(leaf_lb, work.objective(lo))
            continue
        node_bound = max(node_bound, node.bound)
        if np.isfinite(incumbent) and node_bound >= incumbent - gap:
            leaf_lb = min(leaf_lb, node_bound)
            continue

        consider(x)
        consider(_polish(work, np.clip(x, lo, hi), lo, hi, params.polish_steps))
        gap = max(params.gap_abs, params.gap_rel * abs(incumbent)) \
            if np.isfinite(incumbent) else 0.0
        if np.isfinite(incumbent) and node_bound >= incumbent - gap:
            leaf_lb = min(leaf_lb, node_bound)
            continue

        if np.isfinite(incumbent) and params.marginal_reduction:
            # shrink the children's box from the bound's own pieces before
            # choosing where to split it
            reductions += _marginal_reduce(last["red"], last["xf"], last["y"],
                                           lo, hi, int_mask, incumbent - gap,
                                           lo, hi)
            if (lo > hi + tol.primal_feas).any():
                leaf_lb = min(leaf_lb, node_bound)
                continue

        # -- branch: integrality first, then the largest αBB gap term -------- #
        idx = np.flatnonzero(int_mask)
        frac = np.abs(x[idx] - np.round(x[idx]))
        if frac.size and frac.max() > params.integrality:
            j = int(idx[int(np.argmax(frac))])
            v = float(x[j])
            lo_l, hi_l = lo.copy(), hi.copy()
            hi_l[j] = np.floor(v)
            lo_r, hi_r = lo.copy(), hi.copy()
            lo_r[j] = np.ceil(v)
        else:
            width = hi - lo
            term = alpha * np.maximum(x - lo, 0.0) * np.maximum(hi - x, 0.0)
            live = (width > params.branch_eps) & (alpha > 0.0)
            if not live.any():
                # No non-convexity left to split, so the relaxation is (up to
                # the solver's accuracy) exact and the node is a convex QP:
                # solve it better before giving up on closing it.
                closed = False
                for tighten in (1, 2):
                    _, x2, b2, _ = relax(lo, hi, tighten)
                    node_bound = max(node_bound, b2)
                    consider(x2)
                    gap = max(params.gap_abs, params.gap_rel * abs(incumbent)) \
                        if np.isfinite(incumbent) else 0.0
                    if np.isfinite(incumbent) and node_bound >= incumbent - gap:
                        closed = True
                        break
                if not closed:
                    undecided += 1
                leaf_lb = min(leaf_lb, node_bound)
                continue
            score = np.where(live, term, -1.0)
            j = int(np.argmax(score))
            if score[j] <= 0.0:
                # the iterate sits on a bound of every live variable: bisect
                # the widest one instead of making a null child
                j = int(np.argmax(np.where(live, width, -1.0)))
                v = 0.5 * (lo[j] + hi[j])
            else:
                v = float(x[j])
                margin = params.branch_eps * max(1.0, width[j])
                if v <= lo[j] + margin or v >= hi[j] - margin:
                    v = 0.5 * (lo[j] + hi[j])
            if int_mask[j]:
                # an integer with an integral iterate: {<= v} ∪ {>= v+1}
                v = float(np.floor(v))
                lo_l, hi_l = lo.copy(), hi.copy()
                hi_l[j] = v
                lo_r, hi_r = lo.copy(), hi.copy()
                lo_r[j] = v + 1.0
            else:
                lo_l, hi_l = lo.copy(), hi.copy()
                hi_l[j] = v
                lo_r, hi_r = lo.copy(), hi.copy()
                lo_r[j] = v

        for cl, ch in ((lo_l, hi_l), (lo_r, hi_r)):
            if (cl <= ch + tol.primal_feas).all():
                heapq.heappush(frontier, _Node(node_bound, order, cl, ch))
                order += 1

        if params.verbose:
            print(f"  αBB nodes {nodes:>7d}  open {len(frontier):>6d}  "
                  f"bound {node_bound:< 14.8g}  incumbent "
                  f"{incumbent if np.isfinite(incumbent) else float('nan'):< 14.8g}"
                  f"  {time.perf_counter() - t0:6.1f}s")

    if not frontier and status == Status.NODE_LIMIT and not undecided:
        status = Status.OPTIMAL if best_x is not None else Status.INFEASIBLE

    global_lb = min(leaf_lb, min((nd.bound for nd in frontier), default=np.inf))
    if best_x is None:
        return Solution(status=status, nodes=nodes,
                        time=time.perf_counter() - t0,
                        method="alphabb").drop_objective_if_unsolved()

    global_lb = min(global_lb, incumbent)
    sol = Solution(status=status, x=best_x, objective=prob.objective(best_x),
                   nodes=nodes, time=time.perf_counter() - t0, method="alphabb")
    sol.dual_bound = -global_lb if flip else global_lb
    sol.info = {"nodes": nodes, "bound_is_rigorous": True,
                "marginal_reductions": reductions,
                "root_alpha_max": float(alpha0.max(initial=0.0)),
                "root_convex_by_gershgorin": root_convex}
    return sol
