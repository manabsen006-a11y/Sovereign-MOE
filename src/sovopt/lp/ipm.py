"""Primal-dual interior-point method, Mehrotra predictor-corrector.

The third LP engine. The simplex walks vertices and is exact; PDLP is
matrix-free and massively parallel but converges slowly in the tail. An
interior-point method sits between them: it needs a factorisation every
iteration, but it needs only tens of them regardless of problem size, and its
iteration count is famously insensitive to dimension. On a degenerate model,
where the simplex stalls cycling between bases of equal objective, the central
path does not notice degeneracy at all.

The formulation
---------------
The canonical model here is ``min ½xᵀQx + cᵀx  s.t.  rl <= Ax <= ru,
l <= x <= u`` with ``Q`` positive semidefinite or absent -- the LP case is
``Q = 0`` and nothing below changes shape for it.
Introducing a row slack ``s = Ax`` turns every row into an equality and every
constraint into a bound::

    min  cᵀx    s.t.   [A  -I] (x; s) = 0,    (l; rl) <= (x; s) <= (u; ru)

so one code path covers equalities, inequalities and ranged rows: an equality
row is simply a slack whose two bounds coincide, and a free row is a slack with
neither. Writing ``z = (x, s)`` and ``Ā = [A  -I]``, the barrier problem's KKT
conditions are

    Ā z = 0,   Āᵀy + z_l - z_u = c_z,   (z - l)·z_l = μ,   (u - z)·z_u = μ

with ``z_l, z_u >= 0``. Eliminating the bound duals leaves the *augmented
system*, and eliminating the slack block from that leaves what is actually
factorised each iteration::

    ⎡ -(Q+Θx⁻¹)  Aᵀ ⎤ ⎡dx⎤   ⎡ r_x ⎤        Θx⁻¹ = z_l/(x-l) + z_u/(u-x)
    ⎢               ⎥ ⎢  ⎥ = ⎢     ⎥
    ⎣    A       Θs ⎦ ⎣dy⎦   ⎣ r_y ⎦        Θs   = 1 / (same, for the slacks)

A convex quadratic changes exactly two things: ``Q`` joins the (1,1) block,
which stays negative definite because ``Q ⪰ 0``, and the dual objective
acquires ``-½xᵀQx`` (the Wolfe dual). Everything else -- the predictor, the
corrector, the step rule, the refinement -- is the LP code unchanged. That is
the reason this is the QP engine of choice: QPLIB's convex instances, which
the first-order QP solver could not finish in a minute, are ten to thirty
factorisations here.

Why the augmented system and not the normal equations
-----------------------------------------------------
The textbook shortcut is to condense further into ``A Θ Aᵀ`` and take a
Cholesky factor. That is smaller, but a **single dense column of A makes
A Θ Aᵀ completely dense** -- one column with k nonzeros contributes a k×k dense
block -- and the classic remedy, splitting dense columns out and correcting by
Sherman-Morrison, is itself numerically delicate. The augmented form is larger
but inherits A's sparsity directly, and with the two regularisation terms below
it is *quasi-definite* in Vanderbei's sense: a factorisation exists for **any**
symmetric permutation, so the pivot order may be chosen for sparsity alone
without a stability veto. That is what lets this reuse
:func:`sovopt.numerics.lu.lu_factor` -- the same threshold-Markowitz code the
simplex depends on -- instead of needing a separate sparse Cholesky.

Regularisation is not optional here. A free variable has no finite bound, so
its ``Θx⁻¹`` is exactly zero and the (1,1) block is singular; an equality row
has a slack pinned between equal bounds, so its ``Θs`` is exactly zero and the
(2,2) block is singular. Both are the common case, not a corner case. Static
primal and dual regularisation ``δp, δd`` keeps both blocks definite, and the
perturbation it introduces is removed by iterative refinement against the
*unregularised* matrix -- which is why the refinement loop is not a luxury.

What this does not do
---------------------
It returns no basis, so it cannot start a simplex warm and cannot answer a
ranging question; that is the crossover gap recorded in the README. Infeasible
and unbounded models are detected by divergence of the iterates rather than by
a Farkas certificate, so those statuses are reported as
``INFEASIBLE_OR_UNBOUNDED`` and are the weakest claim this module makes.

References
----------
Mehrotra, "On the implementation of a primal-dual interior point method", SIAM
  J. Optimization 2 (1992) 575-601 -- the predictor-corrector scheme, the
  adaptive centring parameter σ = (μ_aff/μ)³ and the starting point.
Wright, *Primal-Dual Interior-Point Methods*, SIAM (1997), ch. 8-11 -- the
  bounded-variable KKT system and the fraction-to-boundary rule.
Vanderbei, "Symmetric quasi-definite matrices", SIAM J. Optimization 5 (1995)
  100-113 -- existence of a factorisation of the regularised augmented system
  under any symmetric permutation.
Altman & Gondzio, "Regularized symmetric indefinite systems in interior point
  methods for linear and quadratic optimization", Optim. Methods Softw. 11
  (1999) 275-302 -- the static primal/dual regularisation used here.
Gondzio, "Interior point methods 25 years later", European J. Oper. Res. 218
  (2012) 587-601 -- survey; the step-length and termination conventions.
Vanderbei, "LOQO: an interior point code for quadratic programming", Optim.
  Methods Softw. 11 (1999) 451-484 -- the quadratic block in the reduced KKT
  system and the Wolfe dual objective.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core.problem import ObjSense, Problem, Solution, Status
from ..core.sparse import IDX, VAL
from ..core.tolerances import DEFAULT, INF, Tolerances
from ..numerics.lu import LUSingular, lu_factor
from ..numerics.ordering import rcm_order
from ..numerics.scaling import scale_problem


@dataclass
class IPMParams:
    """Termination and safeguard settings for :func:`solve_ipm`."""

    eps_p: float = 1e-9
    """Relative primal residual ``||Ax - s||inf / (1 + ||s||inf)``."""

    eps_d: float = 1e-9
    """Relative dual residual, scaled by ``1 + ||c||inf``."""

    eps_gap: float = 1e-10
    """Relative duality gap."""

    feas_cap: float = 1e-6
    """Absolute cap on the *unscaled* row and bound violation at termination.

    The same rule PDLP uses, and for the same reason: a relative test on a
    model with a large right-hand side permits an absolute violation the
    independent verifier rejects. Optimality stays relative, feasibility does
    not, and the engine never claims a status its own checker would refuse."""

    max_iter: int = 200
    time_limit: float = 600.0

    reg_primal: float = 1e-8
    reg_dual: float = 1e-8
    """Static regularisation of the two diagonal blocks. Small enough not to
    bias the step, large enough to keep a free column or an equality row from
    making the block singular. Undone by refinement."""

    refine_rounds: int = 2
    """Iterative-refinement passes per KKT solve, against the unregularised
    matrix. Two is enough to recover the digits regularisation costs; the
    residual is checked, so a converged solve stops early."""

    fraction_to_boundary: float = 0.9995
    """How far along the Newton direction a step may go before it would touch
    the boundary. Mehrotra's value."""

    divergence_norm: float = 1e14
    """Iterate norm beyond which the model is called infeasible or unbounded."""

    stall_limit: int = 15
    """Iterations of no primal-residual progress before the model is called
    infeasible or unbounded. See the stagnation test in :func:`solve_ipm`."""

    gap_stall_accept: float = 1e-7
    """Relative gap accepted as optimal when mu has stopped decreasing with
    both residuals already converged -- the floor on the complementarity
    gaps, not the model, is what holds the gap there. See the loop."""

    ordering: str = "auto"
    """Fill-reducing ordering for the KKT: ``"auto"``, ``"rcm"`` or ``"none"``.

    The KKT *pattern* is identical at every iteration -- only the two diagonal
    blocks move -- so an ordering is chosen once per solve and reused for every
    factorisation. That is what makes it worth choosing carefully, and what
    makes ``"auto"`` cheap: it factorises the first system once per candidate
    and keeps whichever produced the sparser factors, so the whole decision
    costs one extra factorisation out of the twenty or thirty a solve does.

    Neither candidate dominates, which is why there is a race rather than a
    default. ``"none"`` is the LU's own order -- peel singletons, then sort by
    live column count -- which is right for a simplex basis and wrong for a
    saddle-point matrix. Measured:

        plan k=4  (3,840 rows, banded by time)   fill 10-15x -> 2.3-3.4x,
                                                 factorisation 20-45x faster
        mod010    (146 x 2,655, wide)            0.18 s -> 118 s timeout

    A multi-period model's graph is a long thin strip and RCM exploits it
    directly; a wide shallow model has no band to find and RCM's level sets
    only get in the way. No cheap structural test told the two apart, and the
    factor count does, so the factor count decides. See
    :mod:`sovopt.numerics.ordering`."""

    verbose: bool = False
    tol: Tolerances = field(default_factory=lambda: DEFAULT)


# --------------------------------------------------------------------------- #
# the KKT matrix                                                               #
# --------------------------------------------------------------------------- #


class _KKT:
    """Sparsity pattern of ``[[-Dx, Aᵀ], [A, Ds]]``, built once and revalued.

    The pattern never changes across iterations -- only the two diagonals move
    -- so the column counts, row indices and the positions of every entry are
    computed once here and the per-iteration cost is two array writes.
    """

    def __init__(self, A, ordering="auto", Q=None):
        self.A = A
        self.Q = Q
        self.order = None
        self._ordering = ordering
        self._pending = None
        m, n = A.m, A.n
        self.m, self.n = m, n
        nk = n + m
        self.nk = nk
        self.q_diag = np.zeros(n, dtype=VAL)

        if Q is not None and Q.nnz > 0:
            self._build_with_q(A, Q, ordering)
            return

        acp = np.asarray(A.cp, dtype=np.int64)
        arp = np.asarray(A.rp, dtype=np.int64)
        col_counts = acp[1:] - acp[:-1]
        row_counts = arp[1:] - arp[:-1]
        nnz = int(acp[-1])

        # column j < n holds the (j,j) diagonal then A's column j, shifted to
        # rows n.. ; column n+i holds A's row i then the (n+i, n+i) diagonal.
        counts = np.empty(nk, dtype=np.int64)
        counts[:n] = 1 + col_counts
        counts[n:] = row_counts + 1
        kp = np.zeros(nk + 1, dtype=np.int64)
        np.cumsum(counts, out=kp[1:])
        self.kp = kp

        self.diag_x = kp[:n].copy()                 # first entry of column j
        self.diag_s = kp[n + 1:nk + 1] - 1          # last entry of column n+i

        # An entry k of A's CSC lands at kp[j] + 1 + (k - acp[j]); the offset
        # is constant within a column, so one repeat plus one arange places all
        # of them without a Python loop. Same idea for the CSR copy.
        self.pos_csc = (np.repeat(kp[:n] + 1 - acp[:n], col_counts)
                        + np.arange(nnz, dtype=np.int64))
        self.pos_csr = (np.repeat(kp[n:nk] - arp[:m], row_counts)
                        + np.arange(nnz, dtype=np.int64))

        ki = np.empty(int(kp[-1]), dtype=IDX)
        ki[self.diag_x] = np.arange(n, dtype=IDX)
        ki[self.pos_csc] = n + np.asarray(A.ci, dtype=IDX)
        ki[self.pos_csr] = np.asarray(A.ri, dtype=IDX)
        ki[self.diag_s] = n + np.arange(m, dtype=IDX)
        self.ki = ki

        kx = np.empty(int(kp[-1]), dtype=VAL)
        kx[self.pos_csc] = A.cx
        kx[self.pos_csr] = A.rx
        self.kx = kx

        # Once per solve, not once per iteration: the pattern above is fixed
        # for the whole run.
        if ordering == "rcm":
            self.order = rcm_order(kp, ki, nk)
        elif ordering == "auto":
            # candidates raced at the first factorisation, when the diagonal is
            # representative of the rest of the solve
            self._pending = [None, rcm_order(kp, ki, nk)]

    def _build_with_q(self, A, Q, ordering):
        """The same pattern with ``-Q`` in the (1,1) block, built from triplets.

        ``Q`` is constant across iterations, so its entries are written once
        and only the diagonal is revalued: ``-(Q_jj + Θx⁻¹_j)``. A diagonal
        entry is inserted for every column whether or not ``Q`` has one, so
        that ``diag_x`` always exists.
        """
        from ..core.sparse import coo_to_csc
        m, n, nk = self.m, self.n, self.nk
        q_cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(Q.cp))
        q_rows = np.asarray(Q.ci, dtype=np.int64)
        q_vals = np.asarray(Q.cx, dtype=VAL)
        # Q's diagonal is kept aside and revalued with Θ every iteration; the
        # off-diagonal part is written once. Every diagonal position gets a
        # placeholder (non-zero, so the triplet builder keeps it) that factor()
        # overwrites before anything reads it.
        on_diag = q_rows == q_cols
        q_diag = np.zeros(n, dtype=VAL)
        np.add.at(q_diag, q_cols[on_diag], q_vals[on_diag])
        self.q_diag = q_diag
        a_cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.cp))
        a_rows = np.asarray(A.ci, dtype=np.int64)
        ar = np.arange(n, dtype=np.int64)
        am = np.arange(m, dtype=np.int64)
        rows = np.concatenate([q_rows[~on_diag], ar, n + a_rows, a_cols, n + am])
        cols = np.concatenate([q_cols[~on_diag], ar, a_cols, n + a_rows, n + am])
        vals = np.concatenate([-q_vals[~on_diag], np.ones(n),
                               np.asarray(A.cx, dtype=VAL),
                               np.asarray(A.cx, dtype=VAL), np.ones(m)])
        kp, ki, kx = coo_to_csc(rows, cols, vals, nk, nk)
        self.kp = np.asarray(kp, dtype=np.int64)
        self.ki = np.asarray(ki, dtype=IDX)
        self.kx = np.asarray(kx, dtype=VAL)
        col_of = np.repeat(np.arange(nk, dtype=np.int64), np.diff(self.kp))
        diag = np.flatnonzero(self.ki == col_of)
        assert diag.size == nk
        self.diag_x = diag[:n]
        self.diag_s = diag[n:]
        if ordering == "rcm":
            self.order = rcm_order(self.kp, self.ki, nk)
        elif ordering == "auto":
            self._pending = [None, rcm_order(self.kp, self.ki, nk)]

    def factor(self, dx, ds, pivot_tol, drop):
        """Revalue the diagonals and factorise. ``dx``, ``ds`` are positive."""
        self.kx[self.diag_x] = -(self.q_diag + dx)
        self.kx[self.diag_s] = ds
        if self._pending is not None:
            # The incumbent is factorised in full; every challenger is then
            # capped at the incumbent's factor count, so a trial that cannot
            # win is abandoned as soon as it proves that rather than run to
            # completion. Without the cap the losing candidate on mod010 takes
            # 24 s to reach 277x fill, against 0.004 s for the winner -- the
            # choice was right and paying for it was not.
            best = lu_factor(self.kp, self.ki, self.kx, self.nk,
                             tol=pivot_tol, drop=drop, order=self._pending[0])
            best_order = self._pending[0]
            for cand in self._pending[1:]:
                try:
                    lu = lu_factor(self.kp, self.ki, self.kx, self.nk,
                                   tol=pivot_tol, drop=drop, order=cand,
                                   max_nnz=best.nnz)
                except LUSingular:
                    continue                  # over budget: it lost
                if lu.nnz < best.nnz:
                    best, best_order = lu, cand
            self.order = best_order
            self._pending = None
            self.chose_rcm = best_order is not None
            return best
        return lu_factor(self.kp, self.ki, self.kx, self.nk,
                         tol=pivot_tol, drop=drop, order=self.order)

    def matvec(self, dx_diag, ds_diag, v):
        """``K v`` for the *unregularised* K, used by iterative refinement."""
        n = self.n
        vx, vy = v[:n], v[n:]
        out = np.empty_like(v)
        out[:n] = -dx_diag * vx + self.A.rmatvec(vy)
        if self.Q is not None:
            out[:n] -= self.Q.matvec(vx)
        out[n:] = self.A.matvec(vx) + ds_diag * vy
        return out

    def solve(self, lu, dx_diag, ds_diag, rhs, rounds):
        """Solve ``K u = rhs`` and refine against the unregularised ``K``."""
        u = lu.ftran(rhs.copy())
        if rounds <= 0:
            return u
        scale = max(1.0, float(np.abs(rhs).max(initial=0.0)))
        for _ in range(rounds):
            resid = rhs - self.matvec(dx_diag, ds_diag, u)
            if float(np.abs(resid).max(initial=0.0)) <= 1e-14 * scale:
                break
            u += lu.ftran(resid)
        return u


# --------------------------------------------------------------------------- #
# the solver                                                                   #
# --------------------------------------------------------------------------- #


def _starting_point(lo, hi, fixed, has_lo, has_hi):
    """An interior point, pushed a sensible distance off every finite bound.

    A point hugging its bound gives a tiny ``g`` or ``t``, hence an enormous
    ``Θ⁻¹``, hence a first step that is almost entirely centring. The offset is
    therefore proportional to the variable's own scale, so a model in millions
    is not started a distance 1 from its bound.

    **Boxed variables are started near zero, not at the midpoint.** The
    midpoint is the obvious choice and it is wrong whenever a box is wide: on
    mas76 one column is bounded by 1e12 while every other is bounded by 1, and
    the midpoint start puts that column at 5e11 for an initial objective of
    1.28e14. The iteration never recovers -- the primal residual stalls at
    1.2e-2 with the step length pinned at zero, and the model is still
    unsolved 200 iterations later. Clipping zero into the box instead keeps the
    start at the scale the *data* sets rather than the scale the loosest bound
    sets.
    """
    z = np.zeros(lo.size, dtype=VAL)
    both = has_lo & has_hi & ~fixed
    only_lo = has_lo & ~has_hi & ~fixed
    only_hi = has_hi & ~has_lo & ~fixed

    w = np.zeros_like(z)
    np.subtract(hi, lo, out=w, where=both)
    near = np.minimum(np.where(has_lo, np.abs(lo), np.inf),
                      np.where(has_hi, np.abs(hi), np.inf))
    near = np.where(np.isfinite(near), near, 0.0)
    off = np.where(both, np.minimum(0.5 * w, 1.0 + 0.1 * near),
                   np.maximum(1.0, 0.1 * near))

    lo_in = np.where(has_lo, lo + off, -INF)
    hi_in = np.where(has_hi, hi - off, INF)
    lo_in[fixed] = lo[fixed]
    hi_in[fixed] = lo[fixed]

    z[both] = np.clip(0.0, (lo + off)[both], (hi - off)[both])
    z[only_lo] = lo[only_lo] + off[only_lo]
    z[only_hi] = hi[only_hi] - off[only_hi]
    z[fixed] = lo[fixed]
    return z, lo_in, hi_in


def _initial_point(kkt, A, cz, lo, hi, fixed, has_lo, has_hi,
                   free_lo, free_hi, params, tol):
    """Mehrotra's starting point, adapted to two-sided bounds.

    Choosing the primal point from the column bounds alone and letting the
    slacks absorb whatever activity results ignores ``Ax = s`` completely, and
    on a model with many equality rows that is not a slow start, it is a
    failure: khb05250 has 77 equality rows whose right-hand side is zero, its
    columns start near 1.3e4 because that is where their bounds are, and the
    initial primal residual is **2.6e4**. mu then runs to 1e14 and the model is
    still unsolved 200 iterations later.

    The fix is Mehrotra's: take the *minimum-norm* point that hits the desired
    row activity, and the least-squares duals that go with it. Both come from
    one factorisation of the KKT matrix at ``Θ = I`` --

        ⎡ -I   Aᵀ ⎤ ⎡x⎤   ⎡0⎤          ⎡ -I   Aᵀ ⎤ ⎡·⎤   ⎡c⎤
        ⎢         ⎥ ⎢ ⎥ = ⎢ ⎥          ⎢         ⎥ ⎢ ⎥ = ⎢ ⎥
        ⎣  A   δI ⎦ ⎣y⎦   ⎣s⎦          ⎣  A   δI ⎦ ⎣y⎦   ⎣0⎦

    -- the first giving ``x = Aᵀ(AAᵀ + δI)⁻¹ s``, the second the least-squares
    ``y``. The factors are reused, so the whole start costs one factorisation.

    The least-squares ``x`` knows about the rows and nothing about the columns,
    so it is then clipped back inside the shrunk box. That reintroduces some
    residual, but far less than starting from the box alone.
    """
    n, m = A.n, A.m
    nk = n + m
    ones_n = np.ones(n, dtype=VAL)
    reg_m = np.full(m, params.reg_dual, dtype=VAL)
    lu = kkt.factor(ones_n, reg_m, tol.lu_pivot_rel, tol.lu_drop)

    box, lo_in, hi_in = _starting_point(lo, hi, fixed, has_lo, has_hi)

    # Solve for the minimum-norm *correction* to a point that already respects
    # the column bounds, not for a minimum-norm x. Those are not the same
    # problem: khb05250's columns have lower bounds up to 1.3e4, so the
    # minimum-norm x is near zero, the clip back into the box drags every one
    # of them up to its bound again, and the least-squares work is thrown away
    # entirely -- the initial residual is 2.6e4 either way. Correcting a
    # feasible-by-bounds point keeps the correction small, so the clip only
    # trims it.
    rhs = np.zeros(nk, dtype=VAL)
    rhs[n:] = box[n:] - A.matvec(box[:n])
    dx = kkt.solve(lu, ones_n, reg_m, rhs, params.refine_rounds)[:n]
    x = np.clip(box[:n] + dx, lo_in[:n], hi_in[:n])

    rhs = np.zeros(nk, dtype=VAL)
    rhs[:n] = cz[:n]
    y = kkt.solve(lu, ones_n, reg_m, rhs, params.refine_rounds)[n:]

    z = np.empty(nk, dtype=VAL)
    z[:n] = x
    z[n:] = np.clip(A.matvec(x), lo_in[n:], hi_in[n:])
    z[fixed] = lo[fixed]

    # Bound duals from the least-squares reduced costs, split by sign and
    # shifted off zero so no complementarity product starts at the boundary.
    r = cz.copy()
    r[:n] -= A.rmatvec(y)
    r[n:] += y
    shift = 1.0 + 0.1 * float(np.abs(r).max(initial=0.0))
    zl = np.where(free_lo, np.maximum(r, 0.0) + shift, 0.0).astype(VAL)
    zu = np.where(free_hi, np.maximum(-r, 0.0) + shift, 0.0).astype(VAL)
    return z, y, zl, zu


def _gaps(z, lo, hi, has_lo, has_hi, floor):
    """Distances to the finite bounds, held at least ``floor`` away from zero.

    Where a bound does not exist the distance is reported as 1.0 rather than
    infinity: the matching dual is identically zero there, so every product it
    appears in vanishes, and 1.0 keeps the divisions finite without a mask on
    every line."""
    g = np.ones_like(z)
    t = np.ones_like(z)
    np.subtract(z, lo, out=g, where=has_lo)
    np.subtract(hi, z, out=t, where=has_hi)
    np.maximum(g, floor, out=g, where=has_lo)
    np.maximum(t, floor, out=t, where=has_hi)
    return g, t


def _max_step(v, dv, active):
    """Largest a with ``v + a*dv > 0`` on ``active``; 1.0 if nothing binds."""
    bad = active & (dv < 0.0)
    if not bad.any():
        return 1.0
    return float(np.minimum(1.0, (-v[bad] / dv[bad]).min()))


def solve_ipm(prob: Problem, params: IPMParams | None = None) -> Solution:
    """Solve an LP or a convex QP by a primal-dual interior-point method.

    Integer restrictions are ignored -- this solves the relaxation. Convexity
    of ``Q`` is the caller's premise: :mod:`sovopt.qp` checks it before
    routing here, and an indefinite ``Q`` handed in directly makes the (1,1)
    block indefinite, which the factorisation may or may not survive.
    """
    params = params or IPMParams()
    tol = params.tol
    t0 = time.perf_counter()

    flip = prob.sense == ObjSense.MAXIMISE
    work = prob
    if flip:
        work = prob.copy()
        work.c = -work.c
        work.obj_offset = -work.obj_offset
        if work.Q is not None:
            work.Q = work.Q.copy()
            work.Q.cx = -work.Q.cx
            work.Q.rx = -work.Q.rx
        work.sense = ObjSense.MINIMISE

    scaled, sc = scale_problem(work, method="ruiz")
    A = scaled.A
    Q = scaled.Q if (scaled.Q is not None and scaled.Q.nnz > 0) else None
    m, n = A.m, A.n
    nk = n + m

    # ---- the bounded-variable form z = (x, s) ----------------------------- #
    lo = np.concatenate([scaled.col_lb, scaled.row_lb]).astype(VAL)
    hi = np.concatenate([scaled.col_ub, scaled.row_ub]).astype(VAL)
    has_lo = lo > -INF
    has_hi = hi < INF
    fixed = has_lo & has_hi & (hi - lo <= tol.zero * np.maximum(1.0, np.abs(lo)))
    free_lo = has_lo & ~fixed
    free_hi = has_hi & ~fixed
    ncomp = int(free_lo.sum() + free_hi.sum())

    cz = np.zeros(nk, dtype=VAL)
    cz[:n] = scaled.c
    c_norm = 1.0 + float(np.abs(cz).max(initial=0.0))

    if ncomp == 0:
        # Every variable pinned: there is nothing for a barrier to do.
        z = np.where(has_lo, lo, np.where(has_hi, hi, 0.0))
        return _finish(prob, work, scaled, sc, flip, z[:n], np.zeros(m),
                       np.zeros(n), Status.OPTIMAL, 0, t0, params, "ipm")

    # ---- starting point ---------------------------------------------------- #
    kkt = _KKT(A, ordering=params.ordering, Q=Q)
    try:
        z, y, zl, zu = _initial_point(kkt, A, cz, lo, hi, fixed,
                                      has_lo, has_hi, free_lo, free_hi,
                                      params, tol)
    except LUSingular:
        # No least-squares start available; fall back to the box alone. Slower,
        # and on an equality-heavy model much slower, but not wrong.
        z, _, _ = _starting_point(lo, hi, fixed, has_lo, has_hi)
        y = np.zeros(m, dtype=VAL)
        zl = np.where(free_lo, 1.0 + np.abs(cz), 0.0).astype(VAL)
        zu = np.where(free_hi, 1.0 + np.abs(cz), 0.0).astype(VAL)
    floor = 1e-12
    big = 1.0 / params.reg_primal
    status = Status.ITERATION_LIMIT
    it = 0
    history = []
    best_pres = np.inf
    stall = 0
    mu_stall = 0
    mu_prev = np.inf

    for it in range(1, params.max_iter + 1):
        if time.perf_counter() - t0 > params.time_limit:
            status = Status.TIME_LIMIT
            break

        g, t = _gaps(z, lo, hi, has_lo, has_hi, floor)

        # ---- residuals ----------------------------------------------------- #
        rp = -(A.matvec(z[:n]) - z[n:])

        # Reduced costs r = c_z - Āᵀy. Dual feasibility asks r = z_l - z_u,
        # but a *pinned* variable's reduced cost is free in sign: it carries no
        # residual, and it contributes to the dual objective directly rather
        # than through a bound multiplier. Leaving that term out is not a small
        # error -- an equality row's slack is pinned, so on any model with
        # equalities the gap stalls at a constant while the primal residual,
        # the dual residual and mu all reach 1e-16, and the solve never
        # terminates despite having found the optimum.
        r = cz.copy()
        qx = Q.matvec(z[:n]) if Q is not None else None
        if qx is not None:
            r[:n] += qx                      # the gradient of ½xᵀQx + cᵀx
        r[:n] -= A.rmatvec(y)
        r[n:] += y
        rd = r - zl + zu
        rd[fixed] = 0.0

        mu = float((g[free_lo] @ zl[free_lo] + t[free_hi] @ zu[free_hi]) / ncomp)

        quad = 0.5 * float(z[:n] @ qx) if qx is not None else 0.0
        pobj = float(cz @ z) + quad
        # Wolfe dual: the LP dual objective less ½xᵀQx
        dobj = float(lo[free_lo] @ zl[free_lo] - hi[free_hi] @ zu[free_hi]
                     + lo[fixed] @ r[fixed]) - quad
        pres = float(np.abs(rp).max(initial=0.0)) / (1.0 + float(np.abs(z[n:]).max(initial=0.0)))
        dres = float(np.abs(rd).max(initial=0.0)) / c_norm
        gap = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))

        if params.verbose:
            print(f"  ipm {it:3d}  mu={mu:.3e}  pres={pres:.3e}  "
                  f"dres={dres:.3e}  gap={gap:.3e}  obj={pobj:.8e}")
        history.append((it, mu, pres, dres, gap))

        if pres <= params.eps_p and dres <= params.eps_d and gap <= params.eps_gap:
            status = Status.OPTIMAL
            break
        # The complementarity gaps are floored at 1e-12 so a division never
        # blows up, and on a model whose duals reach 1e4 that floor pins mu
        # near 1e-8 for good. Both residuals are then at machine precision,
        # the objective agrees with the published value to eight digits, and
        # the gap test alone stands between the run and OPTIMAL. Measured on
        # QPLIB_8938: 140 iterations of mu = 4.872e-08 exactly. A gap that has
        # stopped moving with the residuals converged is the accuracy this
        # arithmetic can reach, and it is reported as such in info["gap"].
        mu_stall = mu_stall + 1 if mu >= mu_prev * (1.0 - 1e-3) else 0
        mu_prev = mu
        if (mu_stall >= 5 and pres <= params.eps_p and dres <= params.eps_d
                and gap <= params.gap_stall_accept):
            status = Status.OPTIMAL
            break
        if not np.isfinite(mu) or float(np.abs(z).max(initial=0.0)) > params.divergence_norm:
            status = Status.INFEASIBLE_OR_UNBOUNDED
            break

        # Primal infeasibility shows up as a residual that stops moving while
        # everything else keeps converging: there is no primal point to find,
        # so the iterates settle on the least-infeasible one and stay there.
        # Measured on a 3x3 all-equality model whose unique solution violates a
        # lower bound -- dres falls 1e-8 -> 1e-50 and mu 5.2 -> 2e-4 while pres
        # is pinned at 5.556e-02 from iteration 8 onward. Without this the run
        # burns its whole iteration budget and reports ITERATION_LIMIT, which
        # says "I ran out of time" about a model that has no answer.
        #
        # This is a stagnation test and not a Farkas certificate, which is why
        # the status is INFEASIBLE_OR_UNBOUNDED and not INFEASIBLE: it is
        # evidence that the method has stopped making progress, not a proof
        # that no point exists.
        if pres < best_pres * (1.0 - 1e-4):
            best_pres, stall = pres, 0
        else:
            stall += 1
        if stall >= params.stall_limit and pres > 1e3 * params.eps_p:
            status = Status.INFEASIBLE_OR_UNBOUNDED
            break

        # ---- the diagonal blocks ------------------------------------------- #
        theta_inv = np.zeros(nk, dtype=VAL)
        theta_inv[free_lo] += zl[free_lo] / g[free_lo]
        theta_inv[free_hi] += zu[free_hi] / t[free_hi]
        theta_inv[fixed] = big                    # pinned: force dz = 0

        dxd = theta_inv[:n] + params.reg_primal
        ti_s = theta_inv[n:]
        theta_s = np.where(ti_s > 1.0 / big, 1.0 / np.maximum(ti_s, 1e-300), big)
        theta_s[fixed[n:]] = 0.0                  # equality row: ds = 0
        dsd = theta_s + params.reg_dual

        # The clock is checked here as well as at the top of the loop, because
        # a factorisation cannot be interrupted once it starts and at scale it
        # is by far the longest step. This stops a doomed one from *beginning*
        # after the budget is already spent; it cannot shorten one already
        # running, which is why a 120 s limit was measured overrunning to
        # 210 s on a 15360-row planning model. See Known limits.
        if time.perf_counter() - t0 > params.time_limit:
            status = Status.TIME_LIMIT
            break
        try:
            lu = kkt.factor(dxd, dsd, tol.lu_pivot_rel, tol.lu_drop)
        except LUSingular:
            try:                                   # one retry, tighter pivoting
                lu = kkt.factor(dxd, dsd, 0.1, tol.lu_drop)
            except LUSingular:
                status = Status.NUMERICAL
                break

        def newton(rl, ru):
            """Assemble the right-hand side and back out the full direction."""
            r_aug = rd.copy()
            r_aug[free_lo] -= rl[free_lo] / g[free_lo]
            r_aug[free_hi] += ru[free_hi] / t[free_hi]

            rhs = np.empty(nk, dtype=VAL)
            rhs[:n] = r_aug[:n]
            rhs[n:] = rp - theta_s * r_aug[n:]
            u = kkt.solve(lu, dxd - params.reg_primal, dsd - params.reg_dual,
                          rhs, params.refine_rounds)
            d_x, d_y = u[:n], u[n:]

            d_z = np.empty(nk, dtype=VAL)
            d_z[:n] = d_x
            d_z[n:] = -theta_s * (r_aug[n:] + d_y)
            d_z[fixed] = 0.0

            d_zl = np.zeros(nk, dtype=VAL)
            d_zu = np.zeros(nk, dtype=VAL)
            d_zl[free_lo] = (rl[free_lo] - zl[free_lo] * d_z[free_lo]) / g[free_lo]
            d_zu[free_hi] = (ru[free_hi] + zu[free_hi] * d_z[free_hi]) / t[free_hi]
            return d_z, d_y, d_zl, d_zu

        # ---- predictor: the pure Newton step for mu = 0 --------------------- #
        rl_aff = np.zeros(nk, dtype=VAL)
        ru_aff = np.zeros(nk, dtype=VAL)
        rl_aff[free_lo] = -g[free_lo] * zl[free_lo]
        ru_aff[free_hi] = -t[free_hi] * zu[free_hi]
        dz_a, dy_a, dzl_a, dzu_a = newton(rl_aff, ru_aff)

        ap = min(_max_step(g, dz_a, free_lo), _max_step(t, -dz_a, free_hi))
        ad = min(_max_step(zl, dzl_a, free_lo), _max_step(zu, dzu_a, free_hi))
        mu_aff = float(
            ((g[free_lo] + ap * dz_a[free_lo]) @ (zl[free_lo] + ad * dzl_a[free_lo])
             + (t[free_hi] - ap * dz_a[free_hi]) @ (zu[free_hi] + ad * dzu_a[free_hi]))
            / ncomp)
        sigma = (mu_aff / mu) ** 3 if mu > 0.0 else 0.0
        sigma = float(min(max(sigma, 0.0), 1.0))

        # ---- corrector: recentre and cancel the second-order error ---------- #
        rl = np.zeros(nk, dtype=VAL)
        ru = np.zeros(nk, dtype=VAL)
        rl[free_lo] = (sigma * mu - g[free_lo] * zl[free_lo]
                       - dz_a[free_lo] * dzl_a[free_lo])
        ru[free_hi] = (sigma * mu - t[free_hi] * zu[free_hi]
                       + dz_a[free_hi] * dzu_a[free_hi])
        d_z, d_y, d_zl, d_zu = newton(rl, ru)

        # ---- step ----------------------------------------------------------- #
        eta = max(params.fraction_to_boundary, 1.0 - mu)
        ap = eta * min(_max_step(g, d_z, free_lo), _max_step(t, -d_z, free_hi))
        ad = eta * min(_max_step(zl, d_zl, free_lo), _max_step(zu, d_zu, free_hi))
        if ap < 1e-12 and ad < 1e-12:
            status = Status.NUMERICAL
            break

        z = z + ap * d_z
        y = y + ad * d_y
        zl = zl + ad * d_zl
        zu = zu + ad * d_zu

    x_s = z[:n]
    d_s = (zl - zu)[:n]
    sol = _finish(prob, work, scaled, sc, flip, x_s, y, d_s,
                  status, it, t0, params, "ipm",
                  ordering="rcm" if getattr(kkt, "chose_rcm", None) or
                  (params.ordering == "rcm") else "none",
                  gap=history[-1][4] if history else None)
    sol.log = history
    return sol


def _finish(prob, work, scaled, sc, flip, x_scaled, y_scaled, d_scaled,
            status, iterations, t0, params, method, ordering="n/a", gap=None):
    """Unscale, verify absolutely, and report."""
    x = sc.unscale_primal(x_scaled)
    np.clip(x, prob.col_lb, prob.col_ub, out=x)
    # The Lagrangian above is written so that stationarity reads
    # c = Āᵀy + z_l - z_u, hence d = z_l - z_u = c - Aᵀy -- already the
    # convention the simplex reports, so y passes through unnegated. Pinned to
    # the simplex on random LPs: agreement to 1e-15 with the sign as it stands,
    # and a sign error here is invisible in the objective and shows up only in
    # the marginal prices.
    y = sc.unscale_dual(y_scaled)
    d = sc.unscale_reduced(d_scaled)
    if flip:
        y = -y
        d = -d

    obj = prob.objective(x)
    # (row, bound, integrality) -- the relaxation is what was solved, so the
    # integrality entry is not this method's to answer for.
    row_v, col_v, _ = prob.violation(x)
    worst = max(row_v, col_v)

    if status == Status.OPTIMAL and worst > params.feas_cap:
        # Converged in the scaled space but not by the yardstick the verifier
        # uses. Say so rather than claim an optimum the checker would reject.
        status = Status.NUMERICAL

    if worst > params.feas_cap and Status(status).has_solution:
        # An unconverged iterate is not a solution, and ``has_solution`` is
        # true for TIME_LIMIT and ITERATION_LIMIT -- so handing the point back
        # invites a caller to use it. Measured on a 15360-row planning model at
        # a 120 s limit: the returned point failed the independent verifier
        # outright. Withholding it costs a caller nothing it could have
        # trusted.
        x = None

    sol = Solution(status=status, x=x, objective=obj if x is not None
                   else float("nan"),
                   y=y, reduced_costs=d,
                   iterations=iterations, time=time.perf_counter() - t0,
                   method=method)
    sol.dual_bound = obj
    sol.work_units = float(iterations)
    sol.info = {"iterations": iterations, "worst_violation": worst,
                "scaling": sc.method, "kkt_ordering": ordering}
    if gap is not None:
        sol.info["gap"] = gap
    return sol.drop_objective_if_unsolved()
