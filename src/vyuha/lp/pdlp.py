"""First-order linear programming by restarted average PDHG.

The GPU path. Every operation in the inner loop is a sparse product, an
elementwise map, or a reduction -- there is no factorisation, no pivoting and no
sequential dependence, which is exactly the shape a GPU wants and exactly the
shape the simplex is not.

Saddle-point form
-----------------
Our canonical LP is ``min cᵀx  s.t.  rl <= Ax <= ru,  l <= x <= u``. Writing the
row constraint as an indicator of the box ``C = [rl, ru]`` and dualising it:

    min_{x ∈ [l,u]}  max_y   cᵀx + yᵀAx − δ*_C(y),      δ*_C(y) = sup_{z∈C} yᵀz

PDHG alternates a projected gradient step in ``x`` with a proximal ascent step in
``y``. The proximal step for a support function has a closed form via the Moreau
decomposition:

    prox_{σ δ*_C}(w) = w − σ · proj_C(w / σ)

which is one formula covering ``<=``, ``>=``, ``=`` and ranged rows with no
branching on row type -- the reason this method vectorises so cleanly.

Sign convention: ``y`` here is the negative of the textbook dual, because the
Lagrangian above uses ``+yᵀAx``. Reported duals are negated on the way out, so
the API returns ordinary row prices with ``d = c − Aᵀy_std``.

What makes it converge in practice
----------------------------------
Plain PDHG on an LP is uselessly slow. Three ingredients change that, and all
three are implemented here:

1. **Diagonal preconditioning** -- Ruiz equilibration followed by a
   Pock-Chambolle pass. Without it the effective condition number is the raw
   coefficient spread, which on a refinery model is 1e12.
2. **Adaptive step size** -- the step is chosen each iteration from the observed
   local curvature ``|Δyᵀ A Δx|`` rather than from a global ``‖A‖`` bound, which
   is pessimistic almost everywhere.
3. **Adaptive restarts to the running average** -- the ergodic average converges
   at a better rate than the last iterate, and restarting the average whenever
   the KKT error has decayed enough turns a sublinear method into one that is
   linearly convergent on non-degenerate LPs.

Iterations cost exactly two sparse products (one with ``A``, one with ``Aᵀ``)
because ``Ax`` and ``Aᵀy`` are maintained incrementally from ``A·Δx`` and
``Aᵀ·Δy``.

References
----------
Chambolle & Pock, "A first-order primal-dual algorithm for convex problems with
  applications to imaging", J. Math. Imaging Vis. 40 (2011) 120-145.
Pock & Chambolle, "Diagonal preconditioning for first order primal-dual
  algorithms", ICCV 2011.
Applegate, Díaz, Hinder, Lu, Lubin, O'Donoghue & Schudy, "Practical large-scale
  linear programming using primal-dual hybrid gradient", NeurIPS 2021 -- the
  adaptive step size, restart scheme and primal weight update used here.
Applegate, Hinder, Lu & Lubin, "Faster first-order primal-dual methods for
  linear programming using restarts and sharpness", Math. Prog. 201 (2023).
Malitsky & Pock, "A first-order primal-dual algorithm with linesearch",
  SIAM J. Optim. 28 (2018) 411-432.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core.backend import Backend, get_backend
from ..core.problem import ObjSense, Problem, Solution, Status
from ..core.sparse import VAL
from ..core.tolerances import INF
from ..numerics.scaling import scale_problem

__all__ = ["PDLPParams", "solve_pdlp"]


@dataclass
class PDLPParams:
    eps_abs: float = 1e-8
    eps_rel: float = 1e-8

    feas_cap: float = 1e-6
    """Absolute cap on primal and dual infeasibility at termination.

    The relative test ``eps_abs + eps_rel*||rhs||`` alone is not enough: on a
    model with ``||rhs||inf = 1.6e5`` it permits an absolute row violation of
    1.6e-3, and the solver then reports OPTIMAL for a point an independent
    checker rejects. Feasibility is therefore judged absolutely -- the same
    convention the benchmark libraries and the commercial solvers use -- while
    optimality stays relative. The engine must never claim a status its own
    verifier would refuse."""
    max_iter: int = 200_000
    time_limit: float = 600.0

    check_every: int = 64
    stepsize_every: int = 8
    """KKT evaluation costs two extra sparse products, so it is amortised."""

    # stepsize_every: how often the adaptive line search runs. Every inner
    # product it needs is a device stall, so validating the step size every
    # iteration makes the GPU path latency-bound rather than compute-bound.

    # restart scheme
    beta_sufficient: float = 0.2
    beta_necessary: float = 0.8
    beta_artificial: float = 0.36

    # primal weight
    primal_weight_smoothing: float = 0.5

    scaling: str = "pdlp"
    device: str = "auto"
    verbose: bool = False
    log_every: int = 2000


@dataclass
class PDLPInfo:
    iterations: int = 0
    restarts: int = 0
    rejected_steps: int = 0
    primal_res: float = float("nan")
    dual_res: float = float("nan")
    gap: float = float("nan")
    primal_obj: float = float("nan")
    dual_obj: float = float("nan")
    step_size: float = float("nan")
    primal_weight: float = float("nan")
    device: str = "cpu"
    history: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _estimate_norm(bk: Backend, rp, ri, rx, cp, ci, cx, m, n, iters=20):
    """Power iteration for ``‖A‖₂``, used only to seed the step size."""
    rng = np.random.default_rng(0)
    v = bk.to_device(rng.standard_normal(n))
    nv = bk.nrm2(v)
    if nv == 0.0:
        return 1.0
    v /= nv
    av = bk.empty(m)
    atv = bk.empty(n)
    est = 1.0
    for _ in range(iters):
        bk.spmv(rp, ri, rx, v, av, m)
        bk.spmv(cp, ci, cx, av, atv, n)     # CSC of A == CSR of Aᵀ
        nrm = bk.nrm2(atv)
        if nrm <= 0.0:
            return 1.0
        v = atv / nrm
        est = np.sqrt(nrm)
    return float(max(est, 1e-12))


class _KKT:
    """Evaluates the KKT residuals of a primal-dual pair.

    Kept as an object so its scratch buffers persist across the thousands of
    evaluations a solve performs.
    """

    def __init__(self, bk: Backend, rp, ri, rx, cp, ci, cx, m, n,
                 c, rl, ru, lo, hi, obj_offset):
        self.bk = bk
        self.rp, self.ri, self.rx = rp, ri, rx
        self.cp, self.ci, self.cx = cp, ci, cx
        self.m, self.n = m, n
        self.c, self.rl, self.ru, self.lo, self.hi = c, rl, ru, lo, hi
        self.obj_offset = obj_offset
        self._ax = bk.empty(m)
        self._aty = bk.empty(n)

        xp = bk.xp
        self.ru_fin = ru < INF
        self.rl_fin = rl > -INF
        self.lo_fin = lo > -INF
        self.hi_fin = hi < INF
        self.zero = bk.to_device(np.zeros(1))[0] * 0.0
        self._rhs_norm = float(xp.abs(xp.where(self.rl_fin, rl, 0.0)).max()) if m else 0.0
        self._rhs_norm = max(self._rhs_norm,
                             float(xp.abs(xp.where(self.ru_fin, ru, 0.0)).max()) if m else 0.0)
        self._c_norm = float(xp.abs(c).max()) if n else 0.0

    def __call__(self, x, y, ax=None, aty=None):
        bk, xp = self.bk, self.bk.xp
        if ax is None:
            ax = bk.spmv(self.rp, self.ri, self.rx, x, self._ax, self.m)
        if aty is None:
            aty = bk.spmv(self.cp, self.ci, self.cx, y, self._aty, self.n)

        # primal residual: distance of Ax from the row box
        pv = xp.maximum(self.rl - ax, 0.0) + xp.maximum(ax - self.ru, 0.0)
        primal_res = float(xp.abs(pv).max()) if self.m else 0.0

        # reduced cost; d>0 must be absorbed by a finite lower bound, d<0 by a
        # finite upper bound -- anything else is dual infeasibility
        d = self.c + aty
        dpos = xp.maximum(d, 0.0)
        dneg = xp.maximum(-d, 0.0)
        dres = xp.where(self.lo_fin, 0.0, dpos) + xp.where(self.hi_fin, 0.0, dneg)
        dual_res = float(xp.abs(dres).max()) if self.n else 0.0

        primal_obj = float(xp.dot(self.c, x)) + self.obj_offset

        # dual objective: -δ*_C(y) + Σ_j (d_j>0 ? l_j d_j : u_j d_j)
        supp = (xp.where(y > 0.0, xp.where(self.ru_fin, self.ru, 0.0),
                         xp.where(self.rl_fin, self.rl, 0.0)) * y)
        bnd = (xp.where(d > 0.0, xp.where(self.lo_fin, self.lo, 0.0),
                        xp.where(self.hi_fin, self.hi, 0.0)) * d)
        dual_obj = float(-xp.sum(supp) + xp.sum(bnd)) + self.obj_offset

        gap = abs(primal_obj - dual_obj)
        return primal_res, dual_res, gap, primal_obj, dual_obj

    def merit(self, x, y, omega, ax=None, aty=None):
        """A single scalar to compare candidate restart points."""
        p, dres, g, _, _ = self(x, y, ax, aty)
        return float(np.sqrt((omega * p) ** 2 + (dres / omega) ** 2 + g ** 2))

    def converged(self, p, dres, g, pobj, dobj, eps_abs, eps_rel):
        return (p <= eps_abs + eps_rel * self._rhs_norm
                and dres <= eps_abs + eps_rel * self._c_norm
                and g <= eps_abs + eps_rel * (abs(pobj) + abs(dobj)))


# --------------------------------------------------------------------------- #
# solver                                                                       #
# --------------------------------------------------------------------------- #


def solve_pdlp(prob: Problem, params: PDLPParams | None = None) -> Solution:
    """Solve an LP with restarted average PDHG."""
    params = params or PDLPParams()
    t0 = time.perf_counter()

    # maximisation is handled by negating the objective and flipping back
    flip = prob.sense == ObjSense.MAXIMISE
    work = prob
    if flip:
        work = prob.copy()
        work.c = -work.c
        work.obj_offset = -work.obj_offset
        work.sense = ObjSense.MINIMISE

    scaled, sc = scale_problem(work, method=params.scaling)

    bk = get_backend(params.device)
    m, n = scaled.m, scaled.n

    rp = bk.pointer(scaled.A.rp)
    ri = bk.index(scaled.A.ri)
    rx = bk.to_device(scaled.A.rx)
    cp = bk.pointer(scaled.A.cp)
    ci = bk.index(scaled.A.ci)
    cx = bk.to_device(scaled.A.cx)

    c = bk.to_device(scaled.c)
    rl = bk.to_device(scaled.row_lb)
    ru = bk.to_device(scaled.row_ub)
    lo = bk.to_device(scaled.col_lb)
    hi = bk.to_device(scaled.col_ub)

    kkt = _KKT(bk, rp, ri, rx, cp, ci, cx, m, n, c, rl, ru, lo, hi,
               scaled.obj_offset)

    xp = bk.xp
    x = xp.clip(bk.zeros(n), lo, hi)
    y = bk.zeros(m)

    ax = bk.empty(m)
    aty = bk.empty(n)
    bk.spmv(rp, ri, rx, x, ax, m)
    bk.spmv(cp, ci, cx, y, aty, n)

    x_new = bk.empty(n)
    y_new = bk.empty(m)
    adx = bk.empty(m)
    ady = bk.empty(n)
    dx_buf = bk.empty(n)
    dy_buf = bk.empty(m)

    anorm = _estimate_norm(bk, rp, ri, rx, cp, ci, cx, m, n)
    eta = 1.0 / anorm

    cn = float(xp.linalg.norm(c))
    bn = float(xp.linalg.norm(xp.where(rl > -INF, rl, 0.0)
                              + xp.where(ru < INF, ru, 0.0)))
    omega = float(np.exp(np.log(max(cn, 1e-12)) - np.log(max(bn, 1e-12)))) if bn > 0 else 1.0
    omega = float(np.clip(omega, 1e-4, 1e4))

    # restart bookkeeping
    sum_x = bk.zeros(n)
    sum_y = bk.zeros(m)
    sum_eta = 0.0
    x_restart = x.copy()
    y_restart = y.copy()
    mu_restart = kkt.merit(x, y, omega, ax, aty)
    mu_last = float("inf")
    epoch_iters = 0

    info = PDLPInfo(device=bk.device)
    status = Status.ITERATION_LIMIT
    k = 0
    attempts = 0

    while k < params.max_iter:
        if k % params.check_every == 0 and \
                (time.perf_counter() - t0) > params.time_limit:
            status = Status.TIME_LIMIT
            break

        # ---------------------------------------------------------------- #
        # Two step flavours.
        #
        # The line-search flavour validates the step size against the observed
        # local curvature, which needs three inner products -- and every inner
        # product is a device-to-host stall on the GPU. Running it every
        # iteration makes the solver latency-bound: measured at 516 it/s on a
        # 960k-nonzero instance where the actual arithmetic accounts for well
        # under a tenth of the time.
        #
        # So the search runs only every ``stepsize_every`` iterations, and the
        # iterations in between take the validated step size with fused,
        # in-place kernels and no host round-trip at all. The step size is a
        # local estimate that varies slowly, so revalidating it every few
        # iterations loses nothing.
        # ---------------------------------------------------------------- #
        adapt = (k % params.stepsize_every) == 0

        if adapt:
            # out-of-place, so a rejected trial can be rolled back
            accepted = False
            for _ in range(60):
                attempts += 1
                tau = eta / omega
                sigma = eta * omega

                bk.primal_step(x_new, x, c, aty, lo, hi, tau, n)
                dx = x_new - x
                bk.spmv(rp, ri, rx, dx, adx, m)

                # A(2x_new − x) = Ax + 2·A·Δx
                ax_bar = ax + 2.0 * adx
                bk.dual_step(y_new, y, ax_bar, rl, ru, sigma, m)
                dy = y_new - y

                interaction = abs(float(xp.dot(dy, adx)))
                movement = 0.5 * (omega * float(xp.dot(dx, dx))
                                  + float(xp.dot(dy, dy)) / omega)

                if movement == 0.0:      # fixed point reached
                    accepted = True
                    break
                eta_limit = movement / interaction if interaction > 0 \
                    else float("inf")

                # The trial counter is cumulative and is incremented *before*
                # the new step size is formed. Using the outer iteration index
                # makes the shrink factor (1 - t^-0.3) exactly zero on the
                # first step, driving eta -- and with it sigma -- to zero.
                t = attempts + 1
                eta_next = max(min((1.0 - t ** -0.3) * eta_limit,
                                   (1.0 + t ** -0.6) * eta), 1e-14)

                if eta <= eta_limit:
                    accepted = True
                    eta = eta_next
                    break
                info.rejected_steps += 1
                eta = eta_next

            if not accepted:
                status = Status.NUMERICAL
                break

            bk.spmv(cp, ci, cx, dy, ady, n)
            x, x_new = x_new, x
            y, y_new = y_new, y
            ax += adx
            aty += ady
            sum_x += eta * x
            sum_y += eta * y
        else:
            # fused, in-place, zero host synchronisation
            tau = eta / omega
            sigma = eta * omega
            bk.pdhg_primal(x, dx_buf, c, aty, lo, hi, tau, n)
            bk.spmv(rp, ri, rx, dx_buf, adx, m)
            bk.pdhg_dual(y, dy_buf, ax, adx, rl, ru, sigma, m)
            bk.spmv(cp, ci, cx, dy_buf, ady, n)
            bk.pdhg_accum(aty, ady, sum_x, x, eta, n)
            sum_y += eta * y

        sum_eta += eta
        k += 1
        epoch_iters += 1

        # ---- restart / termination ---------------------------------------
        if k % params.check_every == 0:
            p, dres, g, pobj, dobj = kkt(x, y, ax, aty)
            mu_c = float(np.sqrt((omega * p) ** 2 + (dres / omega) ** 2 + g ** 2))

            if sum_eta > 0:
                avg_x = sum_x / sum_eta
                avg_y = sum_y / sum_eta
                mu_a = kkt.merit(avg_x, avg_y, omega)
            else:
                avg_x, avg_y, mu_a = x, y, float("inf")

            use_avg = mu_a < mu_c
            mu_cand = mu_a if use_avg else mu_c

            # termination is judged on the *unscaled* problem
            cand_x = avg_x if use_avg else x
            cand_y = avg_y if use_avg else y
            pu, du, gu, pou, dou = _unscaled_kkt(bk, prob, work, sc, cand_x, cand_y)
            if _converged(pu, du, gu, pou, dou, prob, params):
                x, y = cand_x, cand_y
                status = Status.OPTIMAL
                info.primal_res, info.dual_res, info.gap = pu, du, gu
                info.primal_obj, info.dual_obj = pou, dou
                break

            if params.verbose and k % params.log_every == 0:
                print(f"  it {k:>7d}  pres {pu:.2e}  dres {du:.2e}  "
                      f"gap {gu:.2e}  obj {pou:.10g}  eta {eta:.2e}  w {omega:.2e}")
            info.history.append((k, pu, du, gu, pou))

            do_restart = (
                mu_cand <= params.beta_sufficient * mu_restart
                or (mu_cand <= params.beta_necessary * mu_restart and mu_cand > mu_last)
                or epoch_iters >= params.beta_artificial * max(k, 1)
            )
            mu_last = mu_cand

            if do_restart:
                x = cand_x.copy()
                y = cand_y.copy()
                bk.spmv(rp, ri, rx, x, ax, m)
                bk.spmv(cp, ci, cx, y, aty, n)

                ddx = float(xp.linalg.norm(x - x_restart))
                ddy = float(xp.linalg.norm(y - y_restart))
                if ddx > 1e-12 and ddy > 1e-12:
                    th = params.primal_weight_smoothing
                    omega = float(np.exp(th * np.log(ddy / ddx)
                                         + (1.0 - th) * np.log(omega)))
                    omega = float(np.clip(omega, 1e-6, 1e6))

                x_restart, y_restart = x.copy(), y.copy()
                mu_restart = mu_cand
                sum_x = bk.zeros(n)
                sum_y = bk.zeros(m)
                sum_eta = 0.0
                epoch_iters = 0
                info.restarts += 1

    # ---- unscale and report ---------------------------------------------- #
    x_host = bk.to_host(x)
    y_host = bk.to_host(y)
    x_orig = sc.unscale_primal(x_host)
    x_orig = np.clip(x_orig, prob.col_lb, prob.col_ub)
    # sign flip: our y is the negative of the textbook dual
    y_orig = -sc.unscale_dual(y_host)

    obj = float(prob.c @ x_orig) + prob.obj_offset
    if flip:
        y_orig = -y_orig

    if status == Status.ITERATION_LIMIT and k >= params.max_iter:
        status = Status.ITERATION_LIMIT

    info.iterations = k
    info.step_size = eta
    info.primal_weight = omega

    d = prob.c - prob.A.rmatvec(y_orig)
    sol = Solution(
        status=status, x=x_orig, objective=obj, y=y_orig, reduced_costs=d,
        iterations=k, time=time.perf_counter() - t0,
        method=f"pdlp[{bk.device}]",
    )
    sol.log = info.history
    sol.dual_bound = obj
    sol.work_units = float(k)
    sol.info = info
    return sol.drop_objective_if_unsolved()


def _unscaled_kkt(bk, orig: Problem, work: Problem, sc, x_s, y_s):
    """KKT residuals of the *original* problem, which is what users care about."""
    x = sc.unscale_primal(bk.to_host(x_s))
    y = sc.unscale_dual(bk.to_host(y_s))

    ax = work.A.matvec(x)
    pv = np.maximum(work.row_lb - ax, 0.0) + np.maximum(ax - work.row_ub, 0.0)
    bv = np.maximum(work.col_lb - x, 0.0) + np.maximum(x - work.col_ub, 0.0)
    primal_res = float(max(np.abs(pv).max(initial=0.0), np.abs(bv).max(initial=0.0)))

    d = work.c + work.A.rmatvec(y)
    lo_fin = work.col_lb > -INF
    hi_fin = work.col_ub < INF
    dres = np.where(lo_fin, 0.0, np.maximum(d, 0.0)) + \
           np.where(hi_fin, 0.0, np.maximum(-d, 0.0))
    dual_res = float(np.abs(dres).max(initial=0.0))

    pobj = float(work.c @ x) + work.obj_offset
    supp = np.where(y > 0.0, np.where(work.row_ub < INF, work.row_ub, 0.0),
                    np.where(work.row_lb > -INF, work.row_lb, 0.0)) * y
    bnd = np.where(d > 0.0, np.where(lo_fin, work.col_lb, 0.0),
                   np.where(hi_fin, work.col_ub, 0.0)) * d
    dobj = float(-supp.sum() + bnd.sum()) + work.obj_offset
    return primal_res, dual_res, abs(pobj - dobj), pobj, dobj


def _converged(p, d, g, pobj, dobj, prob: Problem, params: PDLPParams) -> bool:
    rhs = max(float(np.abs(np.where(prob.row_lb > -INF, prob.row_lb, 0.0)).max(initial=0.0)),
              float(np.abs(np.where(prob.row_ub < INF, prob.row_ub, 0.0)).max(initial=0.0)))
    cn = float(np.abs(prob.c).max(initial=0.0))
    ptol = min(params.eps_abs + params.eps_rel * rhs, params.feas_cap)
    dtol = min(params.eps_abs + params.eps_rel * cn, params.feas_cap)
    return (p <= ptol and d <= dtol
            and g <= params.eps_abs + params.eps_rel * (abs(pobj) + abs(dobj)))
