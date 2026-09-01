"""Batched Node Relaxation -- bounding a whole branch-and-bound frontier at once.

The idea
--------
Every node of a branch-and-bound tree solves an LP over the *same* constraint
matrix ``A``, the *same* objective ``c``, and the *same* row bounds. Branching
changes nothing but the variable bounds ``l`` and ``u``. Conventional solvers
still treat each node as a separate LP: they warm-start a dual simplex per node
and walk the tree one node at a time. That is a good strategy on a CPU and a
terrible one on a GPU, because a single node's sparse matrix-vector product has
nowhere near enough parallelism to fill the device.

This module inverts the loop. Take ``K`` open nodes, stack their bounds into
``(n, K)`` matrices, and run the first-order LP method on all ``K`` at once. The
two sparse products of a PDHG iteration become **sparse-times-dense products
with one shared matrix**:

        A  @ X      instead of   K separate  A @ x_k
        Aᵀ @ Y      instead of   K separate  Aᵀ @ y_k

``A`` is read once instead of ``K`` times, arithmetic intensity rises by a factor
of ``K``, and the dense operands are laid out with the node index fastest so a
warp's 32 lanes read 32 consecutive doubles. The measured effect of this layout
change alone, before any GPU is involved, is roughly 3.5x on the CPU kernels.

Why the bounds are still rigorous
---------------------------------
Batched PDHG does not converge in the few dozen iterations a node deserves, so
its dual iterate is not dual feasible and its dual objective is not a bound.
:mod:`vyuha.mip.safebound` fixes this: the Neumaier-Shcherbina correction turns
*any* dual vector into a valid lower bound. Unconverged iterates therefore prune
soundly -- a weak ``y`` simply gives a weak bound, never a wrong one.

This is what the combination buys that neither piece gives alone: the throughput
of a first-order method with the exactness guarantees of branch-and-bound.

Consequences for the search
---------------------------
Because the entire frontier is bounded simultaneously rather than one node at a
time, node selection sees real bounds for every open node instead of parent
estimates. Best-bound selection becomes exact rather than approximate, and nodes
that would have been expanded before their bound was known are pruned first.

Exactness of the scaled bound
-----------------------------
Bounds are computed in the scaled space and divided by the objective scale
factor. Because every scale factor is rounded to a power of two
(:mod:`vyuha.numerics.scaling`), that division is exact in binary floating point
-- the scaling contributes no error at all to a bound that must stay valid.

Provenance
----------
The batching scheme itself is original to this project: no published solver
bounds a whole branch-and-bound frontier in one sparse-times-dense product. It
composes two established ingredients, cited below -- a GPU first-order LP method
for the relaxation, and a correction that makes bounds from approximate duals
rigorous. See ``docs/PROVENANCE.md``.

References
----------
Applegate, Díaz, Hinder, Lu, Lubin, O'Donoghue & Schudy, "Practical large-scale
  linear programming using primal-dual hybrid gradient", NeurIPS 2021 -- the
  relaxation method being batched here.
Neumaier & Shcherbina, "Safe bounds in linear and mixed-integer linear
  programming", Math. Prog. 99 (2004) 283-296 -- why an unconverged dual iterate
  still yields a valid bound.
Lu & Yang, "cuPDLP.jl: a GPU implementation of restarted primal-dual hybrid
  gradient for linear programming", 2023 -- GPU behaviour of the single-instance
  method, the baseline this improves on by batching.
Ralphs, Shinano, Berthold & Koch, "Parallel solvers for mixed integer linear
  optimization", in *Handbook of Parallel Constraint Reasoning*, Springer 2018 --
  survey of conventional tree parallelism, i.e. the approach this replaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.backend import Backend, get_backend
from ..core.problem import Problem
from ..core.tolerances import INF
from .safebound import safe_dual_bound_batch

__all__ = ["BNREngine", "BNRConfig", "BatchResult"]


@dataclass
class BNRConfig:
    batch: int = 128
    """Nodes advanced per call. Larger fills the GPU better but costs memory:
    the working set is about ``(2n + 3m) * batch * 8`` bytes."""

    iters: int = 60
    """PDHG iterations per node visit. Bounds improve monotonically in
    expectation, so a node revisited later continues from its warm start."""

    warm_start: bool = True
    eps: float = 1e-9
    device: str = "auto"

    step_growth: float = 0.3
    step_damp: float = 0.6


@dataclass
class BatchResult:
    bounds: np.ndarray
    """(K,) valid lower bounds, in the *original* objective scale."""

    x: np.ndarray
    """(n, K) primal iterates, unscaled -- feeds rounding heuristics."""

    iterations: int = 0
    infeasible: np.ndarray | None = None
    """(K,) True where the node was detected to have empty bounds."""


class BNREngine:
    """Holds the shared matrix on the device and advances node batches.

    The matrix is uploaded once for the whole solve; each batch only moves the
    ``(n, K)`` bound blocks, which is the small end of the traffic.
    """

    def __init__(self, prob: Problem, scaling, config: BNRConfig | None = None):
        self.cfg = config or BNRConfig()
        self.bk: Backend = get_backend(self.cfg.device)
        self.prob = prob
        self.sc = scaling
        bk = self.bk

        self.m, self.n = prob.m, prob.n
        self.rp = bk.pointer(prob.A.rp)
        self.ri = bk.index(prob.A.ri)
        self.rx = bk.to_device(prob.A.rx)
        self.cp = bk.pointer(prob.A.cp)
        self.ci = bk.index(prob.A.ci)
        self.cx = bk.to_device(prob.A.cx)

        self.c = bk.to_device(prob.c)
        self.rl = bk.to_device(prob.row_lb)
        self.ru = bk.to_device(prob.row_ub)

        self.obj_scale = getattr(scaling, "obj", 1.0) or 1.0
        self.obj_offset = prob.obj_offset

        self._alloc(self.cfg.batch)
        self.total_iterations = 0
        self.total_batches = 0
        self.total_nodes = 0

    # -- buffers ------------------------------------------------------------ #

    def _alloc(self, K: int):
        bk, m, n = self.bk, self.m, self.n
        self.K = K
        self.X = bk.zeros((n, K))
        self.Y = bk.zeros((m, K))
        self.Xn = bk.zeros((n, K))
        self.Yn = bk.zeros((m, K))
        self.AX = bk.zeros((m, K))
        self.AdX = bk.zeros((m, K))
        self.AtY = bk.zeros((n, K))
        self.AtdY = bk.zeros((n, K))
        self.eta = np.ones(K)
        self.omega = np.ones(K)

    def _ensure(self, K: int):
        if K > self.K:
            self._alloc(max(K, self.K * 2))

    # -- the batched iteration ---------------------------------------------- #

    def solve_batch(self, col_lb, col_ub, warm=None) -> BatchResult:
        """Advance ``K`` nodes and return valid lower bounds for each.

        ``col_lb`` / ``col_ub`` are ``(n, K)`` arrays of per-node variable
        bounds, already in the scaled space. ``warm`` is an optional
        ``(X, Y)`` pair from a previous visit to these nodes.
        """
        bk, xp = self.bk, self.bk.xp
        n, m = self.n, self.m
        K = col_lb.shape[1]
        self._ensure(K)

        L = bk.to_device(col_lb)
        U = bk.to_device(col_ub)

        empty = (L > U + 1e-12).any(axis=0)

        X = self.X[:, :K]
        Y = self.Y[:, :K]
        if warm is not None:
            X[...] = bk.to_device(warm[0])
            Y[...] = bk.to_device(warm[1])
        else:
            X[...] = 0.0
            Y[...] = 0.0
        xp.clip(X, L, U, out=X)

        AX = self.AX[:, :K]
        AtY = self.AtY[:, :K]
        AdX = self.AdX[:, :K]
        AtdY = self.AtdY[:, :K]
        Xn = self.Xn[:, :K]
        Yn = self.Yn[:, :K]

        bk.spmm(self.rp, self.ri, self.rx, X, AX, m, K)
        bk.spmm(self.cp, self.ci, self.cx, Y, AtY, n, K)

        eta = bk.to_device(self.eta[:K])
        omega = bk.to_device(self.omega[:K])

        rl_b = self.rl[:, None]
        ru_b = self.ru[:, None]
        c_b = self.c[:, None]

        attempts = 0
        for _ in range(self.cfg.iters):
            attempts += 1
            tau = (eta / omega)[None, :]
            sigma = (eta * omega)[None, :]

            # primal: projected gradient, per-node step size
            xp.subtract(X, tau * (c_b + AtY), out=Xn)
            xp.clip(Xn, L, U, out=Xn)
            dX = Xn - X
            bk.spmm(self.rp, self.ri, self.rx, dX, AdX, m, K)

            # dual: prox of the support function via Moreau
            W = Y + sigma * (AX + 2.0 * AdX)
            xp.subtract(W, sigma * xp.clip(W / sigma, rl_b, ru_b), out=Yn)
            dY = Yn - Y
            bk.spmm(self.cp, self.ci, self.cx, dY, AtdY, n, K)

            # per-node adaptive step size; nodes accept or reject independently
            interaction = xp.abs((dY * AdX).sum(axis=0))
            movement = 0.5 * (omega * (dX * dX).sum(axis=0)
                              + (dY * dY).sum(axis=0) / omega)
            limit = xp.where(interaction > 0.0,
                             movement / xp.maximum(interaction, 1e-300),
                             xp.inf)
            accept = (eta <= limit) | (movement == 0.0)

            a = accept[None, :]
            X = xp.where(a, Xn, X)
            Y = xp.where(a, Yn, Y)
            AX = xp.where(a, AX + AdX, AX)
            AtY = xp.where(a, AtY + AtdY, AtY)

            t = attempts + 1
            eta = xp.minimum((1.0 - t ** -self.cfg.step_growth) * limit,
                             (1.0 + t ** -self.cfg.step_damp) * eta)
            eta = xp.maximum(xp.where(xp.isfinite(eta), eta, 1.0), 1e-14)

        self.X[:, :K] = X
        self.Y[:, :K] = Y
        self.eta[:K] = bk.to_host(eta)
        self.omega[:K] = bk.to_host(omega)

        bounds = safe_dual_bound_batch(
            bk, self.cp, self.ci, self.cx, self.c,
            self.rl, self.ru, L, U, Y, n, m, aty_buf=self.AtY[:, :K])

        # The safe bound is a lower bound on the *scaled* linear objective
        # c_s'x_s, which excludes the offset. The scaled objective as a whole is
        #     obj_s(x_s) = c_s'x_s + offset_s = obj_scale * obj_work(x)
        # so the bound in the working problem's units is (L + offset_s)/scale.
        # Every factor is a power of two, so this conversion is exact.
        bounds_host = (bk.to_host(bounds) + self.obj_offset) / self.obj_scale
        bounds_host = np.where(bk.to_host(empty), np.inf, bounds_host)

        self.total_iterations += self.cfg.iters
        self.total_batches += 1
        self.total_nodes += K

        return BatchResult(
            bounds=bounds_host,
            x=bk.to_host(X),
            iterations=self.cfg.iters,
            infeasible=bk.to_host(empty),
        )

    # -- reporting ---------------------------------------------------------- #

    def stats(self) -> dict:
        return {
            "device": self.bk.device,
            "batches": self.total_batches,
            "nodes_bounded": self.total_nodes,
            "pdhg_iterations": self.total_iterations,
            "avg_batch": self.total_nodes / max(self.total_batches, 1),
        }

    def __repr__(self):
        return (f"BNREngine({self.m}x{self.n} on {self.bk.device}, "
                f"batch={self.cfg.batch}, iters={self.cfg.iters})")
