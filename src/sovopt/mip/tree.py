"""Branch-and-bound driven by batched node relaxation.

The search loop here is deliberately shaped around :mod:`sovopt.mip.bnr`. A
conventional tree pops one node, solves its LP, branches, and repeats -- a
strictly serial dependence that no amount of hardware helps with. This one pops
a *slab* of nodes, bounds them all in a single batched sparse product, and only
then decides what to prune and what to branch on.

Two things follow from that inversion, and both are improvements rather than
compromises:

* **Node selection sees real bounds.** In a serial tree, the frontier is ordered
  by *parent* bounds, because a child's own bound is unknown until it is
  processed. Here every node in the slab has its own bound before any of them is
  expanded, so best-bound selection is exact rather than estimated.
* **Pseudocosts are updated in slabs.** Both children of a branching decision are
  bounded in the same batch, so the up and down objective gains land together
  and the pseudocost table fills in much faster than one branch at a time.

The bounds come from an unconverged first-order method, and are made rigorous by
the Neumaier-Shcherbina correction in :mod:`sovopt.mip.safebound`. A weak bound
costs search effort; it never causes a wrong answer.

Everything runs in the scaled space. Integer columns are pinned to a scale
factor of 1 (see :mod:`sovopt.numerics.scaling`), so branching bounds are
identical in both spaces and no rounding is introduced by the change of
variables.

References
----------
Land & Doig, "An automatic method of solving discrete programming problems",
  Econometrica 28 (1960) 497-520.
Achterberg, Koch & Martin, "Branching rules revisited", Oper. Res. Letters 33
  (2005) 42-54 -- pseudocost and reliability branching.
Linderoth & Savelsbergh, "A computational study of search strategies for mixed
  integer programming", INFORMS J. Computing 11 (1999) 173-187.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.problem import ObjSense, Problem, Solution, Status, VarKind
from ..core.sparse import VAL
from ..core.tolerances import DEFAULT, INF, Tolerances
from ..numerics.scaling import scale_problem
from ..lp.simplex import NodeSolver, SimplexParams
from .bnr import BNRConfig, BNREngine
from .conflict import ConflictAnalyzer
from .cuts import (Cut, CutPool, append_cuts, generate_cover,
                   generate_gomory, generate_mir)
from .heuristics import (HeuristicStats, feasibility_jump, feasibility_pump,
                         fix_and_propagate)
from .propagate import propagate
from .symmetry import breaking_constraints, detect_symmetry

__all__ = ["MIPParams", "solve_mip"]

ACCEPT_FEAS_TOL = 1e-6
"""Absolute feasibility an incumbent must satisfy on the *original* model.

Matched to the independent verifier deliberately: the tree must never adopt a
point the checker would reject.
"""


@dataclass
class MIPParams:
    gap_rel: float = 1e-4
    gap_abs: float = 1e-9
    time_limit: float = 300.0
    node_limit: int = 1_000_000

    batch: int = 64
    """Nodes bounded per batch.

    32-64 is the sweet spot on the **CPU** kernels, where the dense operands
    stop fitting in cache beyond it: measured SpMM throughput on a 240k-nonzero
    matrix falls 13.1 -> 7.4 -> 5.0 GFLOP/s across K = 32, 64, 128 and settles
    near 4.6. The default is 64 because it has to be safe on a machine with no
    GPU.

    The **GPU** kernels have no such ceiling in the range tested -- throughput
    saturates by K = 32 and is flat at ~42 GFLOP/s out to K = 1024 -- so a
    GPU-only deployment can raise this considerably. An earlier version of this
    note attributed the cache fall-off to the sparse kernels generally, on the
    strength of a K = 256 GPU measurement that did not survive re-measurement.
    """

    node_solver: str = "simplex"
    """How each node's relaxation is bounded.

    ``"simplex"``  exact LP per node, dual simplex warm-started from the
                   parent's basis. Tight bounds, few nodes. The default,
                   because a bound that is merely valid is not worth much:
                   the batched path needed 40,000 nodes on a knapsack the
                   exact path closes in a handful.
    ``"bnr"``      batched first-order bounding on the GPU. Cheap per node and
                   massively parallel, but the bound after a fixed iteration
                   budget is loose. Wins when nodes are enormous and the
                   frontier is wide.
    """

    bnr_iters: int = 80
    root_iters: int = 3000

    propagate_rounds: int = 6
    heuristic_every: int = 1

    cut_time_frac: float = 0.25
    """Share of the time limit the root cut loop and heuristics may spend.

    Without a cap they can blow through the whole budget before the tree starts
    a single node -- measured overruns of 69s against a 60s limit."""

    cut_rounds: int = 10
    """Rounds of the root cutting-plane loop. Cuts are separated at the root
    only, where they are globally valid and where most of the gap closure
    lives; local cuts in the tree would have to be tracked per node."""

    cuts_per_round: int = 40
    mir_cuts: bool | None = None
    """MIR cuts on the model's own rows. ``None`` decides per model.

    They are worth roughly a factor of four on p0201 and cost gt2 its root
    closure entirely, and no structural property separates the two: both sit
    under the small-basis exemption, both see MIR displace some GMI cuts, and
    MIR is the *sparser* family in both. What does separate them is the
    quantity actually being optimised, and it is visible at the root before a
    single node is explored:

        p0201   root 7054.62 without MIR,  7185.00 with   -> take them
        gt2     root 21166.00 without,    21104.83 with   -> leave them

    So ``None`` runs the root cut loop both ways and keeps the stronger
    bound, splitting the same cut-time budget between the two attempts
    rather than doubling it. ``True`` or ``False`` forces it and runs one
    loop. Either way the solution reports which way it went, in
    ``info["root_mir_cuts"]``.

    This is the gain-only rule docs/NEGATIVE-RESULTS.md identified as the
    stable quantity after the fill-priced rollback failed, with the one
    degree of freedom removed: a race rather than a threshold, so there is
    nothing to calibrate. It compares bounds, which are comparable, and
    never tries to price a cost against a tree size that is not yet known.
    What it does not do is look past the root -- a model whose root MIR
    helps but whose tree MIR slows would be chosen wrongly, and nothing
    here would notice.
    """
    heuristics: bool = True

    heuristic_restart: int = 400
    """Nodes between re-running the primal heuristics while no incumbent exists.

    They used to run at the root and nowhere else. On dcmulti all three failed
    there -- ``0/1`` each -- and the search then explored 1,811 nodes without
    trying again, returning no answer at all. Which vertex a node LP hands the
    rounding heuristic is luck, so re-drawing costs little and is the
    difference between a plan and nothing. Only fires while the incumbent is
    still infinite, so a solve that has an answer pays nothing."""

    conflict: bool = True
    """Learn a clause from each infeasible node.

    An ordinary tree discards the fact that a subproblem was empty and
    rediscovers it in every sibling repeating the same decisions. Conflict
    analysis records *which* decisions were to blame and forbids that
    combination globally."""

    clause_batch: int = 30
    """Learned clauses are added in batches: each rebuild of the node solver
    invalidates the warm-start basis cache, so doing it per clause would cost
    more than the clauses save."""

    symmetry: bool = True
    """Detect fully interchangeable variable groups and order them.

    Identical process units, identical tanks and interchangeable time slots make
    a tree re-derive the same schedule under every relabelling -- a factor of
    k! of wasted search for k identical objects. Detection is exact (every
    candidate transposition is verified against the model), so this can only
    ever remove duplicates of solutions that remain reachable."""

    symmetry_time: float = 2.0
    fj_iterations: int = 30_000
    device: str = "auto"

    verbose: bool = False
    log_every: float = 1.0
    tol: Tolerances = field(default_factory=lambda: DEFAULT)


class _Node:
    """A node is its branching path; bounds are rebuilt on demand.

    Storing full bound vectors per node costs ``O(n)`` memory each and dominates
    everything else on a large model. The path is ``O(depth)``, and depth is
    small.
    """

    __slots__ = ("path", "bound", "depth", "frac", "_order", "parent_id")
    _counter = 0

    def __init__(self, path, bound, depth, frac=0.0, parent_id=-1):
        self.path = path
        self.bound = bound
        self.depth = depth
        self.frac = frac
        self.parent_id = parent_id
        _Node._counter += 1
        self._order = _Node._counter

    def __lt__(self, other):
        # best-bound first; ties broken by depth (dive) then insertion order
        if self.bound != other.bound:
            return self.bound < other.bound
        if self.depth != other.depth:
            return self.depth > other.depth
        return self._order < other._order

    def bounds(self, lo0, hi0):
        lo = lo0.copy()
        hi = hi0.copy()
        for j, is_lower, v in self.path:
            if is_lower:
                if v > lo[j]:
                    lo[j] = v
            else:
                if v < hi[j]:
                    hi[j] = v
        return lo, hi


class _Pseudocost:
    """Per-variable objective gain per unit of fractional move.

    Uninitialised variables fall back to the global average, which is the
    standard way to avoid the cold-start problem without paying for strong
    branching at every node.
    """

    def __init__(self, n):
        self.sum_down = np.zeros(n)
        self.sum_up = np.zeros(n)
        self.cnt_down = np.zeros(n, dtype=np.int64)
        self.cnt_up = np.zeros(n, dtype=np.int64)
        self.total_down = 0.0
        self.total_up = 0.0
        self.n_down = 0
        self.n_up = 0

    def update(self, j, frac, gain, is_down):
        if frac <= 1e-12 or not np.isfinite(gain) or gain < 0:
            return
        unit = gain / frac
        if is_down:
            self.sum_down[j] += unit
            self.cnt_down[j] += 1
            self.total_down += unit
            self.n_down += 1
        else:
            self.sum_up[j] += unit
            self.cnt_up[j] += 1
            self.total_up += unit
            self.n_up += 1

    def score(self, j, f_down, f_up):
        avg_d = self.total_down / self.n_down if self.n_down else 1.0
        avg_u = self.total_up / self.n_up if self.n_up else 1.0
        pd = self.sum_down[j] / self.cnt_down[j] if self.cnt_down[j] else avg_d
        pu = self.sum_up[j] / self.cnt_up[j] if self.cnt_up[j] else avg_u
        # Achterberg's product score, with a floor so a zero on one side does
        # not annihilate the whole score
        return max(f_down * pd, 1e-6) * max(f_up * pu, 1e-6)


def _round_and_repair(prob, x, int_mask, lo, hi, tol):
    """Cheap primal heuristic: round, propagate, accept if feasible.

    Rounds each integer variable to its nearest integer inside the node bounds,
    fixes them, and propagates. If propagation survives and the continuous part
    is already within tolerance, we have an incumbent. This finds the optimum
    outright on many structured models and costs one propagation sweep.
    """
    cand = x.copy()
    idx = np.flatnonzero(int_mask)
    cand[idx] = np.clip(np.round(cand[idx]), lo[idx], hi[idx])
    cand = np.clip(cand, lo, hi)

    lo2 = lo.copy()
    hi2 = hi.copy()
    lo2[idx] = cand[idx]
    hi2[idx] = cand[idx]

    res = propagate(prob.A, prob.row_lb, prob.row_ub, lo2, hi2,
                    int_mask, max_rounds=3, feas_tol=tol.primal_feas)
    if res.infeasible:
        return None

    # place continuous variables at whatever propagation left them
    out = cand.copy()
    free = ~int_mask
    out[free] = np.clip(out[free], res.lo[free], res.hi[free])

    rv, bv, iv = prob.violation(out)
    if max(rv, bv) <= 1e-6 and iv <= tol.integrality:
        return out
    return None


def _root_cut_loop(scaled, node_lp, lo0, hi0, int_mask, params, tol,
                   deadline=None):
    """Separate globally valid cuts at the root until they stop paying.

    Cuts are generated from the root LP with the *root* bounds, so they hold
    for every node. The loop stops when a round produces nothing worth adding
    or when the bound stops moving -- "tailing off" -- because past that point
    each extra row costs more in LP time than it returns in bound.

    Returns ``(scaled, node_lp, x, bound, basis, n_cuts)``.
    """
    pool = CutPool(max_cuts=params.cuts_per_round)
    n = scaled.n
    total = 0
    x = None
    bound = -np.inf
    basis = None
    # logicals are treated as continuous: row scaling destroys any integrality
    # their coefficients had, and calling a continuous variable integral would
    # make the cut invalid. Conservative here costs strength, never validity.
    int_full = np.concatenate([int_mask, np.zeros(scaled.m, dtype=bool)])

    avg_row_nnz = scaled.nnz / max(scaled.m, 1)

    # The last cut set whose own LP actually solved.
    #
    # A round that cannot be solved must not reach the tree. Every node
    # inherits this relaxation, so an unsolvable root LP does not merely cost
    # the cuts -- it produces a solve that explores no nodes and returns no
    # answer at all. Measured on gt2: round 6 stalls the simplex for 224,352
    # iterations on a 119-row model (the same model needs 1,354 from the same
    # basis under looser bounds, so this is cycling, not difficulty), and it
    # burned the entire 60 s limit before a single node was explored. Falling
    # back to round 5's cuts costs some bound and keeps the answer.
    good = (scaled, node_lp, total)

    def _lp_budget():
        """Cap one cut-loop LP so it cannot outlive the whole root budget.

        ``cut_time_frac`` documented this intent and the loop did not
        implement it: the NodeSolver was built with the *full* time limit, so a
        single stalling LP consumed everything the tree was going to need.
        """
        if deadline is None:
            return params.time_limit
        return max(0.0, min(params.time_limit, deadline - time.perf_counter()))

    for _rnd in range(params.cut_rounds):
        if deadline is not None and time.perf_counter() > deadline:
            break
        node_lp.params.time_limit = _lp_budget()
        r = node_lp.solve(lo0, hi0)
        if r.status != Status.OPTIMAL or r.x is None:
            scaled, node_lp, total = good      # roll back to what solved
            break
        good = (scaled, node_lp, total)
        prev = bound
        x, bound, basis = r.x, r.objective, r.basis

        frac = np.abs(x[int_mask] - np.round(x[int_mask]))
        if frac.size == 0 or frac.max() <= tol.integrality:
            break

        B = node_lp.S.B
        cands = generate_gomory(B, node_lp.S.zB, int_full,
                                max_cuts=params.cuts_per_round)
        cands += generate_cover(scaled, x, int_mask, lo0, hi0,
                                max_cuts=params.cuts_per_round)
        # MIR on the model's own rows reaches inequalities the tableau does not
        # expose, and needs no basis. Validity brute-forced: 133 cuts over 55
        # instances, worst slack 0.0 against every feasible point.
        if params.mir_cuts:
            cands += generate_mir(scaled, x, int_mask, lo0, hi0,
                                  max_cuts=params.cuts_per_round)
        chosen = pool.select(cands, x, n, limit=params.cuts_per_round,
                             avg_row_nnz=avg_row_nnz, m=scaled.m)
        if not chosen:
            break

        scaled = append_cuts(scaled, chosen)
        int_full = np.concatenate([int_mask, np.zeros(scaled.m, dtype=bool)])
        node_lp = NodeSolver(scaled, SimplexParams(
            time_limit=params.time_limit,
            feas_tol=tol.primal_feas, opt_tol=tol.dual_feas))
        total += len(chosen)

        if np.isfinite(prev) and bound - prev <= 1e-9 * max(1.0, abs(bound)):
            break                                  # tailing off

    # The tree re-uses this solver for every node, so give it back the full
    # limit -- the cap above belongs to the root loop alone.
    node_lp.params.time_limit = params.time_limit
    r = node_lp.solve(lo0, hi0)
    if r.status == Status.OPTIMAL and r.x is not None:
        x, bound, basis = r.x, r.objective, r.basis
    return scaled, node_lp, x, bound, basis, total


def solve_mip(prob: Problem, params: MIPParams | None = None) -> Solution:
    """Solve a MILP by batched-relaxation branch-and-bound."""
    params = params or MIPParams()
    tol = params.tol
    t0 = time.perf_counter()

    if prob.Q is not None:
        raise NotImplementedError(
            "solve_mip optimises a linear objective; this model has a "
            "quadratic term, which would be silently discarded")

    if not prob.is_mip:
        from ..lp.pdlp import PDLPParams, solve_pdlp
        return solve_pdlp(prob, PDLPParams(device=params.device))

    flip = prob.sense == ObjSense.MAXIMISE
    work = prob
    if flip:
        work = prob.copy()
        work.c = -work.c
        work.obj_offset = -work.obj_offset
        work.sense = ObjSense.MINIMISE

    scaled, sc = scale_problem(work, method="pdlp")
    int_mask = scaled.integer_mask
    n = scaled.n

    # ---- root propagation ------------------------------------------------- #
    root = propagate(scaled.A, scaled.row_lb, scaled.row_ub,
                     scaled.col_lb, scaled.col_ub, int_mask,
                     max_rounds=params.propagate_rounds,
                     feas_tol=tol.primal_feas)
    if root.infeasible:
        return Solution(status=Status.INFEASIBLE, nodes=0,
                        time=time.perf_counter() - t0, method="bnr-bb")
    lo0, hi0 = root.lo, root.hi

    # ---- static symmetry breaking, before anything else sees the model ---- #
    sym_info = None
    if params.symmetry:
        sym_info = detect_symmetry(scaled, time_limit=params.symmetry_time)
        rows = breaking_constraints(sym_info, scaled.integer_mask)
        if rows:
            # NB: no local ``from .cuts import Cut`` here. Cut is imported at
            # module level, and re-importing it inside this branch made the
            # name local to the whole of solve_mip -- so on a model with no
            # detectable symmetry, where this branch never runs, the conflict
            # clause append below raised UnboundLocalError on default
            # parameters (symmetry and conflict are both on by default).
            scaled = append_cuts(scaled, [Cut(i, v, r, kind="symmetry")
                                          for i, v, r in rows])

    use_bnr = params.node_solver == "bnr"
    engine = None
    node_lp = None
    if use_bnr:
        engine = BNREngine(scaled, sc, BNRConfig(batch=params.batch,
                                                 iters=params.bnr_iters,
                                                 device=params.device))
    else:
        node_lp = NodeSolver(scaled, SimplexParams(
            time_limit=params.time_limit,
            feas_tol=tol.primal_feas, opt_tol=tol.dual_feas))

    # ---- root relaxation, solved harder than an ordinary node ------------- #
    root_basis = None
    n_cuts = 0
    cut_gain = 0.0
    # Which way the root cut loop went, or None if it never ran. The choice is
    # automatic by default, so a run that did not report it could not be
    # reproduced from its parameters alone.
    mir_used: bool | None = None
    if use_bnr:
        root_cfg = engine.cfg.iters
        engine.cfg.iters = params.root_iters
        r = engine.solve_batch(lo0[:, None].copy(), hi0[:, None].copy())
        engine.cfg.iters = root_cfg
        root_bound = float(r.bounds[0])
        root_x = r.x[:, 0]
    else:
        rr = node_lp.solve(lo0, hi0)
        if rr.status == Status.INFEASIBLE:
            return Solution(status=Status.INFEASIBLE, nodes=0,
                            time=time.perf_counter() - t0, method="bb")
        if rr.status != Status.OPTIMAL or rr.x is None:
            root_bound, root_x = -np.inf, np.zeros(n, dtype=VAL)
        else:
            root_bound = rr.objective / sc.obj
            root_x = rr.x
            root_basis = rr.basis

        if params.cut_rounds > 0:
            before = root_bound
            trials = ([False, True] if params.mir_cuts is None
                      else [bool(params.mir_cuts)])
            # The cut budget as an instant, not a duration. Each attempt takes
            # an even share of what is *left* at the moment it starts, rather
            # than a slice carved off the limit in advance: root setup runs
            # before this and spends real time, and a fixed slice charges the
            # first attempt for it. A 2 s symmetry pass against a 20 s limit
            # would leave attempt 0 no rounds at all and hand the decision to
            # attempt 1 by forfeit -- on gt2, exactly the wrong answer. It also
            # returns what an attempt does not use: both close in under a
            # second here, so the second gets nearly the whole budget.
            cut_end = t0 + params.cut_time_frac * params.time_limit
            best = None
            for k, use_mir in enumerate(trials):
                # Every attempt cuts the uncut model -- append_cuts copies, so
                # `scaled` is still what the previous attempt was handed. The
                # solver is built per attempt so the one handed on to the tree
                # belongs to the attempt that won it, instead of being left
                # warm-started and re-timed by the attempt that lost.
                lp_try = (node_lp if len(trials) == 1 else
                          NodeSolver(scaled, SimplexParams(
                              time_limit=params.time_limit,
                              feas_tol=tol.primal_feas,
                              opt_tol=tol.dual_feas)))
                now = time.perf_counter()
                out = _root_cut_loop(
                    scaled, lp_try, lo0, hi0, int_mask,
                    replace(params, mir_cuts=use_mir), tol,
                    deadline=now + max(0.0, cut_end - now) / (len(trials) - k))
                # Higher is stronger: this is the scaled minimisation bound and
                # both attempts cut the same relaxation, so the two are
                # directly comparable. A NaN or -inf attempt loses, which `>`
                # gives for free.
                if best is None or out[3] > best[3]:
                    best, mir_used = out, use_mir
            scaled, node_lp, cx, cb, cbasis, n_cuts = best
            if cx is not None:
                root_x, root_basis = cx, cbasis
                root_bound = cb / sc.obj
            cut_gain = root_bound - before

    if not np.isfinite(root_bound):
        root_bound = -np.inf

    incumbent = np.inf
    best_x = None

    root_cand = _round_and_repair(scaled, root_x, int_mask, lo0, hi0, tol)

    frontier: list[_Node] = []
    root_node = _Node((), root_bound, 0)
    root_node.parent_id = 0          # key 0 is free: _order starts at 1
    heapq.heappush(frontier, root_node)
    pc = _Pseudocost(n)

    nodes = 0
    status = Status.NODE_LIMIT
    last_log = t0
    binary_mask = (int_mask & (lo0 >= -1e-9) & (hi0 <= 1.0 + 1e-9))
    conflict = None
    if params.conflict and not use_bnr and binary_mask.any():
        conflict = ConflictAnalyzer(root_lo=lo0.copy(), root_hi=hi0.copy(),
                                    binary=binary_mask)
    pending_clauses: list = []

    duals: dict = {}
    warm_cache: dict = {}
    if root_basis is not None:
        warm_cache[0] = root_basis
    warm_hits = 0
    warm_tries = 0
    node_infeasible = 0
    node_lp_rows = scaled.m if node_lp is not None else -1
    # Nodes left neither expanded nor proved empty. Any such node makes an
    # exhausted frontier stop being a proof, so the report below must not
    # promote the run to OPTIMAL or read a dual bound off the incumbent.
    undecided = 0

    def _accept(cand_scaled):
        """Unscale a candidate and validate it against the *original* model.

        Feasibility must be judged where the answer will be read, not in the
        scaled space the search happens to run in. Row scale factors here go
        down to 1e-3, so a scaled violation of 1e-6 is an unscaled violation of
        1e-3 -- a thousand times past what the independent verifier accepts.
        Checking in the scaled space lets the tree adopt an infeasible point as
        its incumbent and report it as the answer.

        Returns ``(x, objective)`` in the working problem's space, or None.
        """
        x = np.clip(sc.unscale_primal(cand_scaled), work.col_lb, work.col_ub)
        ii = work.integer_mask
        if ii.any():
            x[ii] = np.round(x[ii])
        rv, bv, iv = work.violation(x)
        if max(rv, bv) > ACCEPT_FEAS_TOL or iv > tol.integrality:
            return None
        return x, float(work.c @ x) + work.obj_offset

    def _obj(xv):
        """Objective in the *working* problem's units.

        The batched bounds are converted to these units in the BNR engine, so
        the incumbent must be too -- otherwise the gap compares two different
        scales and the search terminates at the wrong point.
        """
        return (float(scaled.c @ xv) + scaled.obj_offset) / sc.obj

    if root_cand is not None:
        got = _accept(root_cand)
        if got is not None:
            best_x, incumbent = got[0], got[1]

    # ---- primal heuristics at the root ---------------------------------- #
    # Order matters: cheapest first, and each is skipped once an incumbent
    # exists that it is unlikely to beat. Finding *any* feasible point is the
    # binding constraint on the instances this solver fails.
    heur = HeuristicStats()
    last_heuristic = 0
    heur_spent = 0.0
    heur_budget = params.cut_time_frac * params.time_limit
    """Total wall time the heuristics may consume, root and restarts together.

    Needed because none of them accept a deadline: they stop on iteration
    counts, so a call cannot be cut short once entered and the between-trial
    check cannot bound one that is already running. Without this cap the
    restarts turned a 600 s limit into a 5,445 s run."""
    if params.heuristics:
        def _lp_at(l, h, obj=None):
            saved = None
            if obj is not None:
                saved = scaled.c.copy()
                scaled.c = np.ascontiguousarray(obj, dtype=VAL)
                node_lp.S.B.cost[:n] = scaled.c
            try:
                rl = node_lp.solve(l, h)
                return rl.x if rl.status == Status.OPTIMAL else None
            finally:
                if saved is not None:
                    scaled.c = saved
                    node_lp.S.B.cost[:n] = saved

        def _heuristic_round(x0, hl, hh, deadline, cheap_only=False):
            """Run the primal heuristics from one point and box.

            Callable more than once, which it needs to be. Run only at the
            root, all three failed on dcmulti -- ``0/1`` apiece -- and the
            search then explored 1,811 nodes without trying again and returned
            no answer at all. Which vertex a node LP hands the rounding
            heuristic is luck, and a single draw of it is a thin basis for
            giving up: the comment above says finding *any* feasible point is
            the binding constraint on the instances this solver fails, and
            trying once does not act like it.
            """
            nonlocal best_x, incumbent, heur_spent
            _h0 = time.perf_counter()
            trials = [
                ("fix_and_propagate",
                 lambda: fix_and_propagate(scaled, x0, int_mask, hl, hh,
                                           tol.primal_feas,
                                           lp_solve=(None if use_bnr
                                                     else (lambda l, h: _lp_at(l, h))))),
                ("feasibility_jump",
                 lambda: feasibility_jump(scaled, int_mask, hl, hh, x0=x0,
                                          max_iter=params.fj_iterations)),
            ]
            # The pump costs up to 40 LP solves and has no internal time
            # bound, so it runs at the root only. None of these take a
            # deadline -- they stop on iteration counts -- which means one call
            # cannot be cut short, and that is what makes the budget below
            # necessary rather than tidy.
            if not use_bnr and not cheap_only:
                trials.append(("feasibility_pump",
                               lambda: feasibility_pump(scaled, int_mask, hl, hh,
                                                        _lp_at, x_lp=x0)))
            try:
                for name, fn in trials:
                    if np.isfinite(incumbent):
                        break
                    if time.perf_counter() > deadline:
                        break
                    try:
                        cand = fn()
                    except Exception:
                        cand = None
                    got = _accept(cand) if cand is not None else None
                    heur.record(name, got is not None)
                    if got is not None and got[1] < incumbent:
                        best_x, incumbent = got[0], got[1]
            finally:
                heur_spent += time.perf_counter() - _h0

        _heuristic_round(root_x, lo0, hi0,
                         t0 + params.cut_time_frac * params.time_limit)

    while frontier:
        now = time.perf_counter()
        if now - t0 > params.time_limit:
            status = Status.TIME_LIMIT
            break
        if nodes >= params.node_limit:
            status = Status.NODE_LIMIT
            break

        best_bound = frontier[0].bound
        if np.isfinite(incumbent):
            if incumbent - best_bound <= max(params.gap_abs,
                                             params.gap_rel * abs(incumbent)):
                status = Status.OPTIMAL
                break

        # ---- pop a slab ---------------------------------------------------
        slab: list[_Node] = []
        while frontier and len(slab) < params.batch:
            nd = heapq.heappop(frontier)
            if np.isfinite(incumbent) and nd.bound >= incumbent - max(
                    params.gap_abs, params.gap_rel * abs(incumbent)):
                continue
            slab.append(nd)
        if not slab:
            status = Status.OPTIMAL
            break

        K = len(slab)
        LO = np.empty((n, K), dtype=VAL)
        HI = np.empty((n, K), dtype=VAL)
        alive = []
        for t, nd in enumerate(slab):
            l, h = nd.bounds(lo0, hi0)
            pr = propagate(scaled.A, scaled.row_lb, scaled.row_ub, l, h,
                           int_mask, max_rounds=2, feas_tol=tol.primal_feas,
                           inplace=True)
            if pr.infeasible:
                continue
            LO[:, len(alive)] = pr.lo
            HI[:, len(alive)] = pr.hi
            alive.append(nd)
        if not alive:
            continue

        Ka = len(alive)

        # ---- bound the slab -----------------------------------------------
        # Two strategies, same interface: fill `bounds` (in working-problem
        # objective units) and `xs` (relaxation points for the heuristics).
        bounds = np.empty(Ka, dtype=VAL)
        xs = np.zeros((n, Ka), dtype=VAL)

        if use_bnr:
            # A child's LP differs from its parent's by one bound, so the
            # parent's dual iterate is an excellent starting point. Duals are
            # cached per node rather than stored on it: an (m,)-vector on every
            # open node would dominate memory on a large model.
            warm = None
            if engine.cfg.warm_start:
                Yw = np.zeros((scaled.m, Ka), dtype=VAL)
                Xw = np.zeros((n, Ka), dtype=VAL)
                hit = 0
                for t, nd in enumerate(alive):
                    cached = duals.get(nd.parent_id)
                    if cached is not None:
                        Yw[:, t] = cached[0]
                        Xw[:, t] = cached[1]
                        hit += 1
                if hit:
                    warm = (Xw, Yw)
                warm_hits += hit
                warm_tries += Ka

            res = engine.solve_batch(LO[:, :Ka], HI[:, :Ka], warm=warm)
            bounds[:] = res.bounds
            xs[:] = res.x

            Yout = engine.bk.to_host(engine.Y[:, :Ka])
            for t, nd in enumerate(alive):
                duals[nd._order] = (Yout[:, t].copy(), res.x[:, t].copy())
            while len(duals) > 6 * params.batch:
                duals.pop(next(iter(duals)))
        else:
            # Exact node LPs. The parent's basis is still dual feasible at the
            # child -- reduced costs do not depend on the bounds -- so the dual
            # simplex resumes from it in a few pivots.
            for t, nd in enumerate(alive):
                wb = warm_cache.get(nd.parent_id)
                warm_tries += 1
                if wb is not None:
                    warm_hits += 1
                r = node_lp.solve(LO[:, t], HI[:, t], warm_basis=wb)
                if r.status == Status.INFEASIBLE:
                    bounds[t] = np.inf
                    node_infeasible += 1
                    if conflict is not None and r.farkas is not None:
                        cl = conflict.analyse(scaled.A, scaled.row_lb,
                                              scaled.row_ub, LO[:, t], HI[:, t],
                                              r.farkas)
                        if cl is not None:
                            pending_clauses.append(cl)
                elif r.status == Status.OPTIMAL and r.x is not None:
                    bounds[t] = r.objective / sc.obj
                    xs[:, t] = r.x
                    warm_cache[nd._order] = r.basis
                else:
                    bounds[t] = nd.bound
            while len(warm_cache) > 8 * params.batch:
                k = next(iter(warm_cache))
                if k == 0:
                    warm_cache.pop(k)
                    continue
                warm_cache.pop(k)

        nodes += Ka

        if pending_clauses and len(pending_clauses) >= params.clause_batch:
            scaled = append_cuts(scaled, [Cut(i, v, rr, kind="conflict")
                                          for i, v, rr in pending_clauses])
            node_lp = NodeSolver(scaled, SimplexParams(
                time_limit=params.time_limit,
                feas_tol=tol.primal_feas, opt_tol=tol.dual_feas))
            warm_cache.clear()          # basis size changed
            pending_clauses = []

        # ---- process the slab ---------------------------------------------
        for t, nd in enumerate(alive):
            b = float(bounds[t])
            if not np.isfinite(b):
                b = nd.bound                       # vacuous bound: keep parent's
            b = max(b, nd.bound)                   # bounds only improve downward

            if np.isfinite(incumbent) and b >= incumbent - max(
                    params.gap_abs, params.gap_rel * abs(incumbent)):
                continue

            xv = xs[:, t]
            l = LO[:, t]
            h = HI[:, t]

            # integrality check on this node's relaxation
            xi = xv[int_mask]
            frac_all = np.abs(xi - np.round(xi))
            if frac_all.size == 0 or frac_all.max() <= tol.integrality:
                got = _accept(np.clip(xv, l, h))
                if got is not None and got[1] < incumbent:
                    best_x, incumbent = got[0], got[1]
                # Closing the node here needs this point to be the node's LP
                # *optimum*: an integral optimum is feasible, so nothing below
                # it can be better and the subtree is genuinely finished. An
                # exact node LP delivers that. A BNR iterate does not -- it is
                # an unconverged first-order point that merely looks integral,
                # and _accept rounds before it validates, so it will happily
                # turn that point into a feasible incumbent. Reading the
                # incumbent as proof discards the subtree holding the real
                # optimum: a five-variable model came back OPTIMAL at -134
                # against a true -136, the bound agreeing with the wrong
                # answer because the node that refuted it was never opened.
                # When the bound came from BNR, fall through to the exact
                # re-solve below and decide from a vertex.
                if got is not None and not use_bnr:
                    continue

            # primal heuristic on the node's fractional point
            if params.heuristic_every:
                capt = _round_and_repair(scaled, xv, int_mask, l, h, tol)
                if capt is not None:
                    got = _accept(capt)
                    if got is not None and got[1] < incumbent:
                        best_x, incumbent = got[0], got[1]

            # Still nothing after a stretch of search: run the full heuristics
            # again, from this node's relaxation point and box rather than the
            # root's. Cheap rounding has now failed on hundreds of vertices, so
            # the expensive ones have earned another attempt.
            if (params.heuristics and not np.isfinite(incumbent)
                    and params.heuristic_restart
                    and heur_spent < heur_budget
                    and nodes - last_heuristic >= params.heuristic_restart):
                last_heuristic = nodes
                _heuristic_round(
                    xv, l, h,
                    min(t0 + params.time_limit,
                        time.perf_counter()
                        + max(1.0, min(5.0, 0.02 * params.time_limit))),
                    cheap_only=True)

            # ---- branch ---------------------------------------------------
            idx = np.flatnonzero(int_mask)
            fr = np.abs(xv[idx] - np.round(xv[idx]))
            cands = idx[fr > tol.integrality]
            if cands.size == 0 and use_bnr:
                # No fractional variable, and _accept above did not take the
                # point. With an exact node LP that cannot happen: an
                # integral vertex is feasible, so the node really is done.
                # A BNR iterate is not a certificate of anything -- an
                # unconverged first-order point can look integral while
                # being LP-infeasible, and then there is nothing to branch
                # on. Dropping the node would discard that subtree unproven;
                # on flugpl it discards the whole tree and reports
                # INFEASIBLE on a model with a published optimum. Re-solve
                # the node exactly and decide from a real vertex.
                if node_lp is None or node_lp_rows != scaled.m:
                    node_lp = NodeSolver(scaled, SimplexParams(
                        time_limit=params.time_limit,
                        feas_tol=tol.primal_feas, opt_tol=tol.dual_feas))
                    node_lp_rows = scaled.m
                r_exact = node_lp.solve(l, h)
                if r_exact.status == Status.OPTIMAL and r_exact.x is not None:
                    xv = r_exact.x
                    b = max(b, r_exact.objective / sc.obj)
                    got = _accept(np.clip(xv, l, h))
                    if got is not None and got[1] < incumbent:
                        best_x, incumbent = got[0], got[1]
                    # Recompute the candidates from the *vertex*, whatever
                    # _accept did with it. _accept rounds the integer
                    # variables, so it is a repair heuristic as much as a
                    # validator: on a fractional vertex it often succeeds and
                    # returns a perfectly good incumbent, which says nothing
                    # about whether this node is finished. Gating the
                    # recompute on ``got is None`` therefore dropped precisely
                    # the nodes whose rounding worked -- subtree and all --
                    # and an empty frontier is read below as proof of
                    # optimality, so the search reported OPTIMAL with
                    # dual_bound == incumbent, one unit above the true
                    # optimum on a five-variable model.
                    fr = np.abs(xv[idx] - np.round(xv[idx]))
                    cands = idx[fr > tol.integrality]
                elif r_exact.status != Status.INFEASIBLE:
                    # Neither solved nor proved empty, so nothing at all is
                    # known about this node. INFEASIBLE is the one safe case:
                    # the node really is gone. Anything else falls through to
                    # the drop below, and a dropped node is what turns an
                    # empty frontier into a proof that was never made.
                    undecided += 1
            if cands.size == 0:
                continue

            best_j, best_s = -1, -np.inf
            for j in cands:
                fd = xv[j] - np.floor(xv[j])
                fu = np.ceil(xv[j]) - xv[j]
                s = pc.score(j, fd, fu)
                if s > best_s:
                    best_s, best_j = s, j
            j = int(best_j)
            val = xv[j]
            fd = val - np.floor(val)
            fu = np.ceil(val) - val

            fl = float(np.floor(val))
            cl = float(np.ceil(val))
            # only create a child whose branch actually restricts the node
            if fl >= l[j] - 1e-9:
                heapq.heappush(frontier,
                               _Node(nd.path + ((j, False, fl),), b,
                                     nd.depth + 1, fd, parent_id=nd._order))
            if cl <= h[j] + 1e-9:
                heapq.heappush(frontier,
                               _Node(nd.path + ((j, True, cl),), b,
                                     nd.depth + 1, fu, parent_id=nd._order))

        if params.verbose and time.perf_counter() - last_log > params.log_every:
            last_log = time.perf_counter()
            bb = frontier[0].bound if frontier else incumbent
            gap = (abs(incumbent - bb) / max(1.0, abs(incumbent))
                   if np.isfinite(incumbent) else float("inf"))
            print(f"  nodes {nodes:>8d}  open {len(frontier):>7d}  "
                  f"bound {bb:< 14.8g} incumbent "
                  f"{incumbent if np.isfinite(incumbent) else float('nan'):< 14.8g} "
                  f"gap {gap:8.3%}  {time.perf_counter()-t0:6.1f}s")

    # ---- report ----------------------------------------------------------- #
    # Falling out of the loop with nothing left to explore means the search is
    # exhausted: every node was either pruned or expanded, so the incumbent is
    # proved optimal. Leaving the initial NODE_LIMIT status in place here would
    # report a solved model as merely truncated.
    if not frontier and status in (Status.NODE_LIMIT,) and not undecided:
        status = Status.OPTIMAL

    dual_bound = frontier[0].bound if frontier else incumbent
    if undecided and not frontier:
        # The frontier emptied, but not every node in it was decided, so
        # there is no bound to report and nothing to call proved.
        dual_bound = -float("inf")
    if status == Status.OPTIMAL and not frontier:
        dual_bound = incumbent

    if np.isfinite(incumbent):
        # A dual bound can never exceed an objective actually achieved: the
        # optimum is at most the incumbent, and the bound is at most the
        # optimum. The clamp is needed because the gap rule stops the search
        # with a *non-empty* frontier the moment every remaining node is
        # prunable -- which is precisely when frontier[0].bound has risen past
        # the incumbent. The branch above only rewrites the bound when the
        # frontier emptied, so that exit reported OPTIMAL alongside a bound
        # above the optimum it had just proved.
        dual_bound = min(dual_bound, incumbent)

    # A MAXIMISE model was negated on the way in and the whole search ran as a
    # minimisation, so every bound above lives in that negated space. Both
    # return paths report in the caller's sense, so the flip belongs here,
    # above the split -- applying it only on the path that happens to have an
    # incumbent left the other one reporting the negation of its bound. With
    # all-positive costs that is not merely the wrong sign but a number below
    # every feasible objective, offered as an upper bound. The no-incumbent
    # return is the common case on a model too big to crack inside its limit,
    # which is exactly when a user reads the bound to decide whether to keep
    # going.
    db = -dual_bound if flip else dual_bound

    if best_x is None:
        # Attach the diagnostics here too. They were dropped on this path, so
        # the one outcome that most needs explaining -- no answer at all --
        # was the one that arrived with nothing to explain it: no heuristic
        # counts, no cut counts, no node statistics.
        out = Solution(status=Status.INFEASIBLE if status == Status.OPTIMAL
                       else status,
                       nodes=nodes, time=time.perf_counter() - t0,
                       dual_bound=db, method="bnr-bb")
        out.info = (engine.stats() if engine is not None else node_lp.stats())
        out.info["node_solver"] = params.node_solver
        out.info["root_cuts"] = n_cuts
        out.info["root_mir_cuts"] = mir_used
        out.info["heuristics"] = heur.summary()
        out.info["nodes_infeasible"] = node_infeasible
        out.info["undecided_nodes"] = undecided
        return out

    # best_x is already unscaled and validated by _accept
    x_orig = best_x
    obj = float(prob.c @ x_orig) + prob.obj_offset

    sol = Solution(status=status, x=x_orig, objective=obj,
                   dual_bound=db, nodes=nodes,
                   time=time.perf_counter() - t0, method="bnr-bb")
    sol.info = (engine.stats() if engine is not None else node_lp.stats())
    sol.info["node_solver"] = params.node_solver
    sol.info["root_cuts"] = n_cuts
    if conflict is not None:
        sol.info["conflict"] = conflict.stats()
    if sym_info is not None:
        sol.info["symmetry"] = sym_info.summary()
    sol.info["root_cut_bound_gain"] = cut_gain
    sol.info["root_mir_cuts"] = mir_used
    sol.info["heuristics"] = heur.summary()
    sol.info["warm_start_hit_rate"] = warm_hits / max(warm_tries, 1)
    sol.info["nodes_infeasible"] = node_infeasible
    return sol
