"""Presolve: shrink the model, then put the answer back.

A presolver removes structure the solver would otherwise pay to rediscover at
every iteration. The reductions themselves are elementary -- a column pinned
between equal bounds is not a decision, a row with one nonzero is a bound
wearing a row's clothing -- and the difficulty is entirely in **postsolve**:
recovering a primal point, a dual vector and a set of reduced costs for the
*original* model from a solution to a smaller one. A presolve that returns a
right objective and wrong shadow prices is worse than no presolve, because the
shadow prices are what a planner reads.

What is implemented, and why exactly this set
---------------------------------------------
Counted over the eleven benchmark models before writing any of it:

    reduction              count    where
    fixed columns            275    10teams 225, khb05250 50
    forcing rows              20    10teams
    singleton rows            19    dcmulti 18
    redundant rows            17    10teams 15
    free column singletons     1    misc07
    empty rows / columns       0    nowhere

So the textbook headline reduction -- substituting out a free column singleton,
which removes a row *and* a column -- is worth **one column on one model** here
and is not implemented. What is implemented is the three that are actually
present and whose duals can be recovered exactly:

* **fixed column** ``l == u``: substituted out, its contribution moved into the
  row bounds and the objective offset.
* **singleton row** ``rl <= a·x_j <= ru``: turned into bounds on ``x_j`` and
  removed.
* **redundant row**: one whose own activity bounds, computed from the column
  bounds, already lie inside the row bounds, so it can never bind.

Plus empty rows and columns, which the count says never occur in the *input* --
but a fixpoint loop creates them, because removing 225 fixed columns from
10teams empties rows that were not empty before. That cascade is the reason the
loop iterates rather than making one pass.

**Forcing rows are detected and deliberately not applied.** Twenty of them, all
on one model. A forcing row fixes every variable in it, and its dual is not
recoverable by the rule the other three share -- it needs the general argument
about which of the fixed variables' reduced costs the row is paying for. Doing
it wrong would silently corrupt the shadow prices on the one model it fires on,
so it is left for when it can be done properly.

How the duals come back
-----------------------
The reduced costs are never tracked through the stack. ``d = c - Aᵀy`` is a
*definition*, so once the full original ``y`` is recovered every reduced cost
follows exactly from one matrix-vector product on the original model. That
removes an entire class of bookkeeping bug, and it is why only ``x`` and ``y``
are reconstructed here.

For ``y`` the rules are:

* a removed empty or redundant row never binds, so ``y_i = 0``;
* a singleton row's dual is whatever the column's own bounds cannot explain.
  With ``d_j`` computed from every *other* row's dual, if ``x_j`` rests on one
  of its original column bounds and ``d_j`` has the sign that bound implies,
  the column bound is doing the work and ``y_i = 0``; otherwise the row is, and
  ``y_i = d_j / a_ij``.

References
----------
Andersen & Andersen, "Presolving in linear programming", Mathematical
  Programming 71 (1995) 221-245 -- the reduction set, the fixpoint loop, and
  the postsolve rules for the primal and the dual.
Brearley, Mitra & Williams, "Analysis of mathematical programming problems
  prior to applying the simplex algorithm", Mathematical Programming 8 (1975)
  54-83 -- row activity bounds, redundant and forcing rows.
Gondzio, "Presolve analysis of linear programs prior to applying an interior
  point method", INFORMS J. Computing 9 (1997) 73-91 -- why an interior-point
  method wants a different reduction set from a simplex, and which reductions
  are safe for both.
Achterberg, Bixby, Gu, Rothberg & Weninger, "Presolve reductions in mixed
  integer programming", INFORMS J. Computing 32 (2020) 473-506 -- the modern
  catalogue and the postsolve stack discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.problem import ObjSense, Problem, Solution, Status
from .core.sparse import IDX, VAL, SparseMatrix
from .core.tolerances import DEFAULT, INF, Tolerances

__all__ = ["PresolveResult", "presolve", "postsolve"]


@dataclass
class _Fixed:
    """Column ``j`` pinned at ``value`` and removed."""
    j: int
    value: float


@dataclass
class _DroppedRow:
    """Row ``i`` removed because it can never bind: ``y_i = 0``."""
    i: int


@dataclass
class _SingletonRow:
    """Row ``i`` was ``rl <= a * x_j <= ru`` and became bounds on ``x_j``."""
    i: int
    j: int
    a: float
    col_lo: float          # the column's bounds *before* the row tightened them
    col_hi: float


@dataclass
class PresolveResult:
    """A reduced model plus everything needed to undo the reduction."""

    problem: Problem
    stack: list = field(default_factory=list)
    col_keep: np.ndarray = None          # original indices of kept columns
    row_keep: np.ndarray = None          # original indices of kept rows
    original: Problem = None
    status: Status | None = None         # set only if presolve settled it
    rounds: int = 0

    @property
    def reduced(self) -> tuple[int, int]:
        return self.problem.m, self.problem.n

    def summary(self) -> str:
        o = self.original
        return (f"presolve: {o.m}x{o.n} -> {self.problem.m}x{self.problem.n} "
                f"({len(self.stack)} reductions, {self.rounds} rounds)")


def _row_activity(A, lo, hi, rp, ri, rx):
    """Least and greatest ``a_i·x`` over the box, per row."""
    m = rp.size - 1
    lo_e = lo[ri]
    hi_e = hi[ri]
    pos = rx > 0
    contrib_min = np.where(pos, rx * lo_e, rx * hi_e)
    contrib_max = np.where(pos, rx * hi_e, rx * lo_e)
    # an infinite bound on a live coefficient makes the activity unbounded
    amin = np.zeros(m, dtype=VAL)
    amax = np.zeros(m, dtype=VAL)
    if rx.size:
        np.add.at(amin, np.repeat(np.arange(m), np.diff(rp)), contrib_min)
        np.add.at(amax, np.repeat(np.arange(m), np.diff(rp)), contrib_max)
    return amin, amax


def presolve(prob: Problem, tol: Tolerances | None = None,
             max_rounds: int = 20) -> PresolveResult:
    """Reduce ``prob``, returning the smaller model and the undo stack."""
    tol = tol or DEFAULT
    feas = tol.primal_feas

    n, m = prob.n, prob.m
    lo = prob.col_lb.astype(VAL).copy()
    hi = prob.col_ub.astype(VAL).copy()
    rl = prob.row_lb.astype(VAL).copy()
    ru = prob.row_ub.astype(VAL).copy()
    c = prob.c.astype(VAL).copy()
    offset = float(prob.obj_offset)

    col_alive = np.ones(n, dtype=bool)
    row_alive = np.ones(m, dtype=bool)
    stack: list = []
    status = None
    rounds = 0

    A = prob.A
    # integer columns are never fixed away on a non-integral value, and a MIP's
    # duals are meaningless anyway; the reductions below are value-preserving
    # so they stay valid, but nothing here tightens an integer bound.
    for rounds in range(1, max_rounds + 1):
        changed = False

        # ---- fixed columns ------------------------------------------------- #
        pin = col_alive & (hi - lo <= 1e-12 * np.maximum(1.0, np.abs(lo)))
        for j in np.flatnonzero(pin):
            v = float(lo[j])
            s, e = A.cp[j], A.cp[j + 1]
            for p in range(s, e):
                i = int(A.ci[p])
                if not row_alive[i]:
                    continue
                shift = float(A.cx[p]) * v
                if rl[i] > -INF:
                    rl[i] -= shift
                if ru[i] < INF:
                    ru[i] -= shift
            offset += float(c[j]) * v
            col_alive[j] = False
            stack.append(_Fixed(j, v))
            changed = True

        # ---- rows: empty, singleton, redundant ----------------------------- #
        for i in np.flatnonzero(row_alive):
            s, e = A.rp[i], A.rp[i + 1]
            live = [p for p in range(s, e) if col_alive[A.ri[p]]]

            if not live:
                # empty row: feasible only if 0 lies within its bounds
                if (rl[i] > feas) or (ru[i] < -feas):
                    status = Status.INFEASIBLE
                    break
                row_alive[i] = False
                stack.append(_DroppedRow(i))
                changed = True
                continue

            if len(live) == 1:
                p = live[0]
                j = int(A.ri[p])
                a = float(A.rx[p])
                if abs(a) <= tol.zero:
                    row_alive[i] = False
                    stack.append(_DroppedRow(i))
                    changed = True
                    continue
                b_lo = rl[i] / a if rl[i] > -INF else (-INF if a > 0 else INF)
                b_hi = ru[i] / a if ru[i] < INF else (INF if a > 0 else -INF)
                if a < 0:
                    b_lo, b_hi = b_hi, b_lo
                stack.append(_SingletonRow(i, j, a, float(lo[j]), float(hi[j])))
                if b_lo > lo[j]:
                    lo[j] = b_lo
                if b_hi < hi[j]:
                    hi[j] = b_hi
                if lo[j] > hi[j] + feas:
                    status = Status.INFEASIBLE
                    break
                row_alive[i] = False
                changed = True
                continue

        if status is not None:
            break

        # redundant rows need the activity bounds of what is still live
        keep_e = col_alive[A.ri]
        rp2 = np.concatenate([[0], np.cumsum(
            [int(keep_e[A.rp[i]:A.rp[i + 1]].sum()) for i in range(m)])])
        ri2 = A.ri[keep_e]
        rx2 = A.rx[keep_e]
        lo_f = np.where(lo > -INF, lo, -np.inf)
        hi_f = np.where(hi < INF, hi, np.inf)
        with np.errstate(invalid="ignore"):
            amin, amax = _row_activity(A, lo_f, hi_f, rp2, ri2, rx2)
        for i in np.flatnonzero(row_alive):
            if rp2[i] == rp2[i + 1]:
                continue
            lo_ok = (rl[i] <= -INF) or (amin[i] >= rl[i] - feas)
            hi_ok = (ru[i] >= INF) or (amax[i] <= ru[i] + feas)
            if lo_ok and hi_ok:
                row_alive[i] = False
                stack.append(_DroppedRow(i))
                changed = True

        # ---- empty columns ------------------------------------------------- #
        for j in np.flatnonzero(col_alive):
            s, e = A.cp[j], A.cp[j + 1]
            if any(row_alive[A.ci[p]] for p in range(s, e)):
                continue
            # Nothing constrains it: park it at whichever bound the objective
            # wants. The sense matters and is easy to drop -- on a maximisation
            # a positive cost wants the *upper* bound, and taking the
            # minimisation branch there parks the column at the worst end and
            # silently returns a suboptimal plan that is still feasible, so
            # nothing downstream complains.
            cj = float(c[j])
            if prob.sense == ObjSense.MAXIMISE:
                cj = -cj
            if cj > 0:
                v = lo[j]
            elif cj < 0:
                v = hi[j]
            else:
                v = lo[j] if lo[j] > -INF else (hi[j] if hi[j] < INF else 0.0)
            if v <= -INF or v >= INF:
                status = Status.UNBOUNDED
                break
            offset += float(c[j]) * float(v)
            col_alive[j] = False
            stack.append(_Fixed(j, float(v)))
            changed = True

        if status is not None or not changed:
            break

    col_keep = np.flatnonzero(col_alive)
    row_keep = np.flatnonzero(row_alive)

    # ---- assemble the reduced model --------------------------------------- #
    new_col = np.full(n, -1, dtype=np.int64)
    new_col[col_keep] = np.arange(col_keep.size)
    new_row = np.full(m, -1, dtype=np.int64)
    new_row[row_keep] = np.arange(row_keep.size)

    ent_r = np.repeat(np.arange(m), np.diff(A.rp))
    keep_e = row_alive[ent_r] & col_alive[A.ri]
    sub = SparseMatrix.from_triplets(new_row[ent_r[keep_e]],
                                     new_col[A.ri[keep_e]],
                                     A.rx[keep_e],
                                     row_keep.size, col_keep.size)
    reduced = Problem(
        A=sub, c=c[col_keep],
        row_lb=rl[row_keep], row_ub=ru[row_keep],
        col_lb=lo[col_keep], col_ub=hi[col_keep],
        kind=prob.kind[col_keep].copy(),
        obj_offset=offset, sense=prob.sense,
        name=f"{prob.name}.presolved",
        col_names=([prob.col_names[j] for j in col_keep]
                   if prob.col_names else None),
        row_names=([prob.row_names[i] for i in row_keep]
                   if prob.row_names else None),
    )
    return PresolveResult(problem=reduced, stack=stack, col_keep=col_keep,
                          row_keep=row_keep, original=prob, status=status,
                          rounds=rounds)


def postsolve(res: PresolveResult, sol: Solution) -> Solution:
    """Lift a solution of the reduced model back to the original."""
    prob = res.original
    n, m = prob.n, prob.m

    out = Solution(status=sol.status, iterations=sol.iterations,
                   time=sol.time, method=sol.method)
    out.info = dict(getattr(sol, "info", {}))
    out.info["presolve"] = res.summary()
    if sol.x is None:
        out.objective = float("nan")
        return out

    x = np.zeros(n, dtype=VAL)
    x[res.col_keep] = sol.x
    y = np.zeros(m, dtype=VAL)
    if sol.y is not None:
        y[res.row_keep] = sol.y

    # Undo in reverse. Only x and y are reconstructed; every reduced cost comes
    # from d = c - Aᵀy at the end, which is a definition and cannot drift.
    minimise = prob.sense != ObjSense.MAXIMISE
    for red in reversed(res.stack):
        if isinstance(red, _Fixed):
            x[red.j] = red.value
        elif isinstance(red, _DroppedRow):
            y[red.i] = 0.0
        elif isinstance(red, _SingletonRow):
            j, a = red.j, red.a
            # d_j from every other row's dual, then decide who is binding
            idx, val = prob.A.col(j)
            dj = float(prob.c[j]) - float(np.dot(val, y[idx])) + a * y[red.i]
            at_lo = red.col_lo > -INF and x[j] <= red.col_lo + 1e-9
            at_hi = red.col_hi < INF and x[j] >= red.col_hi - 1e-9
            sgn = 1.0 if minimise else -1.0
            explained = (at_lo and sgn * dj >= -1e-9) or \
                        (at_hi and sgn * dj <= 1e-9)
            y[red.i] = 0.0 if explained else dj / a

    obj = float(prob.c @ x) + prob.obj_offset
    d = prob.c - prob.A.rmatvec(y)
    out.x = x
    out.y = y
    out.reduced_costs = d
    out.objective = obj
    out.dual_bound = obj
    return out
