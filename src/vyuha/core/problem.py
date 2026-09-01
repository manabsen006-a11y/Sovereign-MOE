"""Canonical problem representation.

Internal form -- everything the solver touches is in this shape:

    minimise    ½ xᵀ Q x + cᵀ x + c₀
    subject to  rl ≤ A x ≤ ru
                 l ≤   x  ≤ u
                x_j integral for j ∈ integrality

The **row-bounds** form (rather than ``Ax ≤ b`` with explicit slacks) is what
every modern implementation uses, because it represents ``≤``, ``≥``, ``=`` and
MPS ``RANGES`` rows uniformly with no structural change to ``A``:

    ≤ row : rl = -inf, ru = b
    ≥ row : rl = b,    ru = +inf
    = row : rl = ru = b
    range : both finite

The simplex adds logical (slack) variables later to reach the computational
form ``[A | -I] [x; s] = 0`` with ``rl ≤ s ≤ ru``; see ``vyuha.lp.simplex``.
Keeping that step out of the model means presolve, propagation and the
first-order methods all see the same object.

References
----------
Maros, *Computational Techniques of the Simplex Method*, Kluwer 2003, Ch. 3 --
  general LP form with logical variables.
Achterberg, *Constraint Integer Programming*, PhD thesis, TU Berlin 2007, §2 --
  MIP model conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .sparse import SparseMatrix, VAL
from .tolerances import INF

__all__ = ["Problem", "VarKind", "ObjSense", "Status", "Solution"]


class VarKind(IntEnum):
    CONTINUOUS = 0
    INTEGER = 1
    BINARY = 2
    SEMI_CONTINUOUS = 3
    SEMI_INTEGER = 4


class ObjSense(IntEnum):
    MINIMISE = 1
    MAXIMISE = -1


class Status(IntEnum):
    OPTIMAL = 0
    INFEASIBLE = 1
    UNBOUNDED = 2
    INFEASIBLE_OR_UNBOUNDED = 3
    ITERATION_LIMIT = 4
    TIME_LIMIT = 5
    NODE_LIMIT = 6
    GAP_LIMIT = 7
    NUMERICAL = 8
    INTERRUPTED = 9
    NOT_SOLVED = 10

    @property
    def is_solved(self) -> bool:
        return self in (Status.OPTIMAL,)

    @property
    def has_solution(self) -> bool:
        return self in (Status.OPTIMAL, Status.ITERATION_LIMIT, Status.TIME_LIMIT,
                        Status.NODE_LIMIT, Status.GAP_LIMIT)


@dataclass
class Problem:
    """An LP, MILP, QP or MIQP in canonical form."""

    A: SparseMatrix
    c: np.ndarray
    row_lb: np.ndarray
    row_ub: np.ndarray
    col_lb: np.ndarray
    col_ub: np.ndarray

    kind: np.ndarray = None            # (n,) uint8 of VarKind
    Q: SparseMatrix | None = None      # symmetric Hessian; None for LP/MILP
    obj_offset: float = 0.0
    sense: ObjSense = ObjSense.MINIMISE

    name: str = "model"
    col_names: list[str] | None = None
    row_names: list[str] | None = None

    # populated by presolve so a solution can be lifted back to the original
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.c = np.ascontiguousarray(self.c, dtype=VAL)
        self.row_lb = np.ascontiguousarray(self.row_lb, dtype=VAL)
        self.row_ub = np.ascontiguousarray(self.row_ub, dtype=VAL)
        self.col_lb = np.ascontiguousarray(self.col_lb, dtype=VAL)
        self.col_ub = np.ascontiguousarray(self.col_ub, dtype=VAL)
        if self.kind is None:
            self.kind = np.zeros(self.n, dtype=np.uint8)
        else:
            self.kind = np.ascontiguousarray(self.kind, dtype=np.uint8)
        self._validate()

    def _validate(self):
        m, n = self.A.shape
        for arr, size, what in ((self.c, n, "c"),
                                (self.col_lb, n, "col_lb"),
                                (self.col_ub, n, "col_ub"),
                                (self.kind, n, "kind"),
                                (self.row_lb, m, "row_lb"),
                                (self.row_ub, m, "row_ub")):
            if arr.shape[0] != size:
                raise ValueError(f"{what} has length {arr.shape[0]}, expected {size}")
        if self.Q is not None and self.Q.shape != (n, n):
            raise ValueError(f"Q must be ({n}, {n}), got {self.Q.shape}")

    # -- shape -------------------------------------------------------------- #

    @property
    def m(self) -> int:
        return self.A.m

    @property
    def n(self) -> int:
        return self.A.n

    @property
    def nnz(self) -> int:
        return self.A.nnz

    @property
    def is_mip(self) -> bool:
        return bool((self.kind != VarKind.CONTINUOUS).any())

    @property
    def is_qp(self) -> bool:
        return self.Q is not None

    @property
    def integer_mask(self) -> np.ndarray:
        return (self.kind == VarKind.INTEGER) | (self.kind == VarKind.BINARY)

    @property
    def n_integer(self) -> int:
        return int(self.integer_mask.sum())

    @property
    def n_binary(self) -> int:
        m = self.integer_mask & (self.col_lb >= -1e-9) & (self.col_ub <= 1 + 1e-9)
        return int(m.sum())

    # -- row typing --------------------------------------------------------- #

    def row_types(self):
        """Counts of ``(equality, ranged, less, greater, free)`` rows."""
        lo, hi = self.row_lb, self.row_ub
        fin_lo, fin_hi = lo > -INF, hi < INF
        eq = int((fin_lo & fin_hi & (hi - lo <= 1e-12)).sum())
        rng = int((fin_lo & fin_hi & (hi - lo > 1e-12)).sum())
        le = int((~fin_lo & fin_hi).sum())
        ge = int((fin_lo & ~fin_hi).sum())
        free = int((~fin_lo & ~fin_hi).sum())
        return eq, rng, le, ge, free

    # -- evaluation --------------------------------------------------------- #

    def objective(self, x) -> float:
        """Objective value in the *original* sense, including the offset."""
        x = np.asarray(x, dtype=VAL)
        val = float(self.c @ x) + self.obj_offset
        if self.Q is not None:
            val += 0.5 * float(x @ self.Q.matvec(x))
        return val

    def row_activity(self, x, out=None):
        return self.A.matvec(x, out=out)

    def violation(self, x):
        """``(max row violation, max bound violation, max integrality violation)``
        for a candidate point. This is the definition of feasible used
        everywhere, including by the independent verifier."""
        x = np.asarray(x, dtype=VAL)
        act = self.A.matvec(x)
        row_v = max(
            float(np.maximum(self.row_lb - act, 0.0).max(initial=0.0)),
            float(np.maximum(act - self.row_ub, 0.0).max(initial=0.0)),
        )
        col_v = max(
            float(np.maximum(self.col_lb - x, 0.0).max(initial=0.0)),
            float(np.maximum(x - self.col_ub, 0.0).max(initial=0.0)),
        )
        mask = self.integer_mask
        int_v = float(np.abs(x[mask] - np.round(x[mask])).max(initial=0.0)) if mask.any() else 0.0
        return row_v, col_v, int_v

    # -- reporting ---------------------------------------------------------- #

    def stats(self) -> dict:
        eq, rng, le, ge, free = self.row_types()
        lo, hi = self.A.coeff_range()
        obj_nz = np.abs(self.c[self.c != 0.0])
        return {
            "name": self.name,
            "rows": self.m,
            "cols": self.n,
            "nonzeros": self.nnz,
            "density": self.A.density,
            "integers": self.n_integer,
            "binaries": self.n_binary,
            "continuous": self.n - self.n_integer,
            "eq_rows": eq, "range_rows": rng, "le_rows": le,
            "ge_rows": ge, "free_rows": free,
            "coeff_min": lo, "coeff_max": hi,
            "coeff_ratio": (hi / lo) if lo > 0 else float("inf"),
            "obj_min": float(obj_nz.min()) if obj_nz.size else 0.0,
            "obj_max": float(obj_nz.max()) if obj_nz.size else 0.0,
            "quadratic": self.Q is not None,
        }

    def summary(self) -> str:
        s = self.stats()
        kindstr = "MIQP" if s["quadratic"] and s["integers"] else \
                  "QP" if s["quadratic"] else \
                  "MILP" if s["integers"] else "LP"
        return (
            f"{s['name']}  [{kindstr}]\n"
            f"  {s['rows']:,} rows x {s['cols']:,} cols, {s['nonzeros']:,} nonzeros "
            f"(density {s['density']:.2e})\n"
            f"  rows: {s['eq_rows']:,} eq, {s['le_rows']:,} <=, {s['ge_rows']:,} >=, "
            f"{s['range_rows']:,} ranged, {s['free_rows']:,} free\n"
            f"  cols: {s['continuous']:,} continuous, {s['integers']:,} integer "
            f"({s['binaries']:,} binary)\n"
            f"  |a_ij| in [{s['coeff_min']:.3e}, {s['coeff_max']:.3e}]  "
            f"ratio {s['coeff_ratio']:.2e}"
        )

    def copy(self) -> "Problem":
        return Problem(
            A=self.A.copy(), c=self.c.copy(),
            row_lb=self.row_lb.copy(), row_ub=self.row_ub.copy(),
            col_lb=self.col_lb.copy(), col_ub=self.col_ub.copy(),
            kind=self.kind.copy(),
            Q=self.Q.copy() if self.Q is not None else None,
            obj_offset=self.obj_offset, sense=self.sense, name=self.name,
            col_names=list(self.col_names) if self.col_names else None,
            row_names=list(self.row_names) if self.row_names else None,
            meta=dict(self.meta),
        )


@dataclass
class Solution:
    """Result of a solve."""

    status: Status
    x: np.ndarray | None = None
    objective: float = float("nan")

    # LP duals
    y: np.ndarray | None = None            # row duals
    reduced_costs: np.ndarray | None = None
    basis_status: np.ndarray | None = None

    # MIP
    dual_bound: float = float("nan")
    nodes: int = 0
    gap_rel: float = float("nan")

    iterations: int = 0
    time: float = 0.0
    work_units: float = 0.0
    method: str = ""
    log: list = field(default_factory=list)

    @property
    def gap(self) -> float:
        """Relative MIP gap, |primal - dual| / max(1, |primal|)."""
        if not np.isfinite(self.objective) or not np.isfinite(self.dual_bound):
            return float("inf")
        return abs(self.objective - self.dual_bound) / max(1.0, abs(self.objective))

    def __repr__(self):
        obj = f"{self.objective:.10g}" if np.isfinite(self.objective) else "-"
        s = f"Solution({self.status.name}, obj={obj}"
        if np.isfinite(self.dual_bound) and self.nodes:
            s += f", bound={self.dual_bound:.10g}, gap={self.gap:.3%}, nodes={self.nodes}"
        s += f", iters={self.iterations}, {self.time:.3f}s)"
        return s
