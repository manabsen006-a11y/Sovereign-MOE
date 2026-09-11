"""Read the QPLIB instance format.

QPLIB (Furini et al. 2019) is the reference collection of quadratic programs:
453 instances, each with a classification, a best-known solution point and
its objective value. It is what a QP engine is measured against, the way an
LP engine is measured against Netlib and a MIP engine against MIPLIB. The
instances come in GAMS, AMPL, CPLEX-LP and the library's own ``.qplib``
format; this reads the last, which is a plain sequence of numbers with the
structure decided by the three-letter problem type on its second line.

The format
----------
Every scalar is on its own line, optionally followed by ``# a comment`` -- the
published files carry one on every structural line, which this reader checks
against what it expects, so a misread section fails loudly at the line it
happened on instead of producing a plausible model with the wrong numbers.

    name
    type                 OVC: objective (L,D,C,Q), variables (C,B,M,I,G),
                              constraints (N,B,L,D,C,Q)
    minimize|maximize
    n                    variables
    m                    constraints        -- absent when C is N or B
    Q⁰ entries           -- absent when O is L: count, then "i j v", 1-based,
                            lower triangle (i >= j); the objective is
                            ½ Σ_{i>=j} v_ij x_i x_j + b⁰ᵀx + q⁰, so a symmetric
                            Hessian H with ½xᵀHx has H_ii = v_ii and
                            H_ij = H_ji = v_ij / 2 -- checked against the
                            published objective value at the solution point
    b⁰                   default, count of non-defaults, "i v"
    q⁰
    Qⁱ entries           -- only when C is Q or D or C: not supported here
    A entries            -- when m > 0: count, then "i j v"
    infinity             the value that stands for ∞ in the sections below
    lhs, rhs             -- when m > 0: default, count, "i v" each
    variable bounds      -- absent when V is B: lower then upper, same shape
    variable types       -- when V is M or G: default, count, "i t"
    starting point       primal, constraint duals (when m > 0), bound duals
    names                variable names then constraint names: count, "i name"

Only linear constraints are supported: a type whose third letter is Q, D or
C carries quadratic constraint terms and is refused rather than misread.

References
----------
Furini, Traversi, Belotti, Frangioni, Gleixner, Gould, Liberti, Lodi, Misener,
  Mittelmann, Sahinidis, Vigerske & Wiegele, "QPLIB: a library of quadratic
  programming instances", Math. Prog. Computation 11 (2019) 237-265 -- the
  collection, its classification and the file format in its appendix.
qplib.zib.de -- the instances, their solution points and objective values.
"""

from __future__ import annotations

import os

import numpy as np

from ..core.problem import ObjSense, Problem, VarKind
from ..core.sparse import IDX, SparseMatrix, VAL
from ..core.tolerances import INF

__all__ = ["read_qplib", "read_qplib_solution", "QPLIBError"]


class QPLIBError(ValueError):
    pass


class _Lines:
    def __init__(self, text: str):
        self.raw = text.split("\n")
        self.i = 0

    def next(self, expect=None) -> str:
        """The next non-empty line's value part; check its comment if any.

        ``expect`` is a phrase or a tuple of phrases that the comment must
        all contain, when there is a comment at all.
        """
        while self.i < len(self.raw):
            line = self.raw[self.i]
            self.i += 1
            value, _, comment = line.partition("#")
            value = value.strip()
            if not value:
                continue
            comment = comment.strip()
            if expect is not None and comment:
                phrases = (expect,) if isinstance(expect, str) else expect
                if not all(ph in comment for ph in phrases):
                    raise QPLIBError(
                        f"line {self.i}: expected {' and '.join(map(repr, phrases))}, "
                        f"the file says '{comment}' -- the sections are out "
                        f"of step")
            return value
        raise QPLIBError("unexpected end of file")

    def scalar(self, expect: str | None = None) -> float:
        return float(self.next(expect))

    def count(self, expect: str | None = None) -> int:
        return int(float(self.next(expect)))

    def entries(self, k: int, width: int):
        """``k`` lines of ``width`` numbers."""
        out = np.empty((k, width), dtype=np.float64)
        for r in range(k):
            parts = self.next().split()
            if len(parts) < width:
                raise QPLIBError(f"line {self.i}: expected {width} numbers")
            for c in range(width):
                out[r, c] = float(parts[c])
        return out


def _defaults_then_sparse(L: _Lines, n: int, what: str) -> np.ndarray:
    default = L.scalar(("default", what))
    k = L.count(("non-default", what))
    v = np.full(n, default, dtype=np.float64)
    if k:
        e = L.entries(k, 2)
        v[e[:, 0].astype(int) - 1] = e[:, 1]
    return v


def read_qplib(path: str, name: str | None = None) -> Problem:
    """Read a ``.qplib`` file with linear constraints into a :class:`Problem`."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        L = _Lines(fh.read())

    fname = L.next()
    ptype = L.next().upper()
    if len(ptype) != 3:
        raise QPLIBError(f"problem type {ptype!r} is not three letters")
    otype, vtype, ctype = ptype
    sense_word = L.next().lower()
    sense = ObjSense.MAXIMISE if sense_word.startswith("max") else ObjSense.MINIMISE
    if ctype in ("Q", "D", "C"):
        raise QPLIBError(f"{fname}: constraints of type {ctype} are quadratic; "
                         f"only linear constraints are supported")

    n = L.count("number of variables")
    m = L.count("number of constraints") if ctype not in ("N", "B") else 0

    # -- objective ------------------------------------------------------- #
    Q = None
    if otype != "L":
        nq = L.count("number of quadratic terms in objective")
        e = L.entries(nq, 3) if nq else np.zeros((0, 3))
        i = e[:, 0].astype(np.int64) - 1
        j = e[:, 1].astype(np.int64) - 1
        v = e[:, 2]
        # Each off-diagonal line is the whole coefficient of x_i x_j inside
        # the ½(...) -- the CPLEX-LP export writes the same number in its
        # [ ... ]/2 block -- so the symmetric Hessian carries half of it on
        # each side. Getting this wrong is a factor of two on every cross
        # term and a model that still parses; the solution-point check in
        # bench/qplib.py is what caught it.
        off = i != j
        rows = np.concatenate([i, j[off]])
        cols = np.concatenate([j, i[off]])
        vals = np.concatenate([np.where(off, 0.5 * v, v), 0.5 * v[off]])
        Q = SparseMatrix.from_triplets(rows.astype(IDX), cols.astype(IDX),
                                       vals.astype(VAL), n, n)
    c = _defaults_then_sparse(L, n, "linear coefficients in objective")
    q0 = L.scalar("objective constant")

    # -- constraints ----------------------------------------------------- #
    if m > 0:
        na = L.count("number of linear terms in all constraints")
        e = L.entries(na, 3) if na else np.zeros((0, 3))
        A = SparseMatrix.from_triplets((e[:, 0].astype(np.int64) - 1).astype(IDX),
                                       (e[:, 1].astype(np.int64) - 1).astype(IDX),
                                       e[:, 2].astype(VAL), m, n)
    else:
        A = SparseMatrix.from_triplets(np.zeros(0, dtype=IDX), np.zeros(0, dtype=IDX),
                                       np.zeros(0, dtype=VAL), 0, n)
    infinity = L.scalar("value for infinity")

    def clip_inf(a):
        a = np.array(a, dtype=np.float64)
        a[a >= infinity] = INF
        a[a <= -infinity] = -INF
        return a

    if m > 0:
        row_lb = clip_inf(_defaults_then_sparse(L, m, "left-hand-side"))
        row_ub = clip_inf(_defaults_then_sparse(L, m, "right-hand-side"))
    else:
        row_lb = np.zeros(0)
        row_ub = np.zeros(0)

    # -- variables ------------------------------------------------------- #
    if vtype == "B":
        col_lb = np.zeros(n)
        col_ub = np.ones(n)
    else:
        col_lb = clip_inf(_defaults_then_sparse(L, n, "variable lower bound"))
        col_ub = clip_inf(_defaults_then_sparse(L, n, "variable upper bound"))
    kind = np.zeros(n, dtype=np.uint8)
    if vtype == "B":
        kind[:] = VarKind.BINARY
    elif vtype == "I":
        kind[:] = VarKind.INTEGER
    elif vtype in ("M", "G"):
        t = _defaults_then_sparse(L, n, "variable type")
        integral = t != 0.0
        binary = integral & (col_lb >= 0.0) & (col_ub <= 1.0)
        kind[integral] = VarKind.INTEGER
        kind[binary] = VarKind.BINARY

    # -- starting point and names: read past them, checking the comments -- #
    _defaults_then_sparse(L, n, "variable primal value")
    if m > 0:
        _defaults_then_sparse(L, m, "constraint dual value")
    _defaults_then_sparse(L, n, "variable bound dual value")
    col_names = None
    k = L.count("number of non-default variable names")
    if k:
        col_names = [f"x{j + 1}" for j in range(n)]
        for _ in range(k):
            idx, nm = L.next().split(None, 1)
            col_names[int(idx) - 1] = nm.strip()
    row_names = None
    k = L.count("number of non-default constraint names")
    if k:
        row_names = [f"c{i + 1}" for i in range(m)]
        for _ in range(k):
            idx, nm = L.next().split(None, 1)
            row_names[int(idx) - 1] = nm.strip()

    prob = Problem(A=A, c=c, Q=Q, row_lb=row_lb, row_ub=row_ub,
                   col_lb=col_lb, col_ub=col_ub, kind=kind, obj_offset=q0,
                   sense=sense, name=name or fname,
                   col_names=col_names, row_names=row_names)
    prob.meta["qplib_type"] = ptype
    return prob


def read_qplib_solution(path: str, n: int):
    """A ``.sol`` file: the objective value and the point (zeros implied).

    The files come from the GAMS export, in which ``x1`` is the objective
    variable and the model's variables are ``x2 .. x(n+1)``, named ``x`` when
    continuous, ``b`` when binary and ``i`` when integer -- so ``bk`` in the
    file is variable ``k − 1`` of the ``.qplib`` model. The value of
    ``objvar`` is the published objective value at the point, to more digits
    than the instance page shows. Names other than these are returned in the
    dictionary for the caller to map.
    """
    obj = float("nan")
    x = np.zeros(n, dtype=VAL)
    named = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            key, val = parts[0], float(parts[1])
            if key == "objvar" or key == "x1":
                obj = val                       # x1 is objvar under its GAMS name
            elif key[0] in "xbi" and key[1:].isdigit() and 2 <= int(key[1:]) <= n + 1:
                x[int(key[1:]) - 2] = val
            else:
                named[key] = val
    return obj, x, named
