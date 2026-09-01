"""CPLEX LP format reader.

MPS is a punched-card format: column-aligned, machine-oriented, and unreadable
without a reference card. LP format is what a person actually writes when they
want to state a small model by hand or eyeball what a generator produced:

    Maximize
      obj: 3 x + 2 y
    Subject To
      c1: x + y <= 4
      c2: x + 3 y <= 6
      c3: 1 <= x + y <= 4        \\ ranged rows are written inline
    Bounds
      0 <= x <= 3
      y free
    General
      x
    End

Supporting it costs one module and no change to any solver path -- the result is
an ordinary :class:`~vyuha.core.problem.Problem`, indistinguishable from one that
came out of the MPS reader. The test suite asserts exactly that: the same model
written both ways must parse to the same matrix, bounds, costs and senses.

What is handled
---------------
* ``Maximize`` / ``Minimize`` (and the ``-ise`` spellings, and ``max`` / ``min``)
* ``Subject To`` / ``st`` / ``s.t.`` / ``such that``
* operators ``<=`` ``=<`` ``<`` ``>=`` ``=>`` ``>`` ``=``
* ranged rows written as ``lo <= expr <= hi``
* constants on either side, and terms on both sides of the operator
* ``Bounds`` with ``free``, ``-inf``, ``+infinity``, and ``x = v`` fixing
* ``General`` / ``Generals`` / ``Integer``, ``Binary`` / ``Binaries``,
  ``Semi-Continuous``
* ``\\`` comments to end of line, and blank lines anywhere

Quirks worth knowing
--------------------
A variable may appear more than once in a row (``x + 2 x``); coefficients are
summed, which is what every other reader does. A variable that appears only in
the ``Bounds`` section still exists -- it is simply not in any row. The objective
may carry a constant term, which becomes the objective offset.

References
----------
IBM, *CPLEX LP file format* reference (ILOG CPLEX Optimization Studio).
Gurobi Optimizer reference manual, "LP format" -- the de facto extensions
  (``Semi-Continuous``, multiple objective-sense spellings).
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import re

import numpy as np

from ..core.problem import ObjSense, Problem, VarKind
from ..core.sparse import SparseMatrix, VAL
from ..core.tolerances import INF

__all__ = ["read_lp", "LPFormatError"]


class LPFormatError(ValueError):
    """Malformed LP input, with the offending line number."""

    def __init__(self, message: str, lineno: int = -1, line: str = ""):
        self.lineno = lineno
        self.line = line
        where = f" at line {lineno}" if lineno >= 0 else ""
        super().__init__(f"{message}{where}"
                         + (f": {line.strip()!r}" if line else ""))


_SECTIONS = {
    "maximize": "obj_max", "maximise": "obj_max", "max": "obj_max",
    "minimize": "obj_min", "minimise": "obj_min", "min": "obj_min",
    "subject to": "rows", "subjectto": "rows", "st": "rows",
    "s.t.": "rows", "such that": "rows", "suchthat": "rows",
    "bounds": "bounds", "bound": "bounds",
    "general": "int", "generals": "int", "gen": "int", "integer": "int",
    "integers": "int",
    "binary": "bin", "binaries": "bin", "bin": "bin",
    "semi-continuous": "semi", "semicontinuous": "semi", "semis": "semi",
    "sos": "sos", "end": "end",
}

# a term: optional sign, optional coefficient, optional '*', then a name
_TERM = re.compile(r"""
    (?P<sign>[+-])?\s*
    (?P<coef>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)?\s*
    (?:\*\s*)?
    (?P<name>[A-Za-z!"#$%&()/,;?@_`'{}|~][A-Za-z0-9!"#$%&()/,;?@_`'{}|~.\[\]]*)?
""", re.VERBOSE)

_OPS = ("<=", "=<", ">=", "=>", "=", "<", ">")


def _open(path):
    low = str(path).lower()
    if low.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8",
                                errors="replace")
    if low.endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8",
                                errors="replace")
    if low.endswith(".xz"):
        return io.TextIOWrapper(lzma.open(path, "rb"), encoding="utf-8",
                                errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _parse_expr(text, lineno, raw):
    """``(terms, constant)`` where ``terms`` maps variable name to coefficient."""
    terms: dict[str, float] = {}
    const = 0.0
    pos = 0
    text = text.strip()
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        m = _TERM.match(text, pos)
        if not m or m.end() == pos:
            raise LPFormatError(f"cannot parse expression at {text[pos:pos+20]!r}",
                                lineno, raw)
        pos = m.end()
        sign = -1.0 if m.group("sign") == "-" else 1.0
        coef = m.group("coef")
        name = m.group("name")
        if name is None:
            if coef is None:
                raise LPFormatError("empty term", lineno, raw)
            const += sign * float(coef)      # a bare number is a constant
            continue
        c = sign * (float(coef) if coef is not None else 1.0)
        terms[name] = terms.get(name, 0.0) + c
    return terms, const


def _split_op(text):
    """Split on relational operators, returning ``(pieces, operators)``."""
    pieces, ops = [], []
    i = 0
    start = 0
    while i < len(text):
        for op in _OPS:
            if text.startswith(op, i):
                pieces.append(text[start:i])
                ops.append(op)
                i += len(op)
                start = i
                break
        else:
            i += 1
    pieces.append(text[start:])
    return pieces, ops


def _norm_op(op):
    if op in ("<=", "=<", "<"):
        return "<="
    if op in (">=", "=>", ">"):
        return ">="
    return "="


def read_lp(path, name: str | None = None) -> Problem:
    """Parse a CPLEX LP file into a :class:`~vyuha.core.problem.Problem`."""
    if name is None:
        base = os.path.basename(str(path))
        for suf in (".gz", ".bz2", ".xz"):
            if base.lower().endswith(suf):
                base = base[: -len(suf)]
        name = os.path.splitext(base)[0]

    section = None
    sense = ObjSense.MINIMISE
    obj_terms: dict[str, float] = {}
    obj_const = 0.0

    row_names: list[str] = []
    row_terms: list[dict] = []
    row_lb: list[float] = []
    row_ub: list[float] = []

    bounds_lo: dict[str, float] = {}
    bounds_hi: dict[str, float] = {}
    int_vars: set[str] = set()
    bin_vars: set[str] = set()
    semi_vars: set[str] = set()

    order: list[str] = []
    seen: set[str] = set()

    def note(names):
        for v in names:
            if v not in seen:
                seen.add(v)
                order.append(v)

    pending = ""            # LP statements may wrap across lines
    pending_line = 0

    def flush(lineno, raw):
        nonlocal pending, obj_terms, obj_const
        stmt = pending.strip()
        pending = ""
        if not stmt:
            return
        if section in ("obj_max", "obj_min"):
            body = stmt
            if ":" in body.split("<")[0].split(">")[0]:
                body = body.split(":", 1)[1]
            t, c = _parse_expr(body, lineno, raw)
            for k, v in t.items():
                obj_terms[k] = obj_terms.get(k, 0.0) + v
            obj_const += c
            note(t)
        elif section == "rows":
            label = None
            head = stmt.split("<")[0].split(">")[0].split("=")[0]
            if ":" in head:
                label, stmt = stmt.split(":", 1)
                label = label.strip()
            pieces, ops = _split_op(stmt)
            if not ops:
                raise LPFormatError("constraint has no relational operator",
                                    lineno, raw)
            if len(ops) == 1:
                lt, lc = _parse_expr(pieces[0], lineno, raw)
                rt, rc = _parse_expr(pieces[1], lineno, raw)
                for k, v in rt.items():           # move rhs terms to the left
                    lt[k] = lt.get(k, 0.0) - v
                rhs = rc - lc
                op = _norm_op(ops[0])
                lo = rhs if op in (">=", "=") else -INF
                hi = rhs if op in ("<=", "=") else INF
            elif len(ops) == 2:
                lo_t, lo_c = _parse_expr(pieces[0], lineno, raw)
                lt, lc = _parse_expr(pieces[1], lineno, raw)
                hi_t, hi_c = _parse_expr(pieces[2], lineno, raw)
                if lo_t or hi_t:
                    raise LPFormatError(
                        "a ranged row needs constants on both outer sides",
                        lineno, raw)
                a, b = lo_c - lc, hi_c - lc
                if _norm_op(ops[0]) == ">=":
                    a, b = b, a
                lo, hi = min(a, b), max(a, b)
            else:
                raise LPFormatError("too many operators in one constraint",
                                    lineno, raw)
            row_names.append(label or f"R{len(row_names)}")
            row_terms.append(lt)
            row_lb.append(lo)
            row_ub.append(hi)
            note(lt)
        elif section == "bounds":
            _bound_stmt(stmt, bounds_lo, bounds_hi, note, lineno, raw)
        elif section in ("int", "bin", "semi"):
            names = stmt.replace(",", " ").split()
            target = {"int": int_vars, "bin": bin_vars, "semi": semi_vars}[section]
            target.update(names)
            note(names)

    with _open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("\\", 1)[0]          # \ starts a comment
            stripped = line.strip()
            if not stripped:
                continue

            low = stripped.lower()
            key = None
            for k in sorted(_SECTIONS, key=len, reverse=True):
                if low == k or low.startswith(k + " ") or low.startswith(k + "\t"):
                    key = k
                    break
            if key is not None and (low == key or not _split_op(stripped)[1]):
                flush(lineno, raw)
                section = _SECTIONS[key]
                if section == "obj_max":
                    sense = ObjSense.MAXIMISE
                elif section == "obj_min":
                    sense = ObjSense.MINIMISE
                elif section == "end":
                    break
                rest = stripped[len(key):].strip()
                if rest:
                    pending = rest
                    pending_line = lineno
                continue

            # a new statement begins when the previous one is complete
            if section == "rows" and pending and _split_op(pending)[1]:
                flush(pending_line, raw)
            elif section in ("bounds", "int", "bin", "semi") and pending:
                flush(pending_line, raw)
            if not pending:
                pending_line = lineno
            pending = (pending + " " + stripped).strip()
        flush(pending_line, "")

    if not order:
        raise LPFormatError("no variables found")

    n = len(order)
    index = {v: i for i, v in enumerate(order)}
    m = len(row_terms)

    ri, ci, cv = [], [], []
    for i, t in enumerate(row_terms):
        for k, v in t.items():
            if v != 0.0:
                ri.append(i)
                ci.append(index[k])
                cv.append(v)
    A = SparseMatrix.from_triplets(ri, ci, cv, m, n)

    c = np.zeros(n, dtype=VAL)
    for k, v in obj_terms.items():
        c[index[k]] = v

    lo = np.zeros(n, dtype=VAL)
    hi = np.full(n, INF, dtype=VAL)
    for k, v in bounds_lo.items():
        lo[index[k]] = v
    for k, v in bounds_hi.items():
        hi[index[k]] = v

    kind = np.zeros(n, dtype=np.uint8)
    for v in int_vars:
        kind[index[v]] = VarKind.INTEGER
    for v in bin_vars:
        j = index[v]
        kind[j] = VarKind.INTEGER
        if v not in bounds_lo:
            lo[j] = 0.0
        if v not in bounds_hi:
            hi[j] = 1.0
    for v in semi_vars:
        kind[index[v]] = VarKind.SEMI_CONTINUOUS

    return Problem(A=A, c=c,
                   row_lb=np.array(row_lb, dtype=VAL),
                   row_ub=np.array(row_ub, dtype=VAL),
                   col_lb=lo, col_ub=hi, kind=kind,
                   obj_offset=obj_const, sense=sense, name=name,
                   col_names=order, row_names=row_names)


def _bound_stmt(stmt, lo, hi, note, lineno, raw):
    """One line of the Bounds section."""
    low = stmt.lower()
    if low.endswith(" free") or low == "free":
        v = stmt[: len(stmt) - 4].strip()
        lo[v] = -INF
        hi[v] = INF
        note([v])
        return

    def num(tok):
        t = tok.strip().lower()
        neg = t.startswith("-")
        t2 = t.lstrip("+-")
        if t2 in ("inf", "infinity"):
            return -INF if neg else INF
        try:
            return float(t)
        except ValueError:
            raise LPFormatError(f"expected a number, got {tok!r}",
                                lineno, raw) from None

    pieces, ops = _split_op(stmt)
    if len(ops) == 1:
        left, right = pieces[0].strip(), pieces[1].strip()
        op = _norm_op(ops[0])
        # one side is a name, the other a number
        try:
            val = num(right)
            var = left
        except LPFormatError:
            val = num(left)
            var = right
            op = {"<=": ">=", ">=": "<="}.get(op, op)
        note([var])
        if op == "<=":
            hi[var] = val
        elif op == ">=":
            lo[var] = val
        else:
            lo[var] = hi[var] = val
    elif len(ops) == 2:
        a, var, b = num(pieces[0]), pieces[1].strip(), num(pieces[2])
        note([var])
        if _norm_op(ops[0]) == ">=":
            a, b = b, a
        lo[var], hi[var] = min(a, b), max(a, b)
    else:
        raise LPFormatError("cannot parse bound", lineno, raw)
