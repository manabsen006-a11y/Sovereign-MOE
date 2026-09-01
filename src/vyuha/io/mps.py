"""MPS reader and writer.

MPS is the lingua franca of optimisation benchmarking: Netlib, MIPLIB and the
Mittelmann sets are all distributed in it. Without a correct reader none of the
benchmark claims in the problem statement are testable, so this module is the
first thing that has to be right.

Both free-format and fixed-format files are accepted. Real-world files are
overwhelmingly whitespace-delimited, so the tokeniser splits on whitespace and
falls back to fixed column positions only when a line cannot be interpreted
otherwise (which happens when a row or column name contains a space).

Quirks that are easy to get wrong, and are handled here
-------------------------------------------------------
* An ``RHS`` entry on the objective row sets a **negative** objective constant.
  This is the CPLEX/Gurobi convention and it flips the sign of every reported
  objective if you get it backwards.
* ``RANGES`` means different things on ``L``, ``G`` and ``E`` rows, and on ``E``
  rows the sign of the range value decides which side moves.
* A ``UP`` bound with a negative value on a variable still at its default lower
  bound of zero implicitly sets the lower bound to −∞. Omitting this rule
  silently makes models infeasible.
* ``MARKER`` / ``INTORG`` / ``INTEND`` toggles integrality mid-COLUMNS.
* ``MI`` sets only the lower bound to −∞ (the archaic reading, which also set
  the upper bound to 0, is not used).

References
----------
IBM, *MPS file format*, Optimization Subroutine Library reference.
Gay, "Electronic mail distribution of linear programming test problems",
  Mathematical Programming Society COAL Newsletter 13 (1985) -- Netlib
  conventions.
Koch et al., "MIPLIB 2010", Math. Prog. Computation 3 (2011) -- MPS extensions
  in the MIP setting.
"""

from __future__ import annotations

import gzip
import io
import lzma
import bz2
import os

import numpy as np

from ..core.problem import Problem, ObjSense, VarKind
from ..core.sparse import SparseMatrix, VAL
from ..core.tolerances import INF

__all__ = ["read_mps", "write_mps", "MPSError"]

_SECTIONS = {
    "NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA",
    "OBJSENSE", "OBJSENS", "QUADOBJ", "QMATRIX", "QSECTION",
    "SOS", "INDICATORS", "REFERENCE ROW",
}

# fixed-format field positions (0-based, end-exclusive)
_FIXED = ((1, 3), (4, 12), (14, 22), (24, 36), (39, 47), (49, 61))


class MPSError(ValueError):
    """Malformed MPS input, with the offending line number."""

    def __init__(self, message: str, lineno: int = -1, line: str = ""):
        self.lineno = lineno
        self.line = line
        where = f" at line {lineno}" if lineno >= 0 else ""
        super().__init__(f"{message}{where}" + (f": {line.rstrip()!r}" if line else ""))


def _open_maybe_compressed(path):
    """Open plain, .gz, .bz2 or .xz -- MIPLIB ships compressed."""
    lower = str(path).lower()
    if lower.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    if lower.endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", errors="replace")
    if lower.endswith(".xz") or lower.endswith(".lzma"):
        return io.TextIOWrapper(lzma.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _fixed_fields(line: str):
    """Split a line on fixed MPS column positions, dropping empty fields."""
    out = []
    for a, b in _FIXED:
        if a >= len(line):
            break
        f = line[a:b].strip()
        if f:
            out.append(f)
    return out


def _num(tok: str, lineno: int, line: str) -> float:
    try:
        return float(tok)
    except ValueError:
        raise MPSError(f"expected a number, got {tok!r}", lineno, line) from None


def read_mps(path, name: str | None = None) -> Problem:
    """Parse an MPS file into a :class:`~vyuha.core.problem.Problem`."""
    if name is None:
        base = os.path.basename(str(path))
        for suf in (".gz", ".bz2", ".xz", ".lzma"):
            if base.lower().endswith(suf):
                base = base[: -len(suf)]
        name = os.path.splitext(base)[0]

    # ROWS
    row_index: dict[str, int] = {}
    row_sense: list[str] = []
    row_names: list[str] = []
    obj_row: str | None = None
    free_rows: set[str] = set()          # extra N rows, ignored as constraints

    # COLUMNS
    col_index: dict[str, int] = {}
    col_names: list[str] = []
    col_is_int: list[bool] = []
    tri_r: list[int] = []
    tri_c: list[int] = []
    tri_v: list[float] = []
    obj_c: dict[int, float] = {}

    rhs: dict[int, float] = {}
    ranges: dict[int, float] = {}
    obj_offset = 0.0
    sense = ObjSense.MINIMISE

    bound_lo: dict[int, float] = {}
    bound_hi: dict[int, float] = {}
    forced_int: set[int] = set()
    semicont: dict[int, float] = {}

    quad: list[tuple[int, int, float]] = []

    section = None
    int_marker = False
    pending_objsense = False

    with _open_maybe_compressed(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            if not raw.strip() or raw[0] in "*$":
                continue

            # A section header starts in column 1 (no leading whitespace).
            if not raw[0].isspace():
                head = raw.split()
                key = head[0].upper()
                if key in _SECTIONS:
                    section = key
                    if key == "ENDATA":
                        break
                    if key == "NAME":
                        if len(head) > 1:
                            name = head[1]
                    elif key in ("OBJSENSE", "OBJSENS"):
                        if len(head) > 1:      # value on the same line
                            sense = (ObjSense.MAXIMISE
                                     if head[1].upper().startswith("MAX")
                                     else ObjSense.MINIMISE)
                            pending_objsense = False
                        else:
                            pending_objsense = True
                    elif key in ("QUADOBJ", "QMATRIX", "QSECTION"):
                        section = "QUADOBJ"
                    continue
                # Not a known section but unindented -- some writers do this for
                # OBJSENSE values. Fall through and treat as data.

            tok = raw.split()
            if not tok:
                continue

            if pending_objsense:
                sense = (ObjSense.MAXIMISE if tok[0].upper().startswith("MAX")
                         else ObjSense.MINIMISE)
                pending_objsense = False
                continue

            # ---------------- ROWS ---------------- #
            if section == "ROWS":
                if len(tok) < 2:
                    tok = _fixed_fields(raw)
                if len(tok) < 2:
                    raise MPSError("ROWS entry needs a type and a name", lineno, raw)
                s, rname = tok[0].upper(), tok[1]
                if s == "N":
                    if obj_row is None:
                        obj_row = rname
                    else:
                        free_rows.add(rname)      # secondary N rows are dropped
                    continue
                if s not in ("L", "G", "E"):
                    raise MPSError(f"unknown row type {s!r}", lineno, raw)
                if rname in row_index:
                    raise MPSError(f"duplicate row name {rname!r}", lineno, raw)
                row_index[rname] = len(row_names)
                row_names.append(rname)
                row_sense.append(s)

            # ---------------- COLUMNS ---------------- #
            elif section == "COLUMNS":
                if len(tok) >= 3 and tok[1].upper() == "'MARKER'":
                    joined = raw.upper()
                    if "INTORG" in joined:
                        int_marker = True
                    elif "INTEND" in joined:
                        int_marker = False
                    continue
                if "'MARKER'" in raw.upper():
                    joined = raw.upper()
                    if "INTORG" in joined:
                        int_marker = True
                    elif "INTEND" in joined:
                        int_marker = False
                    continue

                if len(tok) < 3:
                    tok = _fixed_fields(raw)
                if len(tok) < 3:
                    raise MPSError("COLUMNS entry needs name, row, value", lineno, raw)

                cname = tok[0]
                j = col_index.get(cname)
                if j is None:
                    j = len(col_names)
                    col_index[cname] = j
                    col_names.append(cname)
                    col_is_int.append(int_marker)

                for k in range(1, len(tok) - 1, 2):
                    rname = tok[k]
                    val = _num(tok[k + 1], lineno, raw)
                    if rname == obj_row:
                        obj_c[j] = obj_c.get(j, 0.0) + val
                    elif rname in free_rows:
                        continue
                    else:
                        i = row_index.get(rname)
                        if i is None:
                            raise MPSError(f"unknown row {rname!r}", lineno, raw)
                        tri_r.append(i)
                        tri_c.append(j)
                        tri_v.append(val)

            # ---------------- RHS ---------------- #
            elif section == "RHS":
                if len(tok) < 3:
                    tok = _fixed_fields(raw)
                # the RHS-set name is optional; detect by parity
                start = 1 if (len(tok) % 2 == 1) else 0
                for k in range(start, len(tok) - 1, 2):
                    rname = tok[k]
                    val = _num(tok[k + 1], lineno, raw)
                    if rname == obj_row:
                        obj_offset = -val          # CPLEX/Gurobi sign convention
                    elif rname in free_rows:
                        continue
                    else:
                        i = row_index.get(rname)
                        if i is None:
                            raise MPSError(f"unknown row {rname!r} in RHS", lineno, raw)
                        rhs[i] = val

            # ---------------- RANGES ---------------- #
            elif section == "RANGES":
                if len(tok) < 3:
                    tok = _fixed_fields(raw)
                start = 1 if (len(tok) % 2 == 1) else 0
                for k in range(start, len(tok) - 1, 2):
                    rname = tok[k]
                    val = _num(tok[k + 1], lineno, raw)
                    if rname in free_rows or rname == obj_row:
                        continue
                    i = row_index.get(rname)
                    if i is None:
                        raise MPSError(f"unknown row {rname!r} in RANGES", lineno, raw)
                    ranges[i] = val

            # ---------------- BOUNDS ---------------- #
            elif section == "BOUNDS":
                if len(tok) < 3:
                    tok = _fixed_fields(raw)
                if len(tok) < 3:
                    raise MPSError("BOUNDS entry too short", lineno, raw)
                btype = tok[0].upper()
                # tok[1] is the bound-set name; tok[2] the column
                cname = tok[2] if len(tok) >= 3 else None
                if cname not in col_index:
                    # some files omit the bound-set name
                    cname = tok[1]
                    vtok = tok[2:]
                else:
                    vtok = tok[3:]
                j = col_index.get(cname)
                if j is None:
                    # A bound on a column that never appeared in COLUMNS: the
                    # variable exists but is not in any row. Create it.
                    j = len(col_names)
                    col_index[cname] = j
                    col_names.append(cname)
                    col_is_int.append(False)

                val = _num(vtok[0], lineno, raw) if vtok else 0.0

                if btype == "UP":
                    bound_hi[j] = val
                    if val < 0.0 and j not in bound_lo and not col_is_int[j]:
                        bound_lo[j] = -INF        # the classic MPS quirk
                elif btype == "LO":
                    bound_lo[j] = val
                elif btype == "FX":
                    bound_lo[j] = bound_hi[j] = val
                elif btype == "FR":
                    bound_lo[j] = -INF
                    bound_hi[j] = INF
                elif btype == "MI":
                    bound_lo[j] = -INF
                elif btype in ("PL", "BV_PL"):
                    bound_hi[j] = INF
                elif btype == "BV":
                    bound_lo[j] = 0.0
                    bound_hi[j] = 1.0
                    forced_int.add(j)
                elif btype == "LI":
                    bound_lo[j] = val
                    forced_int.add(j)
                elif btype == "UI":
                    bound_hi[j] = val
                    forced_int.add(j)
                elif btype in ("SC", "SI"):
                    semicont[j] = val if val != 0.0 else INF
                    bound_hi[j] = semicont[j]
                    if btype == "SI":
                        forced_int.add(j)
                else:
                    raise MPSError(f"unknown bound type {btype!r}", lineno, raw)

            # ---------------- QUADOBJ ---------------- #
            elif section == "QUADOBJ":
                if len(tok) < 3:
                    tok = _fixed_fields(raw)
                if len(tok) < 3:
                    continue
                a, b = col_index.get(tok[0]), col_index.get(tok[1])
                if a is None or b is None:
                    raise MPSError("unknown column in QUADOBJ", lineno, raw)
                quad.append((a, b, _num(tok[2], lineno, raw)))

            # SOS / INDICATORS are parsed in a later phase; skip quietly for now
            elif section in ("SOS", "INDICATORS"):
                continue

    if obj_row is None:
        raise MPSError("no objective (type N) row found")

    n = len(col_names)
    m = len(row_names)
    if n == 0:
        raise MPSError("model has no columns")

    # -- assemble ----------------------------------------------------------- #
    A = SparseMatrix.from_triplets(tri_r, tri_c, tri_v, m, n)

    c = np.zeros(n, dtype=VAL)
    for j, v in obj_c.items():
        c[j] = v

    row_lb = np.empty(m, dtype=VAL)
    row_ub = np.empty(m, dtype=VAL)
    for i in range(m):
        b = rhs.get(i, 0.0)
        s = row_sense[i]
        if s == "L":
            row_lb[i], row_ub[i] = -INF, b
        elif s == "G":
            row_lb[i], row_ub[i] = b, INF
        else:
            row_lb[i] = row_ub[i] = b

    for i, r in ranges.items():
        s = row_sense[i]
        b = rhs.get(i, 0.0)
        if s == "L":
            row_lb[i] = b - abs(r)
        elif s == "G":
            row_ub[i] = b + abs(r)
        else:                                # E row: sign of r picks the side
            if r >= 0.0:
                row_lb[i], row_ub[i] = b, b + r
            else:
                row_lb[i], row_ub[i] = b + r, b

    col_lb = np.zeros(n, dtype=VAL)
    col_ub = np.full(n, INF, dtype=VAL)
    for j, v in bound_lo.items():
        col_lb[j] = v
    for j, v in bound_hi.items():
        col_ub[j] = v

    kind = np.zeros(n, dtype=np.uint8)
    for j in range(n):
        if col_is_int[j] or j in forced_int:
            kind[j] = VarKind.INTEGER
    for j in semicont:
        kind[j] = VarKind.SEMI_INTEGER if j in forced_int else VarKind.SEMI_CONTINUOUS

    Q = None
    if quad:
        qr = [a for a, _, _ in quad]
        qc = [b for _, b, _ in quad]
        qv = [v for _, _, v in quad]
        # QUADOBJ lists the lower triangle only; mirror it.
        for a, b, v in quad:
            if a != b:
                qr.append(b)
                qc.append(a)
                qv.append(v)
        Q = SparseMatrix.from_triplets(qr, qc, qv, n, n)

    return Problem(
        A=A, c=c,
        row_lb=row_lb, row_ub=row_ub,
        col_lb=col_lb, col_ub=col_ub,
        kind=kind, Q=Q,
        obj_offset=obj_offset, sense=sense,
        name=name, col_names=col_names, row_names=row_names,
    )


def write_mps(prob: Problem, path) -> None:
    """Write an MPS file. Used for round-trip tests and for handing a model to
    a comparator solver during benchmarking."""
    cn = prob.col_names or [f"C{j}" for j in range(prob.n)]
    rn = prob.row_names or [f"R{i}" for i in range(prob.m)]

    out = []
    out.append(f"NAME          {prob.name}")
    if prob.sense == ObjSense.MAXIMISE:
        out.append("OBJSENSE")
        out.append("    MAX")

    out.append("ROWS")
    out.append(" N  COST")
    for i in range(prob.m):
        lo, hi = prob.row_lb[i], prob.row_ub[i]
        if lo <= -INF and hi >= INF:
            continue                                   # free row: drop
        s = "E" if (lo > -INF and hi < INF and hi - lo <= 1e-12) else \
            "L" if lo <= -INF else "G"
        out.append(f" {s}  {rn[i]}")

    out.append("COLUMNS")
    intblock = False
    marker = 0
    for j in range(prob.n):
        want_int = prob.kind[j] in (VarKind.INTEGER, VarKind.BINARY)
        if want_int != intblock:
            tag = "INTORG" if want_int else "INTEND"
            out.append(f"    MARKER{marker:<4d}          'MARKER'                 '{tag}'")
            marker += 1
            intblock = want_int
        if prob.c[j] != 0.0:
            out.append(f"    {cn[j]:<10s}COST      {prob.c[j]!r:>15s}")
        idx, val = prob.A.col(j)
        for i, v in zip(idx, val):
            lo, hi = prob.row_lb[i], prob.row_ub[i]
            if lo <= -INF and hi >= INF:
                continue
            out.append(f"    {cn[j]:<10s}{rn[i]:<10s}{v!r:>15s}")
    if intblock:
        out.append(f"    MARKER{marker:<4d}          'MARKER'                 'INTEND'")

    out.append("RHS")
    if prob.obj_offset != 0.0:
        out.append(f"    RHS       COST      {(-prob.obj_offset)!r:>15s}")
    for i in range(prob.m):
        lo, hi = prob.row_lb[i], prob.row_ub[i]
        if lo <= -INF and hi >= INF:
            continue
        b = hi if lo <= -INF else lo
        if b != 0.0:
            out.append(f"    RHS       {rn[i]:<10s}{b!r:>15s}")

    rng = [(i, prob.row_ub[i] - prob.row_lb[i]) for i in range(prob.m)
           if prob.row_lb[i] > -INF and prob.row_ub[i] < INF
           and prob.row_ub[i] - prob.row_lb[i] > 1e-12]
    if rng:
        out.append("RANGES")
        for i, r in rng:
            out.append(f"    RNG       {rn[i]:<10s}{r!r:>15s}")

    out.append("BOUNDS")
    for j in range(prob.n):
        lo, hi = prob.col_lb[j], prob.col_ub[j]
        if lo == 0.0 and hi >= INF:
            continue
        if lo <= -INF and hi >= INF:
            out.append(f" FR BND       {cn[j]}")
        elif lo == hi:
            out.append(f" FX BND       {cn[j]:<10s}{lo!r:>15s}")
        else:
            if lo <= -INF:
                out.append(f" MI BND       {cn[j]}")
            elif lo != 0.0:
                out.append(f" LO BND       {cn[j]:<10s}{lo!r:>15s}")
            if hi < INF:
                out.append(f" UP BND       {cn[j]:<10s}{hi!r:>15s}")
    out.append("ENDATA")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
