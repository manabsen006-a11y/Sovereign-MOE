"""Expand Netlib's compressed LP files into MPS.

The Netlib LP test set -- the oldest and most cited collection of linear
programs there is, ~90 problems with published optima to ten significant
figures -- is distributed *only* in a compressed format. There is no plain-MPS
copy on netlib.org; every route to the data goes through the `emps` expander.
That is why the set has been listed under `Known limits` as untouched.

What this file is, and what it is not
-------------------------------------
This is a **file-format expander**, not an algorithm. It reads the compressed
container and writes MPS text, which is then handed to the existing, tested
:mod:`sovopt.io.mps` reader -- so nothing here builds a model, and a bug here
cannot produce a subtly different LP that still parses. It either decodes or it
does not, and the checksums built into the format say which.

On the clean-room rule: `CLAUDE.md` forbids importing, linking, vendoring or
reading the source of any *solver* or any LU/Cholesky *factorization* library.
`emps` is neither. It is a data-format converter of the same kind as the MPS,
CPLEX-LP and QPS specifications this project already implements readers from,
and the format it describes is a container, not a method. The decision to read
it was taken explicitly rather than drifted into.

The format
----------
A base-92 digit alphabet (``TRTAB``) encodes everything. After a blank line and
a ``NAME`` line come two statistics lines, then a table of ``ns`` distinct
numeric values, then the row names, then the COLUMNS, RHS, RANGES and BOUNDS
sections. Two ideas do the compressing:

* **Supersparse values.** Most LP matrices reuse a handful of coefficients
  thousands of times, so every distinct value is expanded once into a table and
  every later occurrence is an index into it.
* **Structure by omission.** A column is introduced by an index of zero,
  after which the rest of the line is its name; every following index is a row,
  so nothing repeats a column name per nonzero the way MPS does.

Every 71 data lines are followed by a checksum line, which this expander
*verifies* rather than skips. That is worth the few lines it costs: the decoder
is a state machine over a byte stream, and the failure mode of getting one
index wrong is not an exception but a plausible model with the wrong numbers in
it. A checksum mismatch turns that into an error at the line it happened on.

References
----------
Gay, "Electronic mail distribution of linear programming test problems",
  Mathematical Programming Society COAL Newsletter 13 (1985) 10-12 -- the
  collection and the reason it is compressed.
netlib.org/lp/data/readme -- the problem summary table, whose published optimal
  values are what :mod:`bench.netlib` checks the expansion against.
netlib.org/lp/data/emps.c, D. M. Gay -- the reference expander for the
  container format described above.
"""

from __future__ import annotations

__all__ = ["TRTAB", "expand", "read_netlib"]

# The base-92 digit alphabet. Note there is no backslash: in the C source the
# alphabet is written across two lines and the backslash is a continuation.
TRTAB = ("!\"#$%&'()*+,-./0123456789;<=>?@"
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`"
         "abcdefghijklmnopqrstuvwxyz{|}~")

_INV = {c: i for i, c in enumerate(TRTAB)}
_BAD = 92                       # any byte outside the alphabet, e.g. a space


def _tr(c: str) -> int:
    return _INV.get(c, _BAD)


class _Corrupt(ValueError):
    """The stream stopped making sense -- almost always a desynchronised
    index rather than a damaged byte, which is why the checksums matter."""


class _Stream:
    """The compressed byte stream, one line at a time, with checksums.

    ``emps`` reads with a 77-byte buffer and treats end-of-line as "fetch the
    next line", so the decoder below asks for a byte and the stream decides
    whether that means advancing within a line or reading a new one.
    """

    def __init__(self, text: str):
        self.lines = text.split("\n")
        self.i = 0
        self.z = ""
        self.chk = [" "]        # chkbuf[0] is a space
        self.mystery: list[str] = []

    # -- lines ----------------------------------------------------------- #

    def _raw(self) -> str:
        if self.i >= len(self.lines):
            raise _Corrupt("unexpected end of compressed file")
        line = self.lines[self.i]
        self.i += 1
        return line

    def _checksum_char(self, line: str) -> str:
        x = 0
        for ch in line:
            c = _tr(ch)
            if x & 1:
                x = (x >> 1) + c + 16384
            else:
                x = (x >> 1) + c
            x &= 0xFFFFFFFF
        return TRTAB[x % 92]

    def rdline(self) -> str:
        """The next *data* line, consuming mystery and checksum lines."""
        while True:
            line = self._raw()
            self.chk.append(self._checksum_char(line))
            if len(self.chk) >= 72:
                self._verify()
            if line.startswith(":"):
                self.mystery.append(line[1:])
                continue            # a mystery line is not data
            return line

    def reset_checksum(self) -> None:
        """Start a fresh checksum block.

        The reference expander resets its accumulator after the NAME line and
        again after the two statistics lines, so a block is 71 lines counted
        from the *value table*, not from the start of the file. Getting this
        wrong puts the first checksum line 4 lines off, which on a small
        problem is invisible -- afiro is 70 lines and never reaches one -- and
        fails on everything larger.
        """
        self.chk = [" "]

    def final_checksum(self) -> None:
        """Each problem ends with a checksum over whatever block is open."""
        if len(self.chk) > 1:
            self._verify()

    def _verify(self) -> None:
        while True:
            want = "".join(self.chk) + "\n"
            got = self._raw() + "\n"
            if got == want:
                self.chk = [" "]
                return
            if got.startswith(":") and len(self.chk) <= 72:
                # a mystery line may interrupt a checksum line
                self.chk.pop()
                self.chk.append(self._checksum_char(got[:-1]))
                self.mystery.append(got[1:-1])
                continue
            raise _Corrupt(
                f"checksum mismatch at line {self.i}: the decoder and the file "
                f"disagree about where a line ends")

    # -- bytes ------------------------------------------------------------ #

    def need(self) -> None:
        if not self.z:
            self.z = self.rdline()

    def peek(self) -> str:
        self.need()
        return self.z[0]

    def take(self) -> str:
        self.need()
        c, self.z = self.z[0], self.z[1:]
        return c

    def drop_line(self) -> None:
        """Abandon whatever is left of the current line.

        Each section of the container starts on a line boundary: in the
        reference expander the cursor is a local of the section routine, so
        anything left over when a section ends is simply dropped and the next
        section begins with a fresh read. Carrying the tail across instead
        works on small problems and desynchronises on the first one whose
        section happens to end mid-line.
        """
        self.z = ""

    def rest(self) -> str:
        """Whatever is left of the current line -- possibly nothing.

        Deliberately does *not* fetch a new line. A name is whatever follows
        the index byte on the line that index was on, and an empty remainder
        means the name *is* empty, not that it lives on the next line. blend's
        RHS set is exactly that: a line holding the index byte and nothing
        else. Refilling here silently eats the following line and every index
        after it is read from the wrong place.
        """
        out, self.z = self.z, ""
        return out


def _exindx(s: _Stream) -> int:
    """A variable-length base-46 index; the high half marks 'one digit only'."""
    k = _tr(s.take())
    if k >= 46:
        raise _Corrupt("bad index byte")
    if k >= 23:
        return k - 23
    x = k
    while True:
        k = _tr(s.take())
        x = x * 46 + k
        if k >= 46:
            return x - 46


def _exform(s: _Stream, table: list[str]) -> str:
    """One numeric field, as the text ``emps`` would have written."""
    # Peek rather than consume: in the reference expander the supersparse
    # branch re-reads from the byte it just looked at, because the local
    # cursor is only written back on the non-index paths. Consuming here
    # instead shifts every subsequent index by one byte and produces a model
    # that still parses.
    k = _tr(s.peek())
    if k < 46:
        idx = _exindx(s)
        # The table is 1-indexed: the reference expander biases its base
        # pointer back by one slot, so index k is the k-th value written.
        if not 1 <= idx <= len(table):
            raise _Corrupt(f"supersparse index {idx} outside 1..{len(table)}")
        return table[idx - 1]

    s.take()
    out: list[str] = []
    db: list[str] = []
    k -= 46
    if k >= 23:
        out.append("-")
        k -= 23
        nelim = 11
    else:
        nelim = 12

    if k >= 11:
        # an integer written as a float: digits then a trailing point
        k -= 11
        db.append(".")
        if k >= 6:
            x = k - 6
        else:
            x = k
            while True:
                k = _tr(s.take())
                x = x * 46 + k
                if k >= 46:
                    x -= 46
                    break
        if not x:
            db.append("0")
        else:
            while x:
                db.append(chr(48 + x % 10))
                x //= 10
        out.extend(reversed(db))
        return "".join(out)

    # general floating point: a base-92 mantissa and a biased exponent
    ex = _tr(s.take()) - 50
    x = _tr(s.take())
    y = 0
    while k > 0:
        k -= 1
        if x >= 100000000:
            y = x
            x = _tr(s.take())
        else:
            x = x * 92 + _tr(s.take())
    if y:
        while x > 1:
            db.append(chr(48 + x % 10))
            x //= 10
        while True:
            db.append(chr(48 + y % 10))
            if y < 10:
                break
            y //= 10
    elif x:
        while True:
            db.append(chr(48 + x % 10))
            if x < 10:
                break
            x //= 10
    else:
        db.append("0")

    nd = len(db) + ex
    if ex > 0 and (nd < nelim or ex < 3):
        out.extend(reversed(db))
        out.append("0" * ex)
        out.append(".")
    elif ex <= 0 and nd >= 0:
        out.extend(reversed(db[len(db) - nd:]))
        out.append(".")
        out.extend(reversed(db[:len(db) - nd]))
    elif ex <= 0 and ex > -nelim:
        out.append(".")
        out.append("0" * (-nd))
        out.extend(reversed(db))
    else:
        # exponent notation
        ex += len(db) - 1
        d = list(db)
        if ex == -10:
            ex = -9
        else:
            if 9 < ex <= len(d) + 8:
                while ex > 9:
                    out.append(d.pop())
                    ex -= 1
            out.append(d.pop())
        out.append(".")
        out.extend(reversed(d))
        out.append("E")
        if ex < 0:
            out.append("-")
            ex = -ex
        out.append(str(ex))
    return "".join(out)


_BOUND_TYPES = ["UP", "LO", "FX", "FR", "MI", "PL"]


def _name(raw: str) -> str:
    """MPS names are emitted free-form, so a blank inside one would split it.

    ``emps -b`` does the same substitution. Netlib names do not contain blanks
    in practice; this only guarantees that if one did, it could not silently
    become two tokens.
    """
    return raw.strip().replace(" ", "_") or "_"


def _section(s: _Stream, out: list[str], head: str, nz: int, what: int,
             rows: list[str], cols: list[str], table: list[str]) -> None:
    if not nz:
        if what <= 2:
            out.append(head)
        return
    s.drop_line()
    first = True
    cur = ""
    pending = None
    while nz > 0:
        nz -= 1
        if first:
            out.append(head)
            first = False
        # an index of zero introduces a new column: the rest of the line is
        # its name, and the stream moves on
        while True:
            n = _exindx(s)
            if n:
                break
            if pending is not None:
                out.append(f"    {pending[0]}  {pending[1]}  {pending[2]}")
                pending = None
            raw = s.rest()
            # An empty name is legal in the container and not in free-form
            # MPS, so it takes the section's own name -- which is what the
            # reference expander does when asked to make names blank-safe.
            cur = _name(raw) if raw.strip() else head
            if what == 1:
                cols.append(cur)
            s.z = s.rdline()

        if what >= 4:
            if n >= 7:
                raise _Corrupt(f"bad bound type index {n}")
            col = cols[_exindx(s) - 1]
            if n >= 4:
                out.append(f" {_BOUND_TYPES[n - 1]} {cur}  {col}")
                continue
            val = _exform(s, table)
            out.append(f" {_BOUND_TYPES[n - 1]} {cur}  {col}  {val}")
            continue

        row = rows[n - 1]
        val = _exform(s, table)
        if pending is None:
            pending = (cur, row, val)
        else:
            out.append(f"    {pending[0]}  {pending[1]}  {pending[2]}"
                       f"  {row}  {val}")
            pending = None
    if pending is not None:
        out.append(f"    {pending[0]}  {pending[1]}  {pending[2]}")


def expand(data) -> str:
    """Expand one compressed Netlib LP file into MPS text."""
    if isinstance(data, bytes):
        data = data.decode("ascii", errors="replace")
    s = _Stream(data)
    out: list[str] = []

    line = s.rdline()
    while not line.startswith("NAME"):
        line = s.rdline()
    out.append(line.rstrip())
    s.reset_checksum()

    stats = s.rdline().split()
    if len(stats) != 8:
        raise _Corrupt(f"expected 8 statistics, got {len(stats)}")
    nrow, ncol, _colmx, nz, _nrhs, rhsnz, _nran, ranz = (int(v) for v in stats)
    stats2 = s.rdline().split()
    if len(stats2) != 3:
        raise _Corrupt(f"expected 3 statistics, got {len(stats2)}")
    _nbd, bdnz, ns = (int(v) for v in stats2)
    s.reset_checksum()

    # the value table: every distinct number, expanded once
    table: list[str] = []
    for _ in range(ns):
        table.append(_exform(s, table))

    rows: list[str] = []
    s.drop_line()
    out.append("ROWS")
    for _ in range(nrow):
        line = s.rdline()
        kind, nm = line[0], _name(line[1:])
        rows.append(nm)
        out.append(f" {kind}  {nm}")

    cols: list[str] = []
    _section(s, out, "COLUMNS", nz, 1, rows, cols, table)
    _section(s, out, "RHS", rhsnz, 2, rows, cols, table)
    _section(s, out, "RANGES", ranz, 3, rows, cols, table)
    _section(s, out, "BOUNDS", bdnz, 4, rows, cols, table)
    s.final_checksum()
    out.append("ENDATA")
    if len(cols) != ncol:
        raise _Corrupt(f"expanded {len(cols)} columns, header said {ncol}")
    return "\n".join(out) + "\n"


def read_netlib(path, name: str | None = None):
    """Read a compressed Netlib file straight into a :class:`Problem`.

    The expansion is handed to the ordinary MPS reader rather than parsed
    here, so the model is built by the same tested code path every other
    format uses.
    """
    import os
    import tempfile

    from .mps import read_mps

    with open(path, "rb") as f:
        text = expand(f.read())
    fd, tmp = tempfile.mkstemp(suffix=".mps")
    try:
        with os.fdopen(fd, "w") as out:
            out.write(text)
        prob = read_mps(tmp, name=name)
    finally:
        os.unlink(tmp)

    # An RHS entry on the objective row is a constant term, and its sign is the
    # oldest ambiguity in the MPS format: sovopt.io.mps follows the
    # CPLEX/Gurobi reading, in which the entry is the *negative* of the
    # constant. Netlib's own published optima are stated with no constant term
    # at all, so validating against the table in its readme means dropping it.
    #
    # This is one instance in eighty-nine -- only e226 has one -- and it is
    # checkable rather than a matter of taste: keeping the constant puts e226
    # at -11.6389 against a published -18.7519, and dropping it matches to
    # 1e-11 while changing nothing anywhere else.
    prob.obj_offset = 0.0
    return prob
