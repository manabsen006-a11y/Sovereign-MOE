"""A fixed-column MPS file whose names contain spaces.

Maros and Meszaros' qforplan (Netlib's forplan with a quadratic term)
names its rows "DEDO3 1R" and its columns "DEDO3 11": legal in the
fixed-column format, and a whitespace tokeniser sees a duplicate row
"DEDO3" in ROWS and a number where a row name should be in COLUMNS. Once
a ROWS line does not split into a type and a name, the file is read on
its column positions throughout.
"""

import numpy as np

from sovopt.io.mps import read_mps


def _line(*fields):
    """A data line on the fixed column positions (1-based: 2-3, 5-12,
    15-22, 25-36, 40-47, 50-61)."""
    starts = (1, 4, 14, 24, 39, 49)
    out = ""
    for start, f in zip(starts, fields):
        if f is None:
            continue
        out = out.ljust(start) + f
    return out + "\n"


FIXED = ("NAME          SPACED\n"
         "ROWS\n"
         + _line("N", "COST")
         + _line("E", "DEDO3 1R")
         + _line("L", "LIM 1")
         + "COLUMNS\n"
         + _line(None, "X 1", "COST", "1.0", "DEDO3 1R", "1.0")
         + _line(None, "X 1", "LIM 1", "1.0")
         + _line(None, "Y", "COST", "2.0", "DEDO3 1R", "1.0")
         + _line(None, "Y", "LIM 1", "1.0")
         + "RHS\n"
         + _line(None, "RHS", "DEDO3 1R", "1.0", "LIM 1", "4.0")
         + "BOUNDS\n"
         + _line("UP", "BND", "X 1", "0.5")
         + "ENDATA\n")


def test_names_with_spaces_are_read_on_the_column_positions(tmp_path):
    f = tmp_path / "spaced.mps"
    f.write_text(FIXED, encoding="utf-8")
    p = read_mps(str(f))
    assert p.row_names == ["DEDO3 1R", "LIM 1"]
    assert p.col_names == ["X 1", "Y"]
    assert p.m == 2 and p.n == 2 and p.nnz == 4
    assert list(p.c) == [1.0, 2.0]
    assert p.row_lb[0] == p.row_ub[0] == 1.0 and p.row_ub[1] == 4.0
    assert p.col_ub[0] == 0.5
    # x + y = 1 with x <= 0.5 at min x + 2y: x = 0.5, y = 0.5, cost 1.5
    from sovopt.lp.simplex import solve_simplex
    s = solve_simplex(p)
    assert abs(s.objective - 1.5) < 1e-9
    assert np.allclose(s.x, [0.5, 0.5], atol=1e-9)


def test_an_integer_column_with_no_bound_line_is_binary(tmp_path):
    """The MPSX convention CPLEX, SCIP and MIPLIB keep: an integer column in
    a MARKER block with no bound entry of any kind is [0, 1]; a bound line
    of any type on it (LI, LO, UP) leaves the other side at its default.
    MIPLIB's neos-2626858-aoos is published infeasible under this rule and
    was feasible under [0, +inf), with an exactly integer point the
    verifier accepted -- a point of a different model."""
    f = tmp_path / "ints.mps"
    f.write_text(
        "NAME          INTS\n"
        "ROWS\n"
        " N  COST\n"
        " L  R1\n"
        "COLUMNS\n"
        "    MARKER                 'MARKER'                 'INTORG'\n"
        "    NOBND     COST         1.0   R1           1.0\n"
        "    LIONLY    COST         1.0   R1           1.0\n"
        "    UPONLY    COST         1.0   R1           1.0\n"
        "    MARKER                 'MARKER'                 'INTEND'\n"
        "    CONT      COST         1.0   R1           1.0\n"
        "RHS\n"
        "    RHS       R1           9.0\n"
        "BOUNDS\n"
        " LI BND       LIONLY       2\n"
        " UP BND       UPONLY       5\n"
        "ENDATA\n", encoding="utf-8")
    p = read_mps(str(f))
    b = dict(zip(p.col_names, zip(p.col_lb, p.col_ub, p.integer_mask)))
    assert b["NOBND"] == (0.0, 1.0, True)          # the convention
    assert b["LIONLY"][0] == 2.0 and b["LIONLY"][1] >= 1e29 and b["LIONLY"][2]
    assert b["UPONLY"] == (0.0, 5.0, True)
    assert b["CONT"][1] >= 1e29 and not b["CONT"][2]
