"""CPLEX LP format reader.

The strongest test available is that the *same model written two ways* parses to
the same thing. If the LP reader and the MPS reader disagree on any instance,
one of them is wrong, and the disagreement says which fields to look at.
"""

import os

import numpy as np
import pytest

from vyuha.core.problem import ObjSense, Status, VarKind
from vyuha.core.tolerances import INF
from vyuha.io import read_model
from vyuha.io.lp_format import LPFormatError, read_lp
from vyuha.io.mps import read_mps
from vyuha.lp.simplex import SimplexParams, solve_simplex

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def write(tmp_path, text, name="m.lp"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# agreement with the MPS reader                                                #
# --------------------------------------------------------------------------- #


def test_same_model_in_both_formats_parses_identically():
    a = read_mps(os.path.join(FIX, "testprob.mps"))
    b = read_lp(os.path.join(FIX, "testprob.lp"))

    assert (a.m, a.n) == (b.m, b.n)
    assert np.allclose(a.A.to_dense(), b.A.to_dense())
    assert np.allclose(a.c, b.c)
    assert np.allclose(a.row_lb, b.row_lb)
    assert np.allclose(a.row_ub, b.row_ub)
    assert np.allclose(a.col_lb, b.col_lb)
    assert np.allclose(a.col_ub, b.col_ub)
    assert a.sense == b.sense


def test_both_formats_solve_to_the_same_optimum():
    a = solve_simplex(read_mps(os.path.join(FIX, "testprob.mps")))
    b = solve_simplex(read_lp(os.path.join(FIX, "testprob.lp")))
    assert a.status == b.status == Status.OPTIMAL
    assert abs(a.objective - b.objective) < 1e-9
    assert abs(a.objective - 16.0) < 1e-9


# --------------------------------------------------------------------------- #
# constructs                                                                   #
# --------------------------------------------------------------------------- #


def test_maximise_with_objective_constant(tmp_path):
    p = read_lp(write(tmp_path, """\\ a comment
Maximize
 obj: 3 x + 2 y + 10
Subject To
 c1: x + y <= 4
End
"""))
    assert p.sense == ObjSense.MAXIMISE
    assert p.obj_offset == 10.0
    assert np.allclose(p.c, [3.0, 2.0])


def test_ranged_row(tmp_path):
    p = read_lp(write(tmp_path, """Minimize
 obj: x
Subject To
 r: 1 <= x + 3 y <= 6
End
"""))
    assert p.row_lb[0] == 1.0 and p.row_ub[0] == 6.0


def test_ranged_row_written_downwards(tmp_path):
    p = read_lp(write(tmp_path, """Minimize
 obj: x
Subject To
 r: 6 >= x + 3 y >= 1
End
"""))
    assert p.row_lb[0] == 1.0 and p.row_ub[0] == 6.0


def test_terms_on_both_sides_are_collected(tmp_path):
    """``2x - y + z >= x + 1`` must become ``x - y + z >= 1``."""
    p = read_lp(write(tmp_path, """Minimize
 obj: x
Subject To
 c: 2 x - y + z >= x + 1
End
"""))
    idx = {n: i for i, n in enumerate(p.col_names)}
    row = p.A.to_dense()[0]
    assert row[idx["x"]] == 1.0
    assert row[idx["y"]] == -1.0
    assert row[idx["z"]] == 1.0
    assert p.row_lb[0] == 1.0


def test_repeated_variable_in_a_row_is_summed(tmp_path):
    p = read_lp(write(tmp_path, """Minimize
 obj: x
Subject To
 c: x + 2 x <= 6
End
"""))
    assert p.A.to_dense()[0][0] == 3.0


def test_bounds_forms(tmp_path):
    p = read_lp(write(tmp_path, """Minimize
 obj: a + b + c + d + e
Subject To
 r: a + b + c + d + e <= 100
Bounds
 0 <= a <= 3
 b free
 c >= -inf
 c <= 8
 d = 2.5
 e >= 1
End
"""))
    i = {n: k for k, n in enumerate(p.col_names)}
    assert (p.col_lb[i["a"]], p.col_ub[i["a"]]) == (0.0, 3.0)
    assert p.col_lb[i["b"]] <= -INF and p.col_ub[i["b"]] >= INF
    assert p.col_lb[i["c"]] <= -INF and p.col_ub[i["c"]] == 8.0
    assert p.col_lb[i["d"]] == p.col_ub[i["d"]] == 2.5
    assert p.col_lb[i["e"]] == 1.0 and p.col_ub[i["e"]] >= INF


def test_integer_and_binary_sections(tmp_path):
    p = read_lp(write(tmp_path, """Minimize
 obj: x + y + z
Subject To
 c: x + y + z >= 2
General
 x
Binary
 z
End
"""))
    i = {n: k for k, n in enumerate(p.col_names)}
    assert p.kind[i["x"]] == VarKind.INTEGER
    assert p.kind[i["y"]] == VarKind.CONTINUOUS
    assert p.kind[i["z"]] == VarKind.INTEGER
    assert p.col_lb[i["z"]] == 0.0 and p.col_ub[i["z"]] == 1.0


def test_default_bounds_are_zero_to_infinity(tmp_path):
    p = read_lp(write(tmp_path, """Minimize
 obj: x
Subject To
 c: x >= 1
End
"""))
    assert p.col_lb[0] == 0.0 and p.col_ub[0] >= INF


def test_comments_and_blank_lines_are_ignored(tmp_path):
    p = read_lp(write(tmp_path, """\\ leading comment

Minimize

 obj: x + y   \\ trailing comment

Subject To

 c1: x + y <= 4

End
"""))
    assert p.n == 2 and p.m == 1


def test_unnamed_constraints_get_generated_names(tmp_path):
    p = read_lp(write(tmp_path, """Minimize
 obj: x
Subject To
 x + y <= 4
 x - y >= 0
End
"""))
    assert p.m == 2
    assert len(set(p.row_names)) == 2


# --------------------------------------------------------------------------- #
# dispatch and errors                                                          #
# --------------------------------------------------------------------------- #


def test_read_model_dispatches_by_extension():
    a = read_model(os.path.join(FIX, "testprob.lp"))
    b = read_model(os.path.join(FIX, "testprob.mps"))
    assert np.allclose(a.A.to_dense(), b.A.to_dense())


def test_malformed_input_raises_with_a_line_number(tmp_path):
    with pytest.raises(LPFormatError):
        read_lp(write(tmp_path, """Minimize
 obj: x
Subject To
 c: x + y
End
"""))


def test_empty_model_is_rejected(tmp_path):
    with pytest.raises(LPFormatError):
        read_lp(write(tmp_path, "Minimize\nSubject To\nEnd\n"))
