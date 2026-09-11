"""The QPLIB reader.

Two kinds of check. Synthetic files written here pin the format down section
by section -- which sections exist for which problem type, the half on the
cross terms, the objvar offset in the solution files. Then, when the fetched
instances are on disk, every one of them is evaluated at its published
solution point: the objective must match the published value to 1e-6 and the
point must be feasible for the model we built. That check is the whole reason
the reader can be trusted; it is what caught the factor of two.
"""

import json
import os

import numpy as np
import pytest

from sovopt.core.problem import ObjSense, VarKind
from sovopt.core.tolerances import INF
from sovopt.io.qplib import QPLIBError, read_qplib, read_qplib_solution

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "qplib")


def write(tmp_path, text, name="t.qplib"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


QCL = """\
# a comment line
tiny
QCL
minimize
3 # number of variables
2 # number of constraints
3 # number of quadratic terms in objective
1 1 2.0
2 1 -4.0
3 3 6.0
1.0 # default value for linear coefficients in objective
1 # number of non-default linear coefficients in objective
2 -5.0
7.5 # objective constant
4 # number of linear terms in all constraints
1 1 1.0
1 2 1.0
2 2 1.0
2 3 -1.0
1e20 # value for infinity
-1e20 # default left-hand-side value
1 # number of non-default left-hand-sides
2 0.5
1e20 # default right-hand-side value
1 # number of non-default right-hand-sides
1 1.5
0.0 # default variable lower bound value
1 # number of non-default variable lower bounds
3 -2.0
1e20 # default variable upper bound value
2 # number of non-default variable upper bounds
1 3.0
2 3.0
0.0 # default variable primal value in starting point
0 # number of non-default variable primal values in starting point
0.0 # default constraint dual value in starting point
0 # number of non-default constraint dual values in starting point
0.0 # default variable bound dual value in starting point
0 # number of non-default variable bound dual values in starting point
2 # number of non-default variable names
1 flow
3 yield
1 # number of non-default constraint names
2 demand
"""


def test_a_linear_constrained_qp_parses_section_by_section(tmp_path):
    p = read_qplib(write(tmp_path, QCL))
    assert p.name == "tiny" and p.sense == ObjSense.MINIMISE
    assert p.n == 3 and p.m == 2
    Q = p.Q.to_dense()
    # diagonal as given; the cross term 2 1 -4.0 is the whole coefficient of
    # x1 x2 inside the ½(...), so the symmetric Hessian carries -2 on each side
    assert Q[0, 0] == 2.0 and Q[2, 2] == 6.0
    assert Q[1, 0] == -2.0 and Q[0, 1] == -2.0 and Q[1, 1] == 0.0
    assert np.allclose(p.c, [1.0, -5.0, 1.0])
    assert p.obj_offset == 7.5
    assert np.allclose(p.A.to_dense(), [[1, 1, 0], [0, 1, -1]])
    assert p.row_lb[0] <= -INF and p.row_lb[1] == 0.5
    assert p.row_ub[0] == 1.5 and p.row_ub[1] >= INF
    assert np.allclose(p.col_lb, [0.0, 0.0, -2.0])
    assert p.col_ub[0] == 3.0 and p.col_ub[1] == 3.0 and p.col_ub[2] >= INF
    assert not p.is_mip
    assert p.col_names == ["flow", "x2", "yield"]
    assert p.row_names == ["c1", "demand"]
    assert p.meta["qplib_type"] == "QCL"
    # ½ Σ_{i>=j} v_ij x_i x_j + b'x + q0 at x = (1, 2, 3)
    x = np.array([1.0, 2.0, 3.0])
    expected = 0.5 * (2.0 * 1 + (-4.0) * 1 * 2 + 6.0 * 9) + (1 - 10 + 3) + 7.5
    assert abs(p.objective(x) - expected) < 1e-12


QBN = """\
box
QBN
maximize
2 # number of variables
1 # number of quadratic terms in objective
2 1 3.0
0.0 # default value for linear coefficients in objective
0 # number of non-default linear coefficients in objective
0.0 # objective constant
1e20 # value for infinity
0.0 # default variable primal value in starting point
0 # number of non-default variable primal values in starting point
0.0 # default variable bound dual value in starting point
0 # number of non-default variable bound dual values in starting point
0 # number of non-default variable names
0 # number of non-default constraint names
"""


def test_a_binary_unconstrained_instance_has_no_constraint_or_bound_sections(tmp_path):
    p = read_qplib(write(tmp_path, QBN))
    assert p.sense == ObjSense.MAXIMISE
    assert p.n == 2 and p.m == 0
    assert (p.kind == VarKind.BINARY).all()
    assert (p.col_lb == 0).all() and (p.col_ub == 1).all()
    assert p.Q.to_dense()[0, 1] == 1.5 and p.Q.to_dense()[1, 0] == 1.5
    assert abs(p.objective(np.array([1.0, 1.0])) - 1.5) < 1e-12


QML = """\
mixed
QML
minimize
3 # number of variables
1 # number of constraints
1 # number of quadratic terms in objective
1 1 1.0
0.0 # default value for linear coefficients in objective
0 # number of non-default linear coefficients in objective
0.0 # objective constant
3 # number of linear terms in all constraints
1 1 1.0
1 2 1.0
1 3 1.0
1e20 # value for infinity
-1e20 # default left-hand-side value
0 # number of non-default left-hand-sides
2.0 # default right-hand-side value
0 # number of non-default right-hand-sides
0.0 # default variable lower bound value
0 # number of non-default variable lower bounds
1.0 # default variable upper bound value
1 # number of non-default variable upper bounds
3 5.0
0 # default variable type
2 # number of non-default variable types
2 1
3 1
0.0 # default variable primal value in starting point
0 # number of non-default variable primal values in starting point
0.0 # default constraint dual value in starting point
0 # number of non-default constraint dual values in starting point
0.0 # default variable bound dual value in starting point
0 # number of non-default variable bound dual values in starting point
0 # number of non-default variable names
0 # number of non-default constraint names
"""


def test_variable_types_become_binary_or_integer_by_their_bounds(tmp_path):
    p = read_qplib(write(tmp_path, QML))
    assert p.kind[0] == VarKind.CONTINUOUS
    assert p.kind[1] == VarKind.BINARY            # type 1 on [0, 1]
    assert p.kind[2] == VarKind.INTEGER           # type 1 on [0, 5]
    assert p.n_binary == 1 and p.n_integer == 2


def test_a_comment_out_of_step_is_an_error_not_a_model(tmp_path):
    bad = QCL.replace("2 # number of constraints", "2 # number of quadratic terms in objective")
    with pytest.raises(QPLIBError, match="out of step"):
        read_qplib(write(tmp_path, bad))


def test_quadratic_constraints_are_refused(tmp_path):
    with pytest.raises(QPLIBError, match="quadratic"):
        read_qplib(write(tmp_path, QCL.replace("QCL", "QCQ")))


def test_the_solution_file_is_offset_by_the_objective_variable(tmp_path):
    p = tmp_path / "t.sol"
    p.write_text("objvar   -6.5\nx1   -6.5\nx2   0.25\nb4   1.0\nfoo 1.0\n")
    obj, x, named = read_qplib_solution(str(p), 3)
    assert obj == -6.5
    assert np.allclose(x, [0.25, 0.0, 1.0])        # x2 -> variable 1, b4 -> 3
    assert named == {"foo": 1.0}


# --------------------------------------------------------------------------- #
# the fetched instances at their published points                             #
# --------------------------------------------------------------------------- #


def _fetched():
    if not os.path.isdir(DATA) or not os.path.exists(os.path.join(DATA, "index.json")):
        return []
    index = json.load(open(os.path.join(DATA, "index.json")))
    out = []
    for f in sorted(os.listdir(DATA)):
        if f.endswith(".qplib"):
            name = f[6:-6]
            sol = os.path.join(DATA, f"QPLIB_{name}.sol")
            if name in index and "published" in index[name] and os.path.exists(sol):
                out.append(name)
    return out


@pytest.mark.parametrize("name", _fetched() or ["none"])
def test_published_point_evaluates_to_published_value(name):
    if name == "none":
        pytest.skip("no QPLIB instances fetched; run python -m bench.qplib --fetch")
    index = json.load(open(os.path.join(DATA, "index.json")))
    page_value = index[name]["published"].get("objective")
    if page_value is None:
        pytest.skip(f"{name}: no published objective value")
    p = read_qplib(os.path.join(DATA, f"QPLIB_{name}.qplib"))
    obj, x, named = read_qplib_solution(os.path.join(DATA, f"QPLIB_{name}.sol"), p.n)
    assert not named, f"unmapped names in the solution file: {list(named)[:5]}"
    if not np.isfinite(obj):
        # an empty solution file: the published point is the origin, and
        # the page carries its value
        obj = page_value
    # the page shows ten digits, the solution file fifteen
    assert abs(obj - page_value) <= 1e-8 * max(1.0, abs(page_value))
    ours = p.objective(x)
    assert abs(ours - obj) <= 1e-9 * max(1.0, abs(obj)), (
        f"{name}: model gives {ours} at the published point, QPLIB says {obj}")
    rv, cv, iv = p.violation(x)
    assert max(rv, cv) <= 1e-5 and iv <= 1e-5
    # and the listing's counts agree with what was parsed
    rec = index[name]
    assert p.n == rec["n"] and p.m == rec["m"]
    assert p.n_binary == rec["n_binary"]
