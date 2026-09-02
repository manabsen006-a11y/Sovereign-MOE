"""MPS writer.

The writer had no test at all, and that is exactly how it came to emit
``np.float64(700.0)`` in every numeric field: ``repr`` of a numpy scalar
changed in NumPy 2.0, the writer formatted with ``!r`` for shortest
round-trip exactness, and nothing in the repository read its output back.

A writer's only real test is the reader. If a model survives a round trip
bit-identical, the file was well formed; if it does not, the file was wrong
whatever it looked like.
"""

import numpy as np

from vyuha.core.problem import ObjSense, Problem, VarKind
from vyuha.core.sparse import SparseMatrix
from vyuha.core.tolerances import INF
from vyuha.io import read_model, write_mps


def _model():
    """Awkward on purpose: negatives, fractions, a ranged row, an integer."""
    A = SparseMatrix.from_dense(np.array([[1.0, -2.5, 0.0],
                                          [0.1, 700.0, 1.0 / 3.0],
                                          [1.0, 0.0, 2.0]]))
    return Problem(A=A, c=np.array([700.0, -0.25, 1e-7]),
                   row_lb=np.array([-INF, 2.0, 1.5]),
                   row_ub=np.array([10.0, 8.0, 1.5]),      # L, ranged, E
                   col_lb=np.array([0.0, -3.0, 0.0]),
                   col_ub=np.array([INF, 4.5, 1.0]),
                   kind=np.array([VarKind.CONTINUOUS, VarKind.CONTINUOUS,
                                  VarKind.INTEGER], dtype=np.uint8),
                   sense=ObjSense.MAXIMISE, name="rt",
                   row_names=["L", "RNG", "EQ"],
                   col_names=["X", "Y", "Z"])


def test_written_file_carries_plain_numbers(tmp_path):
    """REGRESSION: numpy scalars were written as ``np.float64(700.0)``."""
    path = tmp_path / "rt.mps"
    write_mps(_model(), str(path))
    text = path.read_text(encoding="utf-8")
    assert "np.float64" not in text, "numpy repr leaked into the file"


def test_round_trip_through_the_reader_is_exact(tmp_path):
    path = tmp_path / "rt.mps"
    p = _model()
    write_mps(p, str(path))
    q = read_model(str(path))

    assert q.sense == p.sense
    assert np.array_equal(q.A.to_dense(), p.A.to_dense())
    assert np.array_equal(q.c, p.c)
    assert np.array_equal(q.row_lb, p.row_lb)
    assert np.array_equal(q.row_ub, p.row_ub)
    assert np.array_equal(q.col_lb, p.col_lb)
    assert np.array_equal(q.col_ub, p.col_ub)
    assert np.array_equal(q.kind, p.kind)
