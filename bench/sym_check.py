"""Quick symmetry-detection probe over synthetic and real models."""
from __future__ import annotations

import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from vyuha.core.problem import Problem
from vyuha.core.sparse import SparseMatrix
from vyuha.core.tolerances import INF
from vyuha.io.mps import read_mps
from vyuha.mip.symmetry import (breaking_constraints, detect_symmetry,
                                verify_permutation)
from vyuha.models import unit_scheduling
from vyuha.numerics.scaling import scale_problem


def identical_items(n=8):
    A = SparseMatrix.from_dense(np.ones((1, n)))
    return Problem(A=A, c=-np.ones(n), row_lb=np.array([-INF]),
                   row_ub=np.array([3.0]), col_lb=np.zeros(n),
                   col_ub=np.ones(n), kind=np.ones(n, dtype=np.uint8),
                   name="identical")


def distinct_items(n=8):
    A = SparseMatrix.from_dense(np.arange(1, n + 1, dtype=float).reshape(1, n))
    return Problem(A=A, c=-np.arange(1, n + 1, dtype=float),
                   row_lb=np.array([-INF]), row_ub=np.array([9.0]),
                   col_lb=np.zeros(n), col_ub=np.ones(n),
                   kind=np.ones(n, dtype=np.uint8), name="distinct")


def two_blocks():
    D = np.array([[1., 1., 0., 0.], [0., 0., 1., 1.]])
    return Problem(A=SparseMatrix.from_dense(D), c=np.ones(4),
                   row_lb=np.array([-INF, -INF]), row_ub=np.array([1., 1.]),
                   col_lb=np.zeros(4), col_ub=np.ones(4),
                   kind=np.ones(4, dtype=np.uint8), name="blocks")


def report(name, p, scale=False):
    if scale:
        p, _ = scale_problem(p, method="pdlp")
    t = time.perf_counter()
    info = detect_symmetry(p, time_limit=10)
    dt = time.perf_counter() - t
    rows = breaking_constraints(info, p.integer_mask)
    bad = sum(1 for g in info.generators
              if not verify_permutation(p, g, _con_for(p, g)))
    print(f"  {name:<26} {info.summary()}  rows={len(rows)}  {dt:.2f}s")
    return info


def _con_for(p, g):
    # re-verification helper is not exposed; generators were already verified
    return np.arange(p.m)


def main():
    print("synthetic:")
    report("identical items (n=8)", identical_items())
    report("distinct items (n=8)", distinct_items())
    report("two identical blocks", two_blocks())

    print("\nrefinery templates:")
    for nu in (3, 4, 6):
        report(f"unit_scheduling identical nu={nu}",
               unit_scheduling(n_units=nu, n_periods=5, identical_units=True),
               scale=True)
    report("unit_scheduling distinct nu=6",
           unit_scheduling(n_units=6, n_periods=5, identical_units=False),
           scale=True)

    print("\nMIPLIB:")
    import glob
    import os
    for path in sorted(glob.glob("data/instances/*.mps")):
        nm = os.path.splitext(os.path.basename(path))[0]
        try:
            p = read_mps(path)
        except Exception:
            continue
        report(nm, p, scale=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
