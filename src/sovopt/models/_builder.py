"""A small builder for models written by name: columns and rows as they
are read off a page or a table, assembled into a :class:`Problem`.

Nothing mathematical happens here. It exists so that a model transcribed
from a book (:mod:`williams`) or from a planner's tables (:mod:`tabular`)
reads like the book or the table and not like a triplet list, and so that
every column and row keeps its name for the report at the end.

References
----------
Williams, H.P., "Model Building in Mathematical Programming", 5th ed.,
  Wiley (2013), chapter 3 -- the naming of variables and constraints the
  builder follows.
"""

from __future__ import annotations

import numpy as np

from ..core.problem import ObjSense, Problem, VarKind
from ..core.sparse import SparseMatrix
from ..core.tolerances import INF

__all__ = ["Builder"]


class Builder:
    """Columns and rows by name; ``problem()`` assembles them."""

    def __init__(self, name, sense=ObjSense.MINIMISE):
        self.name, self.sense = name, sense
        self.cols, self.lb, self.ub, self.kind, self.cost = [], [], [], [], []
        self.ri, self.rj, self.rv = [], [], []
        self.rlb, self.rub, self.rows = [], [], []
        self.index = {}
        self._row_names = set()        # the duplicate check: a list scan made
                                       # building quadratic in the row count

    def col(self, name, lo=0.0, hi=INF, c=0.0, kind=VarKind.CONTINUOUS):
        if name in self.index:
            raise ValueError(f"duplicate column name {name!r}")
        self.index[name] = len(self.cols)
        self.cols.append(name); self.lb.append(lo); self.ub.append(hi)
        self.cost.append(c); self.kind.append(kind)
        return name

    def binary(self, name, c=0.0):
        return self.col(name, 0.0, 1.0, c, VarKind.BINARY)

    def integer(self, name, hi, c=0.0):
        return self.col(name, 0.0, hi, c, VarKind.INTEGER)

    def row(self, name, terms, lo, hi):
        if name in self._row_names:
            raise ValueError(f"duplicate row name {name!r}")
        self._row_names.add(name)
        r = len(self.rows)
        for v, a in terms.items():
            if a:
                self.ri.append(r); self.rj.append(self.index[v]); self.rv.append(float(a))
        self.rlb.append(lo); self.rub.append(hi); self.rows.append(name)

    def eq(self, name, terms, rhs=0.0):
        self.row(name, terms, rhs, rhs)

    def le(self, name, terms, rhs):
        self.row(name, terms, -INF, rhs)

    def ge(self, name, terms, rhs):
        self.row(name, terms, rhs, INF)

    def problem(self):
        n = len(self.cols)
        A = SparseMatrix.from_triplets(np.asarray(self.ri, dtype=np.int32),
                                       np.asarray(self.rj, dtype=np.int32),
                                       np.asarray(self.rv, dtype=np.float64),
                                       len(self.rows), n)
        return Problem(A=A, c=np.asarray(self.cost, dtype=np.float64),
                       row_lb=np.asarray(self.rlb), row_ub=np.asarray(self.rub),
                       col_lb=np.asarray(self.lb), col_ub=np.asarray(self.ub),
                       kind=np.asarray(self.kind, dtype=np.int8),
                       sense=self.sense, name=self.name,
                       col_names=list(self.cols), row_names=list(self.rows))
