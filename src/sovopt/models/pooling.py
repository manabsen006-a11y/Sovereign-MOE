"""Pooling models -- the non-convex core of crude and product blending.

Haverly's pooling problem is the standard benchmark for exactly the phenomenon
that costs refineries money: a blend whose quality is a *product* of decision
variables, where the fixed-point iteration industry uses (distributive recursion
in PIMS and GRTMPS) can settle on a local optimum worth less than the global
one.

The structure
-------------
Two crudes ``A`` and ``B`` feed a **pool**; a third, ``C``, bypasses it and goes
straight to products. Two products ``X`` and ``Y`` each have a maximum sulfur
specification. The difficulty is that the pool has one sulfur content, ``p``,
shared by everything leaving it -- and the sulfur *carried* into a product is
``p`` times the flow, a product of two decisions.

Variables (in this order)::

    xA  xB   crude into the pool
    yX  yY   pool into each product
    cX  cY   crude C direct to each product
    p        pool sulfur          <- the complicating variable
    w1  w2   auxiliaries: w1 = p*yX, w2 = p*yY

Constraints::

    xA + xB = yX + yY                       pool volume balance
    3 xA + 1 xB = w1 + w2                   pool sulfur balance
    w1 - 2.5 yX - 0.5 cX <= 0               product X spec (2.5% max)
    w2 - 1.5 yY + 0.5 cY <= 0               product Y spec (1.5% max)
    yX + cX <= dX,   yY + cY <= dY          demand

Why one pool is useful for testing
----------------------------------
With a single pool there is a single complicating variable ``p``. Fix it and
everything left is a linear program, so the true global optimum can be found by
scanning ``p`` across its range and solving an LP at each point. That gives an
**independent** check on the spatial branch-and-bound rather than a comparison
against a remembered literature value -- which is what `tests/test_pooling.py`
uses.

References
----------
Haverly, "Studies of the behaviour of recursion for the pooling problem",
  ACM SIGMAP Bulletin 25 (1978) 19-28.
Adhya, Tawarmalani & Sahinidis, "A Lagrangian approach to the pooling problem",
  Ind. Eng. Chem. Res. 38 (1999) 1956-1972.
Misener & Floudas, "Advances for the pooling problem: modeling, global
  optimization, and computational studies", Applied and Computational
  Mathematics 8 (2009) 3-22.
"""

from __future__ import annotations

import numpy as np

from ..core.problem import ObjSense, Problem
from ..core.sparse import SparseMatrix
from ..core.tolerances import INF
from ..globalopt.bilinear import BilinearProblem, BilinearTerm

__all__ = ["haverly", "HAVERLY_VARIANTS"]

# index layout
_XA, _XB, _YX, _YY, _CX, _CY, _P, _W1, _W2 = range(9)

HAVERLY_VARIANTS = (1, 2, 3)


def haverly(variant: int = 1,
            cost_A: float = 6.0, cost_B: float = 16.0, cost_C: float = 10.0,
            price_X: float = 9.0, price_Y: float = 15.0,
            demand_X: float = 100.0, demand_Y: float = 200.0,
            sulfur_A: float = 3.0, sulfur_B: float = 1.0,
            sulfur_C: float = 2.0,
            spec_X: float = 2.5, spec_Y: float = 1.5) -> BilinearProblem:
    """Haverly's pooling problem, as a :class:`BilinearProblem`.

    The three published variants differ in one parameter each:

    ``1`` the base case;
    ``2`` product X demand raised to 600;
    ``3`` crude B cheaper, at 13.
    """
    if variant == 2:
        demand_X = 600.0
    elif variant == 3:
        cost_B = 13.0

    n = 9
    rows, cols, vals = [], [], []
    rlo, rub, rnames = [], [], []
    r = 0

    def add(entries, lb, ub, name):
        nonlocal r
        for j, v in entries:
            rows.append(r)
            cols.append(j)
            vals.append(v)
        rlo.append(lb)
        rub.append(ub)
        rnames.append(name)
        r += 1

    # pool volume balance: xA + xB - yX - yY = 0
    add([(_XA, 1.0), (_XB, 1.0), (_YX, -1.0), (_YY, -1.0)], 0.0, 0.0, "pool_vol")

    # pool sulfur balance: sA*xA + sB*xB - w1 - w2 = 0
    add([(_XA, sulfur_A), (_XB, sulfur_B), (_W1, -1.0), (_W2, -1.0)],
        0.0, 0.0, "pool_sulfur")

    # product X spec:  w1 + sC*cX <= specX*(yX + cX)
    add([(_W1, 1.0), (_YX, -spec_X), (_CX, sulfur_C - spec_X)],
        -INF, 0.0, "spec_X")

    # product Y spec:  w2 + sC*cY <= specY*(yY + cY)
    add([(_W2, 1.0), (_YY, -spec_Y), (_CY, sulfur_C - spec_Y)],
        -INF, 0.0, "spec_Y")

    add([(_YX, 1.0), (_CX, 1.0)], -INF, demand_X, "demand_X")
    add([(_YY, 1.0), (_CY, 1.0)], -INF, demand_Y, "demand_Y")

    A = SparseMatrix.from_triplets(rows, cols, vals, r, n)

    c = np.zeros(n)
    c[_XA] = -cost_A
    c[_XB] = -cost_B
    c[_YX] = price_X
    c[_YY] = price_Y
    c[_CX] = price_X - cost_C
    c[_CY] = price_Y - cost_C

    lo = np.zeros(n)
    hi = np.empty(n)
    total = demand_X + demand_Y
    hi[_XA] = hi[_XB] = total
    hi[_YX] = demand_X
    hi[_YY] = demand_Y
    hi[_CX] = demand_X
    hi[_CY] = demand_Y
    # the pool can only be as sour as its sourest feed, or as sweet as its
    # sweetest -- these bounds are what make the envelopes finite
    lo[_P] = min(sulfur_A, sulfur_B)
    hi[_P] = max(sulfur_A, sulfur_B)
    hi[_W1] = hi[_P] * demand_X
    hi[_W2] = hi[_P] * demand_Y

    names = ["xA", "xB", "yX", "yY", "cX", "cY", "p", "w1", "w2"]
    linear = Problem(A=A, c=c,
                     row_lb=np.array(rlo), row_ub=np.array(rub),
                     col_lb=lo, col_ub=hi,
                     sense=ObjSense.MAXIMISE,
                     name=f"haverly{variant}",
                     col_names=names, row_names=rnames)

    terms = [BilinearTerm(w=_W1, x=_P, y=_YX),
             BilinearTerm(w=_W2, x=_P, y=_YY)]
    return BilinearProblem(linear=linear, terms=terms,
                           name=f"haverly{variant}")


def pool_quality_index() -> int:
    """Column index of the pool quality, the single complicating variable."""
    return _P
