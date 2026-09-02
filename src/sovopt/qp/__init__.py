"""Quadratic programming.

Only the convex case is solved, by a proximal primal-dual method that extends
the first-order LP engine. The simplex and branch-and-bound paths still refuse a
quadratic objective outright rather than silently dropping it.
"""

from .proximal import NotConvexError, QPParams, solve_qp

__all__ = ["QPParams", "solve_qp", "NotConvexError"]
