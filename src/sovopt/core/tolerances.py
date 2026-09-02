"""Named numerical tolerances.

Every tolerance in the solver is declared here. No bare ``1e-9`` in algorithm
code -- when a refinery model misbehaves, the first question is always "which
tolerance did it trip", and that question needs one place to look.

Defaults follow the conventions the benchmark libraries assume, so that
solutions we report as optimal are optimal by the same yardstick MIPLIB and
Netlib use.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

INF = 1.0e30
"""Anything at or beyond this magnitude is an infinite bound.

MPS has no way to write infinity, and 1e30 is the universal convention. Real
coefficients never reach it; treating 1e30 as finite produces garbage duals.
"""


@dataclass(frozen=True)
class Tolerances:
    # -- feasibility -------------------------------------------------------- #
    primal_feas: float = 1e-8
    """Max allowed row/bound violation, measured on the *unscaled* model."""

    dual_feas: float = 1e-8
    """Max allowed reduced-cost sign violation."""

    integrality: float = 1e-6
    """Distance from an integer at which a value counts as integral. MIPLIB's
    own checker uses 1e-4; we hold ourselves to 1e-6 and report against both."""

    # -- optimality --------------------------------------------------------- #
    mip_gap_rel: float = 1e-4
    mip_gap_abs: float = 1e-10
    opt_rel: float = 1e-9
    """Relative duality gap at which an LP is declared solved."""

    # -- linear algebra ----------------------------------------------------- #
    pivot: float = 1e-7
    """Smallest acceptable pivot magnitude in the simplex ratio test."""

    lu_pivot_rel: float = 0.01
    """Threshold-Markowitz relative pivot tolerance: a candidate pivot must be
    at least this fraction of the largest magnitude in its column. Lower means
    sparser factors and worse stability; 0.01 is the usual compromise."""

    lu_drop: float = 1e-14
    """Entries below this are dropped from the factors as numerical noise."""

    refactor_residual: float = 1e-9
    """Trigger a refactorisation when the FTRAN residual exceeds this."""

    zero: float = 1e-12
    """Below this, a computed quantity is treated as exactly zero."""

    # -- degeneracy --------------------------------------------------------- #
    harris_relax: float = 1e-9
    """Bound relaxation in the first pass of the Harris ratio test."""

    perturb_base: float = 1e-7
    """Base magnitude for anti-degeneracy cost perturbation."""

    # -- presolve ----------------------------------------------------------- #
    presolve_bound_tighten: float = 1e-9
    """Minimum improvement for a tightened bound to be worth recording."""

    coeff_huge: float = 1e10
    coeff_tiny: float = 1e-10
    """Coefficients outside [tiny, huge] earn a numerical-quality warning."""

    def scaled(self, factor: float) -> "Tolerances":
        """Loosen (or tighten) every feasibility tolerance by a factor.

        Used by heuristics that only need an approximately feasible point, and
        by the safe-bound path that deliberately runs loose then repairs.
        """
        return replace(
            self,
            primal_feas=self.primal_feas * factor,
            dual_feas=self.dual_feas * factor,
            opt_rel=self.opt_rel * factor,
        )


DEFAULT = Tolerances()
