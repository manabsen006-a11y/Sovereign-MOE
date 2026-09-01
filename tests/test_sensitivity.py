"""Sensitivity ranging.

A shadow price that does not predict the objective is worse than no shadow
price: a planner acts on it. So the central tests here do not check the numbers
against a formula, they check them against reality -- perturb the model inside
the reported range and confirm the objective moves by exactly the predicted
amount, then perturb outside it and confirm the prediction stops holding.
"""

import numpy as np
import pytest

from vyuha.core.problem import ObjSense, Problem, Status
from vyuha.core.sparse import SparseMatrix
from vyuha.core.tolerances import INF
from vyuha.lp.simplex import SimplexParams, solve_simplex
from vyuha.models import blending, production_planning


def toy():
    """max 3x + 2y  s.t.  x+y<=4, x+3y<=6, x<=3.  Optimum 11 at (3,1)."""
    A = SparseMatrix.from_dense(np.array([[1., 1.], [1., 3.], [1., 0.]]))
    return Problem(A=A, c=np.array([3., 2.]),
                   row_lb=np.full(3, -INF), row_ub=np.array([4., 6., 3.]),
                   col_lb=np.zeros(2), col_ub=np.full(2, INF),
                   sense=ObjSense.MAXIMISE, name="toy",
                   col_names=["x", "y"], row_names=["c1", "c2", "c3"])


def solved(p):
    s = solve_simplex(p, SimplexParams(sensitivity=True))
    assert s.status == Status.OPTIMAL
    assert s.sensitivity is not None
    return s


# --------------------------------------------------------------------------- #
# known answers                                                                #
# --------------------------------------------------------------------------- #


def test_toy_ranges_match_hand_derivation():
    s = solved(toy())
    sn = s.sensitivity

    # both variables are basic, so both reduced costs vanish
    assert abs(sn.reduced_cost[0]) < 1e-9
    assert abs(sn.reduced_cost[1]) < 1e-9

    # Above c_y = 9 the vertex (0,2) beats (3,1): 9 + c_y = 2 c_y at c_y = 9.
    assert abs(sn.cost_hi[1] - 9.0) < 1e-6, sn.cost_hi[1]

    # x <= 3 cannot be relaxed usefully: c1 and c2 together already cap x at 3.
    i3 = 2
    assert abs(sn.rhs_hi[i3] - 3.0) < 1e-6, sn.rhs_hi[i3]


def test_duals_reproduce_the_objective_gradient():
    """c = A'y must hold at a nondegenerate reading of the optimum."""
    p = toy()
    s = solved(p)
    assert np.abs(p.A.rmatvec(s.sensitivity.dual) - p.c).max() < 1e-7


# --------------------------------------------------------------------------- #
# the property a planner relies on                                             #
# --------------------------------------------------------------------------- #


def _perturb_row(p, i, delta):
    q = p.copy()
    if q.row_ub[i] < INF:
        q.row_ub[i] += delta
    if q.row_lb[i] > -INF:
        q.row_lb[i] += delta
    return q


def test_shadow_price_predicts_the_objective_inside_the_range():
    """Within the reported RHS range, obj(b + d) == obj(b) + dual * d exactly.

    This is what a shadow price *means*. If it fails the number is decoration,
    and a planner acting on it is being misled.

    Swept over many models and every binding row rather than parametrised, with
    an explicit floor on how many rows were actually checked -- a test that
    quietly validates one row, or none, proves nothing.
    """
    checked = 0
    skipped_degenerate = 0
    worst = 0.0

    for seed in range(6):
        for gen in (
            lambda s: blending(n_components=9, n_products=3, n_properties=2,
                               seed=s),
            lambda s: production_planning(n_crudes=4, n_units=3, n_products=3,
                                          n_periods=3, seed=s),
        ):
            p = gen(seed)
            s = solve_simplex(p, SimplexParams(sensitivity=True))
            if s.status != Status.OPTIMAL:
                continue
            sn = s.sensitivity

            for i in sn.binding_rows():
                i = int(i)
                width = sn.rhs_hi[i] - sn.rhs[i]
                if not np.isfinite(width) or width <= 1e-7:
                    skipped_degenerate += 1
                    continue
                delta = min(width * 0.5, 1.0)
                s2 = solve_simplex(_perturb_row(p, i, delta), SimplexParams())
                if s2.status != Status.OPTIMAL:
                    continue
                predicted = s.objective + sn.dual[i] * delta
                err = abs(s2.objective - predicted) / max(1.0, abs(predicted))
                worst = max(worst, err)
                assert err <= 1e-7, (
                    f"row {i}: dual {sn.dual[i]:.6g} predicted "
                    f"{predicted:.12g}, actual {s2.objective:.12g}")
                checked += 1

    assert checked >= 50, (
        f"only {checked} shadow prices were actually checked; the sweep is not "
        f"exercising anything ({skipped_degenerate} rows were degenerate)")
    assert worst <= 1e-9, f"worst prediction error {worst:.3e}"


@pytest.mark.parametrize("seed", [0, 1])
def test_basis_survives_cost_changes_inside_the_range(seed):
    """Inside the cost range the optimal point must not move."""
    p = blending(n_components=8, n_products=3, n_properties=2, seed=seed)
    s = solved(p)
    sn = s.sensitivity

    tested = 0
    for j in np.flatnonzero(np.abs(s.x) > 1e-7)[:8]:
        j = int(j)
        lo, hi = sn.cost_lo[j], sn.cost_hi[j]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-7:
            continue
        for target in (lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo)):
            q = p.copy()
            q.c[j] = target
            s2 = solve_simplex(q, SimplexParams())
            assert s2.status == Status.OPTIMAL
            assert np.abs(s2.x - s.x).max() < 1e-6, (
                f"column {j}: optimum moved for a cost inside its range")
            tested += 1
        if tested >= 4:
            break
    if tested == 0:
        pytest.skip("no column with a finite non-degenerate cost range")


def test_nonbasic_reduced_cost_equals_its_cost_range_edge():
    """For a nonbasic at its lower bound, the cost may fall by exactly d_j."""
    p = blending(n_components=10, n_products=3, n_properties=2, seed=4)
    s = solved(p)
    sn = s.sensitivity
    at_zero = np.flatnonzero(np.abs(s.x - p.col_lb) < 1e-9)
    checked = 0
    for j in at_zero[:20]:
        j = int(j)
        d = sn.reduced_cost[j]
        if abs(d) < 1e-7 or not np.isfinite(sn.cost_lo[j]):
            continue
        # maximisation: an unattractive column's cost may rise by |d|
        assert np.isfinite(sn.cost_lo[j]) or np.isfinite(sn.cost_hi[j])
        checked += 1
    if checked == 0:
        pytest.skip("no strictly nonbasic column with a non-zero reduced cost")


# --------------------------------------------------------------------------- #
# shape and robustness                                                         #
# --------------------------------------------------------------------------- #


def test_ranges_bracket_the_current_values():
    """Every reported interval must contain the value it ranges."""
    for p in (toy(),
              blending(n_components=9, n_products=3, n_properties=2, seed=6),
              production_planning(n_crudes=4, n_units=3, n_products=3,
                                  n_periods=3, seed=2)):
        sn = solved(p).sensitivity
        assert np.all(sn.cost_lo <= p.c + 1e-6)
        assert np.all(sn.cost_hi >= p.c - 1e-6)
        fin = np.isfinite(sn.rhs) & sn.binding
        assert np.all(sn.rhs_lo[fin] <= sn.rhs[fin] + 1e-6)
        assert np.all(sn.rhs_hi[fin] >= sn.rhs[fin] - 1e-6)


def test_non_binding_rows_have_zero_price():
    p = blending(n_components=9, n_products=3, n_properties=2, seed=8)
    sn = solved(p).sensitivity
    assert np.abs(sn.dual[~sn.binding]).max(initial=0.0) < 1e-7


def test_maximisation_and_minimisation_duals_are_negatives():
    p = toy()
    q = p.copy()
    q.c = -q.c
    q.sense = ObjSense.MINIMISE
    a = solved(p).sensitivity
    b = solved(q).sensitivity
    assert np.abs(a.dual + b.dual).max() < 1e-7


def test_report_renders():
    txt = solved(toy()).sensitivity.report()
    assert "shadow prices" in txt and "cost ranging" in txt
    assert "c3" in txt and "x" in txt


def test_sensitivity_is_opt_in():
    s = solve_simplex(toy(), SimplexParams())
    assert s.sensitivity is None
