"""Symmetry detection and symmetry breaking.

The property that matters is that breaking the symmetry does not change the
answer. An unsound symmetry constraint removes real solutions silently -- the
solver reports a worse objective with full confidence and nothing downstream can
tell. Every test here either checks that a detected permutation really is an
automorphism, or that adding the derived constraints leaves the optimum intact.
"""

import numpy as np
import pytest

from vyuha.core.problem import ObjSense, Problem, Status
from vyuha.core.sparse import SparseMatrix
from vyuha.core.tolerances import INF
from vyuha.mip.symmetry import (breaking_constraints, detect_symmetry,
                                find_generators, orbits_from, refine_colours,
                                verify_permutation)
from vyuha.mip.tree import MIPParams, solve_mip
from vyuha.models import unit_scheduling
from vyuha.numerics.scaling import scale_problem


def identical_items(n=8, cap=3.0):
    """n interchangeable binaries in one knapsack row: the group is Sym(n)."""
    A = SparseMatrix.from_dense(np.ones((1, n)))
    return Problem(A=A, c=-np.ones(n), row_lb=np.array([-INF]),
                   row_ub=np.array([cap]), col_lb=np.zeros(n),
                   col_ub=np.ones(n), kind=np.ones(n, dtype=np.uint8),
                   name="identical")


def distinct_items(n=8):
    A = SparseMatrix.from_dense(np.arange(1, n + 1, dtype=float).reshape(1, n))
    return Problem(A=A, c=-np.arange(1, n + 1, dtype=float),
                   row_lb=np.array([-INF]), row_ub=np.array([9.0]),
                   col_lb=np.zeros(n), col_ub=np.ones(n),
                   kind=np.ones(n, dtype=np.uint8), name="distinct")


def two_blocks():
    """Block symmetry: {0,1} <-> {2,3}. No single transposition achieves it."""
    D = np.array([[1., 1., 0., 0.], [0., 0., 1., 1.]])
    return Problem(A=SparseMatrix.from_dense(D), c=np.ones(4),
                   row_lb=np.array([-INF, -INF]), row_ub=np.array([1., 1.]),
                   col_lb=np.zeros(4), col_ub=np.ones(4),
                   kind=np.ones(4, dtype=np.uint8), name="blocks")


# --------------------------------------------------------------------------- #
# detection                                                                    #
# --------------------------------------------------------------------------- #


def test_finds_full_symmetric_group_on_identical_items():
    info = detect_symmetry(identical_items(8))
    assert info.generators, "no symmetry found on an obviously symmetric model"
    assert info.largest_orbit == 8
    assert info.interchangeable and len(info.interchangeable[0]) == 8


def test_finds_nothing_on_a_model_with_no_symmetry():
    info = detect_symmetry(distinct_items(8))
    assert info.generators == []
    assert info.orbits == []
    assert breaking_constraints(info) == []


def test_finds_block_symmetry_no_transposition_can_express():
    """{0,1} <-> {2,3} is a symmetry; swapping only 0 and 2 is not.

    This is the case that motivates individualisation-refinement: real symmetry
    groups permute variables in coordinated blocks, so a search that only ever
    tries single transpositions finds nothing on them.
    """
    p = two_blocks()
    info = detect_symmetry(p)
    assert info.largest_orbit == 4

    swap02 = np.array([2, 1, 0, 3])
    assert not any(verify_permutation(p, swap02, sc)
                   for sc in (np.array([0, 1]), np.array([1, 0])))


def test_detects_identical_process_units():
    """The refinery case the feature exists for."""
    p, _ = scale_problem(unit_scheduling(n_units=5, n_periods=4,
                                         identical_units=True), method="pdlp")
    info = detect_symmetry(p, time_limit=10)
    assert info.generators
    assert info.largest_orbit == 5, info.summary()


def test_no_symmetry_when_units_differ():
    p, _ = scale_problem(unit_scheduling(n_units=5, n_periods=4,
                                         identical_units=False), method="pdlp")
    assert detect_symmetry(p, time_limit=5).generators == []


def test_canonical_colours_regression():
    """REGRESSION: colours were assigned by order of first appearance.

    The search refines two colourings independently -- one with ``v``
    individualised, one with ``w`` -- and then compares them cell by cell. With
    first-appearance numbering the same structural colour gets different
    integers in the two runs, so every cross-colouring comparison compared
    unrelated things and the search returned zero generators on *every* input,
    including ones with obvious Sym(8) symmetry. A feature that silently finds
    nothing is indistinguishable from one that correctly finds nothing, which is
    why this is pinned.
    """
    p = identical_items(8)
    v1, c1 = refine_colours(p)
    v2, c2 = refine_colours(p)
    assert np.array_equal(v1, v2) and np.array_equal(c1, c2)

    # individualising equivalent variables must yield identically-numbered
    # colourings, which is what makes them comparable
    base_v, base_c = refine_colours(p)
    newc = int(base_v.max()) + 1
    a = base_v.copy(); a[0] = newc
    b = base_v.copy(); b[3] = newc
    va, _ = refine_colours(p, start=(a, base_c))
    vb, _ = refine_colours(p, start=(b, base_c))
    assert sorted(va.tolist()) == sorted(vb.tolist())

    gens, tested, _ = find_generators(p, time_limit=5)
    assert tested > 0
    assert gens, "individualisation search found nothing on Sym(8)"


# --------------------------------------------------------------------------- #
# soundness of what is detected                                                #
# --------------------------------------------------------------------------- #


def test_every_reported_generator_is_a_real_automorphism():
    """Re-verify independently of the search that produced them."""
    for p in (identical_items(6), two_blocks()):
        gens, _, _ = find_generators(p, time_limit=5)
        for g in gens:
            assert np.array_equal(np.sort(g), np.arange(p.n))
            # objective, bounds and kinds must be invariant
            assert np.allclose(p.c[g], p.c)
            assert np.allclose(p.col_lb[g], p.col_lb)
            assert np.allclose(p.col_ub[g], p.col_ub)
            assert np.array_equal(p.kind[g], p.kind)


def test_verify_permutation_rejects_a_non_automorphism():
    p = distinct_items(5)
    bad = np.array([1, 0, 2, 3, 4])
    assert not verify_permutation(p, bad, np.arange(p.m))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_verify_permutation_rejects_bounds_only_an_overflow_made_equal():
    """REGRESSION: the colour key overflowed int64 and distinct bounds collided.

    Keys were ``round(value / 1e-9).astype(int64)``, which needs
    ``|value| / 1e-9`` to fit in an int64 -- a cap of about 9.2e9 on the value.
    Bounds are clipped to 1e30 before quantising, so every infinite bound blew
    straight past it. An overflowing cast is undefined: NumPy warns and yields
    ``INT64_MIN``, so all of them landed on one key.

    The two columns here are identical except for their lower bound, and the
    bounds are chosen so the collision is exact: ``-1e30 / 1e-9`` overflows to
    ``INT64_MIN``, and ``-9223372036.854776 / 1e-9`` *is* ``INT64_MIN``. So the
    only field that can reject the swap reported the two as equal, and
    ``verify_permutation`` returned True for a permutation that is not an
    automorphism -- the first column is unbounded below, the second is not.

    A collision in ``_initial_colours`` would merely cost candidates, which the
    exact check rejects. A collision *inside* the exact check is the one thing
    this module is not allowed to do: the search may be incomplete, it may
    never be unsound.
    """
    edge = 9223372036.854776
    p = Problem(A=SparseMatrix.from_dense(np.array([[1.0, 1.0]])),
                c=np.array([1.0, 1.0]),
                row_lb=np.array([-INF]), row_ub=np.array([1.0]),
                col_lb=np.array([-INF, -edge]), col_ub=np.array([10.0, 10.0]),
                name="overflow")
    assert p.col_lb[0] != p.col_lb[1], "the two bounds must actually differ"
    assert not verify_permutation(p, np.array([1, 0]), np.array([0])), (
        "accepted a permutation swapping an infinite lower bound onto a "
        "finite one")


def test_generators_map_feasible_points_to_feasible_points():
    """The defining property, checked directly on sampled solutions."""
    p = identical_items(7, cap=3.0)
    gens, _, _ = find_generators(p, time_limit=5)
    assert gens
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = (rng.random(p.n) < 0.4).astype(float)
        rv, bv, _ = p.violation(x)
        if max(rv, bv) > 1e-9:
            continue
        for g in gens:
            y = np.empty_like(x)
            y[g] = x                      # apply the permutation
            rv2, bv2, _ = p.violation(y)
            assert max(rv2, bv2) <= 1e-9
            assert abs(p.objective(y) - p.objective(x)) < 1e-9


# --------------------------------------------------------------------------- #
# the property that matters: the answer does not change                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n,cap", [(8, 3.0), (9, 4.0)])
def test_breaking_preserves_the_optimum_on_synthetic(n, cap):
    p = identical_items(n, cap)
    off = solve_mip(p, MIPParams(device="cpu", time_limit=60, symmetry=False))
    on = solve_mip(p, MIPParams(device="cpu", time_limit=60, symmetry=True))
    assert off.status == Status.OPTIMAL and on.status == Status.OPTIMAL
    assert abs(off.objective - on.objective) < 1e-6


def test_breaking_preserves_the_optimum_on_identical_units():
    p = unit_scheduling(n_units=4, n_periods=4, identical_units=True)
    off = solve_mip(p, MIPParams(device="cpu", time_limit=120, symmetry=False))
    on = solve_mip(p, MIPParams(device="cpu", time_limit=120, symmetry=True))
    if off.status != Status.OPTIMAL or on.status != Status.OPTIMAL:
        pytest.skip("not solved to optimality within the limit")
    assert abs(off.objective - on.objective) < 1e-6 * max(1.0, abs(off.objective))


def test_breaking_constraints_admit_a_sorted_representative():
    """Each emitted row must be satisfiable by permuting within the group.

    For a fully interchangeable set the sorted (non-increasing) arrangement
    satisfies every ordering row, which is exactly the validity argument.
    """
    p = identical_items(8)
    info = detect_symmetry(p)
    rows = breaking_constraints(info, p.integer_mask)
    assert rows
    x = np.zeros(p.n)
    x[:3] = 1.0                            # sorted non-increasing
    for idx, val, rhs in rows:
        assert float(val @ x[idx]) >= rhs - 1e-9


def test_detection_respects_its_time_budget():
    p, _ = scale_problem(unit_scheduling(n_units=8, n_periods=8,
                                         identical_units=True), method="pdlp")
    import time
    t = time.perf_counter()
    detect_symmetry(p, time_limit=1.0)
    assert time.perf_counter() - t < 6.0
