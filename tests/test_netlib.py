"""The Netlib expander, and the solver bug that adding Netlib found.

The Netlib LP set is distributed only in a compressed container, so reading it
at all needed an expander. The point of doing that was never the expander: it
was that ~90 problems arrive with published optima attached, which is 89 more
independent checks on the LP engine than the repository had. It paid on the
first run -- see the bore3d test at the bottom, which is a soundness bug the
MIPLIB set had never exposed.

These tests skip when the data is not cached, since it is fetched rather than
committed: ``python -m bench.netlib --fetch``.
"""

import os

import numpy as np
import pytest

from sovopt.core.problem import Status
from sovopt.io.netlib import TRTAB, expand, read_netlib
from sovopt.lp.simplex import SimplexParams, solve_simplex

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "netlib")

# name -> published optimum, from the PROBLEM SUMMARY TABLE in lp/data/readme.
# Small, fast problems chosen so the suite stays quick; bench/netlib.py runs
# the whole set.
PUBLISHED = {
    "afiro": -4.6475314286e+02,
    "adlittle": 2.2549496316e+05,
    "blend": -3.0812149846e+01,
    "share2b": -4.1573224074e+02,
    "sc50a": -6.4575077059e+01,
    "beaconfd": 3.3592485807e+04,
    "bandm": -1.5862801845e+02,
    "boeing2": -3.1501872802e+02,
    "e226": -1.8751929066e+01,
}


def _instance(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        pytest.skip("netlib instances not fetched")
    return path


def test_the_alphabet_is_ninety_two_distinct_digits():
    """The container is base-92, and the backslash in the reference source is
    a line continuation rather than a digit -- counting it would shift every
    value that uses a digit above it."""
    assert len(TRTAB) == 92
    assert len(set(TRTAB)) == 92
    assert "\\" not in TRTAB


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_expanded_problems_solve_to_the_published_optimum(name):
    """The expansion is checked by the answer, not by inspection.

    A decoder for a byte-level container fails by producing a *plausible*
    model -- one that parses, solves, and is quietly the wrong LP. Comparing
    against a number published in the same file, to ten significant figures,
    is the check that catches that; reading the MPS output never would.
    """
    prob = read_netlib(_instance(name), name=name)
    sol = solve_simplex(prob, SimplexParams())
    assert sol.status == Status.OPTIMAL
    want = PUBLISHED[name]
    assert abs(sol.objective - want) <= 1e-6 * max(1.0, abs(want))

    row_v, col_v, _ = prob.violation(sol.x)
    assert max(row_v, col_v) < 1e-6


def test_expansion_produces_parseable_mps_sections():
    with open(_instance("afiro"), "rb") as f:
        text = expand(f.read())
    lines = [ln.strip() for ln in text.splitlines()]
    for section in ("ROWS", "COLUMNS", "RHS", "ENDATA"):
        assert section in lines, f"{section} missing from the expansion"
    assert lines[0].startswith("NAME")


def test_a_truncated_file_is_rejected_rather_than_half_decoded():
    """Silence is the dangerous failure here, so the checksums are verified."""
    with open(_instance("adlittle"), "rb") as f:
        raw = f.read()
    with pytest.raises(ValueError):
        expand(raw[:len(raw) // 2])


def test_the_objective_constant_is_dropped_to_match_the_published_table():
    """e226 is the one instance in 89 with a constant in its objective.

    The sign of an RHS entry on the objective row is the oldest ambiguity in
    MPS, and Netlib's published optima are stated with no constant at all.
    Keeping it puts e226 at -11.6389 against a published -18.7519.
    """
    prob = read_netlib(_instance("e226"), name="e226")
    assert prob.obj_offset == 0.0
    sol = solve_simplex(prob, SimplexParams())
    assert abs(sol.objective - PUBLISHED["e226"]) < 1e-6 * 18.75


# --------------------------------------------------------------------------- #
# what adding the set found                                                    #
# --------------------------------------------------------------------------- #


def test_a_tighter_tolerance_does_not_turn_a_feasible_model_infeasible():
    """REGRESSION: bore3d, and the reason a new benchmark family earns its keep.

    The dual simplex reports INFEASIBLE when its ratio test finds no entering
    column that keeps the basis dual feasible. That is a Farkas certificate
    only when it is not numerical -- and at a tight ``opt_tol`` fewer columns
    qualify, so the test can run out of candidates on a problem that has an
    answer. bore3d returned the published 1373.080394 at ``feas_tol=1e-7`` and
    **INFEASIBLE** at 1e-8, which is the tolerance the benchmark harness uses.

    Tightening a tolerance must make a solver more careful, never make it
    reject a feasible model, and INFEASIBLE is the one answer a caller cannot
    check for themselves -- there is no point to hand a verifier. The verdict
    is now confirmed by an independent primal phase 1 before it is reported.
    """
    prob = read_netlib(_instance("bore3d"), name="bore3d")
    for tol in (1e-6, 1e-7, 1e-8, 1e-9):
        sol = solve_simplex(prob.copy(),
                            SimplexParams(feas_tol=tol, opt_tol=tol))
        assert sol.status == Status.OPTIMAL, (
            f"bore3d came back {sol.status.name} at tolerance {tol:.0e}")
        assert abs(sol.objective - 1373.080394) < 1e-4
