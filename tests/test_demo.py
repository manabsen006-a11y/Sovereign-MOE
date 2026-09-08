"""The demo runs, end to end, and shows what the runbook says it shows.

`python -m sovopt.cli demo` is the most visible artefact this project has --
it is the command the whole thing is presented with -- and until now nothing
tested it. Two failure modes matter and neither is caught anywhere else: a
stage raising and taking the demo down with it, and `docs/DEMO.md` drifting out
of step with the stages the code actually prints. The runbook had been claiming
seven stages against the code's eight, with every title from stage 3 onward
numbered one behind, since the arithmetic stage was added.

Run at a small size: this is a smoke test, not a benchmark, and the figures the
demo prints are measured elsewhere by `bench/`.
"""

import contextlib
import io
import re

import pytest

from sovopt.demo import run

SIZE = 6

# The titles the runbook documents, in order. Kept as a literal rather than
# imported so that a change to one has to be made in both places deliberately.
STAGES = [
    "The model: refinery product blending",
    "Solve: two engines of our own, agreeing",
    "The arithmetic, shown rather than asserted",
    "What a planner reads: shadow prices and ranging",
    "Is it right? An independent check",
    "Right by an outside standard? HiGHS on the same model",
    "Does the GPU earn its place? Measured, both ways",
    "Beyond LP: a convex quadratic, and proof it is not a vertex",
]


@pytest.fixture(scope="module")
def demo_output():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run(size=SIZE)
    return rc, buf.getvalue()


def test_demo_runs_to_completion(demo_output):
    rc, out = demo_output
    assert rc == 0
    assert out.strip(), "the demo printed nothing"


def test_every_documented_stage_appears_in_order(demo_output):
    """REGRESSION: docs/DEMO.md drifting out of step with the code.

    The runbook is what the presenter reads from, so a stage table that is off
    by one is a live failure during the one command the project is shown with.
    """
    _, out = demo_output
    seen = re.findall(r"^\s*\[(\d+)\]\s+(.*)$", out, re.M)
    assert [t.strip() for _, t in seen] == STAGES
    assert [int(n) for n, _ in seen] == list(range(1, len(STAGES) + 1))


def test_both_from_scratch_engines_run_and_agree(demo_output):
    """Stage 2's claim: two engines, different mathematics, same number.

    The interior-point half is guarded like every other stage, so a failure
    would degrade to a printed note rather than an exception -- which is
    exactly the silent failure worth asserting against.
    """
    _, out = demo_output
    assert "interior-point stage unavailable" not in out
    assert re.search(r"interior point\s+OPTIMAL", out), \
        "the interior point did not report an optimum"

    m = re.search(r"the two agree to\s+([0-9.eE+-]+)", out)
    assert m, "stage 2 did not report the agreement between the two engines"
    # the blending objective is order 1e6, so this is ~1e-8 relative
    assert float(m.group(1)) < 1e-2


def test_the_verifier_stage_passes(demo_output):
    """The independent check has to actually pass, not merely run."""
    _, out = demo_output
    assert "[PASS] row constraints" in out
    assert "[PASS] column bounds" in out
    assert "[FAIL]" not in out
