"""The answer-key plumbing of bench.fetch, offline.

Every reference value the harness compares against comes through
``read_reference`` from a header comment, and for MIPLIB 2017 that comment
is written by ``fetch_one`` from what ``miplib_reference`` read off the
instance page. A wrong parse here is a wrong "matched the reference"
column, so both ends are pinned to real page and header text."""

from __future__ import annotations

import os

from bench.fetch import _PAGE_ROW, _decompress, read_reference

# the summary row of instance_details_gen-ip002.html, as served in Sept 2026
_PAGE = '''<tr class="odd"> <td>Simon Bowly</td> <td>41</td> <td>24</td>
<td>9.36992e-01</td> <td><a href="tag_easy.html" class="nounderline"><span
class="label label-easy">easy</span></a></td> <td>generated</td>
<td>-4783.733392</td> <td><a href="WebData/instances/gen-ip002.mps.gz">'''


def test_the_page_regex_reads_status_and_objective():
    m = _PAGE_ROW.search(_PAGE)
    assert m is not None
    assert m.group(1) == "easy"
    assert float(m.group(2)) == -4783.733392


def test_the_page_regex_is_not_fooled_by_the_tag_spans():
    """The page opens with a row of tag labels in the same span class; the
    regex anchors on the status cell, which only the summary row has."""
    tags = ('<a href="tag_benchmark.html"><span class="label label-benchmark">'
            'benchmark</span></a> <td>x</td> <td>1.0</td>')
    assert _PAGE_ROW.search(tags) is None
    m = _PAGE_ROW.search(tags + _PAGE.replace("easy", "hard"))
    assert m is not None and m.group(1) == "hard"


def test_read_reference_takes_opt_only_when_marked(tmp_path):
    p = tmp_path / "a.mps"
    p.write_text("*NAME:         a\n*BEST SOLN:    40005.05398999999 (opt)\n"
                 "*SOURCE:       miplib.zib.de instance page, status easy\n"
                 "NAME a\nROWS\n", encoding="utf-8")
    ref = read_reference(str(p))
    assert ref["best_soln"] == 40005.05398999999
    assert ref.get("proved_optimal") is True

    p.write_text("*BEST SOLN:    -563.846\n*SOURCE: page, status hard\nNAME a\n",
                 encoding="utf-8")
    ref = read_reference(str(p))
    assert ref["best_soln"] == -563.846
    assert "proved_optimal" not in ref


def test_read_reference_reads_the_miplib3_header_pair(tmp_path):
    p = tmp_path / "p0033.mps"
    p.write_text("*NAME:         p0033\n*ROWS:         16\n*COLUMNS:      33\n"
                 "*INTEGER:      33\n*NONZERO:      98\n*BEST SOLN:    3089 (opt)\n"
                 "*LP SOLN:      2520.57\nNAME          P0033\n", encoding="utf-8")
    ref = read_reference(str(p))
    assert ref["best_soln"] == 3089 and ref["lp_soln"] == 2520.57
    assert ref["rows"] == 16 and ref["columns"] == 33


def test_decompress_recognises_both_encodings():
    import bz2
    import gzip
    assert _decompress(gzip.compress(b"NAME x\n")) == b"NAME x\n"
    assert _decompress(bz2.compress(b"NAME y\n")) == b"NAME y\n"
    try:
        _decompress(b"<!DOCTYPE html>")
    except ValueError as e:
        assert "not a gzip or bz2" in str(e)
    else:
        raise AssertionError("an HTML error page must not pass as an instance")


def test_fetched_headers_name_their_source():
    """Every instance saved from MIPLIB 2017 or plato carries a *SOURCE:
    line, unless the file came with the library's own header values (a
    few MIPLIB 2017 files still do, dano3_3 among them)."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    seen = 0
    for sub in ("mittelmann", "fctp", "mittelmann-milp"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".mps"):
                with open(os.path.join(d, f), encoding="utf-8", errors="replace") as fh:
                    head = "".join(fh.readline() for _ in range(8)).upper()
                assert "*SOURCE:" in head or "BEST SOLN" in head, f
                seen += 1
    if not seen:
        import pytest
        pytest.skip("no fetched Mittelmann instances")
