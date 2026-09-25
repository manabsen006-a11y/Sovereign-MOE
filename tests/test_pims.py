"""Aspen PIMS blending tables to ``sovopt blend``'s (:mod:`sovopt.io.pims`).

The answer key is a round trip: the tables of examples/blending written
out as PIMS tables -- BUY, SELL, BLNMIX, BLNSPEC, BLNPROP, in either
orientation, as CSV files or one workbook -- must convert back to a model
with the same optimum. examples/pims is a PIMS-layout gasoline blend with
the parts a real export has and a round trip does not: comment rows, a
TEXT column, a BLNMIX that keeps streams out of products, streams the
refinery makes (values.csv), and RVP blended through its index.
"""

import csv
import io
import os

import pytest

from bench.verify import verify
from sovopt.core.problem import Status
from sovopt.io.pims import PimsError, convert_pims, pims_to_csv, read_pims
from sovopt.lp.simplex import solve_simplex
from sovopt.models.tabular import parse_blending_csv, read_blending_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLEND = os.path.join(ROOT, "examples", "blending")
PIMS = os.path.join(ROOT, "examples", "pims")


def _dicts(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_pims(prop_rows_are_properties=True, mix_rows_are_components=True, allow=None):
    """examples/blending written as the five PIMS tables (grids)."""
    comps = _dicts(os.path.join(BLEND, "components.csv"))
    prods = _dicts(os.path.join(BLEND, "products.csv"))
    quals = [k for k in comps[0] if k not in ("name", "cost", "available", "minimum")]
    cn = [c["name"] for c in comps]
    pn = [p["name"] for p in prods]
    buy = [["", "TEXT", "COST", "MIN", "MAX"]] + [
        [c["name"], "", c["cost"], c["minimum"], c["available"]] for c in comps]
    sell = [["", "TEXT", "PRICE", "MIN", "MAX"]] + [
        [p["name"], "", p["price"], p["demand_min"], p["demand_max"]] for p in prods]
    spec = [["", "TEXT"] + pn]
    for q in quals:
        for side, prefix in (("min", "N"), ("max", "X")):
            vals = [p.get(f"{q}_{side}", "") for p in prods]
            if any(vals):
                spec.append([prefix + q.upper(), ""] + vals)
    if prop_rows_are_properties:
        prop = [["", "TEXT"] + cn] + [[q.upper(), ""] + [c[q] for c in comps] for q in quals]
    else:
        prop = [["", "TEXT"] + [q.upper() for q in quals]] + [
            [c["name"], ""] + [c[q] for q in quals] for c in comps]
    allow = allow or {}
    ok = {(c, p): "1" if p in allow.get(c, pn) else "" for c in cn for p in pn}
    if mix_rows_are_components:
        mix = [["", "TEXT"] + pn] + [[c, ""] + [ok[c, p] for p in pn] for c in cn]
    else:
        mix = [["", "TEXT"] + cn] + [[p, ""] + [ok[c, p] for c in cn] for p in pn]
    return {"BUY": buy, "SELL": sell, "BLNSPEC": spec, "BLNPROP": prop, "BLNMIX": mix}


def _write_dir(grids, d):
    os.makedirs(d, exist_ok=True)
    for name, rows in grids.items():
        with open(os.path.join(d, f"{name}.csv"), "w", encoding="utf-8", newline="") as fh:
            csv.writer(fh, lineterminator="\n").writerows(rows)
    return d


def _flow(t, s, component, product):
    """The solved flow of ``component`` into ``product`` (flows are keyed by name)."""
    return s.x[t.problem.col_names.index(t.flow[component, product])]


def _optimum(comps_text, prods_text):
    t = parse_blending_csv(comps_text, prods_text)
    s = solve_simplex(t.problem)
    assert s.status == Status.OPTIMAL
    assert verify(t.problem, s.x, s.objective, y=s.y).ok
    return s.objective, t


REFERENCE = None


def _reference():
    global REFERENCE
    if REFERENCE is None:
        REFERENCE = _optimum(open(os.path.join(BLEND, "components.csv"), encoding="utf-8").read(),
                             open(os.path.join(BLEND, "products.csv"), encoding="utf-8").read())[0]
    return REFERENCE


@pytest.mark.parametrize("prop_rows_are_properties", [True, False])
@pytest.mark.parametrize("mix_rows_are_components", [True, False])
def test_a_round_trip_through_pims_tables_keeps_the_optimum(tmp_path, prop_rows_are_properties,
                                                            mix_rows_are_components):
    """Whichever way round BLNPROP and BLNMIX are written, the converter
    reads the axes from the tags and gets the same model back."""
    d = _write_dir(_as_pims(prop_rows_are_properties, mix_rows_are_components), tmp_path / "t")
    comps, prods, _notes = convert_pims(read_pims(str(d)))
    obj, _ = _optimum(comps, prods)
    assert abs(obj - _reference()) <= 1e-9 * max(1.0, abs(_reference()))


def test_a_workbook_reads_the_same_as_the_csv_files(tmp_path):
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in _as_pims().items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append([float(v) if _isnum(v) else v for v in r])
    path = str(tmp_path / "model.xlsx")
    wb.save(path)
    comps, prods, _ = convert_pims(read_pims(path))
    assert abs(_optimum(comps, prods)[0] - _reference()) <= 1e-9 * abs(_reference())


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def test_blnmix_keeps_a_component_out_of_a_product(tmp_path):
    """Keep Reformate out of Regular: the converted table says so, the plan
    sends none there, and the margin cannot rise. (PIMS tags are
    case-insensitive; the converter writes them upper case.)"""
    grids = _as_pims(allow={"Reformate": ["Premium", "Export"]})
    comps, prods, _ = convert_pims(read_pims(str(_write_dir(grids, tmp_path / "t"))))
    row = next(r for r in csv.DictReader(io.StringIO(comps)) if r["name"] == "REFORMATE")
    assert row["allowed"] == "PREMIUM;EXPORT"
    obj, t = _optimum(comps, prods)
    s = solve_simplex(t.problem)
    assert abs(_flow(t, s, "REFORMATE", "REGULAR")) <= 1e-9
    assert obj <= _reference() + 1e-9


def test_the_example_converts_solves_and_honours_its_mix(tmp_path):
    notes = pims_to_csv(PIMS, str(tmp_path), os.path.join(PIMS, "values.csv"), {"RVP": 1.25})
    assert any("LPG" in n for n in notes)                 # priced, not blended
    t = read_blending_csv(str(tmp_path / "components.csv"), str(tmp_path / "products.csv"))
    assert "rvp_index" in t.qualities
    s = solve_simplex(t.problem)
    assert s.status == Status.OPTIMAL
    assert verify(t.problem, s.x, s.objective, y=s.y).headline()[0] == "ACCEPTED"
    assert abs(_flow(t, s, "LSR", "PRG")) <= 1e-9            # BLNMIX keeps it out
    pr = {p["name"]: p for p in t.products}
    assert abs(pr["URG"]["specs"]["rvp_index"]["max"] - 60 ** 1.25) <= 1e-6


def test_a_stream_the_refinery_makes_needs_its_value(tmp_path):
    """Without values.csv the made streams have no BUY row and no cost; the
    converter names them rather than price them at zero."""
    with pytest.raises(PimsError) as e:
        convert_pims(read_pims(PIMS))
    for tag in ("LSR", "ISO", "RFT", "FCN", "HCN"):
        assert tag in str(e.value)


def test_a_property_missing_from_blnprop_is_named(tmp_path):
    grids = _as_pims()
    grids["BLNPROP"] = [r for r in grids["BLNPROP"] if r[0] != "SULFUR"]
    with pytest.raises(PimsError) as e:
        convert_pims(read_pims(str(_write_dir(grids, tmp_path / "t"))))
    assert "SULFUR" in str(e.value) and "every specified property" in str(e.value)


def test_a_specified_product_must_have_a_price(tmp_path):
    grids = _as_pims()
    grids["SELL"] = [r for r in grids["SELL"] if r[0] != "Export"]
    with pytest.raises(PimsError) as e:
        convert_pims(read_pims(str(_write_dir(grids, tmp_path / "t"))))
    assert "no SELL row" in str(e.value) and "EXPORT" in str(e.value)


def test_a_spec_row_that_is_neither_n_nor_x_is_reported_and_skipped(tmp_path):
    grids = _as_pims()
    grids["BLNSPEC"].append(["ESULFUR", "", "0.03", "", ""])
    comps, prods, notes = convert_pims(read_pims(str(_write_dir(grids, tmp_path / "t"))))
    assert any("ESULFUR" in n for n in notes)
    assert abs(_optimum(comps, prods)[0] - _reference()) <= 1e-9 * abs(_reference())


def test_allowed_names_only_products():
    comps = "name,cost,allowed,sulfur\nA,1,Regular;Nope,0.1\n"
    prods = "name,price,sulfur_max\nRegular,2,0.2\n"
    from sovopt.models.tabular import TableError
    with pytest.raises(TableError) as e:
        parse_blending_csv(comps, prods)
    assert "'Nope'" in str(e.value) and "not a product" in str(e.value)


def test_the_pooling_model_refuses_an_allowed_list():
    from sovopt.models.tabular import TableError
    from sovopt.models.tabular_pooling import parse_pooling_csv
    comps = "name,cost,allowed,direct,sulfur\nA,6,X,,3\nB,16,,*,1\n"
    prods = "name,price,demand_max,sulfur_max\nX,9,100,2.5\nY,15,200,1.5\n"
    with pytest.raises(TableError) as e:
        parse_pooling_csv(comps, prods, "name,capacity,inputs\nP,600,A\n")
    assert "'allowed'" in str(e.value)
