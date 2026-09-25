"""Aspen PIMS model tables to the tables ``sovopt blend`` reads.

A refinery planning model in Aspen PIMS keeps its data in named tables --
worksheets of one workbook, or one CSV file each when saved out. Its
blending lives in five of them:

    BUY       purchased streams: TEXT, COST, MIN, MAX
    SELL      products: TEXT, PRICE, MIN, MAX
    BLNMIX    which components may go into which blended product (a
              non-zero entry where one may)
    BLNSPEC   the blend specifications: a row ``N<prop>`` is a minimum on
              property ``<prop>``, ``X<prop>`` a maximum, one column per
              product
    BLNPROP   the components' blending properties, by property tag

Each is a grid: row tags down the first column, column tags across the
first row, an optional ``TEXT`` column of descriptions, and comment rows
and columns starting with ``*``. Which axis of BLNMIX holds the products,
and which of BLNPROP the properties, is read from the data -- the axis
whose tags the other tables name -- rather than assumed.

The conversion writes

    components.csv   name, cost, available, minimum, allowed, a column per
                     specified property
    products.csv     name, price, demand_min, demand_max, <prop>_min/_max

with BLNMIX carried as each component's ``allowed`` products.

What a PIMS model knows that these five tables do not, and what is done
about it:

* **The value of a stream the refinery makes.** Only purchased streams
  have a BUY row. Every other component needs its value (and its
  production, as ``available``) from ``values`` -- a table of ``name,
  cost, available, minimum``, which a PIMS solution's marginal values and
  stream rates supply. A component with neither is refused by name, never
  given a cost of zero.
* **Properties PIMS computes** (submodels, recursion). Every property a
  specification names must be in BLNPROP for every component; the ones
  missing are listed.
* **Non-linear blending.** PIMS blends vapour pressure, viscosity and the
  like through indices. ``index={"RVP": 1.25}`` raises both the
  components' values and the specification limits to that power (the
  RVP^1.25 index), and names the column ``rvp_index``.
* **Weight-basis properties** are blended by volume here; a PIMS
  property blended by weight is reported, not converted.
* **Multi-period tables and pooling recursion** are not read.

References
----------
Aspen PIMS table conventions -- the BUY, SELL, BLNMIX, BLNSPEC and BLNPROP
  tables and the N/X prefixes of specification rows -- as refinery-planning
  practice describes them. This module was written without AspenTech's
  documentation, a PIMS installation or a PIMS file; that is why table
  orientation is detected rather than assumed, and why a real model's
  export is the test that settles it.
Gary, Handwerk & Kaiser, "Petroleum Refining: Technology and Economics",
  5th ed., CRC Press (2007), ch. 14 -- the RVP^1.25 blending index.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field

__all__ = ["PimsError", "PimsTable", "read_pims", "convert_pims", "pims_to_csv"]

TABLES = ("BUY", "SELL", "BLNMIX", "BLNSPEC", "BLNPROP")


class PimsError(ValueError):
    """PIMS tables that cannot be converted, with the reason."""


@dataclass
class PimsTable:
    name: str
    cols: list                         # column tags, TEXT and comments removed
    rows: dict                         # row tag -> {column tag: cell text}
    text: dict = field(default_factory=dict)

    def num(self, row, col):
        """The cell as a float, ``None`` when blank."""
        s = self.rows.get(row, {}).get(col, "")
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            raise PimsError(f"table {self.name}: row {row}, column {col}: {s!r} is not a number")


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _table(name: str, grid) -> PimsTable:
    """A PIMS grid: the header is the first row carrying a TEXT column, or
    else the first non-blank row that is not a comment."""
    lines = [[_cell(c) for c in r] for r in grid]
    lines = [r for r in lines if any(r)]
    head_at = next((i for i, r in enumerate(lines)
                    if any(c.upper() == "TEXT" for c in r[1:])), None)
    if head_at is None:
        head_at = next((i for i, r in enumerate(lines) if not r[0].startswith("*")), None)
    if head_at is None:
        raise PimsError(f"table {name}: no header row")
    head = [c.upper() for c in lines[head_at]]
    keep = [(j, h) for j, h in enumerate(head)
            if j > 0 and h and not h.startswith("*") and h != "TEXT"]
    text_j = head.index("TEXT") if "TEXT" in head else None
    rows, text = {}, {}
    for r in lines[head_at + 1:]:
        tag = r[0].upper()
        if not tag or tag.startswith("*"):
            continue
        if tag in rows:
            raise PimsError(f"table {name}: row {tag!r} appears twice")
        rows[tag] = {h: (r[j] if j < len(r) else "") for j, h in keep}
        if text_j is not None and text_j < len(r):
            text[tag] = r[text_j]
    return PimsTable(name, [h for _, h in keep], rows, text)


def read_pims(source) -> dict:
    """The PIMS tables in ``source``: a directory of ``<TABLE>.csv`` files, or
    an ``.xlsx`` workbook with a sheet per table. Names are case-insensitive;
    tables other than the five this converter reads are ignored."""
    grids = {}
    if os.path.isdir(source):
        for fn in os.listdir(source):
            stem, ext = os.path.splitext(fn)
            if ext.lower() == ".csv" and stem.upper() in TABLES:
                with open(os.path.join(source, fn), encoding="utf-8-sig", newline="") as fh:
                    grids[stem.upper()] = list(csv.reader(fh))
    elif source.lower().endswith((".xlsx", ".xlsm")):
        try:
            import openpyxl
        except ImportError as e:                       # pragma: no cover
            raise PimsError("reading a workbook needs openpyxl (pip install openpyxl), "
                            "or save each table as CSV into one folder") from e
        wb = openpyxl.load_workbook(source, read_only=True, data_only=True)
        for ws in wb.worksheets:
            if ws.title.strip().upper() in TABLES:
                grids[ws.title.strip().upper()] = [list(r) for r in ws.iter_rows(values_only=True)]
        wb.close()
    else:
        raise PimsError(f"{source}: give a folder of <TABLE>.csv files or an .xlsx workbook")
    return {name: _table(name, g) for name, g in grids.items()}


def _orient(table: PimsTable, names: set):
    """``(outer, inner)`` tag lists of ``table`` with ``inner`` the axis that
    names more of ``names`` -- so ``table.rows[o][i]`` or its transpose."""
    by_col = len(set(table.cols) & names)
    by_row = len(set(table.rows) & names)
    if by_col == 0 and by_row == 0:
        return None
    if by_col >= by_row:
        return list(table.rows), list(table.cols), lambda o, i: table.rows.get(o, {}).get(i, "")
    return list(table.cols), list(table.rows), lambda o, i: table.rows.get(i, {}).get(o, "")


def _read_values(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    out = {}
    for r in rows:
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        name = r.get("name", "").upper()
        if not name:
            continue
        rec = {}
        for k in ("cost", "available", "minimum"):
            if r.get(k, "") != "":
                try:
                    rec[k] = float(r[k])
                except ValueError:
                    raise PimsError(f"values: {name} {k}: {r[k]!r} is not a number")
        out[name] = rec
    return out


def _fmt(v) -> str:
    return "" if v is None else f"{v:.10g}"


def convert_pims(tables: dict, values_text: str | None = None, index: dict | None = None):
    """``(components_csv, products_csv, notes)`` from the PIMS tables."""
    notes = []
    index = {k.upper(): float(p) for k, p in (index or {}).items()}
    for need in ("SELL", "BLNSPEC", "BLNPROP"):
        if need not in tables:
            raise PimsError(f"no {need} table (found: {', '.join(sorted(tables)) or 'none'})")
    sell, spec, prop = tables["SELL"], tables["BLNSPEC"], tables["BLNPROP"]
    buy = tables.get("BUY")
    values = _read_values(values_text) if values_text else {}

    # ---- products and their specifications ---------------------------------
    products = [c for c in spec.cols]
    missing = [p for p in products if p not in sell.rows]
    if missing:
        raise PimsError(f"BLNSPEC names products with no SELL row (no price): {', '.join(missing)}")
    unblended = [p for p in sell.rows if p not in products]
    if unblended:
        notes.append(f"SELL rows with no specification in BLNSPEC, not blended here: "
                     f"{', '.join(unblended)}")
    specs = {p: {} for p in products}
    props = []
    for row in spec.rows:
        side = {"N": "min", "X": "max"}.get(row[0])
        if side is None or len(row) < 2:
            notes.append(f"BLNSPEC row {row!r} is not N<property> or X<property>; skipped")
            continue
        q = row[1:]
        for p in products:
            v = spec.num(row, p)
            if v is not None:
                specs[p].setdefault(q, {})[side] = v
                if q not in props:
                    props.append(q)
    if not props:
        raise PimsError("BLNSPEC holds no N<property>/X<property> limits")

    # ---- which components, allowed into which products -----------------------
    po = _orient(prop, set(props))
    if po is None:
        raise PimsError(f"BLNPROP names none of the specified properties "
                        f"({', '.join(props)})")
    streams, _, prop_cell = po
    allowed = {}
    if "BLNMIX" in tables:
        mo = _orient(tables["BLNMIX"], set(products))
        if mo is None:
            raise PimsError(f"BLNMIX names none of the products ({', '.join(products)})")
        comps, prods_in_mix, mix_cell = mo
        for c in comps:
            ok = []
            for p in prods_in_mix:
                s = mix_cell(c, p)
                try:
                    if s != "" and float(s) != 0.0:
                        ok.append(p)
                except ValueError:
                    raise PimsError(f"BLNMIX: {c}, {p}: {s!r} is not a number")
            ok = [p for p in ok if p in products]
            if ok:
                allowed[c] = ok
        components = [c for c in comps if c in allowed]
    else:
        components = list(streams)
        allowed = {c: list(products) for c in components}
        notes.append("no BLNMIX table: every component may enter every product")
    if not components:
        raise PimsError("no component is allowed into any product")

    # ---- each component's value, availability and properties ---------------
    no_value, no_prop, rows_out = [], [], []
    for c in components:
        rec = dict(values.get(c, {}))
        if buy is not None and c in buy.rows:
            if "cost" not in rec:
                rec["cost"] = buy.num(c, "COST")
            rec.setdefault("available", buy.num(c, "MAX"))
            rec.setdefault("minimum", buy.num(c, "MIN"))
        if rec.get("cost") is None:
            no_value.append(c)
            continue
        if rec.get("available") is None:
            notes.append(f"{c}: no MAX in BUY and no 'available' in values -- unlimited")
        qv = {}
        for q in props:
            s = prop_cell(c, q) if c in streams else ""
            try:
                v = float(s) if s != "" else None
            except ValueError:
                raise PimsError(f"BLNPROP: {c}, {q}: {s!r} is not a number")
            if v is None:
                no_prop.append(f"{c}.{q}")
                continue
            qv[q] = v
        rows_out.append((c, rec, qv))
    if no_value:
        raise PimsError(f"no value for {', '.join(no_value)}: not purchased in BUY and not "
                        f"in the values table (name, cost, available, minimum) -- a stream "
                        f"the refinery makes needs its marginal value from a PIMS solution")
    if no_prop:
        raise PimsError(f"BLNPROP has no value for {', '.join(no_prop)}; every component "
                        f"needs every specified property")

    # ---- indices for what does not blend linearly --------------------------
    def col(q):
        return f"{q.lower()}_index" if q in index else q.lower()

    def tx(q, v):
        if q not in index:
            return v
        if v < 0:
            raise PimsError(f"{q}: {v:g} is negative; a power index needs values >= 0")
        return v ** index[q]
    for q in index:
        if q not in props:
            notes.append(f"index {q}: no specification names it; ignored")

    # ---- write -------------------------------------------------------------
    cbuf = io.StringIO()
    w = csv.writer(cbuf, lineterminator="\n")
    w.writerow(["name", "cost", "available", "minimum", "allowed"] + [col(q) for q in props])
    for c, rec, qv in rows_out:
        a = "" if set(allowed[c]) >= set(products) else ";".join(allowed[c])
        w.writerow([c, _fmt(rec.get("cost")), _fmt(rec.get("available")),
                    _fmt(rec.get("minimum")), a] + [_fmt(tx(q, qv[q])) for q in props])
    pbuf = io.StringIO()
    w = csv.writer(pbuf, lineterminator="\n")
    head = ["name", "price", "demand_min", "demand_max"]
    for q in props:
        head += [f"{col(q)}_min", f"{col(q)}_max"]
    w.writerow(head)
    for p in products:
        row = [p, _fmt(sell.num(p, "PRICE")), _fmt(sell.num(p, "MIN")), _fmt(sell.num(p, "MAX"))]
        for q in props:
            s = specs[p].get(q, {})
            row += [_fmt(tx(q, s["min"])) if "min" in s else "",
                    _fmt(tx(q, s["max"])) if "max" in s else ""]
        w.writerow(row)
    return cbuf.getvalue(), pbuf.getvalue(), notes


def pims_to_csv(source, out_dir, values_path=None, index=None):
    """Read ``source``, convert, and write ``components.csv`` and
    ``products.csv`` into ``out_dir``. Returns the notes."""
    values_text = None
    if values_path:
        with open(values_path, encoding="utf-8-sig") as fh:
            values_text = fh.read()
    comps, prods, notes = convert_pims(read_pims(source), values_text, index)
    os.makedirs(out_dir, exist_ok=True)
    for name, text in (("components.csv", comps), ("products.csv", prods)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    return notes
