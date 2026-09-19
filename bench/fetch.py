"""Download public benchmark instances.

MIPLIB is the primary source because its MPS headers carry reference values:

    *BEST SOLN:    1201500 (opt)
    *LP SOLN:      1167185.73

which give an independently published target for **both** solvers -- the LP
relaxation value validates the first-order path, and the best known solution
validates the branch-and-bound path. No hand-maintained answer key to drift out
of date.

Four sources, one answer-key rule -- every reference value in a saved file
names where it came from:

* MIPLIB 2017 (miplib.zib.de): the current library. Its files carry no
  header values, so the instance page is read for the published objective
  and status, and written into the header as ``*BEST SOLN: v (opt)`` --
  ``(opt)`` only when the page's status is *easy*, which in MIPLIB 2017
  means solved to proven optimality -- with a ``*SOURCE:`` line naming the
  page.
* MIPLIB 3 (the COIN-OR ``Data-miplib3`` mirror): the classical instances
  that MIPLIB 2017 dropped -- p0033, bell5, egout, stein27 and the rest of
  the ``small`` set. Their headers carry the values.
* Mittelmann's LP test set (plato.asu.edu/ftp/lptestset): the LP benchmark
  instances, bz2-compressed MPS. No published optima in the files; the
  harness reports the objective and the verifier's verdict.
* Mittelmann's fixed-charge transportation set (``fctp/`` on the same site):
  small classical MILPs. Same: no values in the files.

* Netlib's infeasible LP set (netlib.org/lp/infeas, Chinneck 1993): 29
  models published *as infeasible* -- five from a petrochemical plant,
  three from BP operations models, the original greenbea. The header says
  ``*INFEASIBLE: yes`` and names the readme; the answer key is the status.
* Netlib's Kennington set (netlib.org/lp/data/kennington): sixteen larger
  LPs, doubly compressed (gzip, then Netlib's own format); the readme's
  table of optimal values is read by machine into ``*LP SOLN:``.
* Mittelmann's LP test set, top level (the files added from MIPLIB 2017 and
  the PDE-constrained models): the next size class, 0.3-5 MB compressed.
* MIPLIB 2017's *easy* listing (miplib.zib.de/tag_easy.html): one table of
  every proven-optimal instance with its size, group, tags and objective.
  ``miplib-draw2`` is every easy instance under ``MIPLIB_DRAW2_MAX_NNZ``
  nonzeros that no earlier set ran, and ``miplib-infeasible`` the easy
  instances whose published status is Infeasible or Unbounded -- the
  status is the answer key. The objective is written from the listing.
* Maros and Meszaros' convex QP set (doc.ic.ac.uk/~im/QPDATA*.ZIP, 138
  QPS files): no published values in a machine-readable form anywhere the
  fetch could find, so the header names the source and the verifier's
  certificate is the check.
* OR-Library's capacitated and uncapacitated warehouse location sets
  (people.brunel.ac.uk/~mastjjb/jeb/orlib): Beasley's data files, saved as
  they are, with ``capopt.txt`` and ``uncapopt.txt`` beside them;
  ``bench.orlib`` builds the models and reads the optima.

    python -m bench.fetch --set small
    python -m bench.fetch --set mittelmann-lp
    python -m bench.fetch --set netlib-infeasible
    python -m bench.fetch --set miplib-draw2
    python -m bench.fetch --list
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import html
import os
import re
import ssl
import sys
import urllib.request

MIPLIB_URL = "https://miplib.zib.de/WebData/instances/{}.mps.gz"
MIPLIB_PAGE = "https://miplib.zib.de/instance_details_{}.html"
MIPLIB3_URL = "https://raw.githubusercontent.com/coin-or-tools/Data-miplib3/master/{}.gz"
PLATO_URL = "https://plato.asu.edu/ftp/lptestset/{}"
NETLIB_INFEAS_URL = "https://netlib.org/lp/infeas/{}"
NETLIB_INFEAS_README = "https://netlib.org/lp/infeas/readme"
KENNINGTON_URL = "https://netlib.org/lp/data/kennington/{}.gz"
KENNINGTON_README = "https://netlib.org/lp/data/kennington/readme"
MIPLIB_EASY = "https://miplib.zib.de/tag_easy.html"
MARMES_URL = "http://www.doc.ic.ac.uk/~im/QPDATA{}.ZIP"
ORLIB_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{}"

# Small, classical instances -- fast enough for a development loop, and all
# with published optima.
SMALL = [
    "flugpl", "gr4x6", "enigma", "p0033", "p0201", "p0282", "bell3a", "bell5",
    "egout", "khb05250", "lseu", "misc03", "misc07", "mod008", "mod010",
    "pp08a", "rgn", "stein27", "stein45", "vpm1", "vpm2", "gt2", "dcmulti",
    "fixnet6", "l152lav", "mas76", "modglob", "qnet1", "set1ch", "10teams",
]

MEDIUM = [
    "air04", "air05", "cap6000", "fiber", "gesa2", "gesa3", "harp2", "mkc",
    "nw04", "p2756", "pk1", "qiu", "rout", "swath", "vpm2", "danoint",
]

# Mittelmann's LP benchmark, the instances within this solver's reach: the
# rest of the set starts at a few million nonzeros. Paths on plato.
MITTELMANN_LP = {
    "qap15": "qap15.mps.bz2", "nug08-3rd": "nug/nug08-3rd.bz2",
    "nug20": "nug/nug20.bz2", "fome11": "fome/fome11.bz2",
    "rail507": "rail/rail507.bz2", "rail516": "rail/rail516.bz2",
    "rail582": "rail/rail582.bz2", "pds-20": "pds/pds-20.bz2",
    "cont1": "misc/cont1.bz2", "cont4": "misc/cont4.bz2",
    "cont11": "misc/cont11.bz2", "neos1": "misc/neos1.bz2",
    "neos2": "misc/neos2.bz2",
}

# Mittelmann's fixed-charge transportation MILPs (fctp/): m sources x n
# sinks, a binary per arc. gr4x6 is the one MIPLIB also carries.
FCTP = ["bk4x3", "bal8x12", "gr4x6", "ran4x64", "ran6x43", "ran8x32",
        "ran10x10a", "ran10x10b", "ran10x10c", "ran10x12", "ran10x26",
        "ran12x12", "ran12x21", "ran13x13", "ran14x18", "ran16x16", "ran17x17"]

# MIPLIB 2017 instances from Mittelmann's MILP benchmark, the ones under
# about 100k nonzeros. Reference values come from the instance pages.
MITTELMANN_MILP = [
    "gen-ip002", "gen-ip021", "markshare_4_0", "markshare2", "mas74",
    "neos5", "timtab1", "tr12-30", "eil33-2", "pg5_34", "pg", "glass4",
    "n5-3", "binkar10_1", "lotsize", "beasleyC3", "ran14x18-disj-8",
    "sp150x300d", "p200x1188c", "mc11", "mik-250-20-75-4", "enlight_hard",
    "gmu-35-40", "cod105", "rocII-5-11", "mad", "neos17", "neos8",
    "swath1", "supportcase18", "ic97_potential", "csched007",
    "exp-1-500-5-5", "50v-10", "graph20-20-1rand", "istanbul-no-cutoff",
    "qap10", "seymour", "roll3000", "dano3_3", "fast0507", "mcsched",
    "nu25-pr12", "drayage-100-23", "neos-3004026-krka", "assign1-5-8",
    "cbs-cta", "neos-1122047",
]

# Netlib's infeasible LP set: every name in the readme's PROBLEM SUMMARY
# TABLE, all published as infeasible (Chinneck 1993).
NETLIB_INFEAS = [
    "bgdbg1", "bgetam", "bgindy", "bgprtr", "box1", "ceria3d", "chemcom",
    "cplex1", "cplex2", "ex72a", "ex73a", "forest6", "galenet", "gosh",
    "gran", "greenbea", "itest2", "itest6", "klein1", "klein2", "klein3",
    "mondou2", "pang", "pilot4i", "qual", "reactor", "refinery", "vol1",
    "woodinfe",
]

# Netlib's Kennington set: sixteen LPs, optimal values in the readme.
KENNINGTON = [
    "cre-a", "cre-b", "cre-c", "cre-d", "ken-07", "ken-11", "ken-13",
    "ken-18", "osa-07", "osa-14", "osa-30", "osa-60", "pds-02", "pds-06",
    "pds-10", "pds-20",
]

# Mittelmann's LP test set, top level: the next size class above the
# thirteen the first campaign ran, everything under 5 MB compressed. The
# two without an mps suffix are Netlib-compressed like the first set.
PLATO2 = {
    "brazil3": "brazil3.mps.bz2",
    "irish-electricity": "irish-electricity.mps.bz2",
    "chromaticindex1024-7": "chromaticindex1024-7.mps.bz2",
    "Linf_520c": "Linf_520c.bz2",
    "supportcase10": "supportcase10.mps.bz2",
    "bdry2": "bdry2.bz2",
    "rmine15": "rmine15.mps.bz2",
    "physiciansched3-3": "physiciansched3-3.mps.bz2",
    "ex10": "ex10.mps.bz2",
    "s250r10": "s250r10.mps.bz2",
    "datt256": "datt256_lp.mps.bz2",
}

MIPLIB_DRAW2_MAX_NNZ = 10000
"""The second MIPLIB draw is every *easy* instance with at most this many
nonzeros that no earlier set ran (114 at 10,000, from a listing of 694)."""

# OR-Library warehouse location: Beasley's file names and the optima files.
ORLIB_CAP = ([f"cap{i}{j}" for i in (4, 6, 7, 8, 9, 10, 11, 12, 13) for j in (1, 2, 3, 4)]
             + ["cap51", "capa", "capb", "capc"])
ORLIB_UNCAP = ([f"cap{i}{j}" for i in (7, 10, 13) for j in (1, 2, 3, 4)]
               + ["capa", "capb", "capc"])
ORLIB_FILES = sorted({f"{n}.txt" for n in ORLIB_CAP} | {"capopt.txt", "uncapopt.txt"})

SETS = {"small": SMALL, "medium": MEDIUM, "all": SMALL + MEDIUM,
        "mittelmann-lp": sorted(MITTELMANN_LP), "fctp": FCTP,
        "mittelmann-milp": MITTELMANN_MILP,
        "netlib-infeasible": NETLIB_INFEAS, "kennington": KENNINGTON,
        "mittelmann-lp2": list(PLATO2),
        "miplib-draw2": None, "miplib-infeasible": None,   # from the listing
        "marmes": None, "orlib": ORLIB_FILES}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data", "instances")
SET_DIRS = {"mittelmann-lp": os.path.join(_ROOT, "data", "mittelmann"),
            "fctp": os.path.join(_ROOT, "data", "fctp"),
            "mittelmann-milp": os.path.join(_ROOT, "data", "mittelmann-milp"),
            "netlib-infeasible": os.path.join(_ROOT, "data", "netlib-infeasible"),
            "kennington": os.path.join(_ROOT, "data", "kennington"),
            "mittelmann-lp2": os.path.join(_ROOT, "data", "mittelmann2"),
            "miplib-draw2": os.path.join(_ROOT, "data", "miplib2"),
            "miplib-infeasible": os.path.join(_ROOT, "data", "miplib-infeasible"),
            "marmes": os.path.join(_ROOT, "data", "marmes"),
            "orlib": os.path.join(_ROOT, "data", "orlib")}
SET_SOURCES = {"mittelmann-lp": "plato", "fctp": "plato", "mittelmann-lp2": "plato",
               "netlib-infeasible": "netlib-infeasible", "kennington": "kennington"}


def _context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _get(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "sovopt-bench/0.1"})
    with urllib.request.urlopen(req, timeout=timeout, context=_context()) as r:
        return r.read()


def _decompress(blob):
    if blob[:2] == b"\x1f\x8b":
        return gzip.decompress(blob)
    if blob[:3] == b"BZh":
        return bz2.decompress(blob)
    raise ValueError("not a gzip or bz2 file: " + repr(blob[:20]))


_PAGE_ROW = re.compile(
    r'<td><a href="tag_(easy|hard|open)\.html".*?</a></td>\s*<td>[^<]*</td>\s*<td>([-+0-9.eE]+)</td>',
    re.S)


def miplib_reference(name, timeout=60):
    """(objective, status) from a MIPLIB 2017 instance page, or None.

    The page's summary row reads Status | Group | Objective; the status
    label is ``easy`` (solved to proven optimality), ``hard`` (the value is
    the best known) or ``open``."""
    try:
        page = _get(MIPLIB_PAGE.format(name), timeout).decode("utf-8", "replace")
    except Exception:                                    # noqa: BLE001
        return None
    m = _PAGE_ROW.search(page)
    if not m:
        return None
    try:
        return float(html.unescape(m.group(2))), m.group(1)
    except ValueError:
        return None


_KENNINGTON_TABLE = None


def kennington_reference(timeout=60):
    """``{name: optimal value}`` from the Kennington readme's table, read
    once. The names there are upper case; the files are lower case."""
    global _KENNINGTON_TABLE
    if _KENNINGTON_TABLE is None:
        text = _get(KENNINGTON_README, timeout).decode("latin-1")
        table = {}
        for m in re.finditer(r"^([A-Z]+-[0-9A-Z]+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(-?[0-9.]+e[+-]\d+)",
                             text, re.M):
            table[m.group(1).lower()] = float(m.group(2))
        if len(table) != len(KENNINGTON):
            raise ValueError(f"the Kennington readme's table gave {len(table)} "
                             f"values, expected {len(KENNINGTON)}")
        _KENNINGTON_TABLE = table
    return _KENNINGTON_TABLE


def fetch_one(name, dest_dir=DATA_DIR, timeout=60, source="miplib", extra=""):
    """Save ``name`` as ``dest_dir/name.mps``, plain text, with any reference
    value the source publishes written into the header comments; ``extra``
    is appended to the header lines on a first fetch."""
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, f"{name}.mps")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out, "cached"
    header = ""
    if source == "netlib-infeasible":
        from sovopt.io.netlib import expand
        blob = _get(NETLIB_INFEAS_URL.format(name), timeout)
        text = expand(blob.decode("latin-1"))
        header = (f"*NAME:         {name}\n"
                  f"*INFEASIBLE:   yes\n"
                  f"*SOURCE:       netlib.org/lp/infeas/readme, PROBLEM SUMMARY TABLE "
                  f"(Chinneck 1993): published as infeasible\n")
        how = "netlib/lp/infeas"
    elif source == "kennington":
        from sovopt.io.netlib import expand
        blob = _get(KENNINGTON_URL.format(name), timeout)
        text = expand(_decompress(blob).decode("latin-1"))
        obj = kennington_reference(timeout)[name]
        header = (f"*NAME:         {name}\n"
                  f"*LP SOLN:      {obj!r}\n"
                  f"*SOURCE:       netlib.org/lp/data/kennington/readme (optimal values "
                  f"computed by Vanderbei's ALPO)\n")
        how = "netlib/kennington"
    elif source == "miplib":
        try:
            blob = _get(MIPLIB_URL.format(name), timeout)
            text = _decompress(blob).decode("utf-8", errors="replace")
            how = "miplib2017"
            ref = miplib_reference(name, timeout)
            if ref is not None and not any(
                    l.startswith("*") and "BEST SOLN" in l.upper()
                    for l in text.splitlines()[:40]):
                obj, status = ref
                header = (f"*NAME:         {name}\n"
                          f"*BEST SOLN:    {obj!r}{' (opt)' if status == 'easy' else ''}\n"
                          f"*SOURCE:       miplib.zib.de instance page, status {status}\n")
                how += f" + page ({status})"
        except Exception:                                # noqa: BLE001
            # MIPLIB 2017 dropped the instance; the MIPLIB 3 mirror has it
            blob = _get(MIPLIB3_URL.format(name), timeout)
            text = _decompress(blob).decode("utf-8", errors="replace")
            how = "miplib3 mirror"
    elif source == "plato":
        path = MITTELMANN_LP.get(name) or PLATO2.get(name) or f"fctp/{name}.mps.bz2"
        blob = _get(PLATO_URL.format(path), timeout)
        text = _decompress(blob).decode("utf-8", errors="replace")
        if not path.endswith(".mps.bz2"):
            # "The files w/o mps subscript are compressed with the MPC
            # utility and need to be uncompressed with EMPS" (plato's
            # 00README): Netlib's compressed MPS, which the repository's
            # own expander reads -- the harness saw twelve parse failures
            # before this line
            from sovopt.io.netlib import expand
            text = expand(text)
        header = f"*NAME:         {name}\n*SOURCE:       plato.asu.edu/ftp/lptestset/{path}\n"
        how = "plato"
        # a few of the fctp instances are in MIPLIB 2017 under the same
        # name (gr4x6, ran12x21, ran13x13), whose pages give the optimum
        ref = miplib_reference(name, timeout) if name not in MITTELMANN_LP else None
        if ref is not None:
            obj, status = ref
            header += (f"*BEST SOLN:    {obj!r}{' (opt)' if status == 'easy' else ''}\n"
                       f"*SOURCE:       miplib.zib.de instance page for the same "
                       f"instance, status {status}\n")
            how += f" + miplib page ({status})"
    else:
        raise ValueError(f"unknown source {source!r}")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(header + extra + text)
    return out, f"{how}, {len(blob):,}B"


# --------------------------------------------------------------------------- #
# MIPLIB 2017's easy listing                                                  #
# --------------------------------------------------------------------------- #

_LISTING = None


def miplib_easy_listing(timeout=120):
    """Every row of miplib.zib.de/tag_easy.html as a dict: name, vars,
    binaries, integers, continuous, rows, nnz, submitter, group, tags, and
    ``objective`` (a float) or ``status`` ("Infeasible" / "Unbounded")
    where the listing shows a word instead of a value. Read once."""
    global _LISTING
    if _LISTING is not None:
        return _LISTING
    page = _get(MIPLIB_EASY, timeout).decode("utf-8", "replace")
    recs = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", " ", c)).split()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        if len(cells) != 12 or cells[1] != ["easy"]:
            continue
        rec = {"name": cells[0][0], "submitter": " ".join(cells[8]),
               "group": " ".join(cells[9]), "tags": cells[11]}
        try:
            (rec["vars"], rec["binaries"], rec["integers"], rec["continuous"],
             rec["rows"], rec["nnz"]) = (int(float(c[0])) for c in cells[2:8])
        except (ValueError, IndexError):
            continue
        try:
            rec["objective"] = float(cells[10][0])
        except ValueError:
            rec["status"] = cells[10][0]
        recs.append(rec)
    if len(recs) < 500:
        raise ValueError(f"the easy listing parsed to {len(recs)} rows")
    _LISTING = recs
    return recs


def miplib_draw2(max_nnz=MIPLIB_DRAW2_MAX_NNZ):
    """The easy instances with a value, at most ``max_nnz`` nonzeros, that
    no earlier set ran; smallest first."""
    done = set(SMALL) | set(MEDIUM) | set(MITTELMANN_MILP) | set(FCTP)
    recs = [r for r in miplib_easy_listing()
            if "objective" in r and r["nnz"] <= max_nnz and r["name"] not in done]
    return sorted(recs, key=lambda r: (r["nnz"], r["name"]))


def miplib_infeasible(max_nnz=100000):
    """The easy instances whose published status is Infeasible or
    Unbounded, at most ``max_nnz`` nonzeros; smallest first."""
    recs = [r for r in miplib_easy_listing() if "status" in r and r["nnz"] <= max_nnz]
    return sorted(recs, key=lambda r: (r["nnz"], r["name"]))


def _listing_header(rec):
    """Header lines for an instance saved from the listing: the value with
    ``(opt)`` (every row of the easy listing is proven), or the status."""
    lines = [f"*NAME:         {rec['name']}\n"]
    if "objective" in rec:
        lines.append(f"*BEST SOLN:    {rec['objective']!r} (opt)\n")
    else:
        lines.append(f"*STATUS:       {rec['status']}\n")
    lines.append("*SOURCE:       miplib.zib.de/tag_easy.html listing (status easy: "
                 "solved to proven optimality)\n")
    lines.append(f"*GROUP:        {rec['group']}\n")
    lines.append(f"*TAGS:         {' '.join(rec['tags'])}\n")
    return "".join(lines)


def fetch_listed(rec, dest_dir, timeout=300):
    """Fetch a listed MIPLIB 2017 instance, header from the listing."""
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, f"{rec['name']}.mps")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out, "cached"
    blob = _get(MIPLIB_URL.format(rec["name"]), timeout)
    text = _decompress(blob).decode("utf-8", errors="replace")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(_listing_header(rec) + text)
    return out, f"miplib2017 + listing, {len(blob):,}B"


# --------------------------------------------------------------------------- #
# Maros and Meszaros, and OR-Library                                          #
# --------------------------------------------------------------------------- #

def fetch_marmes(dest_dir, timeout=600):
    """The 138 QPS files from the three archives, saved as ``<name>.qps``
    with a header naming the archive. Skipped when the directory already
    holds 138 of them."""
    import io as _io
    import zipfile
    os.makedirs(dest_dir, exist_ok=True)
    have = [f for f in os.listdir(dest_dir) if f.endswith(".qps")]
    if len(have) >= 138:
        return len(have), "cached"
    got = 0
    for k in (1, 2, 3):
        blob = _get(MARMES_URL.format(k), timeout)
        zf = zipfile.ZipFile(_io.BytesIO(blob))
        for member in zf.namelist():
            if not member.upper().endswith(".QPS"):
                continue
            name = os.path.splitext(os.path.basename(member))[0].lower()
            out = os.path.join(dest_dir, f"{name}.qps")
            if os.path.exists(out) and os.path.getsize(out) > 0:
                continue
            text = zf.read(member).decode("latin-1")
            header = (f"*NAME:         {name}\n"
                      f"*SOURCE:       doc.ic.ac.uk/~im/QPDATA{k}.ZIP (Maros and Meszaros, "
                      f"A repository of convex quadratic programming problems, OMS 1999); "
                      f"no published value in machine-readable form\n")
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(header + text)
            got += 1
    return got, "doc.ic.ac.uk"


def fetch_orlib(dest_dir, timeout=300):
    """Beasley's warehouse-location data files and the two optima files,
    saved as they are."""
    os.makedirs(dest_dir, exist_ok=True)
    got = 0
    for f in ORLIB_FILES:
        out = os.path.join(dest_dir, f)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        with open(out, "wb") as fh:
            fh.write(_get(ORLIB_URL.format(f), timeout))
        got += 1
    return got, "people.brunel.ac.uk/~mastjjb/jeb/orlib"


_REF = re.compile(r"^\*\s*(BEST SOLN|LP SOLN|ROWS|COLUMNS|INTEGER|NONZERO)\s*:\s*(\S+)",
                  re.IGNORECASE)
_STATUS = re.compile(r"^\*\s*(INFEASIBLE|STATUS|GROUP)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def read_reference(path):
    """Pull the published reference values out of the MPS header comments."""
    ref = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("*"):
                if line.strip().upper().startswith("NAME"):
                    continue
                if not line.startswith("*"):
                    break
            m = _REF.match(line)
            if m:
                key = m.group(1).lower().replace(" ", "_")
                try:
                    ref[key] = float(m.group(2))
                except ValueError:
                    pass
            if "(opt)" in line.lower() and "best" in line.lower():
                ref["proved_optimal"] = True
            m = _STATUS.match(line)
            if m:
                key, val = m.group(1).lower(), m.group(2)
                if key == "infeasible" and val.lower().startswith("y"):
                    ref["expected"] = "infeasible"
                elif key == "status":
                    ref["expected"] = val.lower()          # infeasible / unbounded
                elif key == "group":
                    ref["group"] = val
    return ref


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="small", choices=sorted(SETS))
    ap.add_argument("--only", nargs="*", help="fetch just these instances")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dir", default=DATA_DIR)
    a = ap.parse_args(argv)

    if a.list:
        for k, v in SETS.items():
            print(f"{k:8s} {len(v):3d} instances")
        return 0

    dest = a.dir if a.dir != DATA_DIR else SET_DIRS.get(a.set, DATA_DIR)
    if a.set == "marmes":
        n, how = fetch_marmes(dest)
        print(f"  {n} QPS files ({how}) -> {dest}")
        return 0
    if a.set == "orlib":
        n, how = fetch_orlib(dest)
        print(f"  {n} new files ({how}) -> {dest}")
        return 0
    if a.set in ("miplib-draw2", "miplib-infeasible"):
        recs = miplib_draw2() if a.set == "miplib-draw2" else miplib_infeasible()
        if a.only:
            want = set(a.only)
            recs = [r for r in recs if r["name"] in want]
        ok = fail = 0
        for rec in recs:
            try:
                path, how = fetch_listed(rec, dest)
                what = (f"{rec['objective']:.8g} (opt)" if "objective" in rec
                        else rec["status"])
                print(f"  {rec['name']:<28s} {rec['nnz']:>7d} nnz  {how:<30s} {what}  [{rec['group']}]")
                ok += 1
            except Exception as e:                        # noqa: BLE001
                print(f"  {rec['name']:<28s} FAILED  {type(e).__name__}: {str(e)[:70]}")
                fail += 1
        print(f"\n  {ok} fetched, {fail} failed -> {dest}")
        return 0 if ok else 1

    names = a.only if a.only else SETS[a.set]
    source = SET_SOURCES.get(a.set, "miplib")
    ok = fail = 0
    for nm in names:
        try:
            path, how = fetch_one(nm, dest, source=source, timeout=300)
            ref = read_reference(path)
            bits = []
            if "lp_soln" in ref:
                bits.append(f"LP {ref['lp_soln']:.6g}")
            if "best_soln" in ref:
                bits.append(f"MILP {ref['best_soln']:.6g}")
            if "expected" in ref:
                bits.append(ref["expected"].upper())
            print(f"  {nm:<18s} {how:<34s} {'  '.join(bits)}")
            ok += 1
        except Exception as e:
            print(f"  {nm:<18s} FAILED  {type(e).__name__}: {str(e)[:70]}")
            fail += 1
    print(f"\n  {ok} fetched, {fail} failed -> {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
