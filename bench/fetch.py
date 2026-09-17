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

    python -m bench.fetch --set small
    python -m bench.fetch --set mittelmann-lp
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

SETS = {"small": SMALL, "medium": MEDIUM, "all": SMALL + MEDIUM,
        "mittelmann-lp": sorted(MITTELMANN_LP), "fctp": FCTP,
        "mittelmann-milp": MITTELMANN_MILP}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data", "instances")
SET_DIRS = {"mittelmann-lp": os.path.join(_ROOT, "data", "mittelmann"),
            "fctp": os.path.join(_ROOT, "data", "fctp"),
            "mittelmann-milp": os.path.join(_ROOT, "data", "mittelmann-milp")}


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


def fetch_one(name, dest_dir=DATA_DIR, timeout=60, source="miplib"):
    """Save ``name`` as ``dest_dir/name.mps``, plain text, with any reference
    value the source publishes written into the header comments."""
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, f"{name}.mps")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out, "cached"
    header = ""
    if source == "miplib":
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
        path = MITTELMANN_LP.get(name, f"fctp/{name}.mps.bz2")
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
        fh.write(header + text)
    return out, f"{how}, {len(blob):,}B"


_REF = re.compile(r"^\*\s*(BEST SOLN|LP SOLN|ROWS|COLUMNS|INTEGER|NONZERO)\s*:\s*(\S+)",
                  re.IGNORECASE)


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

    names = a.only if a.only else SETS[a.set]
    dest = a.dir if a.dir != DATA_DIR else SET_DIRS.get(a.set, DATA_DIR)
    source = "plato" if a.set in ("mittelmann-lp", "fctp") else "miplib"
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
            print(f"  {nm:<18s} {how:<34s} {'  '.join(bits)}")
            ok += 1
        except Exception as e:
            print(f"  {nm:<18s} FAILED  {type(e).__name__}: {str(e)[:70]}")
            fail += 1
    print(f"\n  {ok} fetched, {fail} failed -> {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
