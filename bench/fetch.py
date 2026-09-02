"""Download public benchmark instances.

MIPLIB is the primary source because its MPS headers carry reference values:

    *BEST SOLN:    1201500 (opt)
    *LP SOLN:      1167185.73

which give an independently published target for **both** solvers -- the LP
relaxation value validates the first-order path, and the best known solution
validates the branch-and-bound path. No hand-maintained answer key to drift out
of date.

    python -m bench.fetch --set small
    python -m bench.fetch --list
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import ssl
import sys
import urllib.request

MIPLIB_URL = "https://miplib.zib.de/WebData/instances/{}.mps.gz"

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

SETS = {"small": SMALL, "medium": MEDIUM, "all": SMALL + MEDIUM}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "instances")


def _context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_one(name, dest_dir=DATA_DIR, timeout=60):
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, f"{name}.mps")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out, "cached"
    url = MIPLIB_URL.format(name)
    req = urllib.request.Request(url, headers={"User-Agent": "sovopt-bench/0.1"})
    with urllib.request.urlopen(req, timeout=timeout, context=_context()) as r:
        blob = r.read()
    text = gzip.decompress(blob).decode("utf-8", errors="replace")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out, f"{len(blob):,}B gz"


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
    ok = fail = 0
    for nm in names:
        try:
            path, how = fetch_one(nm, a.dir)
            ref = read_reference(path)
            bits = []
            if "lp_soln" in ref:
                bits.append(f"LP {ref['lp_soln']:.6g}")
            if "best_soln" in ref:
                bits.append(f"MILP {ref['best_soln']:.6g}")
            print(f"  {nm:<12s} {how:<12s} {'  '.join(bits)}")
            ok += 1
        except Exception as e:
            print(f"  {nm:<12s} FAILED  {type(e).__name__}: {str(e)[:70]}")
            fail += 1
    print(f"\n  {ok} fetched, {fail} failed -> {a.dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
