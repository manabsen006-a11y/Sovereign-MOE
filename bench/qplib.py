"""Fetch, parse and validate against QPLIB, the quadratic programming library.

QPLIB (Furini et al. 2019) is to QP what Netlib is to LP: the reference set,
with a classification of every instance and a published solution point whose
objective value is known to fifteen digits. That makes it self-checking twice
over. The *reader* is checked by evaluating our model at the published point
-- feasibility and objective must both match, which is how the factor-of-two
convention on cross terms was caught. The *engines* are checked against the
published value: a convex instance must reach it, a non-convex one must never
report a certified bound above it, and an incumbent better than it would be a
new best-known point (or a bug in the reader, which the first check rules
out).

Only instances with linear (or box, or no) constraints are usable here -- the
engines have no quadratic constraints -- and that is 169 of the 453. They
route by class: convex and continuous to the proximal QP, convex with
integers to the MIQP tree, and everything else to the non-convex route. The
default set is the part of that a laptop can run in an evening; ``--all``
takes everything, ``--only`` a list.

    python -m bench.qplib --fetch                 # download the default set
    python -m bench.qplib --fetch --only 8845 0018
    python -m bench.qplib --run --time-limit 120

Instances are cached in data/qplib and never committed; QPLIB is CC-BY 4.0.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import ssl
import sys
import time
import urllib.request

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sovopt.core.problem import Status
from sovopt.io.qplib import read_qplib, read_qplib_solution

BASE = "https://qplib.zib.de"
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "qplib")

# The default set: every linear-constrained class the engines have, sized so
# that a run finishes. Sizes are (variables, constraints) from the listing.
DEFAULT_SET = [
    # convex, continuous -- the proximal QP, up to Mittelmann's sizes
    "8845", "9002", "8938", "8906", "8559", "8567", "8991", "8792", "8515",
    "8790",
    # convex, with binaries -- the MIQP tree
    "10050", "10056", "10069", "3980", "3913", "3871", "4270",
    # non-convex, continuous -- the McCormick route
    "0018", "0343", "2712", "2761",
    # non-convex, binary or mixed
    "3834", "0031", "0633", "10072", "10073", "10074", "0067", "0032",
    "2512", "3714", "5881", "10040", "10041", "10042",
]


def _context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sovopt-bench"})
    with urllib.request.urlopen(req, timeout=timeout, context=_context()) as r:
        return r.read()


# --------------------------------------------------------------------------- #
# the listing: classification of every instance                               #
# --------------------------------------------------------------------------- #


def parse_listing(page: str) -> dict:
    """``name -> record`` from instances.html; the table is the reference."""
    out = {}
    for row in re.findall(r"<TR bgcolor=[^>]*>(.*?)</TR>", page, re.S):
        tds = re.findall(r"<TD[^>]*>(.*?)</TD>", row, re.S)
        if len(tds) != 13:
            continue
        m = re.search(r"QPLIB_(\d+)\.html", tds[0])
        if not m:
            continue
        clean = [html.unescape(re.sub(r"<[^>]+>", "", x)).strip() for x in tds[1:]]

        def num(s):
            return int(s) if s.isdigit() else 0

        out[m.group(1)] = {
            "convex": clean[0] != "-",
            "objective": clean[1], "variables": clean[4], "constraints": clean[8],
            "n": num(clean[5]), "n_binary": num(clean[6]), "n_integer": num(clean[7]),
            "m": num(clean[9]), "m_quadratic": num(clean[10]), "nnz": num(clean[11]),
        }
    return out


def parse_instance_page(page: str) -> dict:
    """The published objective value at the solution point, and its sense."""
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))
    out = {}
    m = re.search(r"solobjvalue (-?[\d.]+(?:[eE][+-]?\d+)?)", text)
    if m:
        out["objective"] = float(m.group(1))
    m = re.search(r"solinfeasibility (-?[\d.]+(?:[eE][+-]?\d+)?)", text)
    if m:
        out["infeasibility"] = float(m.group(1))
    m = re.search(r"objsense (min|max)", text)
    if m:
        out["sense"] = m.group(1)
    m = re.search(r"objcurvature (\w+)", text)
    if m:
        out["curvature"] = m.group(1)
    return out


def usable(rec: dict) -> bool:
    return rec["objective"] != "L" and rec["constraints"] in ("L", "B", "N")


def fetch(dest_dir: str = DATA_DIR, only=None, everything=False) -> int:
    os.makedirs(dest_dir, exist_ok=True)
    index_path = os.path.join(dest_dir, "index.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path))
    else:
        print("fetching the instance listing ...")
        index = parse_listing(_get(f"{BASE}/instances.html").decode("utf-8", "replace"))
        json.dump(index, open(index_path, "w"), indent=1)
    names = list(only) if only else \
        ([n for n, r in index.items() if usable(r)] if everything else DEFAULT_SET)
    got = 0
    for name in names:
        rec = index.get(name)
        if rec is None:
            print(f"  {name}: not in the listing")
            continue
        if not usable(rec):
            print(f"  {name}: {rec['objective']}{rec['variables']}{rec['constraints']} "
                  f"has quadratic constraints; skipped")
            continue
        qfile = os.path.join(dest_dir, f"QPLIB_{name}.qplib")
        if not os.path.exists(qfile):
            print(f"  {name}: fetching ({rec['n']} vars, {rec['m']} rows) ...")
            try:
                open(qfile, "wb").write(_get(f"{BASE}/qplib/QPLIB_{name}.qplib"))
            except Exception as e:                        # noqa: BLE001
                print(f"    failed: {e}")
                continue
        sfile = os.path.join(dest_dir, f"QPLIB_{name}.sol")
        if not os.path.exists(sfile):
            try:
                open(sfile, "wb").write(_get(f"{BASE}/sol/QPLIB_{name}.sol"))
            except Exception as e:                        # noqa: BLE001
                print(f"    no solution file: {e}")
        if "published" not in rec:
            try:
                rec["published"] = parse_instance_page(
                    _get(f"{BASE}/QPLIB_{name}.html").decode("utf-8", "replace"))
                json.dump(index, open(index_path, "w"), indent=1)
            except Exception as e:                        # noqa: BLE001
                print(f"    no instance page: {e}")
        got += 1
    print(f"{got} instances in {dest_dir}")
    return got


# --------------------------------------------------------------------------- #
# run                                                                          #
# --------------------------------------------------------------------------- #


def _route(prob, rec, time_limit, gap):
    from sovopt.qp import NotConvexError, QPParams, solve_qp
    qp = QPParams(time_limit=time_limit, eps_abs=1e-8, eps_rel=1e-8)
    if rec["convex"]:
        if prob.is_mip:
            from sovopt.mip.miqp import MIQPParams, solve_miqp
            return "miqp", solve_miqp(prob, MIQPParams(time_limit=time_limit,
                                                        gap_rel=gap, qp=qp))
        try:
            return "qp", solve_qp(prob, qp)
        except NotConvexError:
            pass                # the listing said convex; the estimate disagrees
    from sovopt.globalopt.nonconvex_qp import NonconvexQPParams, solve_nonconvex_qp
    return "nonconvex", solve_nonconvex_qp(prob, NonconvexQPParams(
        time_limit=time_limit, gap_rel=gap))


def run(dest_dir: str, only, time_limit: float, gap: float, rel_tol: float,
        max_vars: int | None):
    index = json.load(open(os.path.join(dest_dir, "index.json")))
    names = list(only) if only else sorted(
        n[6:-6] for n in os.listdir(dest_dir) if n.endswith(".qplib"))
    names = [n for n in names if n in index]
    if max_vars:
        names = [n for n in names if index[n]["n"] <= max_vars]
    names.sort(key=lambda n: (not index[n]["convex"], index[n]["n"]))

    print(f"QPLIB  time-limit={time_limit:g}s  gap={gap:g}  rel-tol={rel_tol:g}\n")
    print(f"{'instance':<9}{'type':<5}{'n':>7}{'m':>7}{'route':<11}{'status':<12}"
          f"{'objective':>16}{'published':>16}{'relerr':>10}{'bound':>16}"
          f"{'nodes':>7}{'time':>9}  sol  chk")
    counts = {"parsed": 0, "sol_ok": 0, "matched": 0, "bound_ok": 0,
              "bound_checked": 0, "better": 0, "solved": 0, "total": 0}
    rows = []
    for name in names:
        rec = index[name]
        counts["total"] += 1
        path = os.path.join(dest_dir, f"QPLIB_{name}.qplib")
        try:
            prob = read_qplib(path, name=f"QPLIB_{name}")
        except Exception as e:                             # noqa: BLE001
            print(f"{name:<9} parse failed: {e}")
            continue
        counts["parsed"] += 1
        ptype = prob.meta.get("qplib_type", "")

        # the reader's certificate: the published point must evaluate to the
        # published value and be feasible for the model we built
        pub = rec.get("published", {})
        published = pub.get("objective", float("nan"))
        sol_flag = "-"
        sfile = os.path.join(dest_dir, f"QPLIB_{name}.sol")
        if os.path.exists(sfile) and np.isfinite(published):
            try:
                sobj, sx, _ = read_qplib_solution(sfile, prob.n)
                rv, cv, iv = prob.violation(sx)
                ours = prob.objective(sx)
                ok = (abs(ours - published) <= 1e-6 * max(1.0, abs(published))
                      and max(rv, cv) <= 1e-5 and iv <= 1e-5)
                sol_flag = "ok" if ok else f"BAD({ours:.6g})"
                counts["sol_ok"] += ok
            except Exception as e:                         # noqa: BLE001
                sol_flag = f"err"

        t0 = time.perf_counter()
        try:
            route, s = _route(prob, rec, time_limit, gap)
        except Exception as e:                             # noqa: BLE001
            print(f"{name:<9}{ptype:<5}{prob.n:>7}{prob.m:>7}{'-':<11}"
                  f"{'ERROR':<12}  {type(e).__name__}: {str(e)[:60]}")
            continue
        dt = time.perf_counter() - t0

        obj = s.objective if s.x is not None else float("nan")
        relerr = abs(obj - published) / max(1.0, abs(published)) \
            if np.isfinite(obj) and np.isfinite(published) else float("nan")
        chk = ""
        if s.x is not None:
            rv, cv, iv = prob.violation(s.x)
            chk = "ok" if max(rv, cv) <= 1e-6 and iv <= 1e-6 else "VIOL"
        # the engine's certificate: a bound must never sit on the wrong side
        # of a published feasible value
        bound = s.dual_bound
        bflag = ""
        if np.isfinite(bound) and np.isfinite(published):
            counts["bound_checked"] += 1
            minimise = pub.get("sense", "min") == "min"
            slack = 1e-6 * max(1.0, abs(published))
            good = bound <= published + slack if minimise else bound >= published - slack
            counts["bound_ok"] += good
            bflag = "" if good else "  BOUND>PUBLISHED"
        if np.isfinite(relerr) and relerr <= rel_tol:
            counts["matched"] += 1
        if np.isfinite(obj) and np.isfinite(published) and chk == "ok":
            minimise = pub.get("sense", "min") == "min"
            better = (obj < published - 1e-6 * max(1.0, abs(published))) if minimise \
                else (obj > published + 1e-6 * max(1.0, abs(published)))
            counts["better"] += better
        counts["solved"] += s.status == Status.OPTIMAL
        rows.append((name, ptype, prob.n, prob.m, route, s.status.name, obj,
                     published, relerr, bound, s.nodes, dt, sol_flag, chk))
        print(f"{name:<9}{ptype:<5}{prob.n:>7}{prob.m:>7}{route:<11}{s.status.name:<12}"
              f"{obj:>16.8g}{published:>16.8g}{relerr:>10.2e}{bound:>16.8g}"
              f"{s.nodes:>7}{dt:>8.1f}s  {sol_flag:<4} {chk}{bflag}")

    print()
    print(f"  parsed                    {counts['parsed']}/{counts['total']}")
    print(f"  published point verified  {counts['sol_ok']}/{counts['parsed']}")
    print(f"  status OPTIMAL            {counts['solved']}/{counts['parsed']}")
    print(f"  within {rel_tol:g} of published  {counts['matched']}/{counts['parsed']}")
    print(f"  bound never above published  {counts['bound_ok']}/{counts['bound_checked']}")
    print(f"  incumbent better than published  {counts['better']}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--all", action="store_true", help="with --fetch: every usable instance")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--rel-tol", type=float, default=1e-5)
    ap.add_argument("--max-vars", type=int, default=None)
    args = ap.parse_args(argv)
    if args.fetch:
        fetch(args.dir, args.only, args.all)
    if args.run or not args.fetch:
        run(args.dir, args.only, args.time_limit, args.gap, args.rel_tol,
            args.max_vars)


if __name__ == "__main__":
    main()
