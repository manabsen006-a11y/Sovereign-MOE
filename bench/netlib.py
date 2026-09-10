"""Fetch, expand and validate the Netlib LP test set.

Netlib is the oldest and most cited collection of linear programs there is, and
the one every LP solver is expected to have an opinion about. Its readme
carries a PROBLEM SUMMARY TABLE with the optimal value of every problem to ten
significant figures, so the set is self-checking: expand a problem, solve it,
and compare. That makes it a test of three things at once -- the expander in
:mod:`sovopt.io.netlib`, the MPS reader, and the LP engine -- and a wrong
answer from any of them shows up as a mismatch against a number nobody here
chose.

    python -m bench.netlib --fetch            # download once, into data/netlib
    python -m bench.netlib --mode lp          # expand, solve, compare
    python -m bench.netlib --only afiro share

Instances are cached on disk; ``--fetch`` is only needed once.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.request

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sovopt.core.problem import Status
from sovopt.io.netlib import read_netlib

BASE = "https://www.netlib.org/lp/data"
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "netlib")

# Name Rows Cols Nonzeros Bytes [BR] OptimalValue, from lp/data/readme.
_ROW = re.compile(
    r"^([A-Z0-9][A-Z0-9-]*)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"(?:([BR]{1,2})\s+)?(-?[\d.]+E[+-]\d+)")


def summary_table(readme: str) -> dict:
    """Parse the published table: name -> (rows, cols, nonzeros, optimum).

    The table is the reference data, so it is read from the file rather than
    transcribed into this repository -- a transcribed constant is a number
    nobody re-checks.
    """
    out = {}
    for line in readme.splitlines():
        m = _ROW.match(line.strip())
        if not m:
            continue
        name, rows, cols, nz, _bytes, _br, opt = m.groups()
        out[name.lower()] = (int(rows), int(cols), int(nz), float(opt))
    return out


def _get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def fetch(dest_dir: str = DATA_DIR, only=None, timeout: int = 60) -> int:
    """Download the readme and every problem it lists that is not cached."""
    os.makedirs(dest_dir, exist_ok=True)
    readme_path = os.path.join(dest_dir, "readme")
    if not os.path.exists(readme_path):
        with open(readme_path, "wb") as f:
            f.write(_get(f"{BASE}/readme", timeout))
        print(f"  readme -> {readme_path}")
    table = summary_table(open(readme_path, encoding="latin-1").read())

    names = sorted(table)
    if only:
        want = {n.lower() for n in only}
        names = [n for n in names if n in want]

    got = 0
    for name in names:
        path = os.path.join(dest_dir, name)
        if os.path.exists(path):
            continue
        try:
            data = _get(f"{BASE}/{name}", timeout)
        except Exception as e:                       # noqa: BLE001
            print(f"  {name:<10} FAILED  {type(e).__name__}: {str(e)[:50]}")
            continue
        with open(path, "wb") as f:
            f.write(data)
        got += 1
        print(f"  {name:<10} {len(data):>8d} bytes")
    print(f"fetched {got} new instance(s); {len(names)} listed")
    return 0


def run(dest_dir: str, only, method: str, time_limit: float, tol: float,
        rel_tol: float) -> int:
    from bench.verify import verify
    from sovopt.cli import solve

    readme_path = os.path.join(dest_dir, "readme")
    if not os.path.exists(readme_path):
        print("no readme; run with --fetch first")
        return 1
    table = summary_table(open(readme_path, encoding="latin-1").read())

    names = [n for n in sorted(table) if os.path.exists(os.path.join(dest_dir, n))]
    if only:
        want = {n.lower() for n in only}
        names = [n for n in names if n in want]
    if not names:
        print("no cached instances; run with --fetch first")
        return 1

    print(f"{'instance':<10} {'rows':>6} {'cols':>6} {'nnz':>8} {'status':<12} "
          f"{'objective':>18} {'published':>18} {'relerr':>10} {'time':>8} {'chk':>4}")
    print("-" * 116)
    ok = bad = expand_fail = 0
    times, worst = [], (0.0, "")
    for name in names:
        path = os.path.join(dest_dir, name)
        try:
            prob = read_netlib(path, name=name)
        except Exception as e:                       # noqa: BLE001
            print(f"{name:<10} EXPAND FAILED  {type(e).__name__}: {str(e)[:60]}")
            expand_fail += 1
            continue
        _r, _c, _nz, published = table[name]
        t = time.perf_counter()
        try:
            sol = solve(prob, method=method, time_limit=time_limit, tol=tol)
        except Exception as e:                       # noqa: BLE001
            print(f"{name:<10} {prob.m:>6d} {prob.n:>6d} {prob.nnz:>8d} "
                  f"RAISED {type(e).__name__}")
            bad += 1
            continue
        dt = time.perf_counter() - t
        times.append(dt)

        relerr = float("nan")
        if sol.x is not None and np.isfinite(sol.objective):
            relerr = abs(sol.objective - published) / max(1.0, abs(published))
        chk = "-"
        if sol.x is not None:
            chk = "ok" if verify(prob, sol.x, feas_tol=1e-6).ok else "BAD"
        good = (sol.status == Status.OPTIMAL and chk == "ok"
                and relerr <= rel_tol)
        ok += good
        bad += not good
        if np.isfinite(relerr) and relerr > worst[0]:
            worst = (relerr, name)
        flag = "" if good else "   <-"
        print(f"{name:<10} {prob.m:>6d} {prob.n:>6d} {prob.nnz:>8d} "
              f"{sol.status.name:<12} {sol.objective:>18.10g} "
              f"{published:>18.10g} {relerr:>10.2e} {dt:>7.2f}s {chk:>4}{flag}")
        sys.stdout.flush()

    print("-" * 116)
    print(f"  instances            {len(names)}")
    print(f"  matched published    {ok}/{len(names)}")
    if expand_fail:
        print(f"  expansion failed     {expand_fail}")
    if times:
        print(f"  total time           {sum(times):.1f}s")
    if worst[1]:
        print(f"  worst relative error {worst[0]:.2e} ({worst[1]})")
    return 0 if bad == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="download the readme and any missing instances")
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--method", default="simplex",
                    choices=["auto", "simplex", "pdlp", "ipm"])
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--rel-tol", type=float, default=1e-6)
    a = ap.parse_args(argv)

    if a.fetch:
        return fetch(a.dir, a.only)
    return run(a.dir, a.only, a.method, a.time_limit, a.tol, a.rel_tol)


if __name__ == "__main__":
    raise SystemExit(main())
