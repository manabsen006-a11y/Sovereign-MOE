"""OR-Library warehouse location: supply-chain MILPs with published optima.

Beasley's capacitated and uncapacitated warehouse location sets (OR-Library,
people.brunel.ac.uk/~mastjjb/jeb/orlib) are the classical supply-chain
benchmark: ``m`` potential warehouses with a fixed opening cost and (for the
capacitated set) a capacity, ``n`` customers with a demand and a cost of
serving all of it from each warehouse. Sets IV-XIII are 16 warehouses by 50
customers; A, B and C are 100 by 1,000, and each of those files stands for
four problems, one per capacity value in Beasley's Table 1 -- the optima
file lists all four. Every optimum is read from ``capopt.txt`` and
``uncapopt.txt`` beside the data; nothing here is typed from a paper.

The model is the strong formulation, which is what the literature solves::

    min  Σ_i f_i y_i + Σ_ij c_ij x_ij
         Σ_i x_ij = 1                      every customer served
         Σ_j d_j x_ij <= cap_i y_i         (capacitated) capacity if open
         x_ij <= y_i                       served only by an open warehouse
         y_i binary, 0 <= x_ij <= 1

``c_ij`` is the file's cost of allocating *all* of customer j's demand to
warehouse i, so ``x_ij`` is the fraction served. The ``x_ij <= y_i`` rows
are redundant given the capacity rows and are what make the LP bound
tight; without them a 16-warehouse instance is a hard tree, with them it
is a few nodes.

    python -m bench.fetch --set orlib
    python -m bench.orlib --set cap            # sets IV-XIII, and A-C x 4
    python -m bench.orlib --set uncap

References
----------
Beasley, J.E., "An algorithm for solving large capacitated warehouse
  location problems", European J. Operational Research 33 (1988) 314-325.
Beasley, J.E., "Lagrangean heuristics for location problems", European J.
  Operational Research 65 (1993) 383-399.
Akinc, U. and Khumawala, B.M., "An efficient branch and bound algorithm
  for the capacitated warehouse location problem", Management Science 23
  (1977) 585-594 -- the origin of sets IV-XIII.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from bench.fetch import ORLIB_CAP, ORLIB_UNCAP, SET_DIRS
from bench.verify import verify
from sovopt.core.problem import ObjSense, Problem, Status, VarKind
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF

DATA_DIR = SET_DIRS["orlib"]


def read_optima(path):
    """``{name: [(capacity or None, value), ...]}`` from an OR-Library
    optima file. A, B and C list four values with their capacities."""
    out = {}
    current = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^(cap\w*)?\s+([0-9.]+)(?:\s*\(capacity\s+([0-9.]+)\))?\s*$", line)
        if not m:
            continue
        if m.group(1):
            current = m.group(1)
            out[current] = []
        if current is None:
            continue
        cap = float(m.group(3)) if m.group(3) else None
        out[current].append((cap, float(m.group(2))))
    return out


def read_cap(path, capacity=None):
    """``(cap, fixed, demand, cost)`` with ``cost[j, i]`` the cost of serving
    all of customer j from warehouse i. ``capacity`` replaces the word in
    the A-C files."""
    text = open(path, encoding="utf-8", errors="replace").read()
    # the A-C files carry the word "capacity" in place of a value; it is the
    # capacity asked for, or nothing for the uncapacitated problem
    text = text.replace("capacity", repr(float(capacity)) if capacity is not None else "0.0")
    tok = text.split()
    m, n = int(tok[0]), int(tok[1])
    k = 2
    cap = np.empty(m)
    fixed = np.empty(m)
    for i in range(m):
        cap[i], fixed[i] = float(tok[k]), float(tok[k + 1])
        k += 2
    demand = np.empty(n)
    cost = np.empty((n, m))
    for j in range(n):
        demand[j] = float(tok[k])
        k += 1
        cost[j] = [float(t) for t in tok[k:k + m]]
        k += m
    return cap, fixed, demand, cost


def build(cap, fixed, demand, cost, capacitated=True, name="cfl") -> Problem:
    """The strong formulation. Columns: ``y_0..y_{m-1}`` then ``x_{ij}`` in
    customer-major order."""
    n, m = cost.shape
    ny, nx = m, n * m
    rows_i, cols_i, vals = [], [], []
    rl, ru = [], []
    r = 0
    # every customer served
    for j in range(n):
        for i in range(m):
            rows_i.append(r); cols_i.append(ny + j * m + i); vals.append(1.0)
        rl.append(1.0); ru.append(1.0)
        r += 1
    # capacity if open
    if capacitated:
        for i in range(m):
            for j in range(n):
                rows_i.append(r); cols_i.append(ny + j * m + i); vals.append(float(demand[j]))
            rows_i.append(r); cols_i.append(i); vals.append(-float(cap[i]))
            rl.append(-INF); ru.append(0.0)
            r += 1
    # served only by an open warehouse
    for j in range(n):
        for i in range(m):
            rows_i.append(r); cols_i.append(ny + j * m + i); vals.append(1.0)
            rows_i.append(r); cols_i.append(i); vals.append(-1.0)
            rl.append(-INF); ru.append(0.0)
            r += 1
    A = SparseMatrix.from_triplets(np.asarray(rows_i, dtype=np.int32),
                                   np.asarray(cols_i, dtype=np.int32),
                                   np.asarray(vals, dtype=np.float64), r, ny + nx)
    c = np.concatenate([fixed.astype(np.float64), cost.reshape(-1).astype(np.float64)])
    kind = np.full(ny + nx, VarKind.CONTINUOUS, dtype=np.int8)
    kind[:ny] = VarKind.BINARY
    return Problem(A=A, c=c, row_lb=np.asarray(rl), row_ub=np.asarray(ru),
                   col_lb=np.zeros(ny + nx), col_ub=np.ones(ny + nx), kind=kind,
                   name=name, sense=ObjSense.MINIMISE)


def problems(which, dest_dir=DATA_DIR):
    """``(name, Problem, published optimum)`` for every instance of the
    set -- A, B and C once per capacity the optima file lists."""
    capacitated = which == "cap"
    names = ORLIB_CAP if capacitated else ORLIB_UNCAP
    optima = read_optima(os.path.join(dest_dir, "capopt.txt" if capacitated else "uncapopt.txt"))
    for name in names:
        path = os.path.join(dest_dir, f"{name}.txt")
        if not os.path.exists(path):
            continue
        for capacity, value in optima.get(name, [(None, None)]):
            cap, fixed, demand, cost = read_cap(path, capacity)
            label = name if capacity is None else f"{name}@{capacity:g}"
            yield label, build(cap, fixed, demand, cost, capacitated, name=label), value


def run(which, time_limit, gap, only=None, dest_dir=DATA_DIR, threads=4):
    from sovopt.mip.tree import MIPParams, solve_mip
    print(f"OR-Library warehouse location  set={which}  time-limit={time_limit}s  gap={gap:g}")
    print()
    print(f"{'instance':<14} {'wh':>4} {'cust':>5} {'cols':>7} {'status':<10} "
          f"{'objective':>16} {'published':>16} {'relerr':>9} {'nodes':>7} {'time':>8} {'chk':>4}")
    print("-" * 118)
    rows = []
    for name, prob, value in problems(which, dest_dir):
        if only and name.split("@")[0] not in only:
            continue
        m = int((prob.kind == VarKind.BINARY).sum())
        n = (prob.n - m) // m
        t = time.perf_counter()
        sol = solve_mip(prob, MIPParams(time_limit=time_limit, gap_rel=gap, threads=threads))
        dt = time.perf_counter() - t
        obj = sol.objective if sol.x is not None else float("nan")
        relerr = abs(obj - value) / max(1.0, abs(value)) if (value is not None and np.isfinite(obj)) else float("nan")
        chk = "-"
        if sol.x is not None:
            v = verify(prob, sol.x, sol.objective, feas_tol=1e-6, int_tol=1e-6)
            chk = "ok" if v.ok else "BAD"
        print(f"{name:<14} {m:>4} {n:>5} {prob.n:>7} {sol.status.name:<10} {obj:>16.8g} "
              f"{value if value is not None else float('nan'):>16.8g} {relerr:>9.2e} "
              f"{sol.nodes:>7} {dt:>7.2f}s {chk:>4}")
        sys.stdout.flush()
        rows.append(dict(name=name, status=sol.status, obj=obj, ref=value, relerr=relerr,
                         time=dt, check=chk, nodes=sol.nodes))
    print("-" * 118)
    n = len(rows)
    proved = [r for r in rows if r["status"] == Status.OPTIMAL]
    onval = [r for r in rows if np.isfinite(r["relerr"]) and r["relerr"] <= 1e-6]
    better = [r for r in rows if r["check"] == "ok" and np.isfinite(r["relerr"])
              and r["obj"] < r["ref"] - 1e-6 * max(1.0, abs(r["ref"]))]
    accepted = [r for r in rows if r["check"] == "ok"]
    bad = [r for r in rows if r["check"] == "BAD"]
    print(f"  instances            {n}")
    print(f"  proved optimal       {len(proved)}/{n}")
    print(f"  on the published value (1e-6)  {len(onval)}/{n}")
    print(f"  verifier accepted    {len(accepted)}/{n}")
    print(f"  total time           {sum(r['time'] for r in rows):.1f}s")
    if better:
        print(f"  !! below the published optimum: {', '.join(r['name'] for r in better)}")
    if bad:
        print(f"  !! VERIFIER REJECTED: {', '.join(r['name'] for r in bad)}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["cap", "uncap"], default="cap")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--gap", type=float, default=1e-6)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    if not os.path.exists(os.path.join(a.dir, "capopt.txt")):
        print("no data; run:  python -m bench.fetch --set orlib")
        return 1
    run(a.set, a.time_limit, a.gap, a.only, a.dir, a.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
