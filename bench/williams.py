"""Williams' textbook models, every engine, against the book's optima.

    python -m bench.williams

Six models from *Model Building in Mathematical Programming* whose subject
is the problem statement's -- blending with storage, factory planning,
distribution, unit commitment, mining -- built in
:mod:`sovopt.models.williams` with every coefficient as printed, plus the
refinery from :mod:`sovopt.models.refinery`. The LPs are solved by the
simplex, the interior point and PDLP, the MILPs by the tree; every point
goes through the verifier; the reference column is the book's value.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from bench.verify import verify
from sovopt.core.problem import Status
from sovopt.models.refinery import WILLIAMS_OPTIMUM, williams_refinery
from sovopt.models.williams import MODELS, PUBLISHED


def run(time_limit=120.0, device="auto"):
    from sovopt.cli import solve
    models = dict(MODELS)
    models["refinery"] = williams_refinery
    published = dict(PUBLISHED)
    published["refinery"] = WILLIAMS_OPTIMUM
    print(f"Williams' models  time-limit={time_limit}s")
    print()
    print(f"{'model':<22} {'rows':>5} {'cols':>5} {'int':>4} {'engine':<9} {'status':<9} "
          f"{'objective':>16} {'published':>14} {'relerr':>9} {'time':>8} {'chk':>4}")
    print("-" * 118)
    rows = []
    for name, build in models.items():
        p = build()
        engines = ["simplex", "ipm", "pdlp"] if p.n_integer == 0 else ["bnb"]
        for eng in engines:
            t = time.perf_counter()
            s = solve(p, method=eng, device=device, time_limit=time_limit, gap=1e-6)
            dt = time.perf_counter() - t
            pub = published[name]
            obj = s.objective if s.x is not None else float("nan")
            rel = abs(obj - pub) / max(1.0, abs(pub)) if np.isfinite(obj) else float("nan")
            chk = "-"
            if s.x is not None:
                v = verify(p, s.x, s.objective, feas_tol=1e-6, int_tol=1e-6,
                           y=(s.y if p.n_integer == 0 else None))
                feas = [c for c in v.checks if c[0] != "optimality"]
                opt = [c for c in v.checks if c[0] == "optimality"]
                chk = "ok" if all(c[1] for c in feas) else "BAD"
                if chk == "ok" and opt and opt[0][1]:
                    chk = "opt"
            print(f"{name:<22} {p.m:>5} {p.n:>5} {p.n_integer:>4} {eng:<9} {s.status.name:<9} "
                  f"{obj:>16.6f} {pub:>14.3f} {rel:>9.2e} {dt:>7.2f}s {chk:>4}")
            sys.stdout.flush()
            rows.append(dict(name=name, engine=eng, status=s.status, obj=obj, ref=pub,
                             relerr=rel, time=dt, check=chk))
    print("-" * 118)
    # to the penny: the book prints pounds to two decimals and recomputes
    # its values from plans it prints rounded; mining is in millions to
    # three decimals
    on = [r for r in rows if np.isfinite(r["obj"])
          and abs(r["obj"] - r["ref"]) <= (0.0005 if r["name"] == "mining" else 0.01)]
    print(f"  solves               {len(rows)}")
    print(f"  on the book's value  {len(on)}/{len(rows)}   (to the penny; mining to five hundred pounds)")
    print(f"  verifier accepted    {sum(1 for r in rows if r['check'] in ('ok', 'opt'))}/{len(rows)}")
    print(f"  certified optimal    {sum(1 for r in rows if r['check'] == 'opt')} of the LP solves")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    run(a.time_limit, a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
