"""How far the non-convex QP engine goes, and which relaxation to use.

Random indefinite QPs over a box with a few linear rows, at increasing size,
solved to a 1e-4 gap by :mod:`sovopt.globalopt.nonconvex_qp`. ``--relaxation``
picks McCormick envelopes (the default, LP nodes) or αBB (diagonal shift,
convex QP nodes); this table is what decided between them. For αBB two
ablations exist: ``--no-reduction`` turns off the marginals-based box
reduction and ``--no-polish`` the projected-gradient incumbent search.

A random dense indefinite ``Q`` is the *hard* case for both: every variable
couples to every other. Refinery quadratics are sparser and better behaved,
and ``--density`` thins ``Q`` to see that. The table is a floor on the engine,
not a ceiling.

    python -m bench.nonconvex_qp --sizes 5 10 15 20 --time-limit 120
    python -m bench.nonconvex_qp --relaxation alphabb --sizes 5 10 15 20
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

from sovopt.core.problem import Problem, Status
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.globalopt.alphabb import AlphaBBParams, gershgorin_alpha
from sovopt.globalopt.nonconvex_qp import NonconvexQPParams, solve_nonconvex_qp


def instance(n: int, seed: int, m: int | None = None, density: float = 1.0):
    rng = np.random.default_rng(1234 + seed)
    m = max(1, n // 4) if m is None else m
    M = rng.standard_normal((n, n))
    if density < 1.0:
        M *= rng.random((n, n)) < density
    H = 0.5 * (M + M.T)
    H -= 0.6 * np.linalg.eigvalsh(H).max() * np.eye(n)
    c = rng.standard_normal(n)
    A = SparseMatrix.from_dense(np.round(rng.uniform(-1, 3, (m, n))))
    b = np.abs(A.matvec(np.full(n, 1.5))) + rng.uniform(1, 5, m)
    return Problem(A=A, c=c, Q=SparseMatrix.from_dense(H),
                   row_lb=np.full(m, -INF), row_ub=b,
                   col_lb=np.zeros(n), col_ub=np.full(n, 3.0),
                   name=f"ncqp-n{n}-s{seed}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 30])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--relaxation", choices=["mccormick", "alphabb"],
                    default="mccormick")
    ap.add_argument("--no-reduction", action="store_true")
    ap.add_argument("--no-polish", action="store_true")
    args = ap.parse_args(argv)

    extra = (f"  reduction={'off' if args.no_reduction else 'on'}  "
             f"polish={'off' if args.no_polish else 'on'}"
             if args.relaxation == "alphabb" else "")
    print(f"non-convex QP  relaxation={args.relaxation}  gap=1e-4  "
          f"time-limit={args.time_limit:g}s  density={args.density:g}{extra}\n")
    print(f"{'instance':<16}{'n':>5}{'m':>5}{'λmin(Q)':>10}{'α max':>9}"
          f"{'status':<12}{'objective':>16}{'bound':>16}{'gap':>10}{'nodes':>8}"
          f"{'reduce':>8}{'time':>9}")
    for n in args.sizes:
        for seed in range(args.seeds):
            p = instance(n, seed, density=args.density)
            lam = float(np.linalg.eigvalsh(p.Q.to_dense()).min())
            a = gershgorin_alpha(p.Q, p.col_lb, p.col_ub)
            t = time.perf_counter()
            s = solve_nonconvex_qp(p, NonconvexQPParams(
                time_limit=args.time_limit, relaxation=args.relaxation,
                alphabb=AlphaBBParams(
                    marginal_reduction=not args.no_reduction,
                    polish_steps=0 if args.no_polish else 50)))
            dt = time.perf_counter() - t
            obj = f"{s.objective:.8g}" if np.isfinite(s.objective) else "-"
            bnd = f"{s.dual_bound:.8g}" if np.isfinite(s.dual_bound) else "-"
            gap = f"{s.gap:.2e}" if np.isfinite(s.gap) else "-"
            red = s.info.get("marginal_reductions", "-") if s.info else "-"
            print(f"{p.name:<16}{n:>5}{p.m:>5}{lam:>10.3f}{a.max():>9.3f}"
                  f"{s.status.name:<12}{obj:>16}{bnd:>16}{gap:>10}{s.nodes:>8}"
                  f"{red:>8}{dt:>8.1f}s")


if __name__ == "__main__":
    main()
