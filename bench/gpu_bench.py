"""CPU vs GPU benchmark for the first-order LP path and batched node relaxation.

Run:
    python -m bench.gpu_bench            # default sizes
    python -m bench.gpu_bench --big      # include the large instance

Reports honest wall-clock, iterations per second, and objective agreement
between devices. A speedup claim with no agreement check is worthless.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from sovopt.core.backend import get_backend, gpu_selftest
from sovopt.core.problem import Problem
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF
from sovopt.lp.pdlp import PDLPParams, solve_pdlp


def make_lp(m, n, nnz_per_col, seed=0):
    """A random sparse LP with a guaranteed feasible interior point."""
    rng = np.random.default_rng(seed)
    cols = np.repeat(np.arange(n), nnz_per_col)
    rows = rng.integers(0, m, cols.size)
    vals = rng.standard_normal(cols.size)
    A = SparseMatrix.from_triplets(rows, cols, vals, m, n)
    x0 = rng.random(n)
    b = A.matvec(x0) + rng.random(m) * 0.5
    return Problem(A=A, c=rng.standard_normal(n),
                   row_lb=np.full(m, -INF), row_ub=b,
                   col_lb=np.zeros(n), col_ub=np.full(n, 10.0),
                   name=f"rand_{m}x{n}")


def bench_pdlp(sizes, max_iter, eps):
    print("=" * 78)
    print("PDLP  -  first-order LP, CPU vs GPU")
    print("=" * 78)
    ok, err = gpu_selftest()
    devices = ["cpu", "gpu"] if ok else ["cpu"]
    if not ok:
        print(f"  (no usable GPU: {err})")

    for m, n, k in sizes:
        p = make_lp(m, n, k, seed=m)
        print(f"\n  {p.name}   nnz={p.nnz:,}")
        got = {}
        for dev in devices:
            t = time.perf_counter()
            s = solve_pdlp(p, PDLPParams(device=dev, eps_abs=eps, eps_rel=eps,
                                         max_iter=max_iter, time_limit=180))
            dt = time.perf_counter() - t
            got[dev] = (dt, s)
            viol = p.violation(s.x)[0]
            print(f"    {dev:<4s} {s.status.name:<16s} obj={s.objective:< 14.8g} "
                  f"it={s.iterations:<6d} {dt:7.3f}s  "
                  f"{s.iterations/max(dt,1e-9):8.0f} it/s  viol={viol:.1e}")
        if len(got) == 2:
            sp = got["cpu"][0] / got["gpu"][0]
            oc, og = got["cpu"][1].objective, got["gpu"][1].objective
            agree = abs(oc - og) / max(1.0, abs(oc))
            print(f"    -> GPU {sp:5.2f}x   objective agreement {agree:.2e}")


def bench_batch(m, n, k, batches):
    """The core claim behind batched node relaxation: SpMM beats K x SpMV."""
    print()
    print("=" * 78)
    print("Batched node relaxation  -  SpMM vs K separate SpMV")
    print("=" * 78)
    p = make_lp(m, n, k, seed=1234)
    A = p.A
    print(f"  matrix {m}x{n}, nnz={A.nnz:,}")
    print(f"  {'K':>5} {'device':>7} {'K x SpMV':>12} {'SpMM':>12} {'speedup':>9} "
          f"{'GFLOP/s':>9}")

    devices = ["cpu", "gpu"] if gpu_selftest()[0] else ["cpu"]
    rng = np.random.default_rng(0)
    Xh = rng.standard_normal((n, max(batches)))

    def timed(fn, bk, reps):
        fn()                       # warm up: JIT / module load / autotune
        bk.sync()
        best = float("inf")
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            bk.sync()
            best = min(best, time.perf_counter() - t)
        return best               # min, not mean: noise is one-sided

    for dev in devices:
        bk = get_backend(dev)
        rp, ri, rx = bk.pointer(A.rp), bk.index(A.ri), bk.to_device(A.rx)
        yv = bk.zeros(m)
        for K in batches:
            X = bk.to_device(np.ascontiguousarray(Xh[:, :K]))
            Y = bk.zeros((m, K))
            xv = bk.to_device(np.ascontiguousarray(Xh[:, 0]))

            # K separate products is what a conventional tree does: one node
            # at a time, one launch each.
            t_spmv = timed(lambda: [bk.spmv(rp, ri, rx, xv, yv, m)
                                    for _ in range(K)], bk, 5)
            t_spmm = timed(lambda: bk.spmm(rp, ri, rx, X, Y, m, K), bk, 5)

            gf = 2.0 * A.nnz * K / t_spmm / 1e9
            print(f"  {K:>5} {dev:>7} {t_spmv*1e3:>10.2f}ms {t_spmm*1e3:>10.2f}ms "
                  f"{t_spmv/t_spmm:>8.2f}x {gf:>8.2f}")


def bench_kernels(m, n, k):
    """Raw kernel throughput, to separate compute cost from launch latency."""
    print()
    print("=" * 78)
    print("Raw kernel throughput  -  is the inner loop compute- or latency-bound?")
    print("=" * 78)
    p = make_lp(m, n, k, seed=99)
    A = p.A
    for dev in (["cpu", "gpu"] if gpu_selftest()[0] else ["cpu"]):
        bk = get_backend(dev)
        rp, ri, rx = bk.pointer(A.rp), bk.index(A.ri), bk.to_device(A.rx)
        cp_, ci_, cx_ = bk.pointer(A.cp), bk.index(A.ci), bk.to_device(A.cx)
        x = bk.to_device(np.random.default_rng(1).standard_normal(n))
        y = bk.zeros(m)
        z = bk.zeros(n)
        lo = bk.zeros(n)
        hi = bk.to_device(np.full(n, 10.0))
        c = bk.to_device(np.random.default_rng(2).standard_normal(n))
        dx = bk.zeros(n)

        reps = 200
        for label, fn in (
            ("spmv  A@x", lambda: bk.spmv(rp, ri, rx, x, y, m)),
            ("spmv A^T@y", lambda: bk.spmv(cp_, ci_, cx_, y, z, n)),
            ("pdhg_primal", lambda: bk.pdhg_primal(x, dx, c, z, lo, hi, 0.1, n)),
        ):
            fn()
            bk.sync()
            t = time.perf_counter()
            for _ in range(reps):
                fn()
            bk.sync()
            dt = (time.perf_counter() - t) / reps
            print(f"  {dev:>4} {label:<12} {dt*1e6:8.1f} us")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--skip-pdlp", action="store_true")
    a = ap.parse_args(argv)

    sizes = [(2_000, 3_000, 6), (20_000, 30_000, 8)]
    if a.big:
        sizes.append((100_000, 150_000, 10))

    if not a.skip_pdlp:
        bench_pdlp(sizes, a.max_iter, a.eps)
    bench_batch(20_000, 30_000, 8, [1, 8, 32, 64, 128, 256])
    bench_kernels(20_000, 30_000, 8)
    return 0


if __name__ == "__main__":
    sys.exit(main())
