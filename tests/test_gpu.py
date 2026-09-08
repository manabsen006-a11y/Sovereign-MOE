"""GPU backend tests.

The GPU kernels in ``sovopt.core.backend`` are hand-written CUDA, so nothing
outside this file ever checks them against a reference. Every test here states
the same claim in a different place: *the GPU computes what the CPU computes*.
A speedup is not a result if the two devices disagree.

The module is skipped when there is no usable device. "Usable" means
``gpu_selftest``, not ``gpu_available`` -- a machine where CuPy imports but
NVRTC cannot build the kernels must skip rather than fail, because that is a
fact about the environment and not a defect in the code under test.
"""

import numpy as np
import pytest

from sovopt.core.backend import Backend, get_backend, gpu_selftest
from sovopt.core.problem import Problem
from sovopt.core.sparse import SparseMatrix
from sovopt.core.tolerances import INF

_OK, _WHY = gpu_selftest()
pytestmark = pytest.mark.skipif(not _OK, reason=f"no usable GPU: {_WHY}")


def _random_csr(m, n, per_col, seed):
    rng = np.random.default_rng(seed)
    cols = np.repeat(np.arange(n), per_col)
    rows = rng.integers(0, m, cols.size)
    vals = rng.standard_normal(cols.size)
    return SparseMatrix.from_triplets(rows, cols, vals, m, n), rng


def _to(bk, A):
    """The three device arrays a CSR kernel takes."""
    return (bk.pointer(A.rp), bk.index(A.ri), bk.to_device(A.rx))


# --------------------------------------------------------------------------- #
# the environment itself                                                       #
# --------------------------------------------------------------------------- #


def test_auto_picks_the_gpu_when_the_selftest_passes():
    assert get_backend("auto").device == "gpu"
    assert get_backend("gpu").device == "gpu"
    assert get_backend("cpu").device == "cpu"


def test_import_emits_no_cuda_path_warning():
    """Importing the backend must not emit CuPy's CUDA_PATH warning.

    It is cosmetic -- the wheels carry their own CUDA libraries and the kernels
    build without a system toolkit -- but it appeared on stderr of every CLI
    invocation and reads as a load failure, which is how this file came to
    exist.

    This has to run in a subprocess. CuPy emits the warning once, from
    ``_environment`` during its first import; reloading our module inside this
    process would hit the ``sys.modules`` cache and pass no matter what the
    filter did.
    """
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-c", "import sovopt.core.backend"],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "CUDA path" not in r.stderr, r.stderr


# --------------------------------------------------------------------------- #
# sparse products                                                              #
# --------------------------------------------------------------------------- #


def test_spmv_matches_cpu():
    A, rng = _random_csr(400, 600, 5, seed=1)
    x = rng.standard_normal(A.n)

    cpu = get_backend("cpu")
    out_c = cpu.zeros(A.m)
    cpu.spmv(*_to(cpu, A), cpu.to_device(x), out_c, A.m)

    gpu = get_backend("gpu")
    out_g = gpu.zeros(A.m)
    gpu.spmv(*_to(gpu, A), gpu.to_device(x), out_g, A.m)
    gpu.sync()

    np.testing.assert_allclose(gpu.to_host(out_g), out_c, rtol=1e-12, atol=1e-12)


def test_spmv_handles_empty_and_warp_boundary_rows():
    """Row lengths 0, 1, 31, 32 and 33 straddle the warp boundary.

    One warp handles one row, so 32 and 33 take different numbers of trips
    through the strided accumulation loop, and an empty row must still write a
    zero rather than leave the output untouched.
    """
    lens = [0, 1, 31, 32, 33]
    m, n = len(lens), 64
    rng = np.random.default_rng(2)
    rows, cols, vals = [], [], []
    for i, length in enumerate(lens):
        c = rng.choice(n, size=length, replace=False)
        rows += [i] * length
        cols += list(c)
        vals += list(rng.standard_normal(length))
    A = SparseMatrix.from_triplets(np.array(rows), np.array(cols),
                                   np.array(vals), m, n)
    x = rng.standard_normal(n)

    gpu = get_backend("gpu")
    out = gpu.zeros(m) + 12345.0        # poison, so a skipped row is caught
    gpu.spmv(*_to(gpu, A), gpu.to_device(x), out, m)
    gpu.sync()
    np.testing.assert_allclose(gpu.to_host(out), A.matvec(x),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("k", [1, 8, 32, 33, 64])
def test_spmm_matches_cpu(k):
    """Batch widths on either side of the 32-lane tile."""
    A, rng = _random_csr(300, 500, 4, seed=k)
    X = rng.standard_normal((A.n, k))

    cpu = get_backend("cpu")
    Yc = cpu.zeros((A.m, k))
    cpu.spmm(*_to(cpu, A), cpu.to_device(X), Yc, A.m, k)

    gpu = get_backend("gpu")
    Yg = gpu.zeros((A.m, k))
    gpu.spmm(*_to(gpu, A), gpu.to_device(X), Yg, A.m, k)
    gpu.sync()

    np.testing.assert_allclose(gpu.to_host(Yg), Yc, rtol=1e-12, atol=1e-12)


def test_spmv_is_bitwise_reproducible():
    """The shuffle reduction has a fixed order, so repeats must be identical.

    Deterministic mode rests on this, and ``allclose`` would not test it.
    """
    A, rng = _random_csr(500, 700, 6, seed=3)
    x = rng.standard_normal(A.n)
    gpu = get_backend("gpu")
    dev, xd = _to(gpu, A), gpu.to_device(x)

    first = None
    for _ in range(4):
        out = gpu.zeros(A.m)
        gpu.spmv(*dev, xd, out, A.m)
        gpu.sync()
        got = gpu.to_host(out)
        if first is None:
            first = got
        else:
            assert np.array_equal(got, first)


def test_float32_kernels_run():
    """The mixed-precision path picks the f32 kernels off the operand dtype."""
    A, rng = _random_csr(200, 300, 4, seed=4)
    x = rng.standard_normal(A.n)
    gpu = Backend("gpu", dtype=np.float32)
    out = gpu.zeros(A.m)
    gpu.spmv(*_to(gpu, A), gpu.to_device(x), out, A.m)
    gpu.sync()
    assert out.dtype == np.float32
    np.testing.assert_allclose(gpu.to_host(out), A.matvec(x),
                               rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# fused elementwise steps                                                      #
# --------------------------------------------------------------------------- #


def test_primal_and_dual_steps_match_cpu():
    rng = np.random.default_rng(5)
    n, m = 1000, 700
    x, c, aty = (rng.standard_normal(n) for _ in range(3))
    lo, hi = -np.abs(rng.standard_normal(n)), np.abs(rng.standard_normal(n))
    y, ax = rng.standard_normal(m), rng.standard_normal(m)
    rl, ru = -np.abs(rng.standard_normal(m)), np.abs(rng.standard_normal(m))
    tau, sigma = 0.37, 1.9

    cpu, gpu = get_backend("cpu"), get_backend("gpu")
    exp_x = cpu.primal_step(cpu.zeros(n),
                            *(cpu.to_device(v) for v in (x, c, aty, lo, hi)),
                            tau, n)
    got_x = gpu.primal_step(gpu.zeros(n),
                            *(gpu.to_device(v) for v in (x, c, aty, lo, hi)),
                            tau, n)
    exp_y = cpu.dual_step(cpu.zeros(m),
                          *(cpu.to_device(v) for v in (y, ax, rl, ru)),
                          sigma, m)
    got_y = gpu.dual_step(gpu.zeros(m),
                          *(gpu.to_device(v) for v in (y, ax, rl, ru)),
                          sigma, m)
    gpu.sync()
    np.testing.assert_allclose(gpu.to_host(got_x), exp_x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(gpu.to_host(got_y), exp_y, rtol=1e-13, atol=1e-13)


def test_fused_pdhg_steps_match_cpu():
    """The fused kernels must equal the unfused reference they replaced."""
    rng = np.random.default_rng(6)
    n, m = 800, 600
    x0, c, aty = (rng.standard_normal(n) for _ in range(3))
    lo, hi = -np.ones(n), np.ones(n)
    y0, ax0, adx = (rng.standard_normal(m) for _ in range(3))
    rl, ru = -np.ones(m), np.ones(m)
    tau, sigma, eta = 0.25, 1.4, 0.6

    # These kernels write their operands in place, and on the CPU backend
    # ``to_device`` hands back the caller's array rather than a copy -- so a
    # shared input would be mutated by the CPU pass and the GPU pass would
    # then start from different data. Copy on the way in.
    out = {}
    for name in ("cpu", "gpu"):
        bk = get_backend(name)
        dev = lambda v: bk.to_device(v.copy())      # noqa: E731
        x, dx = dev(x0), bk.zeros(n)
        bk.pdhg_primal(x, dx, *(dev(v) for v in (c, aty, lo, hi)), tau, n)
        y, dy, ax = dev(y0), bk.zeros(m), dev(ax0)
        bk.pdhg_dual(y, dy, ax, *(dev(v) for v in (adx, rl, ru)), sigma, m)
        acc, run = dev(aty), bk.zeros(n)
        bk.pdhg_accum(acc, dx, run, x, eta, n)
        bk.sync()
        out[name] = [bk.to_host(v) for v in (x, dx, y, dy, ax, acc, run)]

    for expected, got in zip(out["cpu"], out["gpu"]):
        np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)


def test_reductions_match_cpu():
    rng = np.random.default_rng(7)
    a, b = rng.standard_normal(4096), rng.standard_normal(4096)
    cpu, gpu = get_backend("cpu"), get_backend("gpu")
    ac, bc = cpu.to_device(a), cpu.to_device(b)
    ag, bg = gpu.to_device(a), gpu.to_device(b)
    assert gpu.dot(ag, bg) == pytest.approx(cpu.dot(ac, bc), rel=1e-12)
    assert gpu.nrm2(ag) == pytest.approx(cpu.nrm2(ac), rel=1e-12)
    assert gpu.nrm_inf(ag) == pytest.approx(cpu.nrm_inf(ac), rel=1e-12)
    assert gpu.nrm_inf(gpu.zeros(0)) == 0.0


# --------------------------------------------------------------------------- #
# end to end                                                                   #
# --------------------------------------------------------------------------- #


def test_pdlp_cpu_and_gpu_reach_the_same_optimum():
    """The claim the benchmark rests on, as a test rather than a printout."""
    from sovopt.lp.pdlp import PDLPParams, solve_pdlp

    rng = np.random.default_rng(11)
    m, n = 300, 400
    A, _ = _random_csr(m, n, 5, seed=11)
    b = A.matvec(rng.random(n)) + rng.random(m) * 0.5
    p = Problem(A=A, c=rng.standard_normal(n),
                row_lb=np.full(m, -INF), row_ub=b,
                col_lb=np.zeros(n), col_ub=np.full(n, 10.0), name="agree")

    kw = dict(eps_abs=1e-7, eps_rel=1e-7, max_iter=20_000)
    sc = solve_pdlp(p, PDLPParams(device="cpu", **kw))
    sg = solve_pdlp(p, PDLPParams(device="gpu", **kw))

    assert sc.status == sg.status
    assert abs(sc.objective - sg.objective) / max(1.0, abs(sc.objective)) < 1e-5
