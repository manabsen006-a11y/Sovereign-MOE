"""CPU / GPU array backend with hand-written sparse kernels.

The first-order LP method is written once against this interface and runs
unchanged on NumPy or CuPy. Everything it needs is here: sparse products, a few
fused elementwise operations, and deterministic reductions.

Why hand-written CUDA rather than a vendor sparse library
---------------------------------------------------------
Two reasons, one principled and one practical.

*Principled*: the problem statement requires the engine be built from
mathematical foundations. CUDA itself is a compiler and cuBLAS-level elementwise
math is vendor arithmetic, but a sparse solver kernel is the substance of the
work. Writing our own removes the argument entirely.

*Practical*: the kernels we actually need are not the ones a general library
optimises for. The batched node relaxation in ``vyuha.mip.bnr`` needs an SpMM
where the *same* matrix multiplies a block of right-hand sides, with the block
laid out so that consecutive threads read consecutive nodes. That memory layout
is the entire performance argument, and it is not what a generic SpMM gives you.

Kernel design
-------------
``csr_spmv_warp`` assigns one warp per row and reduces with ``__shfl_down_sync``.
The shuffle tree has a fixed reduction order, so results are bitwise reproducible
across runs -- which is what lets the solver offer a deterministic mode.

``csr_spmm_warp`` assigns one warp per (row, block-of-columns) pair. The dense
operand is stored row-major with the batch index fastest, so the 32 lanes of a
warp read 32 consecutive doubles. This is the layout that makes batched node
bounding worth doing at all.

Mixed precision
---------------
Consumer GPUs cripple fp64 (the RTX 3050 runs it at 1/32 rate). Sparse products
are bandwidth-bound rather than flop-bound so fp64 still wins over CPU, but the
fp32 kernels roughly halve the bytes moved. The first-order method uses fp32
during its early, low-accuracy phase and switches to fp64 to finish -- the
accuracy that matters is the accuracy at termination.

References
----------
Bell & Garland, "Implementing sparse matrix-vector multiplication on
  throughput-oriented processors", SC'09 -- the row-per-warp CSR kernel and its
  load-balancing trade-offs.
Merrill & Garland, "Merge-based parallel sparse matrix-vector multiplication",
  SC'16 -- the alternative decomposition for highly irregular row lengths.
Baskaran & Bordawekar, "Optimizing sparse matrix-vector multiplication on GPUs",
  IBM RC24704 (2009) -- coalescing and the dense-operand layout that makes the
  batched kernel worth writing.
NVIDIA, *CUDA C++ Programming Guide* -- warp shuffle semantics. The fixed
  reduction order of ``__shfl_down_sync`` is what makes these kernels bitwise
  reproducible, which the deterministic mode depends on.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["Backend", "get_backend", "gpu_available", "gpu_selftest",
           "GPU_ERROR"]

GPU_ERROR: str | None = None

# CuPy's Windows start-up warns "CUDA path could not be detected" whenever
# CUDA_PATH is unset. That is a guess, not a diagnosis. The ``cupy-cuda12x``
# wheel takes nvrtc, cudart and cuBLAS from the ``nvidia-*-cu12`` wheels
# installed beside it and never needs a system CUDA Toolkit, so on a machine
# with only the driver the warning fires while every kernel below compiles and
# runs. Setting CUDA_PATH to silence it would point CuPy's DLL search at a
# directory that does not exist. A real load failure still surfaces, as the
# exception caught here and recorded in ``GPU_ERROR``; only the speculation is
# suppressed.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="CUDA path could not be detected")
    try:  # pragma: no cover - environment dependent
        import cupy as _cp
        _cp.zeros(1)                  # force context creation now, not later
        _HAVE_CUPY = True
    except Exception as _e:  # pragma: no cover
        _cp = None
        _HAVE_CUPY = False
        GPU_ERROR = f"{type(_e).__name__}: {_e}"


def gpu_available() -> bool:
    """True when CuPy imported and a device context was created.

    This is the cheap half of the question. It does not prove the kernels
    below will build: NVRTC compiles them at run time and the *driver* links
    the result, so a driver older than the runtime the wheels ship, a missing
    nvrtc, or an unsupported architecture all fail later, not here. See
    :func:`gpu_selftest`.
    """
    return _HAVE_CUPY


# --------------------------------------------------------------------------- #
# CUDA source                                                                  #
# --------------------------------------------------------------------------- #

_CUDA_SRC = r"""
extern "C" {

/* ---- one warp per row; shuffle reduction has a fixed order ------------- */
__global__ void csr_spmv_warp_f64(
    const long long* __restrict__ rp, const int* __restrict__ ri,
    const double* __restrict__ rx, const double* __restrict__ x,
    double* __restrict__ y, const int m)
{
    int tid  = blockDim.x * blockIdx.x + threadIdx.x;
    int row  = tid >> 5;
    int lane = tid & 31;
    if (row >= m) return;
    long long s = rp[row], e = rp[row + 1];
    double acc = 0.0;
    for (long long p = s + lane; p < e; p += 32) acc += rx[p] * x[ri[p]];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

__global__ void csr_spmv_warp_f32(
    const long long* __restrict__ rp, const int* __restrict__ ri,
    const float* __restrict__ rx, const float* __restrict__ x,
    float* __restrict__ y, const int m)
{
    int tid  = blockDim.x * blockIdx.x + threadIdx.x;
    int row  = tid >> 5;
    int lane = tid & 31;
    if (row >= m) return;
    long long s = rp[row], e = rp[row + 1];
    float acc = 0.0f;
    for (long long p = s + lane; p < e; p += 32) acc += rx[p] * x[ri[p]];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

/* ---- batched: Y[row, :] = sum_p rx[p] * X[ri[p], :] --------------------
   X and Y are row-major with the batch index fastest, so the 32 lanes of a
   warp touch 32 consecutive doubles. One warp handles one row for one tile
   of 32 batch entries.                                                    */
__global__ void csr_spmm_warp_f64(
    const long long* __restrict__ rp, const int* __restrict__ ri,
    const double* __restrict__ rx, const double* __restrict__ X,
    double* __restrict__ Y, const int m, const int k)
{
    int tid   = blockDim.x * blockIdx.x + threadIdx.x;
    int warp  = tid >> 5;
    int lane  = tid & 31;
    int tiles = (k + 31) >> 5;
    int row   = warp / tiles;
    int tile  = warp - row * tiles;
    if (row >= m) return;
    int col = (tile << 5) + lane;
    if (col >= k) return;

    long long s = rp[row], e = rp[row + 1];
    double acc = 0.0;
    for (long long p = s; p < e; ++p)
        acc += rx[p] * X[(long long)ri[p] * k + col];
    Y[(long long)row * k + col] = acc;
}

__global__ void csr_spmm_warp_f32(
    const long long* __restrict__ rp, const int* __restrict__ ri,
    const float* __restrict__ rx, const float* __restrict__ X,
    float* __restrict__ Y, const int m, const int k)
{
    int tid   = blockDim.x * blockIdx.x + threadIdx.x;
    int warp  = tid >> 5;
    int lane  = tid & 31;
    int tiles = (k + 31) >> 5;
    int row   = warp / tiles;
    int tile  = warp - row * tiles;
    if (row >= m) return;
    int col = (tile << 5) + lane;
    if (col >= k) return;

    long long s = rp[row], e = rp[row + 1];
    float acc = 0.0f;
    for (long long p = s; p < e; ++p)
        acc += rx[p] * X[(long long)ri[p] * k + col];
    Y[(long long)row * k + col] = acc;
}

/* ---- projection onto a box, fused with the gradient step ---------------
   x_new = clamp(x - tau * (c + Aty), lo, hi)                              */
__global__ void primal_step_f64(
    double* __restrict__ xnew, const double* __restrict__ x,
    const double* __restrict__ c, const double* __restrict__ aty,
    const double* __restrict__ lo, const double* __restrict__ hi,
    const double tau, const int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double v = x[i] - tau * (c[i] + aty[i]);
    double l = lo[i], u = hi[i];
    xnew[i] = v < l ? l : (v > u ? u : v);
}

/* ---- dual step: y_new = w - sigma * clamp(w/sigma, rl, ru) -------------
   the Moreau identity for the prox of a support function                  */
__global__ void dual_step_f64(
    double* __restrict__ ynew, const double* __restrict__ y,
    const double* __restrict__ ax, const double* __restrict__ rl,
    const double* __restrict__ ru, const double sigma, const int m)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= m) return;
    double w = y[i] + sigma * ax[i];
    double t = w / sigma;
    double l = rl[i], u = ru[i];
    double pr = t < l ? l : (t > u ? u : t);
    ynew[i] = w - sigma * pr;
}

/* ---- fused PDHG steps --------------------------------------------------
   The inner loop is latency-bound, not flop-bound: a naive implementation
   issues ~20 elementwise kernels per iteration and stalls twice on
   device->host reductions. These three kernels collapse all of the
   elementwise work into three launches and leave the reductions to be done
   only when the step size is actually re-adapted.                          */

/* xnew = clamp(x - tau*(c + aty), lo, hi);  dx = xnew - x;  x = xnew */
__global__ void pdhg_primal_f64(
    double* __restrict__ x, double* __restrict__ dx,
    const double* __restrict__ c, const double* __restrict__ aty,
    const double* __restrict__ lo, const double* __restrict__ hi,
    const double tau, const int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    double v = xi - tau * (c[i] + aty[i]);
    double l = lo[i], u = hi[i];
    v = v < l ? l : (v > u ? u : v);
    dx[i] = v - xi;
    x[i]  = v;
}

/* ax <- ax + adx  (so ax is now A*x_new)
   w  = y + sigma*(ax_new + adx)      [ = y + sigma*A*(2 x_new - x_old) ]
   y  <- w - sigma*clamp(w/sigma, rl, ru);  dy = y_new - y_old            */
__global__ void pdhg_dual_f64(
    double* __restrict__ y, double* __restrict__ dy,
    double* __restrict__ ax, const double* __restrict__ adx,
    const double* __restrict__ rl, const double* __restrict__ ru,
    const double sigma, const int m)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= m) return;
    double a  = adx[i];
    double axn = ax[i] + a;
    ax[i] = axn;
    double yo = y[i];
    double w  = yo + sigma * (axn + a);
    double t  = w / sigma;
    double l = rl[i], u = ru[i];
    double pr = t < l ? l : (t > u ? u : t);
    double yn = w - sigma * pr;
    dy[i] = yn - yo;
    y[i]  = yn;
}

/* aty <- aty + atdy;  sum <- sum + eta*v   (running ergodic average) */
__global__ void pdhg_accum_f64(
    double* __restrict__ acc, const double* __restrict__ inc,
    double* __restrict__ sum, const double* __restrict__ v,
    const double eta, const int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    acc[i] += inc[i];
    sum[i] += eta * v[i];
}

}  /* extern "C" */
"""

_MODULE = None
_KERNELS: dict = {}


def _kernels():
    global _MODULE, _KERNELS
    if _MODULE is None:
        _MODULE = _cp.RawModule(code=_CUDA_SRC, options=("-std=c++14",))
        for nm in ("csr_spmv_warp_f64", "csr_spmv_warp_f32",
                   "csr_spmm_warp_f64", "csr_spmm_warp_f32",
                   "primal_step_f64", "dual_step_f64",
                   "pdhg_primal_f64", "pdhg_dual_f64", "pdhg_accum_f64"):
            _KERNELS[nm] = _MODULE.get_function(nm)
    return _KERNELS


# --------------------------------------------------------------------------- #
# backend                                                                      #
# --------------------------------------------------------------------------- #


class Backend:
    """Array namespace plus sparse kernels, on CPU or GPU."""

    def __init__(self, device: str = "cpu", dtype=np.float64):
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        if device == "gpu" and not _HAVE_CUPY:
            raise RuntimeError(f"GPU backend unavailable ({GPU_ERROR})")
        self.device = device
        self.dtype = np.dtype(dtype)
        self.xp = _cp if device == "gpu" else np

    # -- transfer ----------------------------------------------------------- #

    def to_device(self, a, dtype=None):
        """Move ``a`` to this device.

        On the CPU backend this is a no-op for an array that already has the
        right dtype and layout: ``ascontiguousarray`` returns the *caller's*
        array, not a copy. That is deliberate -- the CPU path would otherwise
        copy every operand of every transfer -- but it means an in-place write
        to the result writes through to the caller. Everything in the solve
        path treats these as read-only operands or writes only into buffers it
        allocated itself; copy first if you need to do otherwise.
        """
        dt = self.dtype if dtype is None else np.dtype(dtype)
        if self.device == "cpu":
            return np.ascontiguousarray(a, dtype=dt)
        return _cp.ascontiguousarray(_cp.asarray(a, dtype=dt))

    def to_host(self, a):
        if self.device == "cpu":
            return np.asarray(a)
        return _cp.asnumpy(a)

    def index(self, a):
        """Index arrays stay int32, pointer arrays int64."""
        return self.to_device(a, dtype=np.int32)

    def pointer(self, a):
        return self.to_device(a, dtype=np.int64)

    def sync(self):
        if self.device == "gpu":
            _cp.cuda.Stream.null.synchronize()

    # -- allocation --------------------------------------------------------- #

    def zeros(self, shape, dtype=None):
        return self.xp.zeros(shape, dtype=self.dtype if dtype is None else dtype)

    def empty(self, shape, dtype=None):
        return self.xp.empty(shape, dtype=self.dtype if dtype is None else dtype)

    # -- sparse products ---------------------------------------------------- #

    def spmv(self, rp, ri, rx, x, out, m):
        """``out = A x`` with ``A`` in CSR."""
        if self.device == "cpu":
            from .sparse import _csr_spmv
            _csr_spmv(rp, ri, rx, x, out)
            return out
        k = _kernels()["csr_spmv_warp_f32" if rx.dtype == np.float32
                       else "csr_spmv_warp_f64"]
        threads = 128
        blocks = (m * 32 + threads - 1) // threads
        k((blocks,), (threads,), (rp, ri, rx, x, out, np.int32(m)))
        return out

    def spmm(self, rp, ri, rx, X, out, m, k_batch):
        """``out = A X`` with ``X`` dense ``(n, k)``, batch index fastest."""
        if self.device == "cpu":
            from .sparse import _csr_spmm
            _csr_spmm(rp, ri, rx, X, out)
            return out
        kern = _kernels()["csr_spmm_warp_f32" if rx.dtype == np.float32
                          else "csr_spmm_warp_f64"]
        tiles = (k_batch + 31) // 32
        threads = 128
        total_warps = m * tiles
        blocks = (total_warps * 32 + threads - 1) // threads
        kern((blocks,), (threads,),
             (rp, ri, rx, X, out, np.int32(m), np.int32(k_batch)))
        return out

    # -- fused steps -------------------------------------------------------- #

    def primal_step(self, xnew, x, c, aty, lo, hi, tau, n):
        """``xnew = clamp(x - tau (c + Aᵀy), lo, hi)``."""
        if self.device == "cpu" or self.dtype != np.float64:
            self.xp.subtract(x, tau * (c + aty), out=xnew)
            self.xp.clip(xnew, lo, hi, out=xnew)
            return xnew
        threads = 256
        blocks = (n + threads - 1) // threads
        _kernels()["primal_step_f64"](
            (blocks,), (threads,),
            (xnew, x, c, aty, lo, hi, np.float64(tau), np.int32(n)))
        return xnew

    def dual_step(self, ynew, y, ax, rl, ru, sigma, m):
        """``ynew = w - sigma · proj_[rl,ru](w/sigma)`` with ``w = y + sigma·Ax``.

        This is the Moreau decomposition of the prox of the support function of
        the row box -- one formula that covers ``<=``, ``>=``, ``=`` and ranged
        rows without any branching on row type.
        """
        if self.device == "cpu" or self.dtype != np.float64:
            w = y + sigma * ax
            pr = self.xp.clip(w / sigma, rl, ru)
            self.xp.subtract(w, sigma * pr, out=ynew)
            return ynew
        threads = 256
        blocks = (m + threads - 1) // threads
        _kernels()["dual_step_f64"](
            (blocks,), (threads,),
            (ynew, y, ax, rl, ru, np.float64(sigma), np.int32(m)))
        return ynew

    # -- fused PDHG iteration ----------------------------------------------- #
    #
    # These keep the whole elementwise part of an iteration on the device with
    # no host round-trip, which is what the inner loop is actually limited by.

    def pdhg_primal(self, x, dx, c, aty, lo, hi, tau, n):
        """``x <- clamp(x - tau(c + Aᵀy), lo, hi)``, also writing ``dx``."""
        if self.device == "cpu" or self.dtype != np.float64:
            xnew = self.xp.clip(x - tau * (c + aty), lo, hi)
            self.xp.subtract(xnew, x, out=dx)
            x[...] = xnew
            return
        threads = 256
        blocks = (n + threads - 1) // threads
        _kernels()["pdhg_primal_f64"](
            (blocks,), (threads,),
            (x, dx, c, aty, lo, hi, np.float64(tau), np.int32(n)))

    def pdhg_dual(self, y, dy, ax, adx, rl, ru, sigma, m):
        """Dual prox step, updating ``ax`` in place and writing ``dy``."""
        if self.device == "cpu" or self.dtype != np.float64:
            ax += adx
            w = y + sigma * (ax + adx)
            ynew = w - sigma * self.xp.clip(w / sigma, rl, ru)
            self.xp.subtract(ynew, y, out=dy)
            y[...] = ynew
            return
        threads = 256
        blocks = (m + threads - 1) // threads
        _kernels()["pdhg_dual_f64"](
            (blocks,), (threads,),
            (y, dy, ax, adx, rl, ru, np.float64(sigma), np.int32(m)))

    def pdhg_accum(self, acc, inc, run_sum, v, eta, n):
        """``acc += inc`` and ``run_sum += eta·v`` in one pass."""
        if self.device == "cpu" or self.dtype != np.float64:
            acc += inc
            run_sum += eta * v
            return
        threads = 256
        blocks = (n + threads - 1) // threads
        _kernels()["pdhg_accum_f64"](
            (blocks,), (threads,),
            (acc, inc, run_sum, v, np.float64(eta), np.int32(n)))

    # -- reductions --------------------------------------------------------- #

    def dot(self, a, b) -> float:
        return float(self.xp.dot(a, b))

    def nrm2(self, a) -> float:
        return float(self.xp.linalg.norm(a))

    def nrm_inf(self, a) -> float:
        return float(self.xp.abs(a).max()) if a.size else 0.0

    def __repr__(self):
        return f"Backend({self.device}, {self.dtype.name})"


_SELFTEST: tuple[bool, str | None] | None = None


def gpu_selftest() -> tuple[bool, str | None]:
    """Build the kernels and check one known answer. Result is cached.

    ``gpu_available`` only proves CuPy loaded. Everything after that -- NVRTC
    compiling ``_CUDA_SRC``, the driver linking it for this architecture, the
    launch, the warp-shuffle reduction, the copy back -- can still fail, and
    without this it fails in the middle of a solve. Paying 0.3 s once to turn
    that into a fallback decision is worth it; CuPy caches the compiled module
    on disk, so later runs cost milliseconds.

    Returns ``(ok, error)``, where ``error`` is ``None`` when ``ok``.
    """
    global _SELFTEST
    if _SELFTEST is not None:
        return _SELFTEST
    if not _HAVE_CUPY:
        _SELFTEST = (False, GPU_ERROR)
        return _SELFTEST
    try:  # pragma: no cover - environment dependent
        # [[1, 0, 2], [0, 3, 0]] @ [1, 2, 3] = [7, 6]. Two rows of unequal
        # length, so the shuffle reduction has to handle a partial warp.
        bk = Backend("gpu")
        rp = bk.pointer(np.array([0, 2, 3], dtype=np.int64))
        ri = bk.index(np.array([0, 2, 1], dtype=np.int32))
        rx = bk.to_device(np.array([1.0, 2.0, 3.0]))
        x = bk.to_device(np.array([1.0, 2.0, 3.0]))
        y = bk.zeros(2)
        bk.spmv(rp, ri, rx, x, y, 2)
        bk.sync()
        got = bk.to_host(y)
        if np.array_equal(got, np.array([7.0, 6.0])):
            _SELFTEST = (True, None)
        else:
            _SELFTEST = (False, f"spmv returned {got.tolist()}, expected [7.0, 6.0]")
    except Exception as e:  # pragma: no cover
        _SELFTEST = (False, f"{type(e).__name__}: {e}")
    return _SELFTEST


def get_backend(device: str = "auto", dtype=np.float64) -> Backend:
    """``'auto'`` uses the GPU when one is usable and falls back silently.

    Usable means the kernels compile and give the right answer, not merely
    that CuPy imported -- otherwise ``auto`` picks a GPU that cannot run
    anything and the failure lands somewhere far from its cause.
    """
    if device == "auto":
        device = "gpu" if gpu_selftest()[0] else "cpu"
    elif device == "gpu":
        ok, err = gpu_selftest()
        if not ok:
            raise RuntimeError(f"GPU backend unavailable ({err})")
    return Backend(device, dtype)
