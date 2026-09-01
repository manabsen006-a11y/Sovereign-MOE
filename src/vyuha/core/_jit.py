"""JIT compilation shim.

Numba is the compiler for every sequential hot loop in this project: sparse LU
factorisation, hypersparse triangular solves, ratio tests, bound propagation.
These are pointer-chasing loops that NumPy cannot vectorise, and they are where
a pure-Python solver dies.

If Numba is unavailable the decorators degrade to no-ops so the package still
imports and runs; callers that care about speed check ``HAVE_NUMBA`` and pick a
vectorised NumPy path instead.

No algorithm lives in this file. It is build infrastructure.
"""

from __future__ import annotations

import os

__all__ = ["njit", "prange", "HAVE_NUMBA", "NUMBA_THREADS", "jit_kernel"]

try:  # pragma: no cover - environment dependent
    from numba import njit as _njit, prange as _prange, get_num_threads

    HAVE_NUMBA = True
    NUMBA_THREADS = get_num_threads()
except Exception:  # pragma: no cover - environment dependent
    HAVE_NUMBA = False
    NUMBA_THREADS = 1

    def _njit(*args, **kwargs):
        """No-op stand-in for :func:`numba.njit`."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap

    _prange = range


njit = _njit
prange = _prange

# Disable the on-disk compile cache when the source tree is read-only or when a
# benchmark run wants cold-compile timings to be honest.
_CACHE = os.environ.get("VYUHA_JIT_CACHE", "1") != "0"


def jit_kernel(parallel: bool = False, fastmath: bool = False, inline: str = "never"):
    """Project-standard ``njit`` configuration.

    ``fastmath`` is opt-in per kernel and only permitted where reassociation
    cannot change the sign of a residual -- mark such kernels ``# SAFE-FASTMATH``.
    """

    def _decorate(fn):
        if not HAVE_NUMBA:
            return fn
        return _njit(
            cache=_CACHE,
            parallel=parallel,
            fastmath=fastmath,
            inline=inline,
            nogil=True,
        )(fn)

    return _decorate
