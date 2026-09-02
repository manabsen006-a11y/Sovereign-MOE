"""Model file readers."""

import os

from .lp_format import LPFormatError, read_lp
from .mps import MPSError, read_mps, write_mps

__all__ = ["read_model", "read_mps", "write_mps", "read_lp",
           "MPSError", "LPFormatError"]


def read_model(path, name=None):
    """Read a model, choosing the reader by file extension.

    ``.lp`` goes to the CPLEX LP reader, everything else to MPS. Compression
    suffixes are looked through, so ``model.lp.gz`` still picks the LP reader.
    """
    base = str(path).lower()
    for suf in (".gz", ".bz2", ".xz", ".lzma"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    if base.endswith(".lp"):
        return read_lp(path, name=name)
    return read_mps(path, name=name)
