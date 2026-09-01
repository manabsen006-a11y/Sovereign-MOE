"""Fail the build if the clean-room policy is broken.

Two checks:
  1. no forbidden solver or factorisation import appears under src/vyuha/
  2. every algorithm module carries a References section naming its sources
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "vyuha")

FORBIDDEN = [
    r"\bscipy\.optimize\b", r"\bfrom scipy import optimize\b",
    r"\bsplu\b", r"\bsuperlu\b", r"\bumfpack\b", r"\bcholmod\b",
    r"\bmumps\b", r"\bpardiso\b",
    r"\bimport highs\b", r"\bimport pyscipopt\b", r"\bimport cylp\b",
    r"\bimport pulp\b", r"\bimport mip\b(?!\.)", r"\bortools\b", r"\bcuopt\b",
    r"\bimport scipy\b",
]

# modules that are plumbing, not algorithms
# Files that implement no mathematics of their own: entry points, data
# containers and presentation wrappers. Everything that implements an
# algorithm must cite where the algorithm comes from.
EXEMPT = {"__init__.py", "_jit.py", "cli.py", "tolerances.py", "problem.py",
          "demo.py"}


def main() -> int:
    bad = []
    missing = []
    for dirpath, _dirs, files in os.walk(SRC):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            text = open(path, encoding="utf-8").read()
            rel = os.path.relpath(path, ROOT)
            for pat in FORBIDDEN:
                for m in re.finditer(pat, text):
                    line = text[:m.start()].count("\n") + 1
                    bad.append(f"{rel}:{line}  forbidden: {m.group(0)}")
            if fn not in EXEMPT and "References" not in text:
                missing.append(rel)

    for b in bad:
        print("FORBIDDEN IMPORT  " + b)
    for m in missing:
        print("NO CITATION HEADER  " + m)

    if bad or missing:
        print(f"\nprovenance check FAILED: {len(bad)} forbidden, "
              f"{len(missing)} uncited")
        return 1
    print("provenance check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
