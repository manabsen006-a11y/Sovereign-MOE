"""Independent solution verifier.

Deliberately separate from the solver, and deliberately simple. Its only job is
to answer "is this vector actually feasible, and is this objective value
actually its objective value" using nothing the solver computed. It re-reads the
model from disk, re-reads the solution from disk, and recomputes everything.

A solver that grades its own homework will report the bug and the wrong answer
with equal confidence. Every performance number in this project is expected to
pass through here before it is quoted.

Row activities are accumulated with the compensated dot product from
:mod:`sovopt.numerics.refine`, so the verdict does not itself dissolve into
rounding error on a badly scaled model -- which is precisely where a solver is
most likely to be wrong.

Optimality, when the solution carries duals. Feasibility says the point is
allowed; it says nothing about whether it is the best. A dual vector ``y``
gives a *certified* lower bound on every feasible point's objective --
Neumaier & Shcherbina's arithmetic, the same line the branch-and-bound
prunes on -- valid for any ``y`` whatever, and exact at an optimal pair. The
gap between that bound and the point's own objective is then a proof of
near-optimality that owes nothing to the solver's termination test. This is
what settled the eight Netlib instances whose objective disagreed with the
library's readme: the verifier certifies the vertex, and the readme is what
is off (see the README's Netlib section, and Koch 2004).

The certificate's arithmetic is :func:`sovopt.mip.safebound.certified_bound`,
one definition for the whole repository: the tree prunes on it, the interior
point applies it to its own final point before reporting, and this verifier
checks with it. The verifier's independence is in its data, not in a second
copy of a theorem -- it re-reads the model and the point from disk and
recomputes every activity with compensated arithmetic, and the bound it
then forms is valid for whatever ``y`` the solver handed over.

Usage:
    python -m bench.verify model.mps solution.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from sovopt.core.problem import ObjSense
from sovopt.io.mps import read_mps
from sovopt.mip.safebound import certified_bound
from sovopt.numerics.refine import compensated_residual


class Verdict:
    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []
        self.certified_bound = None

    def add(self, name, ok, detail=""):
        self.checks.append((name, bool(ok), detail))

    @property
    def ok(self) -> bool:
        return all(c[1] for c in self.checks)

    def report(self) -> str:
        w = max(len(c[0]) for c in self.checks) if self.checks else 10
        lines = []
        for name, ok, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            lines.append(f"  [{mark}] {name:<{w}}  {detail}")
        lines.append("")
        lines.append(f"  VERDICT: {'ACCEPTED' if self.ok else 'REJECTED'}")
        return "\n".join(lines)


def verify(prob, x, claimed_obj=None, feas_tol=1e-6, int_tol=1e-6,
           y=None, opt_tol=1e-9) -> Verdict:
    v = Verdict()
    x = np.asarray(x, dtype=np.float64)

    if x.shape[0] != prob.n:
        v.add("dimension", False, f"got {x.shape[0]}, model has {prob.n} columns")
        return v
    v.add("dimension", True, f"{prob.n} columns")

    if not np.isfinite(x).all():
        v.add("finite", False, "solution contains NaN or Inf")
        return v
    v.add("finite", True, "")

    # --- column bounds ---
    lo_v = float(np.maximum(prob.col_lb - x, 0.0).max(initial=0.0))
    hi_v = float(np.maximum(x - prob.col_ub, 0.0).max(initial=0.0))
    bnd = max(lo_v, hi_v)
    v.add("column bounds", bnd <= feas_tol, f"max violation {bnd:.3e}")

    # --- row activities, compensated ---
    zero = np.zeros(prob.m)
    act = -compensated_residual(prob.A.rp, prob.A.ri, prob.A.rx, x, zero)
    rlo = float(np.maximum(prob.row_lb - act, 0.0).max(initial=0.0))
    rhi = float(np.maximum(act - prob.row_ub, 0.0).max(initial=0.0))
    rowv = max(rlo, rhi)
    v.add("row constraints", rowv <= feas_tol, f"max violation {rowv:.3e}")

    # --- integrality ---
    mask = prob.integer_mask
    if mask.any():
        iv = float(np.abs(x[mask] - np.round(x[mask])).max(initial=0.0))
        v.add("integrality", iv <= int_tol,
              f"max distance to integer {iv:.3e} over {int(mask.sum())} vars")
    else:
        v.add("integrality", True, "no integer variables")

    # --- objective ---
    obj = float(prob.c @ x) + prob.obj_offset
    if prob.Q is not None:
        obj += 0.5 * float(x @ prob.Q.matvec(x))
    if claimed_obj is not None:
        d = abs(obj - claimed_obj) / max(1.0, abs(obj))
        v.add("objective", d <= 1e-6,
              f"recomputed {obj:.12g}, claimed {claimed_obj:.12g}, rel diff {d:.2e}")
    else:
        v.add("objective", True, f"recomputed {obj:.12g}")

    # --- optimality, if duals were supplied and the model is continuous ---
    if y is not None and not mask.any():
        y = np.asarray(y, dtype=np.float64)
        if y.shape[0] != prob.m or not np.isfinite(y).all():
            v.add("optimality", False, "dual vector has the wrong length or is not finite")
        else:
            bnd, pert = certified_bound(prob, x, y)
            v.certified_bound = bnd
            if not np.isfinite(bnd):
                v.add("optimality", False,
                      f"the duals certify nothing: a reduced cost of {pert:.2e} "
                      f"points at an infinite column bound")
            else:
                gap = (obj - bnd) if prob.sense == ObjSense.MINIMISE else (bnd - obj)
                rel = gap / max(1.0, abs(obj))
                v.add("optimality", rel <= opt_tol,
                      f"certified bound {bnd:.12g}, gap {gap:.3e} ({rel:.2e} relative)"
                      + (f", costs perturbed by {pert:.1e}" if pert else ""))

    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description="Independently verify a solution.")
    ap.add_argument("model")
    ap.add_argument("solution")
    ap.add_argument("--feas-tol", type=float, default=1e-6)
    ap.add_argument("--int-tol", type=float, default=1e-6)
    a = ap.parse_args(argv)

    prob = read_mps(a.model)
    with open(a.solution, encoding="utf-8") as fh:
        data = json.load(fh)

    y = None
    if isinstance(data, dict) and "x" in data:
        if isinstance(data["x"], dict):
            names = {nm: i for i, nm in enumerate(prob.col_names or [])}
            x = np.zeros(prob.n)
            for nm, val in data["x"].items():
                if nm in names:
                    x[names[nm]] = val
        else:
            x = np.asarray(data["x"], dtype=float)
        claimed = data.get("objective")
        if data.get("y") is not None:
            if isinstance(data["y"], dict):
                rnames = {nm: i for i, nm in enumerate(prob.row_names or [])}
                y = np.zeros(prob.m)
                for nm, val in data["y"].items():
                    if nm in rnames:
                        y[rnames[nm]] = val
            else:
                y = np.asarray(data["y"], dtype=float)
    else:
        x = np.asarray(data, dtype=float)
        claimed = None

    print(f"Verifying {prob.name}: {prob.m} rows x {prob.n} cols")
    v = verify(prob, x, claimed, a.feas_tol, a.int_tol, y=y)
    print(v.report())
    return 0 if v.ok else 1


if __name__ == "__main__":
    sys.exit(main())
