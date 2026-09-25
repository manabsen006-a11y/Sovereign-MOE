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
from sovopt.core.tolerances import INF
from sovopt.io.mps import read_mps
from sovopt.mip.safebound import certified_bound
from sovopt.numerics.refine import compensated_residual


class Verdict:
    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []
        self.certified_bound = None
        self.gap_rel = None          # the optimality gap, when duals were checked
        self.feas_tol = None
        self.opt_tol = None

    def add(self, name, ok, detail=""):
        self.checks.append((name, bool(ok), detail))

    @property
    def ok(self) -> bool:
        return all(c[1] for c in self.checks)

    def headline(self, extra=None):
        """``(verdict, detail)`` for a plan's footer, in words.

        ``ACCEPTED``  every check passed -- certified optimal when duals were
                      given, feasible when they were not;
        ``FEASIBLE``  every check but optimality: the point meets the model,
                      and its duals bound it only to a gap above ``opt_tol``;
        ``REJECTED``  the point fails the model itself.

        A bare REJECTED used to cover the middle case too, and read as "the
        plan is wrong" for a month of hourly blending that met every row to
        5.4e-10 and was optimal to 7.3e-9. ``extra`` is one more ``(name,
        ok, detail)`` check of the caller's -- a pooling plan's bilinear
        identities.
        """
        checks = list(self.checks) + ([extra] if extra is not None else [])
        failed = [c for c in checks if not c[1]]
        hard = [c for c in failed if c[0] != "optimality"]
        if hard:
            return "REJECTED", "; ".join(f"{name}: {detail}" for name, _ok, detail in hard)
        feas = f"feasible to {self.feas_tol:.0e}" if self.feas_tol else "feasible"
        opt = f"{self.opt_tol:.0e}" if self.opt_tol else "the certificate's tolerance"
        if not failed:
            if any(c[0] == "optimality" for c in checks):
                return "ACCEPTED", f"{feas}, and certified optimal to {opt}"
            return "ACCEPTED", f"{feas}; optimality not checked"
        gap = self.gap_rel
        if gap is None or not np.isfinite(gap):
            return "FEASIBLE", f"{feas}; its duals certify no bound"
        if gap < 0:
            return "FEASIBLE", (f"{feas}; the objective sits {-gap:.1e} past the certified "
                                f"bound -- rounding at the bound's resolution -- so it is "
                                f"not certified at {opt}")
        return "FEASIBLE", f"{feas}, and optimal to {gap:.1e} -- short of the {opt} certificate"

    def report(self) -> str:
        w = max(len(c[0]) for c in self.checks) if self.checks else 10
        lines = []
        for name, ok, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            lines.append(f"  [{mark}] {name:<{w}}  {detail}")
        lines.append("")
        lines.append(f"  VERDICT: {'ACCEPTED' if self.ok else 'REJECTED'}")
        return "\n".join(lines)


def verify_infeasible(prob, ray, rel_tol=1e-9, dual_tol=1e-9) -> Verdict:
    """Check a claim of infeasibility from the solver's dual ray.

    For any ``y``, ``Σ_i y_i·(rl_i if y_i>0 else ru_i) + Σ_j d_j·(lo_j if
    d_j>0 else hi_j)`` with ``d = -Aᵀy`` is a lower bound on zero over every
    point in the box that satisfies the rows, so a strictly positive value
    proves that no such point exists. This is the arithmetic the tree
    prunes on (:mod:`sovopt.mip.conflict`), evaluated on the original model
    with the ray the solver handed over. A ray's sign is the solver's
    convention, so both are tried and the verdict says which held.

    Rounding is handled as the optimality certificate handles it. A ray
    component of 1e-19 on a row with no bound on that side, or a reduced
    ray ``d_j`` of 1e-16 on a column with none, makes the value ``-inf``
    however small it is -- and a phase-1 basis leaves exactly such residues
    on its basic columns (the infeasible set: 20 of 29 valid rays refused
    for terms of 1e-19 to 1e-11 of the ray's largest entry). Those, when no
    larger than ``dual_tol`` of the ray's scale, are dropped -- the value
    is then exact for a model whose matrix differs from this one's by at
    most that much -- and the verdict reports the perturbation. Larger ones
    stay, and the ray certifies nothing, which is the right answer.

    The acceptance threshold is relative to the ray's scale on the finite
    bounds, ``1 + |y|·(|rl|+|ru|)``, so a certificate is neither refused for
    being small on a small model nor accepted for being rounding on a large
    one.
    """
    v = Verdict()
    if ray is None:
        v.add("certificate", False, "no dual ray was returned with the verdict")
        return v
    y = np.asarray(ray, dtype=np.float64)
    if y.shape[0] != prob.m or not np.isfinite(y).all():
        v.add("certificate", False, "dual ray has the wrong length or is not finite")
        return v
    rl, ru, lo, hi = prob.row_lb, prob.row_ub, prob.col_lb, prob.col_ub
    big = float(np.abs(y).max(initial=0.0))
    if big == 0.0:
        v.add("certificate", False, "the ray is zero")
        return v
    amax = float(np.abs(prob.A.rx).max(initial=0.0))
    noise = dual_tol * big * max(1.0, amax)
    # INF is a sentinel, not np.inf, so finite bounds are picked by comparison
    finite = np.where(rl > -INF, np.abs(rl), 0.0) + np.where(ru < INF, np.abs(ru), 0.0)
    scale = 1.0 + float(np.abs(y) @ finite)

    best = (-np.inf, "+", 0.0, "")
    for sgn, yy in (("+", y), ("-", -y)):
        ypos = yy > 0.0
        row_bad = np.where(ypos, rl <= -INF, ru >= INF) & (yy != 0.0)
        if row_bad.any():
            if float(np.abs(yy[row_bad]).max()) > noise:
                continue                                  # a real term, unbounded
            yy = np.where(row_bad, 0.0, yy)
        d = -prob.A.rmatvec(yy)
        dpos = d > 0.0
        col_bad = np.where(dpos, lo <= -INF, hi >= INF) & (d != 0.0)
        pert = float(np.abs(d[col_bad]).max(initial=0.0))
        if pert > noise:
            continue
        row_total = float(np.sum(np.where(yy > 0.0, rl, ru) * yy))
        terms = np.where(dpos, lo, hi) * d
        val = row_total + float(terms[~col_bad].sum())
        rowp = float(np.abs(y[row_bad]).max(initial=0.0)) if row_bad.any() else 0.0
        if val > best[0]:
            best = (val, sgn, max(pert, rowp), "")
    val, sign, pert, _ = best
    if not np.isfinite(val):
        v.add("certificate", False,
              "the ray certifies nothing: a term larger than rounding points at an infinite bound")
        return v
    ok = val > rel_tol * scale
    detail = (f"Farkas value {val:.6g} ({val / scale:.2e} of the ray's scale), sign {sign}"
              + (f", terms of {pert:.1e} at infinite bounds dropped" if pert else ""))
    v.add("certificate", ok, detail)
    return v


def verify(prob, x, claimed_obj=None, feas_tol=1e-6, int_tol=1e-6,
           y=None, opt_tol=1e-9) -> Verdict:
    v = Verdict()
    v.feas_tol, v.opt_tol = feas_tol, opt_tol
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
                v.gap_rel = float("inf")
                v.add("optimality", False,
                      f"the duals certify nothing: a reduced cost of {pert:.2e} "
                      f"points at an infinite column bound")
            else:
                gap = (obj - bnd) if prob.sense == ObjSense.MINIMISE else (bnd - obj)
                rel = gap / max(1.0, abs(obj))
                v.gap_rel = rel
                # A gap below -opt_tol is a point that beats a valid bound
                # on every feasible point -- which is to say a point that
                # is not feasible at the bound's resolution, whatever the
                # 1e-6 feasibility line above made of it. Maros-Meszaros'
                # liswet1: 3e-7 off its rows, 0.19% below the bound.
                if rel < -opt_tol:
                    v.add("optimality", False,
                          f"objective {gap:.3e} BELOW the certified bound {bnd:.12g} "
                          f"({-rel:.2e} relative): the point is infeasible at the "
                          f"bound's resolution, not optimal")
                else:
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
