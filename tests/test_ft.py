"""Forrest-Tomlin update of the basis factors.

What has to hold: after any sequence of column replacements the factor
solves the *current* basis, to the accuracy a fresh factorisation would
give, and refuses -- leaving itself intact -- when it could not. The dense
identity ``L R⁻¹ Ũ = P B Q̃`` is checked outright on a small case, because a
solve that happens to be right is weaker evidence than a factorisation that
is.
"""

import numpy as np
import pytest

from sovopt.core.problem import Status
from sovopt.core.sparse import coo_to_csc
from sovopt.lp.basis import Basis
from sovopt.lp.simplex import SimplexParams, solve_simplex
from sovopt.numerics.ft import FTFactor
from sovopt.numerics.lu import LUSingular, lu_factor

from tests.test_simplex import random_lp


def sparse_basis(rng, m, density):
    while True:
        B = np.where(rng.random((m, m)) < density, rng.standard_normal((m, m)), 0.0)
        B[np.arange(m), np.arange(m)] += rng.uniform(1, 3, m) * rng.choice([-1, 1], m)
        if abs(np.linalg.slogdet(B)[1]) < 100:
            return B


def factor(B):
    m = B.shape[0]
    r, c = np.nonzero(B)
    cp, ci, cx = coo_to_csc(r, c, B[r, c], m, m)
    return lu_factor(cp, ci, cx, m)


def backward_error(B, x, b):
    return np.abs(B @ x - b).max() / (np.abs(b).max() + np.abs(B).max() * np.abs(x).max())


def dense_pieces(ft):
    """``(L, R, Ũ_live, P, bpos)`` reconstructed from the factor's arrays."""
    lu, m = ft.lu, ft.m
    L = np.zeros((m, m))
    for k in range(m):
        for p in range(lu.Lp[k], lu.Lp[k + 1]):
            L[lu.Li[p], k] = lu.Lx[p]
    P = np.zeros((m, m))
    P[lu.pinv, np.arange(m)] = 1.0
    U = np.zeros((m, ft.n_slots))
    for rid in range(m):
        for p in range(ft.rstart[rid], ft.rstart[rid] + ft.rlen[rid]):
            if ft.alive[ft.rslot[p]]:
                U[rid, ft.rslot[p]] += ft.rval[p]
    for s in range(ft.n_slots):
        if ft.alive[s]:
            U[ft.urow[s], s] += ft.udiag[s]
    R = np.eye(m)
    for e in range(ft.n_eta):
        Re = np.eye(m)
        for p in range(ft.estart[e], ft.estart[e + 1]):
            Re[ft.erho[e], ft.erow[p]] -= ft.emult[p]
        R = Re @ R
    live = np.flatnonzero(ft.alive[:ft.n_slots])
    return L, R, U[:, live], P, ft.bpos_of_slot[live]


# --------------------------------------------------------------------------- #
# the factor                                                                   #
# --------------------------------------------------------------------------- #


def test_no_update_is_the_lu_itself():
    rng = np.random.default_rng(0)
    B = sparse_basis(rng, 20, 0.2)
    lu = factor(B)
    ft = FTFactor(lu)
    b = rng.standard_normal(20)
    assert np.allclose(ft.ftran(b.copy()), lu.ftran(b.copy()), atol=1e-13)
    assert np.allclose(ft.btran(b.copy()), lu.btran(b.copy()), atol=1e-13)


@pytest.mark.parametrize("seed", range(4))
def test_the_factorisation_identity_holds_after_updates(seed):
    """``L R⁻¹ Ũ = P B Q̃`` densely, and ``Ũ`` is triangular in position order."""
    rng = np.random.default_rng(seed)
    m = 9
    B = sparse_basis(rng, m, 0.35)
    ft = FTFactor(factor(B), max_updates=20)
    for _ in range(6):
        r = int(rng.integers(m))
        a = np.where(rng.random(m) < 0.5, rng.standard_normal(m), 0.0)
        a[rng.integers(m)] += 2.0
        Bn = B.copy()
        Bn[:, r] = a
        if abs(np.linalg.slogdet(Bn)[0]) < 0.5:
            continue
        idx = np.flatnonzero(a)
        try:
            ft.update(r, idx, a[idx])
        except LUSingular:
            continue
        B = Bn
        L, R, U, P, bpos = dense_pieces(ft)
        assert np.abs(L @ np.linalg.inv(R) @ U - P @ B[:, bpos]).max() < 1e-10
        live = np.flatnonzero(ft.alive[:ft.n_slots])
        for i in range(m):
            for jj, s in enumerate(live):
                if ft.rowpos[i] > ft.colpos[s]:
                    assert U[i, jj] == 0.0


@pytest.mark.parametrize("m,density,n_updates,cap,tol", [
    (30, 0.15, 120, 1e4, 1e-9), (120, 0.04, 300, 1e4, 1e-9),
    (120, 0.04, 300, 1e8, 1e-6)])
def test_hundreds_of_updates_stay_accurate_with_refactorisation_on_refusal(
        m, density, n_updates, cap, tol):
    """Random replacement columns are adversarial -- the trailing matrix goes
    ill-conditioned fast -- and the growth cap is what keeps the solves at
    1e-9 through it at a cap of 1e4. The simplex runs the default cap of
    1e8, where the ratio test rather than the cap keeps pivots reasonable
    (2e-12 measured on woodw); here, without a ratio test, that cap holds
    1e-6. A refusal is answered the way the basis answers it: refactorise
    and carry on."""
    rng = np.random.default_rng(m)
    B = sparse_basis(rng, m, density)
    ft = FTFactor(factor(B), max_updates=n_updates + 5, max_mult=cap)
    worst = 0.0
    refused = 0
    for _ in range(n_updates):
        b = rng.standard_normal(m)
        worst = max(worst, backward_error(B, ft.ftran(b.copy()), b),
                    backward_error(B.T, ft.btran(b.copy()), b))
        r = int(rng.integers(m))
        a = np.where(rng.random(m) < max(density, 3.0 / m), rng.standard_normal(m), 0.0)
        a[rng.integers(m)] += rng.uniform(1, 2)
        Bn = B.copy()
        Bn[:, r] = a
        if abs(np.linalg.slogdet(Bn)[0]) < 0.5:
            continue
        idx = np.flatnonzero(a)
        try:
            ft.update(r, idx, a[idx])
        except LUSingular:
            refused += 1
            try:
                ft = FTFactor(factor(Bn), max_updates=n_updates + 5, max_mult=cap)
            except LUSingular:
                continue                  # genuinely singular: FT was right
        B = Bn
    assert worst < tol, worst
    assert refused < n_updates // 5


def test_a_refused_update_leaves_the_factor_intact():
    """One good update first, because a fresh factor accepts anything but an
    exact zero (see the module header); then a column that would make the
    basis singular, which the growth cap or the pivot test must refuse."""
    rng = np.random.default_rng(5)
    m = 15
    B = sparse_basis(rng, m, 0.3)
    ft = FTFactor(factor(B), max_updates=10)
    a0 = B[:, 1] * 1.5
    a0[1] += 1.0
    ft.update(1, np.flatnonzero(a0), a0[np.flatnonzero(a0)])
    B[:, 1] = a0
    b = rng.standard_normal(m)
    before = ft.ftran(b.copy())
    assert backward_error(B, before, b) < 1e-12
    r, other = 3, 7
    a = B[:, other].copy()                   # a copy of another column
    idx = np.flatnonzero(a)
    with pytest.raises(LUSingular):
        ft.update(r, idx, a[idx])
    assert ft.n_updates == 1 and ft.n_refused == 1
    assert int(ft.alive[:ft.n_slots].sum()) == m and ft.n_slots == m + 1
    assert np.allclose(ft.ftran(b.copy()), before)


def test_slots_and_etas_grow_one_per_update():
    rng = np.random.default_rng(2)
    m = 25
    B = sparse_basis(rng, m, 0.2)
    ft = FTFactor(factor(B), max_updates=40)
    done = 0
    for _ in range(30):
        r = int(rng.integers(m))
        a = np.where(rng.random(m) < 0.2, rng.standard_normal(m), 0.0)
        a[r] += 2.0
        Bn = B.copy()
        Bn[:, r] = a
        if abs(np.linalg.slogdet(Bn)[0]) < 0.5:
            continue
        idx = np.flatnonzero(a)
        try:
            ft.update(r, idx, a[idx])
        except LUSingular:
            continue
        B = Bn
        done += 1
    assert ft.n_updates == done and ft.n_eta == done
    assert ft.n_slots == m + done
    assert int(ft.alive[:ft.n_slots].sum()) == m         # one live slot per position
    assert sorted(ft.bpos_of_slot[:ft.n_slots][ft.alive[:ft.n_slots]]) == list(range(m))


def test_the_update_budget_is_enforced():
    rng = np.random.default_rng(3)
    m = 10
    B = sparse_basis(rng, m, 0.3)
    ft = FTFactor(factor(B), max_updates=2)
    for k in range(3):
        a = B[:, k].copy() * 1.5
        a[k] += 1.0
        idx = np.flatnonzero(a)
        if k < 2:
            ft.update(k, idx, a[idx])
        else:
            with pytest.raises(LUSingular):
                ft.update(k, idx, a[idx])


# --------------------------------------------------------------------------- #
# inside the simplex                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(6))
def test_the_simplex_reaches_the_same_optimum_under_both_updates(seed):
    p = random_lp(seed=seed)
    a = solve_simplex(p.copy(), SimplexParams(basis_update="pfi"))
    b = solve_simplex(p.copy(), SimplexParams(basis_update="ft"))
    assert a.status == b.status
    if a.status == Status.OPTIMAL:
        assert abs(a.objective - b.objective) <= 1e-8 * max(1.0, abs(a.objective))
        assert b.info["update"] == "ft"


def test_a_long_budget_under_ft_keeps_the_answer():
    """The point of the update is to skip refactorisations; at a budget of
    2000 a solve of a few thousand pivots refactorises only on refusal."""
    p = random_lp(seed=1, m=60, n=140)
    a = solve_simplex(p.copy(), SimplexParams(basis_update="pfi"))
    b = solve_simplex(p.copy(), SimplexParams(basis_update="ft", refactor_freq=2000))
    assert a.status == b.status == Status.OPTIMAL
    assert abs(a.objective - b.objective) <= 1e-8 * max(1.0, abs(a.objective))


def test_an_unknown_update_is_refused():
    p = random_lp(seed=0)
    with pytest.raises(ValueError):
        Basis(p, update="magic")
