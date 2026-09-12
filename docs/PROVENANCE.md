# Provenance and clean-room policy

The problem statement requires the engine be built "from scratch from
mathematical foundation" and **not** upon any existing open source solver
library. This document is the evidence for that claim.

## Policy

1. No solver source is read, imported, linked, or vendored. The forbidden list
   is in `CLAUDE.md` and includes CBC, Clp, HiGHS, SCIP, GLPK, SoPlex,
   OR-Tools, cuOpt, and `scipy.optimize`.
2. No factorisation library is used: no SuperLU, UMFPACK, CHOLMOD, MUMPS, or
   `scipy.sparse.linalg.splu`. The LU in `numerics/lu.py` is our own.
3. Algorithms are implemented from papers and textbooks. Every module carries a
   `References` section naming what it implements.
4. Permitted dependencies are compilers and array arithmetic only:
   `numpy` (dense arrays), `numba` (JIT), `cupy` (GPU arrays). The CUDA kernels
   in `core/backend.py` are hand-written; no vendor sparse solver is called.
5. `scipy` may appear only in `tests/` and `bench/` for constructing or
   cross-checking test data, never inside `src/sovopt/`.

## Citation index

| Module | Implements | Primary sources |
|---|---|---|
| `core/sparse.py` | CSC/CSR storage, transpose by counting sort | Duff-Erisman-Reid 2017 ch.2; Davis 2006 ch.2 |
| `numerics/scaling.py` | Ruiz equilibration; Curtis-Reid least squares; Pock-Chambolle | Ruiz 2001; Curtis & Reid 1972; Pock & Chambolle 2011; Tomlin 1975 |
| `numerics/lu.py` | Gilbert-Peierls left-looking LU, threshold Markowitz, singleton peeling, DFS reachability | Gilbert & Peierls 1988; Markowitz 1957; Suhl & Suhl 1990; Davis 2006 ch.6 |
| `numerics/refine.py` | Error-free transformations, compensated dot product, iterative refinement, Hager condition estimate | Dekker 1971; Knuth TAOCP 2 §4.2.2; Ogita-Rump-Oishi 2005; Wilkinson 1963; Higham 2002 ch.12,15; Hager 1984 |
| `lp/basis.py` | Product form of the inverse, augmented form with logicals | Dantzig & Orchard-Hays 1954; Chvatal 1983 ch.7,24; Maros 2003 ch.5,9; Forrest & Tomlin 1972 |
| `lp/simplex.py` | Bounded-variable primal and dual revised simplex, Harris two-pass ratio tests, Devex pricing, anti-degeneracy perturbation | Dantzig 1963; Harris 1973; Gill et al. 1989 (EXPAND); Forrest & Goldfarb 1992; Fourer 1994; Koberstein 2005; Maros 2003 |
| `lp/pdlp.py` | Restarted average PDHG, adaptive step size, adaptive restarts, primal weight | Chambolle & Pock 2011; Applegate et al. NeurIPS 2021; Applegate et al. Math.Prog. 2023; Malitsky & Pock 2018 |
| `lp/ipm.py` | Mehrotra predictor-corrector on the bounded-variable augmented system, LP and convex QP; static regularisation refined away, or proximal-point (opt-in) | Mehrotra 1992; Wright 1997; Vanderbei 1995, 1999; Altman & Gondzio 1999; Friedlander & Orban 2012; Pougkakiotis & Gondzio 2021 |
| `numerics/ldl.py` | Up-looking LDLᵀ for quasi-definite matrices on a fixed ordering, elimination-tree reach | Vanderbei 1995; Davis 2006 ch.4; Liu 1990; Altman & Gondzio 1999 |
| `numerics/ordering.py` | Approximate minimum degree, reverse Cuthill-McKee, symbolic fill | Amestoy-Davis-Duff 1996; George & Liu 1981; Cuthill & McKee 1969; Davis 2006 |
| `numerics/ft.py` | Forrest-Tomlin update with a row file | Forrest & Tomlin 1972; Suhl & Suhl 1993; Chvátal 1983; Maros 2003 |
| `globalopt/alphabb.py` | αBB underestimators with scaled Gershgorin α (opt-in) | Androulakis-Maranas-Floudas 1995; Adjiman et al. 1998; Neumaier & Shcherbina 2004; Horst & Tuy 1996 |
| `mip/safebound.py` | Valid bounds from arbitrary duals | Neumaier & Shcherbina 2004; Cook et al. 2013; Althaus & Dumitriu 2012 |
| `mip/propagate.py` | Activity-based bound tightening | Savelsbergh 1994; Brearley-Mitra-Williams 1975; Achterberg 2007 §7.1 |
| `mip/tree.py` | Branch and bound, pseudocost branching | Land & Doig 1960; Achterberg-Koch-Martin 2005; Linderoth & Savelsbergh 1999 |
| `mip/bnr.py` | Batched node relaxation | Original to this project; builds on the two rows above |
| `models/refinery.py` | Blending, planning, scheduling structures | Haverly 1978; Lee et al. 1996; Pinto-Joly-Moro 2000 |

## Original contribution

`mip/bnr.py` — **Batched Node Relaxation** — is not taken from any published
implementation. It combines two known ingredients (GPU first-order LP; safe
dual bounds from approximate duals) into a search that bounds an entire
branch-and-bound frontier in a single sparse-times-dense product. The
observation it rests on is that all nodes of a tree share `A`, `c` and the row
bounds, differing only in variable bounds, so bounding them is a batched
operation rather than a sequential one.

## Verification

`tools/check_provenance.py` fails if any module under `src/sovopt/` implementing
an algorithm lacks a `References` section, and greps for forbidden imports.
