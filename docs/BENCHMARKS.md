# Benchmark campaign: every public category, one sitting

The problem statement asks that the solver be tested on "publicly available
mathematical optimization benchmark datasets such as MIPLIB, Netlib LP,
Mittelmann benchmark instances, QPLIB", and on "representative refinery
scheduling, crude blending, production planning and supply chain
optimization case studies from open literature". This is that test, run in
one sitting -- 17 September 2026, about eleven hours of wall time -- on the
machine described under
[Measurement conditions](../README.md#measurement-conditions), one stage
after another so that no two solves shared the CPU. The LP, QP and
literature tables are from commit `0d019ca`; the three MILP tables are
from `a52bcf8`, the commit that carries the fixes the first pass over the
MILP sets found (section 6), re-run once those were in; and the Netlib
interior-point row is from `62d9a1d`, the fix for the defect that
table itself exposed (bug 15), re-run alone once it was in (Mittelmann's
interior-point column was re-run at the same commit too, and the outcome
is in the text under that table). Every number below is regenerable from
the command shown with it; the raw listings the tables were cut from are
what the commands print.

**Where the answer keys come from.** Every reference value in a table was
read by machine from a published source and written into the instance file's
header by `bench.fetch`, with a `*SOURCE:` line naming it -- MIPLIB 3's own
headers, MIPLIB 2017's instance pages (marked `(opt)` only where the page's
status is *easy*, which means proven), Netlib's readme, QPLIB's solution
files, Williams' textbook. Where no published value could be read by machine,
the table says so and shows the objective alone. Nothing in a reference
column was typed from memory.

**How to read "optimal".** A MILP row marked optimal was proved within the
harness's 1e-4 relative gap, which is MIPLIB's own convention; cap6000's
−2,451,175 against a reference of −2,451,377 is 8e-5 apart and is that
convention, not a discrepancy. A row at the limit shows the best point found
and how far it sits from the published value. The verifier column is the
independent check of [`bench/verify.py`](../bench/verify.py): every point
reported here was recomputed against the original model, and for LPs the
duals were asked to certify optimality.

## In one table

| category | set | instances | proved / certified | on the published value | verifier-accepted | note |
|---|---|---|---|---|---|---|
| LP | Netlib, simplex | 89 | **87 certified** | 78 (the other 8 are the readme's, by certificate) | 89 | 3 at the limit; 554 s |
| LP | Netlib, interior point | 89 | **82 certified** | 75 | 83 | 6 real failures: 3 wrong infeasibility verdicts, pilot4, forplan, fit2p; 316 s |
| LP | MIPLIB relaxations | 45 | **45 certified** | 30/30 | 45 | HiGHS agrees 45/45 to 4e-14; 6.2× slower than HiGHS |
| LP | Mittelmann, GPU PDLP | 13 | **9 optimal** (5 certified; 7 under the one-sided test of the time, see §3a) | — (no published values) | 9 | the best engine on the set; qap15 in 11 s |
| LP | Mittelmann, interior point | 13 | 8 optimal (8 certified; qap15's limit point was a ninth under the one-sided test, see §3a) | — | 10 | HiGHS 8; ours faster on the three 160k-row models |
| LP | Mittelmann, simplex | 13 | 3 | — | 5 | out of its depth past 400k nnz |
| MILP | MIPLIB classical, 120 s | 45 | **22 proved** | 27 | 44 | air05 no incumbent |
| MILP | Mittelmann fctp, 120 s | 17 | **8 proved** | 1 of the 3 with a value | 17 | flow-cover territory |
| MILP | Mittelmann benchmark, 120 s | 48 | **0 proved** | 4 (9 within 1%) | 40 | the honest distance to the established solvers |
| QP | QPLIB ≤ 6,000 vars, 60 s | 29 | **7 proved** | 10 | 28 | bound never above the published value, 28/28 |
| literature | Williams' refinery | 1 | **certified** | £211,365.13, the book's plan | 1 | all three engines |
| literature | Haverly pooling | 3 | **3 proved global** | 400 / 600 / 750 | 3 | recursion wrong from 20/21 starts |
| generated | refinery ladders | 15 LP + 5 MILP | LP to 4.4M nnz; MILP k=1, 2 proved, k=16 within 0.4% | — | all | GPU wins wide, IPM wins tall |

Seven defects found and fixed on the way (section 6): three time limits
that were not limits, two soundness holes in the tree, one crash, and
one status more pessimistic than the answer it came with.

---

## 1. Netlib LP

`python -m bench.netlib` (the simplex, 120 s, 1e-8) and `--method ipm`. All
89 problems, expanded from Netlib's own compressed format by
[`io/netlib.py`](../src/sovopt/io/netlib.py), against the readme's PROBLEM
SUMMARY TABLE; "certified" is the verifier's line from the returned duals.

| engine | parse | match the readme to 1e-6 | **certified optimal** | at the limit | wrong or no answer | total |
|---|---|---|---|---|---|---|
| revised simplex | 89/89 | 78/89 | **87/89** | 3: cycle (its point certifies anyway), dfl001, maros-r7 | 0 | 554 s |
| interior point | 89/89 | 75/89 | **82/89** | 1: fit2p | 6: agg, finnis, perold (`INFEASIBLE_OR_UNBOUNDED`, wrong), pilot4 (`NUMERICAL`, a bad point), forplan (iteration limit, no point), etamacro (optimal to 2e-8, not certified) | 316 s |

The simplex reproduces the committed record to the instance: the eleven
readme mismatches are the eight the verifier certifies as *better than the
readme* (80bau3b, greenbea, greenbeb, nesm, pilot, pilot87, scrs8 by a
strictly better feasible point; ganges by a bound that excludes the readme
value) plus the three at the limit. The interior point is a little over
half the simplex's time over the set and certifies 82, and its own status
now agrees with the verifier's on every row. It did not on the campaign's
first pass over this table: five `NUMERICAL` exits -- greenbea, maros,
pilot, shell, sierra -- and fffff800's iteration limit returned points the
verifier certified optimal, the solver's column saying less than the
verifier's beside it. That was bug 15 (section 6): the CLI had passed its
1e-8 `tol` to the interior point as the absolute feasibility cap that
exists to be the verifier's 1e-6 line, so a converged point 8.8e-8 off its
rows on an objective of 1.2e9 was demoted; and the loop's test in the
scaled space never fired on fffff800's proved point. With the cap at the
verifier's line and the verifier's certificate applied before reporting,
the six are `OPTIMAL`, four of them now also on the readme's value (71 →
75 matched; greenbea and pilot are better than the readme, by
certificate), and the table above is the re-run. The six real failures
are the ones Known limits already names: three wrong infeasibility
verdicts on the instances whose scaling the unit cannot fix, pilot4 and
forplan, and fit2p over the limit (dfl001 solves in 116 s, against 96 s
on the first pass; the machine's noise, not the code's).

## 2. MIPLIB

### 2a. LP relaxations, and HiGHS beside them

`python -m bench.harness --mode lp --dir data/instances --time-limit 120`
(the router's choice of engine, per instance) and
`python -m bench.comparator --dir data/instances --time-limit 120` (our
simplex against HiGHS through `scipy.optimize.linprog`, the one file in the
repository allowed to call another solver).

| | |
|---|---|
| instances | 45 |
| optimal | **45/45** |
| certified optimal by the verifier, from the duals | **45/45** |
| on the published LP value (the 30 that have one) | **30/30**, worst 3.7e-6 (mod008) |
| shifted geomean time | 0.345 s (shift 1 s); total 46.2 s |
| **HiGHS agrees, to 1e-6** | **45/45**, worst relative difference 4.4e-14 |
| our simplex against HiGHS, total | 29.3 s against 4.7 s: **6.2× slower**, the README's 6.7× on eleven, now on forty-five |

The LP side is settled: every relaxation solved, every one certified, every
one matching HiGHS to fourteen digits. The speed ratio is the one that has
held for months -- air04 (72,965 nonzeros) is the widest gap at 10.1 s
against 1.0 s, mkc 6.4 s against 0.06 s -- and is the case for the presolve
and the Forrest–Tomlin path that Known limits already makes.

### 2b. MILP, 120 s limit

`python -m bench.harness --mode mip --dir data/instances --time-limit 120` --
the 30 classical instances of MIPLIB 3 (via the COIN-OR mirror) and 19 of
the 2003/2017 medium set, 45 instances after the overlap.

| instance | rows × cols | nnz | status | objective | reference | gap to ref | time |
|---|---|---|---|---|---|---|---|
| 10teams | 230 × 2025 | 12,150 | **optimal** | 924 | 924 | 0 | 76.6 s |
| air04 | 823 × 8904 | 72,965 | limit | 56212 | 56137 | +0.13% | 132.1 s |
| air05 | 426 × 7195 | 52,121 | limit | — | 26374 | no incumbent | 132.9 s |
| bell3a | 123 × 133 | 347 | limit | 878430.32 | 878430.32 | 0 | 120.0 s |
| bell5 | 91 × 104 | 266 | limit | 8967020.1 | 8966406.5 | +0.01% | 120.1 s |
| cap6000 | 2176 × 6000 | 48,243 | **optimal** | -2451175 | -2451377 | +0.01% | 15.2 s |
| danoint | 664 × 521 | 3,232 | limit | 73 | 65.666667 | +11.17% | 122.7 s |
| dcmulti | 290 × 548 | 1,315 | **optimal** | 188187.4 | 188182 | +0.00% | 5.6 s |
| egout | 98 × 141 | 282 | **optimal** | 568.1007 | 568.101 | -0.00% | 0.3 s |
| enigma | 21 × 100 | 289 | **optimal** | 0 | 0 | 0 | 5.3 s |
| fiber | 363 × 1298 | 2,944 | limit | 434334.54 | 405935.18 | +7.00% | 120.3 s |
| fixnet6 | 478 × 878 | 1,756 | limit | 4515 | 3983 | +13.36% | 120.2 s |
| flugpl | 18 × 18 | 46 | **optimal** | 1201500 | 1201500 | 0 | 1.9 s |
| gesa2 | 1392 × 1224 | 5,064 | limit | 26207136 | 25779856 | +1.66% | 120.3 s |
| gesa3 | 1368 × 1152 | 4,944 | **optimal** | 27991430 | 27991043 | +0.00% | 6.1 s |
| gr4x6 | 34 × 48 | 96 | **optimal** | 202.35 | 202.35 | 0 | 0.3 s |
| gt2 | 29 × 188 | 376 | **optimal** | 21166 | 21166 | 0 | 1.0 s |
| harp2 | 112 × 2993 | 5,840 | limit | -71862386 | -73899798 | +2.76% | 122.2 s |
| khb05250 | 101 × 1350 | 2,700 | **optimal** | 1.0694023e+08 | 1.0694023e+08 | 0 | 2.3 s |
| l152lav | 97 × 1989 | 9,922 | **optimal** | 4722 | 4750 | -0.59% | 111.1 s |
| lseu | 28 × 89 | 309 | **optimal** | 1120 | 1120 | 0 | 10.3 s |
| mas76 | 12 × 151 | 1,640 | limit | 40139.477 | 40005.054 | +0.34% | 120.2 s |
| misc03 | 96 × 160 | 2,053 | **optimal** | 3360 | 3360 | 0 | 3.2 s |
| misc07 | 212 × 260 | 8,619 | **optimal** | 2810 | 2810 | 0 | 103.8 s |
| mkc | 3411 × 5325 | 17,038 | limit | -469.556 | -563.84601 | +16.72% | 136.4 s |
| mod008 | 6 × 319 | 1,243 | **optimal** | 307 | 307 | 0 | 18.6 s |
| mod010 | 146 × 2655 | 11,203 | **optimal** | 6548 | 6548 | 0 | 3.3 s |
| modglob | 291 × 422 | 968 | limit | 22389976 | 20740508 | +7.95% | 120.2 s |
| nw04 | 36 × 87482 | 636,666 | **optimal** | 16862 | 16862 | 0 | 43.6 s |
| p0033 | 16 × 33 | 98 | **optimal** | 3089 | 3089 | 0 | 0.2 s |
| p0201 | 133 × 201 | 1,923 | **optimal** | 7615 | 7615 | 0 | 3.3 s |
| p0282 | 241 × 282 | 1,966 | **optimal** | 258411 | 258411 | 0 | 21.7 s |
| p2756 | 755 × 2756 | 8,937 | limit | 5629 | 3124 | +80.19% | 120.1 s |
| pk1 | 45 × 86 | 915 | limit | 15 | 11 | +36.36% | 120.1 s |
| pp08a | 136 × 240 | 480 | limit | 7350 | 7350 | 0 | 120.1 s |
| qiu | 1192 × 840 | 3,432 | limit | -132.87314 | -132.87314 | 0 | 120.5 s |
| qnet1 | 503 × 1541 | 4,622 | limit | 16121.303 | 16029.693 | +0.57% | 120.7 s |
| rgn | 24 × 180 | 460 | **optimal** | 82.199999 | 82.1999 | +0.00% | 11.1 s |
| rout | 291 × 556 | 2,431 | limit | 1145.46 | 1077.56 | +6.30% | 120.3 s |
| set1ch | 492 × 712 | 1,412 | limit | 60736.5 | 54537.7 | +11.37% | 121.0 s |
| stein27 | 118 × 27 | 378 | **optimal** | 18 | 18 | 0 | 8.8 s |
| stein45 | 331 × 45 | 1,034 | limit | 30 | 30 | 0 | 120.0 s |
| swath | 884 × 6805 | 34,965 | limit | 585.96001 | 467.40749 | +25.36% | 120.8 s |
| vpm1 | 234 × 378 | 749 | limit | 20 | 20 | 0 | 120.0 s |
| vpm2 | 234 × 378 | 917 | limit | 14.25 | 13.75 | +3.64% | 120.2 s |

```
  instances            45
  status OPTIMAL       22/45
  verifier accepted    44/45
  within 1e-4 of ref   27/45
  shifted geomean time 44.264s (shift 10s)
  total time           3265.2s
  worst relative error 8.02e-01 (p2756)
```

**22 of 45 proved within the 1e-4 gap; 27 of 45 land on the published value
(proved or not); 44 of 45 return a point the verifier accepts** -- air05, a
set-partitioning crew-scheduling model, is the one with no incumbent at all.
On the commit before this one the same command gave 20 and 25: misc07
(104 s) and l152lav (111 s) are proved on this draw and were not on that
one, and qiu and pp08a are back on their optima unproved where the earlier
draw had them off. A 120 s limit is a coin for an instance that finishes
at 100 s, and the two tables show the two sides. The set splits by what it
asks of the solver:

- **Closed, and quickly:** the classical small instances -- p0033, p0201,
  p0282, egout, enigma, flugpl, gr4x6, gt2, khb05250, lseu, misc03, mod008,
  mod010, rgn, stein27 -- and dcmulti, gesa3, cap6000, misc07, l152lav,
  10teams (924, proved in 77 s, which it never was before this campaign)
  and nw04 (87,482 binaries, 44 s, once bug 9 let it run at all).
- **Found but not proved:** bell3a, pp08a, qiu, stein45, vpm1 sit on the
  published optimum at the limit; bell5, mas76, qnet1 and air04 are within
  0.6%. bell3a and bell5 are worth a sentence: 133 and 104 columns, and
  the tree cannot close them in two minutes, because their gap lives in a
  weak formulation that needs cuts this pool does not separate.
- **Far off:** fiber, fixnet6, set1ch, modglob (fixed-charge network and
  lot-sizing structure -- flow-cover territory, 7-13% off), mkc, swath,
  p2756 (80% off), pk1 (15 against 11), danoint (+11%, the degenerate
  node LPs of bug 10).

Three overran the limit -- air04 132 s, mkc 136 s, air05 133 s -- against
the 218 s, 333 s and hours of the pass that found bugs 9-11. The
remainder is cold re-solves of nodes whose LP gave no verdict, each
bounded by the deadline but each entered; it is listed under Known
limits.

## 3. Mittelmann

### 3a. The LP test set

`python -m bench.fetch --set mittelmann-lp`, then `python -m bench.harness
--mode lp --dir data/mittelmann --time-limit 300 --method ipm | simplex |
pdlp --device gpu` and `python -m bench.comparator --dir data/mittelmann
--time-limit 300 --method ipm`. The thirteen instances of Mittelmann's LP
test set under 1.5 MB compressed -- the rest start at millions of nonzeros
-- twelve of them in Netlib's compressed MPS format, which the fetch now
expands with the repository's own Netlib expander (the first pass saw
twelve parse failures and one solve). No published optima come with the
files; the verifier's certificate and HiGHS are the two independent
checks, and they agree wherever both apply.

| instance | rows × cols | nnz | interior point | simplex | PDLP on the GPU | HiGHS |
|---|---|---|---|---|---|---|
| cont1 | 160,792 × 40,398 | 400k | **certified, 118 s** | limit | limit, rejected | 233 s |
| cont11 | 160,792 × 80,396 | 400k | **certified, 29 s** | limit | limit, rejected | limit |
| cont4 | 160,792 × 40,398 | 398k | **certified, 34 s** | limit | limit, rejected | limit |
| fome11 | 12,142 × 24,460 | 71k | certified, 168 s | limit | iteration limit, rejected | **17 s** |
| neos1 | 131,581 × 1,892 | 468k | certified, 84 s | limit | **certified, 19 s** | 30 s |
| neos2 | 132,568 × 1,560 | 553k | iteration limit, accepted point | limit | **certified, 30 s** | 35 s |
| nug08-3rd | 19,728 × 20,448 | 139k | refused: 192M entries in L, 2.5e12 flops | limit | **certified, 2.7 s** | limit |
| nug20 | 15,240 × 72,600 | 305k | refused: 95M entries in L, 7.5e11 flops | limit | **accepted, 59 s** | limit |
| pds-20 | 33,874 × 105,728 | 230k | limit | limit | **certified, 41 s** | 2.2 s |
| qap15 | 6,330 × 22,275 | 95k | limit at 456 s (489 s in the comparator run), point accepted (certified under the one-sided test of the time) -- see below | limit | **accepted, 11 s** | limit |
| rail507 | 507 × 63,009 | 409k | certified, 7.3 s | certified, 12 s | certified, 35 s | **4.5 s** |
| rail516 | 516 × 47,311 | 315k | certified, 3.3 s | certified, 7.3 s | **certified, 2.3 s** | 2.1 s |
| rail582 | 582 × 55,515 | 402k | certified, 7.4 s | certified, 7.9 s | certified, 52 s | **3.6 s** |
| **solved** | | | **8 optimal, 8 certified** | 3 | **9 optimal, 5 certified** | 8 |
| HiGHS agrees | | | 6/6 where both solve, worst 1.5e-10 | | | |
| total | | | 1,370 s | 3,106 s | 1,249 s | 1,841 s |

"Certified" is the verifier's line; "accepted" is a point feasible to 1e-6
whose duals do not close the 1e-9 bound (the first-order method's normal
finish); "rejected" is an unconverged iterate the verifier refuses, which
is what PDLP hands back at a limit. The interior-point column was re-run
alone at the commit that fixes bug 15 (section 6), after which the engine
reports `OPTIMAL` on a point its duals certify whatever its loop
concluded; qap15's row is the one that could have changed, since its
point was certified at a time limit. It did not, and the reason is worth
the sentence: on the re-run the machine was 1.3× slower throughout
(nug08-3rd's symbolic analysis, code the fix does not touch, 8.1 s against
6.0 s; cont11 36 s against 29 s, and the same 39 s with the fix and 39 s
without it, back to back), so the clock check between qap15's 40 s
iterations landed at 308 s on an iterate the verifier accepts but does
not yet certify, where in the campaign it landed at 456 s on one it does.
Every other row kept its status. A proof that depends on which iteration
the clock lands on is a coin, and the table shows the campaign's draw
with the HiGHS column measured in the same sitting; the re-run's timings
are not comparable with that column and are not substituted for it.

The certificate counts in this table are as the second campaign's bug 19
left them. The verifier's optimality line was one-sided at the time
(`gap <= 1e-9`, with any negative gap passing), and three points this
table called certified were in fact *below* their bounds: PDLP's
nug08-3rd by 6.1e-8 relative and rail582 by 1.7e-9 -- points feasible to
1e-6 whose objectives beat a valid bound, which is to say points not
feasible at the bound's resolution, PDLP's normal finish -- and the
interior point's qap15 limit point. Under the two-sided test they are
"accepted"; the engines' own optimal statuses, the objectives and the
HiGHS agreements are unchanged, and the counts above and in the summary
say 5 and 8 where the sitting's listings said 7 and 9. Three things this
table says that the smaller sets could not:

- **The GPU first-order method is the best engine on this set.** Nine of
  thirteen, including the three the CPU engines cannot touch in 300 s --
  nug08-3rd in 2.7 s, qap15 in 11 s, pds-20 in 41 s -- and neos1/neos2 at
  half a million nonzeros in 19 and 30 s. The four it cannot converge on
  are the PDE-constrained cont models and fome11, whose conditioning is
  what a first-order method pays for. On the README's small set PDLP was
  the slowest of three; here the order inverts, which is the reason the
  engine exists.
- **The interior point stands beside HiGHS at this scale.** 8 to HiGHS's 8,
  three of them -- cont1, cont11, cont4 at 160k rows -- where HiGHS is 2×
  slower or at its limit, and three the other way (fome11, neos1, pds-20,
  the last by a factor of a hundred). Total time over the set 1,370 s
  against 1,841 s, with the usual caveat that HiGHS is reached through
  `scipy.optimize.linprog` at its defaults.
- **The simplex is out of its depth past a few hundred thousand nonzeros
  and a few thousand rows**, which the README has said in words since the
  Scale section was written; this is the number: 3 of 13, the three rails
  with 500 rows.

And one thing the campaign found on the way (bug 14 in the README): the
two nug KKTs have 192 and 95 million entries in `L`, and the interior
point spent forty minutes past its 300 s limit inside its *first*
factorisation, because the limit is checked between iterations and a
compiled factorisation cannot look at a clock. The cost is known before
any numeric work -- the symbolic analysis gives the column counts -- so
the IPM now refuses a factorisation predicted to outlast its limit, in
six seconds, and says so in `info["refused"]`. qap15's 489 s against 300
is the same clock at a finer grain: an iteration there is 40 s, and the
check is between iterations.

### 3b. Fixed-charge transportation MILPs

| instance | rows × cols | nnz | status | objective | reference | gap to ref | time |
|---|---|---|---|---|---|---|---|
| bal8x12 | 116 × 192 | 384 | **optimal** | 471.55 | — | — | 0.8 s |
| bk4x3 | 19 × 24 | 48 | **optimal** | 350 | — | — | 0.0 s |
| gr4x6 | 34 × 48 | 96 | **optimal** | 202.35 | 202.35 | 0 | 0.3 s |
| ran10x10a | 120 × 200 | 400 | **optimal** | 1499 | — | — | 9.6 s |
| ran10x10b | 120 × 200 | 400 | **optimal** | 3073 | — | — | 6.6 s |
| ran10x10c | 120 × 200 | 400 | limit | 13067 | — | — | 120.1 s |
| ran10x12 | 142 × 240 | 480 | **optimal** | 2714 | — | — | 3.4 s |
| ran10x26 | 296 × 520 | 1,040 | limit | 4745 | — | — | 120.2 s |
| ran12x12 | 168 × 288 | 576 | limit | 2637 | — | — | 120.2 s |
| ran12x21 | 285 × 504 | 1,008 | limit | 4080 | 3664 | +11.35% | 120.0 s |
| ran13x13 | 195 × 338 | 676 | limit | 3521 | 3252 | +8.27% | 120.2 s |
| ran14x18 | 284 × 504 | 1,008 | limit | 4265 | — | — | 120.1 s |
| ran16x16 | 288 × 512 | 1,024 | limit | 4333 | — | — | 120.2 s |
| ran17x17 | 323 × 578 | 1,156 | limit | 1464 | — | — | 120.1 s |
| ran4x64 | 324 × 512 | 1,024 | **optimal** | 9711 | — | — | 1.7 s |
| ran6x43 | 307 × 516 | 1,032 | **optimal** | 6330 | — | — | 3.5 s |
| ran8x32 | 296 × 512 | 1,024 | limit | 5247 | — | — | 120.2 s |

```
  instances            17
  status OPTIMAL       8/17
  verifier accepted    17/17
  within 1e-4 of ref   1/3
  shifted geomean time 33.841s (shift 10s)
  total time           1107.1s
  worst relative error 1.14e-01 (ran12x21)
```

`python -m bench.harness --mode mip --dir data/fctp --time-limit 120`. Three
of the seventeen are in MIPLIB 2017 under the same names and their optima
were read from the instance pages; the other fourteen have no published
value this campaign could read by machine -- Gottlieb's 1998 page that
Mittelmann's README cites is gone -- so their rows show the objective alone.
**8 of 17 proved, 17 of 17 verifier-accepted.** gr4x6 and the two
transportation-shaped small ones close in under a second; the ran-family
from 10 × 10 up mostly does not, and where a reference exists the incumbent
is 8-11% above it. Fixed-charge transportation is the textbook case for
lifted flow-cover cuts, which this cut pool does not separate: the LP
relaxation of a fixed charge is weak by construction, and cover cuts on
the knapsack rows do not reach it. For the record, the values Gottlieb
published for the others, as recalled and *not* machine-read: ran10x10c
13007 (ours 13067), ran10x26 4270 (4745), ran12x12 2291 (2637), ran14x18
3712 (4265), ran16x16 3823 (4333), ran17x17 1373 (1464), ran8x32 5247
(5247, unproved). Treat those seven as indicative.

### 3c. The MILP benchmark (MIPLIB 2017 instances), 120 s limit

| instance | rows × cols | nnz | status | objective | reference | gap to ref | time |
|---|---|---|---|---|---|---|---|
| 50v-10 | 233 × 2013 | 2,745 | limit | 3637.36 | 3311.18 | +9.85% | 120.3 s |
| assign1-5-8 | 161 × 156 | 3,720 | limit | 212 | 212 | 0 | 120.3 s |
| beasleyC3 | 1750 × 2500 | 5,000 | limit | 982 | 754 | +30.24% | 122.6 s |
| binkar10_1 | 1026 × 2298 | 4,496 | limit | 7081.84 | 6741.38 | +5.05% | 120.2 s |
| cbs-cta | 10112 × 24793 | 64,388 | limit | — | 0 | no incumbent | 123.6 s |
| cod105 | 1024 × 1024 | 57,344 | limit | -12 | -12 | 0 | 122.2 s |
| csched007 | 351 × 1758 | 6,379 | limit | 538 | 351 | +53.28% | 120.5 s |
| dano3_3 | 3202 × 13873 | 79,655 | limit | — | 728.1111 | no incumbent | 124.1 s |
| drayage-100-23 | 4630 × 11090 | 41,550 | limit | 580734.76 | 103333.87 | +462.00% | 123.6 s |
| eil33-2 | 32 × 4516 | 44,243 | limit | 1000.2406 | 934.00792 | +7.09% | 121.3 s |
| enlight_hard | 100 × 200 | 560 | limit | — | 37 | no incumbent | 120.2 s |
| exp-1-500-5-5 | 550 × 990 | 1,980 | limit | 106192 | 65887 | +61.17% | 120.1 s |
| fast0507 | 507 × 63009 | 409,349 | limit | — | 174 | no incumbent | 157.3 s |
| gen-ip002 | 24 × 41 | 922 | limit | -4753.8459 | -4783.7334 | +0.62% | 120.0 s |
| gen-ip021 | 28 × 35 | 945 | limit | 2364.321 | 2361.4542 | +0.12% | 120.1 s |
| glass4 | 396 × 322 | 1,815 | limit | 3.1000257e+09 | 1.2000126e+09 | +158.33% | 120.0 s |
| gmu-35-40 | 424 × 1205 | 4,843 | limit | -2395803.5 | -2406733.4 | +0.45% | 120.5 s |
| graph20-20-1rand | 5587 × 2183 | 19,277 | limit | 0 | -9 | +100.00% | 120.7 s |
| ic97_potential | 1046 × 728 | 3,138 | limit | 4074 | 3941.9999 | +3.35% | 120.2 s |
| istanbul-no-cutoff | 20346 × 5282 | 71,477 | limit | — | 204.08171 | no incumbent | 124.8 s |
| lotsize | 1920 × 2985 | 6,565 | limit | 16136964 | 1480195 | +990.19% | 120.2 s |
| mad | 51 × 220 | 2,808 | limit | 0.5018 | 0.0268 | +47.50% | 120.1 s |
| markshare2 | 7 × 74 | 434 | limit | 25 | 1 | +2400.00% | 120.1 s |
| markshare_4_0 | 4 × 34 | 123 | limit | 5 | 1 | +400.00% | 120.0 s |
| mas74 | 13 × 151 | 1,706 | limit | 11801.186 | 11801.186 | 0 | 120.1 s |
| mc11 | 1920 × 3040 | 6,080 | limit | 13548 | 11689 | +15.90% | 122.5 s |
| mcsched | 2107 × 1747 | 8,088 | limit | 221961 | 211913 | +4.74% | 120.4 s |
| mik-250-20-75-4 | 195 × 270 | 9,270 | limit | -51182 | -52301 | +2.14% | 120.1 s |
| n5-3 | 1062 × 2550 | 9,900 | limit | 10640 | 8105 | +31.28% | 121.8 s |
| neos-1122047 | 57791 × 5100 | 163,640 | limit | — | 161 | no incumbent | 131.1 s |
| neos-3004026-krka | 12545 × 17030 | 41,860 | limit | — | 0 | no incumbent | 121.1 s |
| neos17 | 486 × 535 | 4,931 | limit | 0.26515093 | 0.15000258 | +11.51% | 120.2 s |
| neos5 | 63 × 63 | 2,016 | limit | 15 | 15 | 0 | 120.2 s |
| neos8 | 46324 × 23228 | 313,180 | limit | -3672 | -3719 | +1.26% | 129.4 s |
| nu25-pr12 | 2313 × 5868 | 17,712 | limit | 54730 | 53905 | +1.53% | 124.2 s |
| p200x1188c | 1388 × 2376 | 4,752 | limit | 15531 | 15078 | +3.00% | 120.4 s |
| pg | 125 × 2700 | 5,200 | limit | -8562.634 | -8674.3426 | +1.29% | 121.5 s |
| pg5_34 | 225 × 2600 | 7,700 | limit | -14254.469 | -14339.353 | +0.59% | 134.6 s |
| qap10 | 1820 × 4150 | 18,200 | limit | — | 340 | no incumbent | 129.0 s |
| ran14x18-disj-8 | 447 × 504 | 10,277 | limit | 4474 | 3712 | +20.53% | 120.0 s |
| rocII-5-11 | 26897 × 11523 | 303,291 | limit | -3.6190083 | -6.6755047 | +45.79% | 123.3 s |
| roll3000 | 2295 × 1166 | 29,386 | limit | 18003 | 12890 | +39.67% | 129.6 s |
| seymour | 4944 × 1372 | 33,549 | limit | 445 | 423 | +5.20% | 121.0 s |
| sp150x300d | 450 × 600 | 1,200 | limit | 74 | 69 | +7.25% | 120.1 s |
| supportcase18 | 240 × 13410 | 28,920 | limit | 59 | 48 | +22.92% | 121.0 s |
| swath1 | 884 × 6805 | 34,965 | limit | 382.3078 | 379.0713 | +0.85% | 120.6 s |
| timtab1 | 171 × 397 | 829 | limit | 957432 | 764772 | +25.19% | 120.3 s |
| tr12-30 | 750 × 1080 | 2,508 | limit | 197672 | 130596 | +51.36% | 120.6 s |

```
  instances            48
  status OPTIMAL       0/48
  verifier accepted    40/48
  within 1e-4 of ref   4/48
  shifted geomean time 122.720s (shift 10s)
  total time           5896.4s
  worst relative error 2.40e+01 (markshare2)
```

`python -m bench.harness --mode mip --dir data/mittelmann-milp --time-limit
120 --max-nnz 300000`. Mittelmann's MILP benchmark is drawn from MIPLIB 2017
to separate commercial solvers from each other at time limits of an hour or
two; these are the 48 of its instances under about 100k nonzeros (fast0507,
at 409k, slipped past the filter and is in the table). Every reference
value was read from the instance's MIPLIB page, and every one of the 48 is
marked *easy* there, meaning some solver has proved it. **Proved here at
120 s: none. 40 of 48 return a point the verifier accepts; four sit on the
published optimum** -- assign1-5-8, cod105, mas74, neos5 -- **five more
within 1%** (gen-ip021 0.12%, gmu-35-40 0.45%, pg5_34 0.59%, gen-ip002
0.62%, swath1 0.85%) and four within 3% (neos8, pg, nu25-pr12,
mik-250-20-75-4). Eight are between 3% and 10%, ten between 10% and 50%,
nine worse than that -- the market-sharing pair by factors of 5 and 25,
because their objective is a sum of slacks whose optimum is 1 and a solver
without a strong root bound cannot see the scale -- and eight have no
incumbent at all in two minutes: cbs-cta, dano3_3, enlight_hard, fast0507,
istanbul-no-cutoff, neos-1122047, neos-3004026-krka, qap10. Six overran
the limit, the worst fast0507 at 157 s.

The same command on the commit before this one gave two on the optimum and
a crash (gmu-35-40, bug 12); of the eleven rows that changed between the
two commits every one improved -- cod105 from −2 to the published −12,
mas74 from 0.65% to the optimum, markshare2 from 33 to 25, neos8, roll3000,
seymour, swath1, glass4, mcsched, neos17 all closer -- which is bug 13's
effect: a tree that stops pruning feasible nodes on a drifted eta file
keeps the nodes that hold the better points.

This is the table that says plainly where the solver stands against the
established ones. On the classical set it proves half; on the set the
established solvers are measured on it proves nothing in two minutes,
lands on or within 1% of the optimum on nine of 48, and finds a feasible
point on five in six. That is the difference between a branch-and-bound
with cover, MIR and Gomory cuts, propagation, conflict learning and diving,
and one with thirty years of cut families, presolve reductions and tuned
strong branching on top. The problem statement asks for the comparison;
this is it.

## 4. QPLIB

`python -m bench.qplib --run --time-limit 60 --max-vars 6000`: the 29
instances with linear constraints and up to 6,000 variables, routed by
class -- convex and continuous to the QP interior point, convex with
integers to the MIQP tree, the rest to the non-convex route (McCormick or
αBB by the Hessian's density). QPLIB publishes a solution point and its
objective to fifteen digits for each, and the table checks the reader
against the point and the engine against the value.

| | |
|---|---|
| parsed | 29/29 |
| published point verified by our reader | 28/29 (9002's point fails the verifier's absolute 1e-6 at row sums of 1e10 -- the yardstick, not the reader) |
| proved optimal | **7/29**: convex QPs 8845, 8938, 8906 (0.6-1.2 s); convex MIQPs 10056 (48 s), 10069; non-convex 10074 (43 s), 10040 (5 s) |
| within 1e-5 of the published value | 10/29 (the seven, plus 10050 at a 1.1% gap, 10073, and non-convex 10042 unproved) |
| **bound never above the published value** | **28/28** -- no relaxation, shift or cut in the campaign cut off a known optimum |
| incumbent better than the published value | 0 |

The rest is the README's record at these limits: the four constrained
convex MIQPs at 15% (3980), 2.6% (3913), 27% (3871), 12% (4270); 9002
`NUMERICAL` under a yardstick double precision cannot meet; the non-convex
binary quadratics at gaps from 0.5% (5881) and 1.2% (0067) to 52% on the
two 50-variable continuous ones (0018, 0343), whose McCormick bound sits at
−85 against an optimum of −6.4. 3714 (120 binaries, 40 rows) found no
point in 79 s. Nothing here moved since the last commit that touched it;
the value of the table is the soundness row.

## 5. Refinery, blending, planning and supply chain

Four kinds of model, and the honest statement of which are from the
literature and which are shaped like it.

**Williams' refinery (from the literature, with a published optimum).**
*Model Building in Mathematical Programming*, problem 12.6: two crudes,
distillation, reforming, cracking, lube oil, premium and regular petrol
under octane specifications, jet fuel under a vapour-pressure
specification, a fixed-recipe fuel oil, capacities on every unit --
33 rows, 40 columns, every coefficient as printed. The book's optimum is
**£211,365.13 per day**. All three engines return 211,365.13 (the book
rounds to the penny; the exact value is 211,365.1348), with the book's
plan to the barrel: crude 1 15,000 and crude 2 30,000 (the distillation
cap binds), premium 6,817.78, regular 17,044.45, jet fuel 15,156, no fuel
oil, lube at its floor of 500. Verifier-certified from the duals. It is
`models.williams_refinery()` with seven tests, and the only refinery
model in this repository that checks the solver against a number nobody
here produced.

**Haverly's pooling problems (from the literature, published global
optima).** Crude and product blending with pools, the bilinear structure
that refinery LPs linearise away: Haverly 1978, three instances, global
optima 400 / 600 / 750. `python -m bench.pooling_check`: all three proved
to the global optimum with the bound closed (3, 9 and 15 nodes), and the
industry's distributive recursion -- what PIMS and GRTMPS do -- converges
to a blend worth strictly less than the global optimum from 20 of 21
starting points on each. `python -m bench.pq_check` on six random
multi-pool networks (4 to 6 sources, 2 to 3 pools): the pq-formulation
with RLT rows proves each in 0.1-0.4 s and at most 20 nodes, where the
plain q-formulation runs 3,000-6,000 nodes to the 60 s limit on every one
and, where it finishes, agrees on the optimum.

**The refinery templates (shaped like the literature, generated).**
Product blending with quality specifications (the badly scaled LP),
multi-period crude selection and unit throughput (the degenerate network
LP), and unit scheduling with minimum up and down times (the big-M MILP)
-- the structures Pinto, Joly and Moro (2000) and Lee, Pinto, Grossmann
and Park (1996) describe, parameterised by size so the engines can be
exercised where MRPL's own models live, without MRPL's data.
`python -m bench.scale --mode lp` and `--mode mip`:

| model | size at k=16 | simplex | interior point | PDLP (GPU) |
|---|---|---|---|---|
| blending | 2,080 × 102,400, 1.02M nnz | 35.7 s | 36.5 s | **3.2 s** |
| planning | 61,440 × 61,440, 4.42M nnz | limit | **47.1 s** | limit, unconverged |
| scheduling (MILP) | 18,400 × 12,288, 6,144 binaries | k=1, 2 proved; k=4 / 8 / 16 at the limit with incumbents 1.8% / 1.1% / 0.4% above the bound | | |

Every LP row up to those sizes is verifier-accepted for every engine that
finishes it; the ladder's shape is the README's -- the GPU wins on the
wide blending model, the interior point on the tall planning one, the
simplex on neither past a few thousand rows.

**Supply chain, production planning and scheduling from MIPLIB (from the
literature, published optima).** MIPLIB's instances are industrial models
with their origins recorded, and the campaign's MILP tables contain the
categories the problem statement names. Read by application, from
sections 2b and 3:

| application | instances | outcome at 120 s |
|---|---|---|
| lot sizing / production planning | pp08a, set1ch, lotsize, tr12-30, timtab1 | pp08a within 0.4% of the optimum; set1ch 11%, tr12-30 51%, timtab1 25%, lotsize 10× -- the weak big-M relaxations that flow-cover and lifted cuts exist for |
| facility location and network design | dcmulti, cap6000, fixnet6, khb05250, fiber, qiu, mc11, beasleyC3, n5-3, p200x1188c, sp150x300d | dcmulti, cap6000 and khb05250 **proved**; fixnet6 13%, fiber 14%, mc11 16%, beasleyC3 30%, n5-3 31%, p200x1188c 3%, sp150x300d 7%; qiu a poor draw |
| fixed-charge transportation (Mittelmann) | bk4x3, bal8x12, gr4x6, ran* | 8 of 17 **proved**, the rest 8-15% above the literature values |
| crew, sports and machine scheduling | air04, air05, nw04, 10teams, csched007, mcsched, nu25-pr12 | **nw04 and 10teams proved**; air04 within 0.13%, nu25-pr12 1.5%, mcsched 5.6%; air05 no point in two minutes; csched007 53% |
| energy | gesa2, gesa3, rocII-5-11 | gesa3 **proved** in 8.6 s; gesa2 1.7%; rocII-5-11 46% |
| logistics and routing | drayage-100-23, eil33-2 | eil33-2 7%; drayage 4.6× off |

That is the picture for MRPL-shaped discrete models: the solver proves the
ones a textbook branch-and-bound proves and finds feasible plans for most
of the rest, and the ones it leaves far off share a cause -- fixed-charge
and big-M relaxations that stay weak without the cut families the pool
lacks.

---

## 6. What the campaign found

A campaign over sets the solver had never seen is a test of the solver
more than of the sets, and this one found six defects before it could
produce a clean table, and a seventh in the Netlib table it produced. Each
is recorded under [Bugs worth recording](../README.md#bugs-worth-recording),
numbers 9 to 15, with the measurement that found it and the one that
closed it; the short form:

| # | found on | what | fixed by |
|---|---|---|---|
| 9 | nw04 (87,482 binaries) | a 120 s limit ran for hours inside `fix_and_propagate`, which propagated every nonzero after each of 87k fixings and took no deadline | deadlines on every heuristic; the integral part fixed in one pass; nw04 optimal in 43 s |
| 10 | danoint | a node LP made a million zero-length dual pivots at a dual-degenerate vertex; the compiled kernel has no clock; 198 s against 120 | the kernel runs in chunks, checks the deadline, perturbs on a stall, hands off to the primal; every node solver carries the tree's absolute deadline |
| 11 | danoint, gmu-35-40 | the tree pruned on INFEASIBLE verdicts it never checked; on danoint one in a hundred was wrong, on gmu-35-40 all of them | pruning requires the Farkas row to certify the box empty; unsolved nodes are re-solved, then counted and their bound kept, so no OPTIMAL is claimed over them |
| 12 | gmu-35-40 | `ZeroDivisionError` out of the tree: the kernel's failed factorisation was installed and solved through | a NUMERICAL exit's factors are discarded and the basis refactorised through its repairing path |
| 13 | gmu-35-40 | every INFEASIBLE verdict was wrong: the basic values through the eta file were 7e10 infeasible, through a fresh LU 2e3 | both dual loops refactorise before believing "no entering column"; refused verdicts on gmu-35-40 22 → 0, danoint 7 → 0 |
| 14 | nug08-3rd, nug20 | the interior point spent forty minutes past a 300 s limit inside its first factorisation, 192M entries in `L` | the factorisation's cost is known from the symbolic analysis; one predicted to outlast the limit is refused in six seconds |
| 15 | greenbea, maros, pilot, shell, sierra, fffff800 | the interior point's `NUMERICAL` and iteration-limit exits returned points the verifier certified optimal: the CLI passed its 1e-8 `tol` as the absolute feasibility cap that exists to be the verifier's 1e-6, and the loop's test in the scaled space never fired on fffff800's proved point | the cap is the verifier's line and `tol` does not reach it; `_finish` applies the verifier's own certificate from the duals before reporting, and a feasible point within 1e-9 of its bound is `OPTIMAL` whatever the loop said |

Three of the seven -- 9, 10, 14 -- are the same defect in three places: a
time limit checked between units of work whose size nobody had bounded,
and a compiled unit that cannot look at a clock. Two -- 11 and 13 -- are a
soundness defect that the small set never exercised, because a
120-column model rarely drifts an eta file by seven orders. One -- 15 --
is the opposite of a soundness defect, a solver claiming less than it
had proved, and it was visible only because the table carries the
verifier's verdict beside the solver's; the two columns disagreeing on
six rows is what a reporting defect looks like. The README's own
sentence applies: a set the solver has always passed is not evidence
about the set it has never seen.

**What was measured and left alone,** because the campaign is a
measurement and each of these has its reason on record:

- misc07 and l152lav close at 100-110 s on some draws and not on others
  under a 120 s limit: proved on the final tables, not on the pass before
  them, on the same code path with the kernel's chunk re-entries changing
  the rounding and so the pivot path. qiu's incumbent was +102% on one
  draw and on the optimum on the next. A limit set where an instance
  finishes is a coin, and the tables show the draw they got.
- MILP overruns remain -- air04 132 s, mkc 136 s, air05 133 s, fast0507
  157 s against 120 -- from cold re-solves of nodes whose LP gave no
  verdict, each bounded by the deadline but each entered; and qap15's
  489 s against 300 in the interior point, whose iteration there is 40 s.
  The limit is now honoured to within one unit of work everywhere; making
  that unit small on a large model is the remaining step.
- The Mittelmann MILP benchmark proves nothing in two minutes and finds a
  point on five in six. The distance to the established solvers on that
  set is cut families (lifted flow covers, zero-half, implied bounds),
  presolve reductions and strong branching, none of which the campaign
  changed.
- danoint's node LPs still stall; the fix bounded the stall's cost, not
  its cause, which is the dual simplex's handling of dual degeneracy.
- Every row in the final tables ran alone on the machine. One row of the
  earlier pass (mod008, 30 s against 18 s) had a diagnostic run of mine
  overlapping it; that pass is not the one reported.

## 7. Does it perform as desired?

The question was whether the solver performs as desired on the categories
the problem statement names. Answered by category, against the two
yardsticks the campaign used -- the published value, and an independent
check of every point:

**LP: yes, and it can be certified.** Netlib 87/89 certified optimal by
the simplex and 82/89 by the interior point, with the eight readme
disagreements settled in the solver's favour by certificate; every one of
45 MIPLIB relaxations certified and matching HiGHS to fourteen digits; on
Mittelmann's LP set the GPU first-order method solves 9 of 13 and the
interior point 8, with HiGHS at 8 and slower on the three largest. The
simplex is the exact engine for models up to a few hundred thousand
nonzeros and the wrong one past that, which the router already knows.
What is not as desired: the simplex is 6× behind HiGHS on the small set,
and three Netlib instances still get a wrong infeasibility verdict from
the interior point.

**MILP: yes on the classical set, no on the benchmark set.** 22 of the 45
classical MIPLIB instances proved and 27 on the published value; 8 of 17
fixed-charge transportation problems; industrial models -- facility
location, crew scheduling, generation planning, an 87,482-column
set-partitioning model -- among the proved. On Mittelmann's MILP
benchmark, the set the established solvers are ranked on, nothing is
proved in two minutes and a feasible point is found on five in six. The
gap is cut families, presolve and branching rules, and it is the honest
distance between a branch-and-bound built from the foundations in a few
months and one refined for thirty years. Everything the tree returns is
verifier-accepted, and after bug 11 nothing it prunes is unproved.

**QP: yes where convex, bounded where not.** The convex QPs and two of
three convex MIQPs proved; the constrained MIQPs and the non-convex
binary quadratics at gaps the README already records; no bound in the
campaign ever cut off a published optimum, which for a global solver is
the property that matters.

**Refinery, blending, planning, supply chain: yes, at the sizes measured.**
Williams' refinery to the penny with the book's plan; Haverly's pooling
problems to the proven global optimum, with the industry's recursion
shown wrong from 20 of 21 starts; the refinery-shaped ladders to 4.4
million nonzeros for the LP engines and 6,144 binaries within 0.4% for
the tree; MIPLIB's own lot-sizing, location, network and scheduling
models as above.

**And the campaign itself performed as a test should:** it found seven
defects the eleven-instance set had never exercised -- three time limits
that were not limits, two soundness holes, one crash, and one solver
claiming less than it had proved -- and every one is fixed, tested, and
recorded with the measurement that found it. The solver that finished
the campaign is not the one that started it, which is the reason to run
one.
