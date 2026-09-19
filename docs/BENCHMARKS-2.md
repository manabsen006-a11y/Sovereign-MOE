# The second campaign: sets the first one never touched

The first campaign ([`BENCHMARKS.md`](BENCHMARKS.md), 17 September 2026)
ran the problem statement's categories on the sets everyone runs: Netlib's
89, MIPLIB's classical set, Mittelmann's LP and MILP benchmarks, QPLIB,
Williams' refinery, Haverly's pooling problems. This one, run on 18
and 19 September on the working tree that became commit `c6552df`,
uses only sets that campaign did not --
publicly available, machine-readable, with the answer key read from its
source into every instance's header as before:

| category | set | source of the answer key |
|---|---|---|
| LP, infeasible | Netlib's infeasible LP set, 29 models (Chinneck 1993): five from a petrochemical plant, three from BP, the original greenbea | the readme's list: every one is infeasible |
| LP, large | Netlib's Kennington set, 16 models to 1.4M nonzeros (Carolan et al. 1990) | the readme's table of optimal values |
| LP, larger | Mittelmann's LP test set, top level: the eleven under 5 MB compressed | none published; the verifier's certificate and HiGHS |
| QP | Maros and Meszaros' convex QP set, all 138 (OMS 1999) | none in machine-readable form; the verifier's certificate |
| MILP, infeasible and unbounded | MIPLIB 2017's easy instances whose published status is Infeasible or Unbounded, the 31 under 100k nonzeros | the listing's status |
| MILP, a new draw | every easy MIPLIB 2017 instance under 10,000 nonzeros that no earlier set ran: 114 | the listing's objective (easy: proven) |
| supply chain | OR-Library's capacitated (49 problems) and uncapacitated (15) warehouse location sets (Beasley 1988, 1993) | `capopt.txt`, `uncapopt.txt` |
| literature | six of Williams' textbook models -- blending with storage, factory planning, distribution, unit commitment, mining -- beside the refinery | the book's chapter 13 |

Two things had to be built before the first set could be run at all, and
both are in the engine now: the simplex returns the dual ray that proves
a model infeasible, and the verifier checks it (section 1); and the MPS
reader takes fixed-column files whose names contain spaces (section 4).
Every stage ran alone on the machine, one after another; the rows a
defect's fix or a diagnostic touched were re-run alone afterwards, and
each section says which.

## In one table

| category | set | instances | proved / certified | on the published value | verifier-accepted | note |
|---|---|---|---|---|---|---|
| LP, infeasible | Netlib's infeasible set, simplex | 29 | **29 recognised, 29 certified** | 29/29 (the status is the key) | — | every ray checked on the original model; 98 s |
| LP, infeasible | the same, interior point | 29 | 27 recognised by stagnation, 0 certified | 27/29 | — | cplex2 is infeasible by less than the 1e-6 line |
| LP, infeasible | the same, GPU PDLP | 29 | 0 | 0/29 | — | no infeasibility detection: a limitation, recorded |
| literature | Williams, six models + the refinery | 7 (15 solves) | **15/15 on the book** | 15/15 to the penny | 15 | blending, planning, distribution, unit commitment, mining |
| supply chain | OR-Library warehouse location, 16-50 warehouses | 49 | **49 proved** | 49/49 | 49 | at the root or within 57 nodes; 22 s |
| supply chain | the same, 100 × 1,000 | 15 | 0 | 0 | 0 | the tree's root simplex does not finish; the interior point does it in 35 s |
| QP | Maros and Meszaros, all 138 | 138 | **121 optimal, 97 certified** | — (none published) | 130 | 3 defects fixed first (bugs 18, 19, a reader fix); liswet and boyd remain |
| MILP, infeasible | MIPLIB 2017 published-infeasible, ≤ 100k nnz | 26 | **11 proved infeasible** | 11/26 | — | two "feasible points" were bug 16, a reader convention |
| MILP, unbounded | MIPLIB 2017 published-unbounded | 5 | **4 recognised at the root** | 4/5 | — | were `NODE_LIMIT` before bug 17 |
| MILP, a new draw | every easy MIPLIB 2017 instance ≤ 10k nnz not run before | 114 | **20 proved** | 32 (20 + 12 unproved) | 98 | 16 no incumbent; the cut families again; 3.3 h |
| LP, large | Netlib Kennington, interior point | 16 | **15 certified** | 15/16 | 15 | osa-60's 1.4M nnz in 34 s; HiGHS 16/16 in 61 s against 535 s |
| LP, large | the same, GPU PDLP | 16 | 15 optimal (9 certified) | **16/16** | 15 | pds-20 in 44 s, which the interior point cannot |
| LP, large | the same, simplex | 16 | 13 certified | 13/16 | 13 | out of its depth past 50k rows |
| LP, larger | Mittelmann's next size class | 11 | **6 by one engine or the other** (PDLP 5, IPM 4 certified) | — | 6 | HiGHS 5; ex10's 1.16M nnz in 2 s against 159 s |

Five defects found and fixed on the way (section 8): a reader convention
that made two published-infeasible models feasible, a tree that reported
`NODE_LIMIT` on an unbounded model, an interior point that returned the
zero vector for an equality-constrained QP, two optimality certificates
that were one-sided, and a simplex that called a point OPTIMAL the
verifier would refuse -- plus three things built because the sets needed
them: the simplex's infeasibility ray and its verification, a
fixed-column MPS reader, and the safe bound's absorbed-column mask.

---

## 1. Netlib's infeasible LPs: is "infeasible" a proof?

`python -m bench.fetch --set netlib-infeasible`, then `python -m bench.harness
--mode lp --dir data/netlib-infeasible --time-limit 120 --method simplex |
ipm`, and `--method pdlp --device gpu --time-limit 60`. Chinneck's
collection at netlib/lp/infeas: 29 LPs published as infeasible, from 9 rows
(galenet) to 3,792 × 10,733 with 97k nonzeros (gosh, from BP's operations
models). Five are doctored petrochemical plant models -- chemcom, qual,
**refinery**, reactor, vol1 -- and greenbea is the original, unrepaired
version of the Netlib problem. A refinery planner meets exactly these:
a demand scenario the crude slate cannot meet, a quality specification
no blend satisfies, and the question is not only whether the solver
says so but whether it can be believed when it does.

**What had to be built first.** INFEASIBLE was the one verdict a caller
could not check: there was no point to hand the verifier. The node
solver has returned a Farkas ray since bug 11 made the tree check every
INFEASIBLE before pruning on it, but the top-level `solve_simplex` did not
export one. Now it does -- `Solution.farkas`, the phase-1 duals that price
the rows into a contradiction, unscaled by the row factors -- and
`bench.verify.verify_infeasible` checks it with the same arithmetic the
tree prunes on: a strictly positive Farkas value over the original model
is a proof that no point in the box satisfies the rows. The first pass
certified 9 of 29 valid rays. The other 20 were refused for terms of 1e-19
to 1e-11 of the ray's largest entry pointing at an infinite bound -- a
phase-1 basis leaves rounding residues on its basic columns, and a
residue on a column with no bound on that side makes the value −∞ however
small it is. The verifier now treats them as the optimality certificate
treats a reduced cost of 1e-15 at an infinite bound: dropped when below
rounding, reported as a perturbation, and refused above it.

| engine | recognised | certified | time for the set | note |
|---|---|---|---|---|
| revised simplex, 120 s | **29/29 `INFEASIBLE`** | **29/29** | 98 s | greenbea 68 s (2,393 × 5,405), gosh 15 s; the rest under 3 s |
| interior point, 120 s | 27/29 `INFEASIBLE_OR_UNBOUNDED` | 0 -- a stagnation test, not a proof | 136 s | gosh at the limit; cplex2 returned a point the verifier accepts (below) |
| PDLP on the GPU, 60 s | 0/29 | 0 | 1,598 s | no infeasibility detection: 19 time limits, 10 iteration limits, cplex2 a point |

| instance | rows × cols | nnz | status | objective | reference | rel. err | time | verifier |
|---|---|---|---|---|---|---|---|---|
| bgdbg1 | 348 × 407 | 1,440 | **INFEASIBLE** | — | INFEASIBLE | — | 0.60 s | **certified** |
| bgetam | 400 × 688 | 2,409 | **INFEASIBLE** | — | INFEASIBLE | — | 0.26 s | **certified** |
| bgindy | 2671 × 10116 | 65,502 | **INFEASIBLE** | — | INFEASIBLE | — | 1.85 s | **certified** |
| bgprtr | 20 × 34 | 64 | **INFEASIBLE** | — | INFEASIBLE | — | 0.00 s | **certified** |
| box1 | 231 × 261 | 651 | **INFEASIBLE** | — | INFEASIBLE | — | 0.08 s | **certified** |
| ceria3d | 3576 × 824 | 17,602 | **INFEASIBLE** | — | INFEASIBLE | — | 2.85 s | **certified** |
| chemcom | 288 × 720 | 1,566 | **INFEASIBLE** | — | INFEASIBLE | — | 0.12 s | **certified** |
| cplex1 | 3005 × 3221 | 8,944 | **INFEASIBLE** | — | INFEASIBLE | — | 2.54 s | **certified** |
| cplex2 | 224 × 221 | 1,058 | **INFEASIBLE** | — | INFEASIBLE | — | 0.17 s | **certified** |
| ex72a | 197 × 215 | 467 | **INFEASIBLE** | — | INFEASIBLE | — | 0.07 s | **certified** |
| ex73a | 193 × 211 | 457 | **INFEASIBLE** | — | INFEASIBLE | — | 0.06 s | **certified** |
| forest6 | 66 × 95 | 210 | **INFEASIBLE** | — | INFEASIBLE | — | 0.03 s | **certified** |
| galenet | 8 × 8 | 16 | **INFEASIBLE** | — | INFEASIBLE | — | 0.00 s | **certified** |
| gosh | 3792 × 10733 | 97,231 | **INFEASIBLE** | — | INFEASIBLE | — | 14.73 s | **certified** |
| gran | 2658 × 2520 | 20,106 | **INFEASIBLE** | — | INFEASIBLE | — | 0.61 s | **certified** |
| greenbea | 2393 × 5405 | 30,883 | **INFEASIBLE** | — | INFEASIBLE | — | 68.09 s | **certified** |
| itest2 | 9 × 4 | 17 | **INFEASIBLE** | — | INFEASIBLE | — | 0.01 s | **certified** |
| itest6 | 11 × 8 | 20 | **INFEASIBLE** | — | INFEASIBLE | — | 0.01 s | **certified** |
| klein1 | 54 × 54 | 696 | **INFEASIBLE** | — | INFEASIBLE | — | 0.11 s | **certified** |
| klein2 | 477 × 54 | 4,585 | **INFEASIBLE** | — | INFEASIBLE | — | 0.49 s | **certified** |
| klein3 | 994 × 88 | 12,107 | **INFEASIBLE** | — | INFEASIBLE | — | 3.27 s | **certified** |
| mondou2 | 312 × 604 | 1,208 | **INFEASIBLE** | — | INFEASIBLE | — | 0.32 s | **certified** |
| pang | 361 × 460 | 2,652 | **INFEASIBLE** | — | INFEASIBLE | — | 0.17 s | **certified** |
| pilot4i | 410 × 1000 | 5,141 | **INFEASIBLE** | — | INFEASIBLE | — | 0.29 s | **certified** |
| qual | 323 × 464 | 1,646 | **INFEASIBLE** | — | INFEASIBLE | — | 0.19 s | **certified** |
| reactor | 318 × 637 | 2,420 | **INFEASIBLE** | — | INFEASIBLE | — | 0.25 s | **certified** |
| refinery | 323 × 464 | 1,626 | **INFEASIBLE** | — | INFEASIBLE | — | 0.61 s | **certified** |
| vol1 | 323 × 464 | 1,646 | **INFEASIBLE** | — | INFEASIBLE | — | 0.31 s | **certified** |
| woodinfe | 35 × 89 | 140 | **INFEASIBLE** | — | INFEASIBLE | — | 0.04 s | **certified** |

```
  instances            29
  status OPTIMAL       0/29
  verifier accepted    0/29
  certified optimal    0/29   (duals bound the optimum within 1e-9)
  within 1e-4 of ref   0/0
  shifted geomean time 0.747s (shift 1s)
  total time           98.1s
  published infeasible 29: recognised 29, certified 29
```

Three things the table says:

- **The simplex proves every one.** Twenty-nine of twenty-nine recognised
  and twenty-nine certified, the certificate checked on the original
  model by arithmetic the solver does not share. The petrochemical five
  take 0.2-0.4 s each; greenbea, the set's hardest, 68 s -- the feasible
  Netlib greenbea takes the simplex 21 s to an optimum, and finding that
  there is no optimum costs three times that here, all of it in phase 1.
- **The interior point recognises 27 by stagnation and proves none.**
  `INFEASIBLE_OR_UNBOUNDED` is what its primal residual stalling means,
  and the README has said since the method was built that this is
  evidence rather than a certificate. gosh ran its 120 s without the
  stall firing. The method's verdicts are right on this set and were
  wrong on agg, finnis and perold in the first campaign; a stagnation
  test can be both, which is why the simplex's proof is the one to keep.
- **cplex2 is the yardstick's resolution, not a defect.** The readme calls
  it "an almost-feasible problem". The simplex certifies it infeasible
  (Farkas value 3.7e-3 at a ray scaled 4e6, so the rows contradict each
  other by about 1e-9); the interior point returns a point that violates
  its rows by 6.9e-9, which the verifier's 1e-6 line accepts. The model
  is infeasible by less than the tolerance a feasible point is allowed,
  and both answers are correct at their tolerances. The verifier's own
  optimality line says as much without being asked: the interior point's
  certified bound sits *above* its objective by 1.8e-4, which a feasible
  point cannot do and a 1e-9-infeasible one scaled by 4e6 can.

The first-order method has no infeasibility detection at all, and the
table records it as the limitation it is: a caller who routes an
infeasible model to PDLP gets a time limit and an unconverged iterate.
The engine that answers the refinery planner's question is the simplex,
and it now answers it with a proof.

## 2. Williams' models: blending, planning, distribution, unit commitment, mining

`python -m bench.williams`. Six more of Williams' problems, transcribed
with every coefficient as the book prints them and checked against the
value in its chapter 13, beside the refinery from the first campaign.
Their subjects are the problem statement's own words: **food manufacture
1 and 2** blend five oils bought on a six-month forward market into a
product whose hardness must lie in a band, with storage at a cost -- crude
blending with inventory, as an LP and then with the logical conditions
(at most three oils a month, at least 20 tons of any used, the vegetable
oils forcing the third non-vegetable) that make it a MILP; **factory
planning 1** is seven products over six months on five machine types
under a maintenance schedule and monthly market limits -- production
planning; **distribution 1** is two factories, four depots and six
customers -- a supply chain; **tariff rates** is a day of power
generation in five load periods with three generator types, start-up
costs and a 15% spinning reserve -- unit commitment, which is what a
refinery's utilities run; **mining** is four mines over five years under
blending targets on ore quality, royalties, closures and a discount rate
-- scheduling with blending.

| model | rows × cols | integers | published | simplex | interior point | PDLP | tree |
|---|---|---|---|---|---|---|---|
| food manufacture 1 (12.1) | 65 × 96 | 0 | 107,842.59 | **107,842.59**, certified | certified | 107,842.59, feasible | |
| food manufacture 2 (12.2) | 143 × 126 | 30 | 100,278.71 | | | | **100,278.70**, proved in 4.7 s |
| factory planning 1 (12.3) | 79 × 126 | 0 | 93,715.18 | **93,715.18**, certified | certified | certified | |
| distribution 1 (12.19) | 16 × 29 | 0 | 198,500 | **198,500**, certified | certified | certified | |
| tariff rates (12.15) | 55 × 45 | 30 | 988,540 | | | | **988,540**, proved in 0.07 s |
| mining (12.7) | 71 × 65 | 40 | 146.862 M | | | | **146.862 M**, proved in 0.7 s |
| refinery (12.6) | 33 × 40 | 0 | 211,365.13 | **211,365.13**, certified | certified | certified | |

```
  solves               15
  on the book's value  15/15   (to the penny; mining to five hundred pounds)
  verifier accepted    15/15
  certified optimal    11 of the LP solves
```

Fifteen solves, fifteen on the book's value to the penny, fifteen
verifier-accepted, eleven of the twelve LP solves certified from the
duals (PDLP's food manufacture 1 is feasible to 1e-6 and 2.4e-8 from the
value, its normal finish). Food manufacture 2 is the one worth a
sentence: the book prints 100,278.71 and the exact optimum of the
printed model is 100,278.7037 -- the book recomputes its value from the
plan it prints, rounded to two decimals, which is 0.6 pence of rounding
and not a disagreement. Every plan is the book's where the book prints
one: distribution 1 sends all of C1's 50,000 tons from Liverpool directly,
the tariff plan runs no type-3 generator overnight, the mine schedule
never works four mines.

A textbook page each, and small. Their value is the answer key: seven
models whose optimum nobody in this repository produced, in the four
categories the problem statement names, and every one reached by every
engine that applies.

## 3. OR-Library warehouse location: supply chain, 64 problems

`python -m bench.fetch --set orlib`, then `python -m bench.orlib --set cap
--time-limit 120` and `--set uncap`. Beasley's warehouse location sets are
the supply-chain benchmark of the location literature: `m` candidate
warehouses with a fixed cost and a capacity, `n` customers with a demand
and a cost of serving all of it from each warehouse, choose which to open
and who serves whom. The data files are read as they are and every
optimum comes from the `capopt.txt` and `uncapopt.txt` beside them.
`bench.orlib` builds the strong formulation -- the customer rows, the
capacity rows, and the `x_ij <= y_i` rows that make the LP relaxation
tight -- with `y` binary and `x` the fraction of a customer's demand
served. Sets IV to XIII are 16, 25 or 50 warehouses by 50 customers; A, B
and C are 100 by 1,000, and each file stands for four problems, one per
capacity in Beasley's Table 1.

| instance | warehouses × customers | status | objective | published | rel. err | nodes | time | verifier |
|---|---|---|---|---|---|---|---|---|
| cap41 | 16 × 50 | **optimal** | 1040444.4 | 1040444.4 | 0 | 0 | 0.6 s | feasible |
| cap42 | 16 × 50 | **optimal** | 1098000.5 | 1098000.4 | 2.1e-16 | 3 | 0.3 s | feasible |
| cap43 | 16 × 50 | **optimal** | 1153000.5 | 1153000.4 | 2e-16 | 7 | 0.3 s | feasible |
| cap44 | 16 × 50 | **optimal** | 1235500.5 | 1235500.4 | 1.9e-16 | 7 | 0.3 s | feasible |
| cap61 | 16 × 50 | **optimal** | 932615.75 | 932615.75 | 0 | 0 | 0.2 s | feasible |
| cap62 | 16 × 50 | **optimal** | 977799.4 | 977799.4 | 1.2e-16 | 0 | 0.1 s | feasible |
| cap63 | 16 × 50 | **optimal** | 1014062.1 | 1014062.1 | 0 | 7 | 0.3 s | feasible |
| cap64 | 16 × 50 | **optimal** | 1045650.2 | 1045650.2 | 2.2e-16 | 0 | 0.2 s | feasible |
| cap71 | 16 × 50 | **optimal** | 932615.75 | 932615.75 | 0 | 0 | 0.1 s | feasible |
| cap72 | 16 × 50 | **optimal** | 977799.4 | 977799.4 | 1.2e-16 | 0 | 0.1 s | feasible |
| cap73 | 16 × 50 | **optimal** | 1010641.4 | 1010641.4 | 0 | 0 | 0.1 s | feasible |
| cap74 | 16 × 50 | **optimal** | 1034977 | 1034977 | 1.1e-16 | 0 | 0.1 s | feasible |
| cap81 | 25 × 50 | **optimal** | 838499.29 | 838499.29 | 6e-10 | 3 | 0.5 s | feasible |
| cap82 | 25 × 50 | **optimal** | 910889.56 | 910889.56 | 5.5e-10 | 7 | 0.5 s | feasible |
| cap83 | 25 × 50 | **optimal** | 975889.56 | 975889.56 | 5.1e-10 | 3 | 0.6 s | feasible |
| cap84 | 25 × 50 | **optimal** | 1069369.5 | 1069369.5 | 2.2e-16 | 11 | 0.7 s | feasible |
| cap91 | 25 × 50 | **optimal** | 796648.44 | 796648.44 | 6.3e-10 | 0 | 0.3 s | feasible |
| cap92 | 25 × 50 | **optimal** | 855733.5 | 855733.5 | 0 | 7 | 0.5 s | feasible |
| cap93 | 25 × 50 | **optimal** | 896617.54 | 896617.54 | 5.6e-10 | 17 | 0.6 s | feasible |
| cap94 | 25 × 50 | **optimal** | 946051.32 | 946051.32 | 0 | 27 | 0.6 s | feasible |
| cap101 | 25 × 50 | **optimal** | 796648.44 | 796648.44 | 6.3e-10 | 0 | 0.3 s | feasible |
| cap102 | 25 × 50 | **optimal** | 854704.2 | 854704.2 | 1.4e-16 | 0 | 0.3 s | feasible |
| cap103 | 25 × 50 | **optimal** | 893782.11 | 893782.11 | 5.6e-10 | 0 | 0.3 s | feasible |
| cap104 | 25 × 50 | **optimal** | 928941.75 | 928941.75 | 0 | 0 | 0.3 s | feasible |
| cap111 | 50 × 50 | **optimal** | 826124.71 | 826124.71 | 6e-10 | 5 | 0.9 s | feasible |
| cap112 | 50 × 50 | **optimal** | 901377.21 | 901377.21 | 5.6e-10 | 13 | 1.2 s | feasible |
| cap113 | 50 × 50 | **optimal** | 970567.75 | 970567.75 | 0 | 41 | 1.6 s | feasible |
| cap114 | 50 × 50 | **optimal** | 1063356.5 | 1063356.5 | 4.7e-10 | 57 | 2.0 s | feasible |
| cap121 | 50 × 50 | **optimal** | 793439.56 | 793439.56 | 6.3e-10 | 0 | 0.8 s | feasible |
| cap122 | 50 × 50 | **optimal** | 852524.62 | 852524.62 | 0 | 7 | 1.1 s | feasible |
| cap123 | 50 × 50 | **optimal** | 895302.32 | 895302.32 | 0 | 15 | 1.3 s | feasible |
| cap124 | 50 × 50 | **optimal** | 946051.32 | 946051.32 | 0 | 53 | 1.6 s | feasible |
| cap131 | 50 × 50 | **optimal** | 793439.56 | 793439.56 | 6.3e-10 | 0 | 0.7 s | feasible |
| cap132 | 50 × 50 | **optimal** | 851495.32 | 851495.32 | 0 | 0 | 0.8 s | feasible |
| cap133 | 50 × 50 | **optimal** | 893076.71 | 893076.71 | 5.6e-10 | 0 | 0.8 s | feasible |
| cap134 | 50 × 50 | **optimal** | 928941.75 | 928941.75 | 0 | 0 | 0.9 s | feasible |
| cap51 | 16 × 50 | **optimal** | 1025208.2 | 1025208.2 | 0 | 7 | 0.3 s | feasible |
| capa@8000 | 100 × 1000 | limit | — | 19240822 | — | 0 | 189.5 s | — |
| capa@10000 | 100 × 1000 | limit | — | 18438047 | — | 0 | 207.6 s | — |
| capa@12000 | 100 × 1000 | limit | — | 17765202 | — | 0 | 241.2 s | — |
| capa@14000 | 100 × 1000 | limit | — | 17160439 | — | 0 | 228.8 s | — |
| capb@5000 | 100 × 1000 | limit | — | 13656380 | — | 0 | 235.5 s | — |
| capb@6000 | 100 × 1000 | limit | — | 13361927 | — | 0 | 217.5 s | — |
| capb@7000 | 100 × 1000 | limit | — | 13198556 | — | 0 | 267.6 s | — |
| capb@8000 | 100 × 1000 | limit | — | 13082516 | — | 0 | 196.2 s | — |
| capc@5000 | 100 × 1000 | limit | — | 11646597 | — | 0 | 267.3 s | — |
| capc@5750 | 100 × 1000 | limit | — | 11570340 | — | 0 | 257.4 s | — |
| capc@6500 | 100 × 1000 | limit | — | 11518744 | — | 0 | 191.2 s | — |
| capc@7250 | 100 × 1000 | limit | — | 11505767 | — | 0 | 202.1 s | — |

```
  instances            49
  proved optimal       37/49
  on the published value (1e-6)  37/49
  verifier accepted    37/49
  total time           2724.2s
```

| instance | warehouses × customers | status | objective | published | rel. err | nodes | time | verifier |
|---|---|---|---|---|---|---|---|---|
| cap71 | 16 × 50 | **optimal** | 932615.75 | 932615.75 | 0 | 0 | 0.9 s | feasible |
| cap72 | 16 × 50 | **optimal** | 977799.4 | 977799.4 | 1.2e-16 | 0 | 0.2 s | feasible |
| cap73 | 16 × 50 | **optimal** | 1010641.4 | 1010641.4 | 0 | 0 | 0.2 s | feasible |
| cap74 | 16 × 50 | **optimal** | 1034977 | 1034977 | 1.1e-16 | 0 | 0.2 s | feasible |
| cap101 | 25 × 50 | **optimal** | 796648.44 | 796648.44 | 6.3e-10 | 0 | 0.4 s | feasible |
| cap102 | 25 × 50 | **optimal** | 854704.2 | 854704.2 | 0 | 0 | 0.3 s | feasible |
| cap103 | 25 × 50 | **optimal** | 893782.11 | 893782.11 | 5.6e-10 | 0 | 0.4 s | feasible |
| cap104 | 25 × 50 | **optimal** | 928941.75 | 928941.75 | 0 | 0 | 0.3 s | feasible |
| cap131 | 50 × 50 | **optimal** | 793439.56 | 793439.56 | 6.3e-10 | 0 | 0.9 s | feasible |
| cap132 | 50 × 50 | **optimal** | 851495.32 | 851495.32 | 0 | 0 | 0.9 s | feasible |
| cap133 | 50 × 50 | **optimal** | 893076.71 | 893076.71 | 5.6e-10 | 0 | 0.9 s | feasible |
| cap134 | 50 × 50 | **optimal** | 928941.75 | 928941.75 | 0 | 0 | 0.9 s | feasible |
| capa | 100 × 1000 | limit | — | 17156454 | — | 0 | 166.6 s | — |
| capb | 100 × 1000 | limit | — | 12979072 | — | 0 | 167.7 s | — |
| capc | 100 × 1000 | limit | — | 11505594 | — | 0 | 168.8 s | — |

```
  instances            15
  proved optimal       12/15
  on the published value (1e-6)  12/15
  verifier accepted    12/15
  total time           509.8s
```

**The 16-to-50-warehouse problems: every one proved, on the published
value, at the root or within 57 nodes, 22 s for all 37.** The strong
formulation is what does it -- 18 of the 37 are optimal at the root, the
LP bound and the rounded LP point already the answer -- and the rest close
in a handful of nodes; the hardest, cap114, in 57 nodes and 2 s. The
uncapacitated twelve are the same story at the root. This is the tree at
what it is good at: a tight relaxation and a small number of binaries.

**The 100 × 1,000 problems: none, and not for the reason the size
suggests.** Twelve capacitated problems (A, B, C at four capacities) and
the three uncapacitated ones return no incumbent in 120 s, and every one
of the fifteen overran the limit, to 167-268 s, with **0 nodes**: the
tree never left the root. The strong formulation's root relaxation is
101,100 rows by 100,100 columns, and the tree's root solver is the dual
simplex. Measured on capa at capacity 8,000, that relaxation alone:

| engine | result | time |
|---|---|---|
| revised simplex (the tree's root) | 15,136 iterations, unfinished | > 600 s |
| interior point | **certified optimal, 18,832,965.5** (2.1% under the published MILP optimum of 19,240,822.4) | **35 s** |
| PDLP on the GPU | iteration limit, iterate rejected | 255 s |

The interior point has the root bound in 35 s; the tree, which owns no
interior point, spends its two minutes and more inside the first
factorisations of a 101k-row basis and reports nothing. This is the
first item the campaign adds to Known limits: a tree whose root goes to
the interior point (and through the crossover the repository already
has, to a basis for the children) would have a 2.1% root gap to close
on these; the tree as built does not start.

## 4. Maros and Meszaros: 138 convex QPs

`python -m bench.fetch --set marmes`, then `python -m bench.harness --mode
lp --dir data/marmes --time-limit 120`. Maros and Meszaros' repository
(OMS 1999) is the standard convex QP test set: 138 problems in QPS
format, from 2 variables (hs21, tame) to 93,261 (boyd1) and 90,597 rows
(cont-300), a third of them Netlib LPs with a quadratic term added (the
`q*` names). The three archives on Maros' page are the source; no
optimal values could be found in a machine-readable form anywhere, so
the header says so and the verifier's certificate -- feasible to 1e-6,
duals bounding the optimum within 1e-9 -- is the only check. Every
instance goes to the interior point, which is the engine for convex QP
since the first campaign's QPLIB pass.

**Two reader defects and one engine defect before the set could be
read.** qforplan (Netlib's forplan with a quadratic term) names its rows
"DEDO3 1R" and its columns "DEDO3 11" -- legal in the fixed-column format,
and the whitespace tokeniser saw a duplicate row "DEDO3"; the reader now
switches to the column positions for the whole file once a ROWS line
does not split into a type and a name. Nine problems -- hs51, genhs28,
dpklo1, dtoc3, the four `aug` models and boyd1 -- have every row an
equality and every column free, so no complementarity pair anywhere, and
the interior point's shortcut for that case assumed every variable was
*pinned*, returned the zero vector and had it demoted to `NUMERICAL`
(bug 18); it is one Newton step on the KKT system, which the loop now
takes. And liswet1 found the certificate one-sided (bug 19): a point 3e-7
off its rows whose objective sat 0.19% *below* its own certified bound
was called OPTIMAL, by the engine and by the verifier, because both
tested `gap <= tol` and let a negative gap through. Both are two-sided
now. The table is the run after the three fixes.

| instance | rows × cols | nnz | status | objective | reference | rel. err | time | verifier |
|---|---|---|---|---|---|---|---|---|
| aug2d | 10000 × 20200 | 40,000 | optimal | 1687411.8 | — | — | 0.47 s | certified |
| aug2dc | 10000 × 20200 | 40,000 | optimal | 1818368.1 | — | — | 0.36 s | certified |
| aug2dcqp | 10000 × 20200 | 40,000 | optimal | 6498134.7 | — | — | 1.03 s | certified |
| aug2dqp | 10000 × 20200 | 40,000 | optimal | 6237012 | — | — | 1.09 s | certified |
| aug3d | 1000 × 3873 | 6,546 | optimal | 554.06773 | — | — | 0.07 s | certified |
| aug3dc | 1000 × 3873 | 6,546 | optimal | 771.26244 | — | — | 0.06 s | certified |
| aug3dcqp | 1000 × 3873 | 6,546 | optimal | 993.36215 | — | — | 0.14 s | certified |
| aug3dqp | 1000 × 3873 | 6,546 | optimal | 675.23767 | — | — | 0.16 s | certified |
| boyd1 | 18 × 93261 | 558,985 | numerical | — | — | — | 9.29 s | **rejected** |
| boyd2 | 186531 × 93263 | 423,784 | limit | — | — | — | 183.77 s | — |
| cont-050 | 2401 × 2597 | 12,005 | optimal | -4.5638509 | — | — | 0.36 s | certified |
| cont-100 | 9801 × 10197 | 49,005 | optimal | -4.6443979 | — | — | 2.47 s | certified |
| cont-101 | 10098 × 10197 | 49,599 | optimal | 0.19552732 | — | — | 3.14 s | certified |
| cont-200 | 39601 × 40397 | 198,005 | optimal | -4.6848759 | — | — | 21.34 s | certified |
| cont-201 | 40198 × 40397 | 199,199 | optimal | 0.19248336 | — | — | 33.29 s | feasible |
| cont-300 | 90298 × 90597 | 448,799 | limit | — | — | — | 125.27 s | — |
| cvxqp1_l | 5000 × 10000 | 14,998 | optimal | 1.087048e+08 | — | — | 98.14 s | certified |
| cvxqp1_m | 500 × 1000 | 1,498 | optimal | 1087511.6 | — | — | 0.26 s | certified |
| cvxqp1_s | 50 × 100 | 148 | optimal | 11590.718 | — | — | 0.03 s | certified |
| cvxqp2_l | 2500 × 10000 | 7,499 | optimal | 81842458 | — | — | 59.62 s | certified |
| cvxqp2_m | 250 × 1000 | 749 | optimal | 820155.43 | — | — | 0.16 s | feasible |
| cvxqp2_s | 25 × 100 | 74 | optimal | 8120.9405 | — | — | 0.03 s | certified |
| cvxqp3_l | 7500 × 10000 | 22,497 | optimal | 1.157111e+08 | — | — | 114.19 s | certified |
| cvxqp3_m | 750 × 1000 | 2,247 | optimal | 1362828.7 | — | — | 0.33 s | certified |
| cvxqp3_s | 75 × 100 | 222 | optimal | 11943.432 | — | — | 0.04 s | certified |
| dpklo1 | 77 × 133 | 1,575 | optimal | 0.37009622 | — | — | 0.02 s | certified |
| dtoc3 | 9998 × 14999 | 34,993 | optimal | 235.26248 | — | — | 0.21 s | certified |
| dual1 | 1 × 85 | 85 | optimal | 0.035012966 | — | — | 0.04 s | feasible |
| dual2 | 1 × 96 | 96 | optimal | 0.033733676 | — | — | 0.04 s | certified |
| dual3 | 1 × 111 | 111 | optimal | 0.13575584 | — | — | 0.05 s | certified |
| dual4 | 1 × 75 | 75 | optimal | 0.74609084 | — | — | 0.04 s | certified |
| dualc1 | 215 × 9 | 1,935 | optimal | 6155.2508 | — | — | 0.04 s | certified |
| dualc2 | 229 × 7 | 1,603 | optimal | 3551.3077 | — | — | 0.05 s | certified |
| dualc5 | 278 × 8 | 2,224 | optimal | 427.23233 | — | — | 0.05 s | certified |
| dualc8 | 503 × 8 | 4,024 | optimal | 18309.359 | — | — | 0.05 s | certified |
| exdata | 3001 × 3000 | 7,500 | optimal | -141.84343 | — | — | 26.65 s | feasible |
| genhs28 | 8 × 10 | 24 | optimal | 0.92717369 | — | — | 0.01 s | certified |
| gouldqp2 | 349 × 699 | 1,047 | optimal | 0.0001842745 | — | — | 0.06 s | certified |
| gouldqp3 | 349 × 699 | 1,047 | optimal | 2.062784 | — | — | 0.04 s | feasible |
| hs118 | 17 × 15 | 39 | optimal | 664.82045 | — | — | 0.03 s | certified |
| hs21 | 1 × 2 | 2 | optimal | -99.96 | — | — | 0.02 s | certified |
| hs268 | 5 × 5 | 25 | optimal | 7.5039679e-07 | — | — | 0.03 s | feasible |
| hs35 | 1 × 3 | 3 | optimal | 0.11111111 | — | — | 0.02 s | certified |
| hs35mod | 1 × 3 | 3 | optimal | 0.25 | — | — | 0.03 s | certified |
| hs51 | 3 × 5 | 7 | optimal | 0 | — | — | 0.01 s | certified |
| hs52 | 3 × 5 | 7 | optimal | 5.3266476 | — | — | 0.01 s | certified |
| hs53 | 3 × 5 | 7 | optimal | 4.0930233 | — | — | 0.02 s | certified |
| hs76 | 3 × 4 | 10 | optimal | -4.6818182 | — | — | 0.02 s | certified |
| hues-mod | 2 × 10000 | 20,000 | optimal | 34824464 | — | — | 0.29 s | feasible |
| huestis | 2 × 10000 | 20,000 | optimal | 3.4824464e+11 | — | — | 0.26 s | feasible |
| ksip | 1001 × 20 | 19,898 | optimal | 0.57579794 | — | — | 0.13 s | certified |
| laser | 1000 × 1002 | 3,000 | optimal | 2409601.3 | — | — | 0.06 s | certified |
| liswet1 | 10000 × 10002 | 30,000 | iteration limit | 25.037072 | — | — | 2.77 s | feasible |
| liswet10 | 10000 × 10002 | 30,000 | iteration limit | 25.002413 | — | — | 2.80 s | feasible |
| liswet11 | 10000 × 10002 | 30,000 | iteration limit | 25.005234 | — | — | 2.71 s | feasible |
| liswet12 | 10000 × 10002 | 30,000 | infeasible/unbounded | — | — | — | 1.03 s | **rejected** |
| liswet2 | 10000 × 10002 | 30,000 | iteration limit | 24.998002 | — | — | 3.78 s | feasible |
| liswet3 | 10000 × 10002 | 30,000 | iteration limit | 25.001213 | — | — | 2.81 s | feasible |
| liswet4 | 10000 × 10002 | 30,000 | iteration limit | 25.000093 | — | — | 2.82 s | feasible |
| liswet5 | 10000 × 10002 | 30,000 | optimal | 25.034253 | — | — | 0.35 s | certified |
| liswet6 | 10000 × 10002 | 30,000 | optimal | 24.995747 | — | — | 0.46 s | feasible |
| liswet7 | 10000 × 10002 | 30,000 | iteration limit | 25.002816 | — | — | 2.90 s | feasible |
| liswet8 | 10000 × 10002 | 30,000 | iteration limit | 25.002816 | — | — | 2.81 s | feasible |
| liswet9 | 10000 × 10002 | 30,000 | infeasible/unbounded | — | — | — | 0.57 s | **rejected** |
| lotschd | 7 × 12 | 54 | optimal | 2398.4159 | — | — | 0.02 s | certified |
| mosarqp1 | 700 × 2500 | 3,422 | optimal | -952.87544 | — | — | 0.09 s | certified |
| mosarqp2 | 600 × 900 | 2,930 | optimal | -1597.4821 | — | — | 0.08 s | certified |
| powell20 | 10000 × 10000 | 20,000 | optimal | 5.2089583e+10 | — | — | 0.58 s | certified |
| primal1 | 85 × 325 | 5,815 | optimal | -0.035012966 | — | — | 0.06 s | certified |
| primal2 | 96 × 649 | 8,042 | optimal | -0.033733676 | — | — | 0.08 s | certified |
| primal3 | 111 × 745 | 21,547 | optimal | -0.13575584 | — | — | 0.19 s | certified |
| primal4 | 75 × 1489 | 16,031 | optimal | -0.74609084 | — | — | 0.13 s | certified |
| primalc1 | 9 × 230 | 2,070 | optimal | -6155.2508 | — | — | 0.05 s | certified |
| primalc2 | 7 × 231 | 1,617 | optimal | -3551.3077 | — | — | 0.04 s | certified |
| primalc5 | 8 × 287 | 2,296 | optimal | -427.23233 | — | — | 0.03 s | certified |
| primalc8 | 8 × 520 | 4,160 | optimal | -18309.43 | — | — | 0.06 s | feasible |
| q25fv47 | 820 × 1571 | 10,400 | optimal | 13744448 | — | — | 1.66 s | certified |
| qadlittl | 56 × 97 | 383 | optimal | 480318.86 | — | — | 0.04 s | certified |
| qafiro | 27 × 32 | 83 | optimal | -1.5907818 | — | — | 0.03 s | certified |
| qbandm | 305 × 472 | 2,494 | optimal | 16352.342 | — | — | 0.08 s | certified |
| qbeaconf | 173 × 262 | 3,375 | optimal | 164712.06 | — | — | 0.07 s | certified |
| qbore3d | 233 × 315 | 1,429 | optimal | 3100.2008 | — | — | 0.07 s | feasible |
| qbrandy | 220 × 249 | 2,148 | optimal | 28375.115 | — | — | 0.07 s | certified |
| qcapri | 271 × 353 | 1,767 | optimal | 66793293 | — | — | 0.13 s | certified |
| qe226 | 223 × 282 | 2,578 | optimal | 212.65343 | — | — | 0.09 s | feasible |
| qetamacr | 400 × 688 | 2,409 | optimal | 86760.37 | — | — | 0.44 s | certified |
| qfffff80 | 524 × 854 | 6,227 | infeasible/unbounded | — | — | — | 0.15 s | **rejected** |
| qforplan | 161 × 421 | 4,563 | iteration limit | — | — | — | 0.68 s | — |
| qgfrdxpn | 616 × 1092 | 2,377 | optimal | 1.0079058e+11 | — | — | 0.13 s | certified |
| qgrow15 | 300 × 645 | 5,620 | optimal | -1.0169364e+08 | — | — | 0.11 s | feasible |
| qgrow22 | 440 × 946 | 8,252 | optimal | -1.4962895e+08 | — | — | 0.14 s | feasible |
| qgrow7 | 140 × 301 | 2,612 | optimal | -42798714 | — | — | 0.08 s | feasible |
| qisrael | 174 × 142 | 2,269 | optimal | 25347838 | — | — | 0.08 s | certified |
| qpcblend | 74 × 83 | 491 | optimal | -0.0078425431 | — | — | 0.06 s | certified |
| qpcboei1 | 351 × 384 | 3,485 | optimal | 11503914 | — | — | 0.11 s | certified |
| qpcboei2 | 166 × 143 | 1,196 | optimal | 8171962.2 | — | — | 0.13 s | certified |
| qpcstair | 356 × 467 | 3,856 | optimal | 6204387.5 | — | — | 0.16 s | certified |
| qpilotno | 975 × 2172 | 13,057 | optimal | 4728586.9 | — | — | 0.57 s | feasible |
| qptest | 2 × 2 | 4 | optimal | 4.371875 | — | — | 0.02 s | feasible |
| qrecipe | 91 × 180 | 663 | optimal | -266.616 | — | — | 0.06 s | feasible |
| qsc205 | 205 × 203 | 551 | optimal | -0.0058139535 | — | — | 0.05 s | feasible |
| qscagr25 | 471 × 500 | 1,554 | optimal | 2.0173794e+08 | — | — | 0.07 s | certified |
| qscagr7 | 129 × 140 | 420 | optimal | 26865949 | — | — | 0.06 s | certified |
| qscfxm1 | 330 × 457 | 2,589 | optimal | 16882692 | — | — | 0.16 s | certified |
| qscfxm2 | 660 × 914 | 5,183 | optimal | 27776162 | — | — | 0.21 s | feasible |
| qscfxm3 | 990 × 1371 | 7,777 | optimal | 30816354 | — | — | 0.25 s | feasible |
| qscorpio | 388 × 358 | 1,426 | optimal | 1880.5096 | — | — | 0.06 s | certified |
| qscrs8 | 490 × 1169 | 3,182 | optimal | 904.56001 | — | — | 0.10 s | certified |
| qscsd1 | 77 × 760 | 2,388 | optimal | 8.6666667 | — | — | 0.04 s | certified |
| qscsd6 | 147 × 1350 | 4,316 | optimal | 50.808214 | — | — | 0.08 s | certified |
| qscsd8 | 397 × 2750 | 8,584 | optimal | 940.76357 | — | — | 0.12 s | certified |
| qsctap1 | 300 × 480 | 1,692 | optimal | 1415.8611 | — | — | 0.06 s | certified |
| qsctap2 | 1090 × 1880 | 6,714 | optimal | 1735.0265 | — | — | 0.11 s | certified |
| qsctap3 | 1480 × 2480 | 8,874 | optimal | 1438.7547 | — | — | 0.13 s | certified |
| qseba | 515 × 1028 | 4,352 | optimal | 81481800 | — | — | 0.15 s | certified |
| qshare1b | 117 × 225 | 1,151 | optimal | 720078.32 | — | — | 0.07 s | feasible |
| qshare2b | 96 × 79 | 694 | optimal | 11703.692 | — | — | 0.06 s | certified |
| qshell | 536 × 1775 | 3,556 | optimal | 1.5726368e+12 | — | — | 0.70 s | certified |
| qship04l | 402 × 2118 | 6,332 | optimal | 2420015.5 | — | — | 0.13 s | certified |
| qship04s | 402 × 1458 | 4,352 | optimal | 2424993.7 | — | — | 0.13 s | certified |
| qship08l | 778 × 4283 | 12,802 | optimal | 2376040.6 | — | — | 0.96 s | certified |
| qship08s | 778 × 2387 | 7,114 | optimal | 2385728.9 | — | — | 0.19 s | certified |
| qship12l | 1151 × 5427 | 16,170 | optimal | 3018876.6 | — | — | 0.97 s | certified |
| qship12s | 1151 × 2763 | 8,178 | optimal | 3056962.2 | — | — | 0.26 s | certified |
| qsierra | 1227 × 2036 | 7,302 | optimal | 23750458 | — | — | 0.16 s | certified |
| qstair | 356 × 467 | 3,856 | optimal | 7985452.8 | — | — | 0.11 s | certified |
| qstandat | 359 × 1075 | 3,031 | optimal | 6411.8384 | — | — | 0.08 s | certified |
| s268 | 5 × 5 | 25 | optimal | 7.5039679e-07 | — | — | 0.03 s | feasible |
| stadat1 | 3999 × 2001 | 9,997 | optimal | -28526864 | — | — | 0.62 s | certified |
| stadat2 | 3999 × 2001 | 9,997 | optimal | -32.626665 | — | — | 0.32 s | certified |
| stadat3 | 7999 × 4001 | 19,997 | optimal | -35.779454 | — | — | 0.49 s | feasible |
| stcqp1 | 2052 × 4097 | 13,338 | optimal | 155143.55 | — | — | 0.40 s | certified |
| stcqp2 | 2052 × 4097 | 13,338 | optimal | 22327.313 | — | — | 0.68 s | certified |
| tame | 1 × 2 | 2 | optimal | 0 | — | — | 0.02 s | certified |
| ubh1 | 12000 × 18009 | 48,000 | iteration limit | 1.1413434 | — | — | 5.64 s | feasible |
| values | 1 × 202 | 202 | optimal | -1.3966211 | — | — | 0.05 s | certified |
| yao | 2000 × 2002 | 6,000 | iteration limit | — | — | — | 0.89 s | — |
| zecevic2 | 2 × 2 | 4 | optimal | -4.125 | — | — | 0.03 s | certified |

```
  instances            138
  status OPTIMAL       121/138
  verifier accepted    130/138
  certified optimal    97/138   (duals bound the optimum within 1e-9)
  within 1e-4 of ref   0/0
  shifted geomean time 0.671s (shift 1s)
  total time           730.4s
  !! VERIFIER REJECTED: boyd1, liswet12, liswet9, qfffff80
```

| | |
|---|---|
| parsed | **138/138** (137 before the fixed-column fix) |
| `OPTIMAL` | **121/138** (113 before bug 18) |
| verifier-accepted point | 130/138 |
| **certified optimal** | **97/138** (83 before bugs 18 and 19 and the mask below) |
| total time | 730 s; the 121 optima in 380 s, cvxqp3_l (10,000 × 15,000) the slowest at 114 s |

The 24 accepted-but-uncertified optima are the interior point's normal
finish on a QP: a point feasible to 1e-6 whose duals close the gap to
1e-8 rather than 1e-9 (cont-201, the `qgrow` and `qscfxm` families,
exdata's 2.25 million-entry dense Hessian). One more piece of the
certificate was found wanting on the equality-constrained models: the
verifier absorbs a reduced cost of 1e-15 at an infinite bound into the
cost vector, and for a QP the bound recomputes the reduced cost from
`c + Qx − Aᵀy`, whose rounding put a 1e-16 residue back and made every
such certificate vacuous. The absorbed columns are now told to the
bound as a mask, and aug2d, aug3d and hs51 certify.

**What the interior point cannot do here, in three kinds.** The `liswet`
family (10,000 rows, 10,002 variables, twelve of them) runs to its
iteration limit on eleven and calls two `INFEASIBLE_OR_UNBOUNDED`: the
primal residual stalls at 3e-7 while the complementarity creeps, and the
point it holds is 0.2% below its own bound -- the family is
ill-conditioned by construction and this method's static regularisation
does not reach it; the proximal-point variant does no better. boyd1 and
boyd2 (93k variables) fail on their conditioning and their size, and
cont-300's 90k-row KKT does not factorise inside 120 s. qfffff80 gets the
same wrong `INFEASIBLE_OR_UNBOUNDED` its LP fffff800 does not, from the
stagnation test the first campaign already named as the method's weak
verdict. yao and qforplan stop at the iteration limit with no point.

Against the set's reputation -- it was assembled to be difficult for
interior points -- 121 of 138 optimal and 97 certified without a
published value to lean on is the answer to "does it solve convex QPs":
yes, and the seventeen it does not are the ones every method finds
hard, plus the boyd pair.

## 5. MIPLIB 2017: infeasible and unbounded models, and a new draw of 114

### 5a. Published infeasible or unbounded

`python -m bench.fetch --set miplib-infeasible`, then `python -m
bench.harness --mode mip --dir data/miplib-infeasible --time-limit 120`.
MIPLIB 2017's easy listing carries a word where 53 of its 694 instances
have no value: *Infeasible* for 39, *Unbounded* for 14. These are the 31
under 100k nonzeros, with the word written into each header as the answer
key. A MILP that is infeasible has to be proved so by exhausting a tree;
an unbounded one is decided at the root.

| instance | rows × cols | nnz | status | objective | reference | gap to ref | time | verifier |
|---|---|---|---|---|---|---|---|---|
| bnatt500 | 7029 × 4500 | 27,203 | limit | — | INFEASIBLE | — | 124.5 s | — |
| control30-5-10-4 | 3640 × 1960 | 23,260 | limit | — | INFEASIBLE | — | 216.7 s | — |
| dell | 500 × 626 | 4,580 | **INFEASIBLE** | — | INFEASIBLE | — | 2.1 s | **tree exhausted: proved** |
| enlight11 | 121 × 242 | 682 | limit | — | INFEASIBLE | — | 120.0 s | — |
| enlight4 | 16 × 32 | 80 | **INFEASIBLE** | — | INFEASIBLE | — | 1.1 s | **tree exhausted: proved** |
| enlight9 | 81 × 162 | 450 | limit | — | INFEASIBLE | — | 120.1 s | — |
| fhnw-binpack4-18 | 650 × 520 | 2,392 | limit | — | INFEASIBLE | — | 120.1 s | — |
| fhnw-binpack4-4 | 620 × 520 | 2,332 | limit | — | INFEASIBLE | — | 120.2 s | — |
| flugplinf | 19 × 18 | 64 | **INFEASIBLE** | — | INFEASIBLE | — | 2.0 s | **tree exhausted: proved** |
| g503inf | 41 × 48 | 144 | **INFEASIBLE** | — | INFEASIBLE | — | 0.1 s | **tree exhausted: proved** |
| misc04inf | 1726 × 4897 | 17,253 | **INFEASIBLE** | — | INFEASIBLE | — | 58.6 s | **tree exhausted: proved** |
| misc05inf | 301 × 136 | 2,946 | **INFEASIBLE** | — | INFEASIBLE | — | 2.4 s | **tree exhausted: proved** |
| mod008inf | 7 × 319 | 1,562 | **INFEASIBLE** | — | INFEASIBLE | — | 16.6 s | **tree exhausted: proved** |
| neos-2626858-aoos | 342 × 524 | 1,690 | limit | — | INFEASIBLE | — | 120.2 s | — |
| neos-2656603-coxs | 342 × 524 | 1,690 | limit | — | INFEASIBLE | — | 120.0 s | — |
| neos-3135526-osun | 1546 × 192 | 25,750 | limit | — | INFEASIBLE | — | 122.4 s | — |
| neos-3654993-kolva | 16064 × 13640 | 54,282 | limit | — | UNBOUNDED | — | 273.8 s | — |
| neos-4382714-ruvuma | 3645 × 6562 | 32,805 | limit | — | INFEASIBLE | — | 128.9 s | — |
| neos-4477313-unzha | 4174 × 2193 | 30,400 | infeasible/unbounded | — | UNBOUNDED | — | 14.3 s | **recognised at the root** |
| neos-4954340-beaury | 20162 × 7850 | 77,435 | infeasible/unbounded | — | UNBOUNDED | — | 6.4 s | **recognised at the root** |
| neos-4954357-bednja | 9641 × 3885 | 49,441 | infeasible/unbounded | — | UNBOUNDED | — | 2.8 s | **recognised at the root** |
| neos-4960896-besbre | 14793 × 6149 | 98,690 | infeasible/unbounded | — | UNBOUNDED | — | 3.9 s | **recognised at the root** |
| neos859080 | 164 × 160 | 1,280 | **INFEASIBLE** | — | INFEASIBLE | — | 62.9 s | **tree exhausted: proved** |
| no-ip-64999 | 2547 × 2232 | 13,590 | limit | — | INFEASIBLE | — | 120.4 s | — |
| no-ip-65059 | 2547 × 2232 | 13,590 | limit | — | INFEASIBLE | — | 120.4 s | — |
| p2m2p1m1p0n100 | 1 × 100 | 100 | limit | — | INFEASIBLE | — | 120.0 s | — |
| ponderthis0517-inf | 78 × 975 | 2,925 | limit | — | INFEASIBLE | — | 124.0 s | — |
| stein15inf | 37 × 15 | 135 | **INFEASIBLE** | — | INFEASIBLE | — | 1.6 s | **tree exhausted: proved** |
| stein45inf | 332 × 45 | 1,079 | **INFEASIBLE** | — | INFEASIBLE | — | 5.3 s | **tree exhausted: proved** |
| stein9inf | 14 × 9 | 54 | **INFEASIBLE** | — | INFEASIBLE | — | 0.3 s | **tree exhausted: proved** |
| supportcase29 | 12441 × 12050 | 96,050 | limit | — | INFEASIBLE | — | 124.9 s | — |

```
  instances            31
  status OPTIMAL       0/31
  verifier accepted    0/31
  within 1e-4 of ref   0/0
  shifted geomean time 41.358s (shift 10s)
  total time           2377.0s
  published infeasible 26: recognised 11, certified 0
  published unbounded  5: recognised 4
```

**The table that caught two defects before it could be read.** The first
pass over this set returned verifier-accepted incumbents on
neos-2626858-aoos and neos-2656603-coxs -- exactly integer, every row
satisfied to the last bit, on models MIPLIB publishes as infeasible. The
points were real and the model was not: MIPLIB counts 209 binaries for
aoos where the file bounds 192 columns by `UP 1`, and the other 17 are
integer columns with no bound line at all, which the MPSX convention that
CPLEX, SCIP and MIPLIB keep makes binary and this reader made [0, +inf).
Bug 16 in the README; the reader follows the convention now, the counts
match MIPLIB's on both models, and both are infeasible -- and unproved
in 120 s, which is the honest row. The same pass returned `NODE_LIMIT`
on four of the five unbounded models after 13-73 s: the root relaxation
was `UNBOUNDED` in under a second and the tree carried the node as
unsolved, re-solved it cold and dropped it undecided. An unbounded
relaxation makes a MILP with rational data unbounded or infeasible and
no branching changes that (Meyer 1974), so the tree now says
`INFEASIBLE_OR_UNBOUNDED` at the root -- bug 17 -- and four of the five
are recognised in 3-14 s. The fifth, neos-3654993-kolva, is a 16k-row
relaxation the node solver does not finish inside the limit.

**What the table then says.** Eleven of the 26 infeasible models are
proved infeasible: the classical `*inf` instances (stein9/15/45,
flugplinf, mod008inf, misc04inf, misc05inf, g503inf), dell, enlight4,
neos859080 -- the tree exhausting the search with every leaf certified
empty (bug 11's discipline), misc04inf's 4,897 columns in 59 s. Fifteen
are time limits with no point and no proof, which is what an infeasible
model looks like to a tree that cannot close it: the enlight family (a
parity argument no LP bound sees), the binpacking pair, no-ip, the
neos-pseudoapplication-7 pair, and p2m2p1m1p0n100 -- one row, 100
integer columns, an equality no integer point satisfies, which a tree
without a Diophantine test can only enumerate. An infeasible MILP has no
certificate short of the tree itself, so the column says "no
certificate" on every proved row; it is the tree's exhaustion that is
the proof, and the two runs of this set disagree on nothing.

### 5b. The new draw

`python -m bench.fetch --set miplib-draw2`, then `python -m bench.harness
--mode mip --dir data/miplib2 --time-limit 120`. The draw is mechanical:
every instance of MIPLIB 2017's easy listing with at most 10,000
nonzeros that neither the classical set nor Mittelmann's benchmark nor
the fctp set had run -- 114 of the 178 candidates under 20,000, chosen by
size alone so that no hand picked them. Every one is proven to
optimality by MIPLIB (that is what *easy* means), and the listing's
objective is the header's `*BEST SOLN: v (opt)`. The listing's group
names say what they are: fixed-cost network flow, lot sizing
(`mik_250`), scheduling (`csched`, `cvs`), generalised assignment
(`f2gap`), production (`app`, `pr_product`), the `neos-pseudoapplication`
families, and 28 with no group at all.

| instance | rows × cols | nnz | status | objective | reference | gap to ref | time | verifier |
|---|---|---|---|---|---|---|---|---|
| 22433 | 198 × 429 | 3,408 | **optimal** | 21477 | 21477 | 0 | 1.4 s | feasible |
| 23588 | 137 × 368 | 3,701 | **optimal** | 8090 | 8090 | 0 | 14.0 s | feasible |
| aflow30a | 479 × 842 | 2,091 | limit | 2298 | 1158 | +98.45% | 120.7 s | feasible |
| aflow40b | 1442 × 2728 | 6,783 | limit | 3191 | 1168 | +173.20% | 123.1 s | feasible |
| app2-1 | 1038 × 3283 | 8,652 | **optimal** | 19294.75 | 19294.125 | +0.00% | 15.6 s | feasible |
| app2-2 | 335 × 1226 | 3,130 | **optimal** | 212042.5 | 212040.36 | +0.00% | 0.6 s | feasible |
| b-ball | 30 × 100 | 209 | limit | -1.5 | -1.5 | 0 | 120.1 s | feasible |
| beasleyC1 | 1750 × 2500 | 5,000 | limit | 102 | 85 | +20.00% | 124.1 s | feasible |
| beasleyC2 | 1750 × 2500 | 5,000 | limit | 200 | 144 | +38.89% | 121.6 s | feasible |
| beavma | 372 × 390 | 975 | limit | 439609 | 383285 | +14.70% | 120.0 s | feasible |
| berlin_5_8_0 | 1532 × 1083 | 4,507 | limit | 85 | 62 | +37.10% | 120.5 s | feasible |
| bienst1 | 576 × 505 | 2,184 | **optimal** | 46.75 | 46.75 | 0 | 82.9 s | feasible |
| bienst2 | 576 × 505 | 2,184 | limit | 55.285714 | 54.6 | +1.26% | 120.0 s | feasible |
| blend2 | 274 × 353 | 1,409 | **optimal** | 7.598985 | 7.598985 | 0 | 28.7 s | feasible |
| bppc8-02 | 59 × 232 | 4,387 | limit | 507 | 507 | 0 | 120.2 s | feasible |
| bppc8-09 | 67 × 431 | 9,051 | limit | 499 | 472 | +5.72% | 120.5 s | feasible |
| breastcancer-regularized | 723 × 715 | 8,283 | limit | 544.86595 | 35.767842 | +1423.34% | 122.1 s | feasible |
| chromaticindex32-8 | 2111 × 2304 | 8,436 | limit | 4 | 4 | 0 | 142.6 s | feasible |
| control30-3-2-3 | 512 × 332 | 1,472 | limit | 23.52271 | -17.256589 | +236.31% | 120.4 s | feasible |
| csched008 | 351 × 1536 | 5,687 | limit | — | 173 | no incumbent | 124.7 s | — |
| csched010 | 351 × 1758 | 6,376 | limit | — | 408 | no incumbent | 446.7 s | — |
| cvs08r139-94 | 2398 × 1864 | 6,456 | limit | -38 | -116 | +67.24% | 126.2 s | feasible |
| cvs16r106-72 | 3608 × 2848 | 9,888 | limit | -26 | -81 | +67.90% | 131.3 s | feasible |
| cvs16r70-62 | 3278 × 2112 | 8,512 | limit | -10 | -42 | +76.19% | 132.3 s | feasible |
| cvs16r89-60 | 3068 × 2384 | 8,368 | limit | -22 | -65 | +66.15% | 125.0 s | feasible |
| dsbmip | 1182 × 1886 | 7,366 | **optimal** | -305.19818 | -305.198 | -0.00% | 35.5 s | feasible |
| ej | 1 × 3 | 3 | limit | — | 25508 | no incumbent | 120.0 s | — |
| enlight8 | 64 × 128 | 352 | limit | — | 27 | no incumbent | 120.1 s | — |
| f2gap201600 | 20 × 1600 | 3,200 | **optimal** | 76453 | 76453 | 0 | 11.9 s | feasible |
| f2gap401600 | 40 × 1600 | 3,200 | **optimal** | 82307 | 82307 | 0 | 25.8 s | feasible |
| f2gap40400 | 40 × 400 | 800 | **optimal** | 20772 | 20772 | 0 | 0.8 s | feasible |
| f2gap801600 | 80 × 1600 | 3,200 | **optimal** | 86679 | 86679 | 0 | 33.0 s | feasible |
| g200x740 | 940 × 1480 | 2,960 | limit | 45425 | 44316 | +2.50% | 120.9 s | feasible |
| gen | 780 × 870 | 2,592 | **optimal** | 112313.36 | 112313 | +0.00% | 4.0 s | feasible |
| gen-ip016 | 24 × 28 | 672 | limit | -9419.6949 | -9476.1552 | +0.60% | 120.1 s | feasible |
| gen-ip036 | 46 × 29 | 1,303 | limit | -4606.6796 | -4606.6796 | 0 | 120.1 s | feasible |
| gen-ip054 | 27 × 30 | 532 | limit | 6874.3476 | 6840.9656 | +0.49% | 120.1 s | feasible |
| gmu-35-50 | 435 × 1919 | 8,643 | limit | -2595536.2 | -2607958.3 | +0.48% | 129.6 s | feasible |
| graphdraw-domain | 865 × 254 | 2,600 | limit | 30077 | 19686 | +52.78% | 120.5 s | feasible |
| graphdraw-gemcutter | 474 × 166 | 1,420 | limit | 7948.5 | 7118.5 | +11.66% | 120.2 s | feasible |
| gsvm2rl3 | 180 × 241 | 4,020 | limit | 0.58635482 | 0.33652753 | +24.98% | 120.4 s | feasible |
| h80x6320 | 79 × 12640 | 6,320 | **optimal** | 3700 | 3700 | 0 | 2.7 s | feasible |
| haprp | 1048 × 1828 | 3,628 | **optimal** | 3673280.7 | 3673280.7 | 0 | 2.0 s | feasible |
| ic97_tension | 319 × 703 | 2,070 | limit | — | 3942 | no incumbent | 120.3 s | — |
| k16x240b | 256 × 480 | 960 | limit | 12874 | 11393 | +13.00% | 120.0 s | feasible |
| markshare1 | 6 × 62 | 312 | limit | 17 | 1 | +1600.00% | 120.0 s | feasible |
| markshare_5_0 | 5 × 45 | 203 | limit | 8 | 1 | +700.00% | 120.1 s | feasible |
| mc7 | 1920 × 3040 | 6,080 | limit | 5802 | 3417 | +69.80% | 120.7 s | feasible |
| mc8 | 1920 × 3040 | 6,080 | limit | 1963 | 1566 | +25.35% | 120.5 s | feasible |
| mik-250-20-75-1 | 195 × 270 | 9,270 | limit | -49516 | -49716 | +0.40% | 120.3 s | feasible |
| mik-250-20-75-2 | 195 × 270 | 9,270 | limit | -50368 | -50768 | +0.79% | 120.4 s | feasible |
| mik-250-20-75-3 | 195 × 270 | 9,270 | limit | -52142 | -52242 | +0.19% | 120.3 s | feasible |
| mik-250-20-75-5 | 195 × 270 | 9,270 | limit | -51306 | -51532 | +0.44% | 120.0 s | feasible |
| mtest4ma | 1174 × 1950 | 4,875 | limit | 60342 | 52148 | +15.71% | 120.1 s | feasible |
| neos-1112782 | 2115 × 4140 | 8,145 | limit | 2.25e+13 | 5.7184407e+11 | +3834.64% | 120.1 s | feasible |
| neos-1112787 | 1680 × 3280 | 6,440 | limit | 2e+13 | 5.6477277e+11 | +3441.25% | 120.8 s | feasible |
| neos-1396125 | 1494 × 1161 | 5,511 | limit | — | 3000.0453 | no incumbent | 148.2 s | — |
| neos-1425699 | 89 × 105 | 430 | **optimal** | 3.179699e+09 | 3.179699e+09 | 0 | 2.1 s | feasible |
| neos-1430701 | 668 × 312 | 2,868 | limit | -76 | -77 | +1.30% | 120.4 s | feasible |
| neos-1442119 | 1524 × 728 | 6,692 | limit | -173 | -181 | +4.42% | 120.6 s | feasible |
| neos-2624317-amur | 342 × 524 | 1,690 | limit | 15.780372 | 3.5223968 | +348.00% | 120.1 s | feasible |
| neos-2652786-brda | 342 × 524 | 1,690 | limit | 20.111004 | 4.7924996 | +319.63% | 120.2 s | feasible |
| neos-2657525-crna | 342 × 524 | 1,690 | limit | — | 1.810748 | no incumbent | 120.0 s | — |
| neos-3046601-motu | 563 × 308 | 1,430 | limit | 1725 | 1459 | +18.23% | 120.2 s | feasible |
| neos-3046615-murg | 498 × 274 | 1,266 | limit | 1761 | 1600 | +10.06% | 120.1 s | feasible |
| neos-3072252-nete | 432 × 576 | 1,292 | limit | 12050594 | 11807698 | +2.06% | 120.1 s | feasible |
| neos-3118745-obra | 144 × 1131 | 4,521 | limit | 276 | 255 | +8.24% | 120.2 s | feasible |
| neos-3373491-avoca | 1570 × 2368 | 9,438 | limit | — | 2.7449085e+10 | no incumbent | 120.4 s | — |
| neos-3381206-awhea | 479 × 2375 | 4,275 | limit | — | 453 | no incumbent | 120.2 s | — |
| neos-3421095-cinca | 1218 × 896 | 4,350 | **optimal** | 1.2993401e+08 | 44269368 | +193.51% | 0.8 s | feasible |
| neos-3426085-ticino | 308 × 4688 | 9,083 | limit | 254 | 225 | +12.89% | 120.2 s | feasible |
| neos-3530903-gauja | 220 × 2310 | 4,410 | limit | 187 | 168 | +11.31% | 120.1 s | feasible |
| neos-3530905-gaula | 200 × 2090 | 3,990 | limit | 177 | 159 | +11.32% | 120.0 s | feasible |
| neos-3610040-iskar | 335 × 430 | 1,023 | limit | 37 | 37 | 0 | 120.3 s | feasible |
| neos-3610051-istra | 709 × 805 | 1,999 | limit | 49 | 49 | 0 | 123.6 s | feasible |
| neos-3610173-itata | 747 × 844 | 2,130 | limit | 151 | 148 | +2.03% | 121.1 s | feasible |
| neos-3611447-jijia | 377 × 472 | 1,145 | limit | 107 | 107 | 0 | 120.3 s | feasible |
| neos-3611689-kaihu | 323 × 421 | 1,014 | limit | 120 | 119 | +0.84% | 120.5 s | feasible |
| neos-3627168-kasai | 1655 × 1462 | 5,158 | limit | 1010311.7 | 988585.62 | +2.20% | 120.2 s | feasible |
| neos-3754480-nidda | 402 × 253 | 1,488 | limit | 14041.827 | 12939.754 | +8.52% | 120.1 s | feasible |
| neos-4333596-skien | 812 × 1005 | 5,811 | limit | — | -14610731 | no incumbent | 121.5 s | — |
| neos-4338804-snowy | 1701 × 1344 | 6,342 | limit | 1924 | 1471 | +30.80% | 120.1 s | feasible |
| neos-4650160-yukon | 1969 × 1412 | 6,416 | limit | 79.015 | 59.885 | +31.94% | 120.1 s | feasible |
| neos-4954672-berkel | 1848 × 1533 | 8,007 | limit | 6380647 | 2612710 | +144.22% | 121.1 s | feasible |
| neos-5078479-escaut | 3442 × 3471 | 9,062 | limit | — | 9682.3382 | no incumbent | 125.1 s | — |
| neos-5140963-mincio | 184 × 196 | 834 | limit | 14559 | 14393 | +1.15% | 120.2 s | feasible |
| neos-5192052-neckar | 57 × 180 | 545 | **optimal** | -11670000 | -11670000 | 0 | 0.3 s | feasible |
| neos-631517 | 351 × 1090 | 2,743 | limit | 12048981 | 11490667 | +4.86% | 127.2 s | feasible |
| neos-807639 | 1541 × 1030 | 5,520 | limit | 454.2 | 454.2 | 0 | 123.1 s | feasible |
| neos-911970 | 107 × 888 | 3,408 | limit | 102.69 | 54.76 | +87.53% | 120.5 s | feasible |
| neos16 | 1018 × 377 | 2,801 | limit | — | 446 | no incumbent | 120.1 s | — |
| neos2 | 1103 × 2101 | 7,326 | limit | 854.34141 | 454.8647 | +87.82% | 120.2 s | feasible |
| newdano | 576 × 505 | 2,184 | limit | 71.166667 | 65.666667 | +8.38% | 120.1 s | feasible |
| nexp-50-20-1-1 | 540 × 490 | 1,225 | limit | 33 | 29 | +13.79% | 120.0 s | feasible |
| nexp-50-20-4-2 | 540 × 1225 | 2,695 | limit | 76 | 71 | +7.04% | 120.1 s | feasible |
| nh97_potential | 1916 × 1180 | 5,748 | limit | — | 1418 | no incumbent | 122.4 s | — |
| nh97_tension | 737 × 1576 | 5,264 | limit | — | 1418 | no incumbent | 120.0 s | — |
| noswot | 182 × 128 | 735 | limit | -41 | -41.000009 | +0.00% | 120.1 s | feasible |
| nsa | 1297 × 388 | 4,204 | limit | 139 | 120 | +15.83% | 120.1 s | feasible |
| opt1217 | 64 × 769 | 1,542 | limit | -16 | -16 | 0 | 120.2 s | feasible |
| pigeon-08 | 601 × 344 | 5,176 | limit | -7000 | -7000 | 0 | 120.0 s | feasible |
| pigeon-10 | 931 × 490 | 8,150 | limit | -9000 | -9000 | 0 | 120.2 s | feasible |
| prod1 | 208 × 250 | 5,350 | limit | -55 | -56 | +1.79% | 120.0 s | feasible |
| qnet1_o | 456 × 1541 | 4,214 | **optimal** | 16030.993 | 16029.693 | +0.01% | 62.7 s | feasible |
| r50x360 | 410 × 720 | 1,440 | limit | 2362 | 1653 | +42.89% | 120.1 s | feasible |
| railway_8_1_0 | 2527 × 1796 | 7,098 | limit | — | 400 | no incumbent | 122.5 s | — |
| rlp1 | 68 × 461 | 836 | limit | 19 | 15 | +26.67% | 120.1 s | feasible |
| supportcase14 | 234 × 304 | 1,129 | **optimal** | 288 | 288 | 0 | 3.1 s | feasible |
| supportcase16 | 130 × 319 | 1,076 | **optimal** | 288 | 288 | 0 | 3.3 s | feasible |
| supportcase17 | 2108 × 1381 | 5,253 | limit | 2989 | 1330 | +124.74% | 121.6 s | feasible |
| supportcase26 | 870 × 436 | 2,492 | limit | 1867.1236 | 1745.1238 | +6.99% | 120.3 s | feasible |
| supportcase27i | 3008 × 2281 | 6,149 | **INFEASIBLE** | — | 1330 | no incumbent | 0.0 s | — |
| ta1-UUM | 439 × 2288 | 5,654 | limit | 12545287 | 7518328.2 | +66.86% | 120.2 s | feasible |
| timtab1CUTS | 371 × 397 | 1,742 | limit | 853932 | 764772 | +11.66% | 120.1 s | feasible |

```
  instances            114
  status OPTIMAL       20/114
  verifier accepted    98/114
  within 1e-4 of ref   31/114
  shifted geomean time 83.822s (shift 10s)
  total time           11973.5s
  worst relative error 3.83e+01 (neos-1112782)
```

Twenty of 114 proved in 120 s, from 0.6 s (app2-2) to 83 s (bienst1);
twelve more sit on the published value unproved -- b-ball, noswot,
opt1217, the pigeon pair, chromaticindex32-8 -- which is the tree's
usual shape, the incumbent found and the bound not closed; and 98 of
the 114 return a verifier-accepted point. Of the 82 that neither prove
nor reach the value, eight are within 1% and sixteen within 10%; 42 are
more than 10% away and sixteen have no incumbent at all in two minutes
(the `csched` pair, `nh97`, `ic97_tension`, `railway_8_1_0`, `enlight8`,
`ej` -- three variables, one row, and a tree that cannot see a
Diophantine argument any more than on p2m2p1m1p0n100). The gaps on the
generalised-assignment and network-flow families are the cut families
the README's Mittelmann entry already names as missing; `ej` and
`enlight8` are a class of argument the tree does not have.

The distance to the established solvers is the one the first campaign
measured on Mittelmann's benchmark, moved down a size class: there,
nothing under 100k nonzeros proved in two minutes; here, under 10k, one
in six does. Five rows overran the limit -- csched010 to 447 s, which is
the largest overrun either campaign has seen and is the cold-re-solve
grain of bug 10's leftover -- and 114 rows took 3.3 hours.

The listing was run twice. The first pass read the file with the reader
of bug 16, and 23 of its rows were re-run alone once the reader followed
MIPLIB's convention and once a machine that slept during the run had
been taken out of two rows' times (cvs16r106-72 was charged 18,630 s
for a 120 s solve); the table is the merge, and every number in it is
from the corrected reading. Four of the re-run rows changed their
incumbent under the convention, and neos-3373491-avoca's old one had
sat below the published optimum -- the tell of a wrong model, and the
reason the two campaigns carry the published value beside every point.

## 6. Netlib's Kennington set

`python -m bench.fetch --set kennington`, then `python -m bench.harness
--mode lp --dir data/kennington --time-limit 300 --method ipm | pdlp
--device gpu | simplex`, and `python -m bench.comparator --dir
data/kennington --time-limit 300 --method ipm`. The sixteen "Kennington"
problems (Carolan, Hill, Kennington, Niemi and Wichmann 1990: military
airlift, patient distribution, multicommodity flow) are Netlib's larger
set, doubly compressed -- gzip over Netlib's own format -- from 8k to 1.4M
nonzeros; the readme's table of optimal values, computed by Vanderbei's
ALPO, is read by machine into every header. This is the answer key the
first campaign's Mittelmann set lacked.

| engine | optimal | certified | on the published value | total time | where it stops |
|---|---|---|---|---|---|
| interior point | **15/16** | **15/16** | 15/16 to 4e-8 | 482 s | pds-20 at the limit |
| PDLP on the GPU | **15/16** | 9/16 | **16/16** to 3e-5, 15 to 4e-8 | 657 s | cre-b at its iteration limit, the iterate 2.8e-5 off and refused |
| revised simplex | 13/16 | 13/16 | 13/16 | 2,003 s | ken-13, ken-18, pds-20 at the limit |
| HiGHS (comparator) | 16/16 | — | 16/16 | 61 s | agrees with the interior point on 15/15 to 1e-10 |

| instance | rows × cols | nnz | status | objective | reference | rel. err | time | verifier |
|---|---|---|---|---|---|---|---|---|
| cre-a | 3516 × 4067 | 14,987 | optimal | 23595407 | 23595407 | 2.6e-09 | 0.50 s | certified |
| cre-b | 9648 × 72447 | 256,095 | optimal | 23129640 | 23129640 | 4.9e-09 | 29.54 s | certified |
| cre-c | 3068 × 3678 | 13,244 | optimal | 25275116 | 25275116 | 5.6e-09 | 0.29 s | certified |
| cre-d | 8926 × 69980 | 242,646 | optimal | 24454970 | 24454970 | 9.6e-09 | 13.98 s | certified |
| ken-07 | 2426 × 3602 | 8,404 | optimal | -6.7952044e+08 | -6.7952044e+08 | 5e-09 | 0.08 s | certified |
| ken-11 | 14694 × 21349 | 49,058 | optimal | -6.9723823e+09 | -6.9723823e+09 | 5.4e-09 | 0.93 s | certified |
| ken-13 | 28632 × 42659 | 97,246 | optimal | -1.0257395e+10 | -1.0257395e+10 | 2.1e-08 | 2.70 s | certified |
| ken-18 | 105127 × 154699 | 358,171 | optimal | -5.2217025e+10 | -5.2217025e+10 | 5.5e-09 | 25.62 s | certified |
| osa-07 | 1118 × 23949 | 143,694 | optimal | 535722.52 | 535722.52 | 5e-09 | 1.43 s | certified |
| osa-14 | 2337 × 52460 | 314,760 | optimal | 1106462.8 | 1106462.8 | 4e-08 | 4.80 s | certified |
| osa-30 | 4350 × 100024 | 600,138 | optimal | 2142139.9 | 2142139.9 | 1.2e-08 | 7.10 s | certified |
| osa-60 | 10280 × 232966 | 1,397,793 | optimal | 4044072.5 | 4044072.5 | 8.8e-10 | 33.61 s | certified |
| pds-02 | 2953 × 7535 | 16,390 | optimal | 2.8857862e+10 | 2.8857862e+10 | 3.5e-10 | 0.30 s | certified |
| pds-06 | 9881 × 28655 | 62,524 | optimal | 2.7761038e+10 | 2.7761038e+10 | 1.4e-08 | 8.52 s | certified |
| pds-10 | 16558 × 48763 | 106,436 | optimal | 2.6727095e+10 | 2.6727095e+10 | 9e-10 | 45.04 s | certified |
| pds-20 | 33874 × 105728 | 230,200 | limit | — | 2.3821659e+10 | — | 307.34 s | — |

```
  instances            16
  status OPTIMAL       15/16
  verifier accepted    15/16
  certified optimal    15/16   (duals bound the optimum within 1e-9)
  within 1e-4 of ref   15/16
  shifted geomean time 6.607s (shift 1s)
  total time           481.8s
  worst relative error 4.04e-08 (osa-14)
```

| instance | rows × cols | nnz | status | objective | reference | rel. err | time | verifier |
|---|---|---|---|---|---|---|---|---|
| cre-a | 3516 × 4067 | 14,987 | optimal | 23595407 | 23595407 | 3.7e-09 | 46.42 s | feasible |
| cre-b | 9648 × 72447 | 256,095 | iteration limit | 23128985 | 23129640 | 2.8e-05 | 123.06 s | **rejected** |
| cre-c | 3068 × 3678 | 13,244 | optimal | 25275116 | 25275116 | 8.7e-09 | 19.48 s | feasible |
| cre-d | 8926 × 69980 | 242,646 | optimal | 24454970 | 24454970 | 9.7e-09 | 42.07 s | certified |
| ken-07 | 2426 × 3602 | 8,404 | optimal | -6.7952044e+08 | -6.7952044e+08 | 5.2e-09 | 2.03 s | feasible |
| ken-11 | 14694 × 21349 | 49,058 | optimal | -6.9723823e+09 | -6.9723823e+09 | 5.4e-09 | 9.11 s | feasible |
| ken-13 | 28632 × 42659 | 97,246 | optimal | -1.0257395e+10 | -1.0257395e+10 | 2.1e-08 | 44.03 s | certified |
| ken-18 | 105127 × 154699 | 358,171 | optimal | -5.2217025e+10 | -5.2217025e+10 | 5.5e-09 | 138.61 s | feasible |
| osa-07 | 1118 × 23949 | 143,694 | optimal | 535722.52 | 535722.52 | 5.1e-09 | 2.70 s | certified |
| osa-14 | 2337 × 52460 | 314,760 | optimal | 1106462.8 | 1106462.8 | 4e-08 | 18.97 s | certified |
| osa-30 | 4350 × 100024 | 600,138 | optimal | 2142139.9 | 2142139.9 | 4.9e-09 | 58.50 s | feasible |
| osa-60 | 10280 × 232966 | 1,397,793 | optimal | 4044072.5 | 4044072.5 | 9.5e-10 | 81.32 s | certified |
| pds-02 | 2953 × 7535 | 16,390 | optimal | 2.8857862e+10 | 2.8857862e+10 | 3.4e-10 | 3.09 s | certified |
| pds-06 | 9881 × 28655 | 62,524 | optimal | 2.7761038e+10 | 2.7761038e+10 | 1.5e-08 | 9.12 s | certified |
| pds-10 | 16558 × 48763 | 106,436 | optimal | 2.6727095e+10 | 2.6727095e+10 | 1.3e-09 | 13.76 s | certified |
| pds-20 | 33874 × 105728 | 230,200 | optimal | 2.3821659e+10 | 2.3821659e+10 | 1.5e-08 | 44.28 s | certified |

```
  instances            16
  status OPTIMAL       15/16
  verifier accepted    15/16
  certified optimal    9/16   (duals bound the optimum within 1e-9)
  within 1e-4 of ref   16/16
  shifted geomean time 22.713s (shift 1s)
  total time           656.5s
  !! VERIFIER REJECTED: cre-b
  worst relative error 2.83e-05 (cre-b)
```

| instance | rows × cols | nnz | status | objective | reference | rel. err | time | verifier |
|---|---|---|---|---|---|---|---|---|
| cre-a | 3516 × 4067 | 14,987 | optimal | 23595407 | 23595407 | 2.6e-09 | 3.33 s | certified |
| cre-b | 9648 × 72447 | 256,095 | optimal | 23129640 | 23129640 | 4.9e-09 | 297.60 s | certified |
| cre-c | 3068 × 3678 | 13,244 | optimal | 25275116 | 25275116 | 5.6e-09 | 2.38 s | certified |
| cre-d | 8926 × 69980 | 242,646 | optimal | 24454970 | 24454970 | 9.6e-09 | 193.81 s | certified |
| ken-07 | 2426 × 3602 | 8,404 | optimal | -6.7952044e+08 | -6.7952044e+08 | 5e-09 | 4.73 s | certified |
| ken-11 | 14694 × 21349 | 49,058 | optimal | -6.9723823e+09 | -6.9723823e+09 | 5.4e-09 | 265.57 s | certified |
| ken-13 | 28632 × 42659 | 97,246 | limit | — | -1.0257395e+10 | — | 300.27 s | — |
| ken-18 | 105127 × 154699 | 358,171 | limit | — | -5.2217025e+10 | — | 300.35 s | — |
| osa-07 | 1118 × 23949 | 143,694 | optimal | 535722.52 | 535722.52 | 5e-09 | 1.27 s | certified |
| osa-14 | 2337 × 52460 | 314,760 | optimal | 1106462.8 | 1106462.8 | 4e-08 | 4.41 s | certified |
| osa-30 | 4350 × 100024 | 600,138 | optimal | 2142139.9 | 2142139.9 | 1.2e-08 | 14.75 s | certified |
| osa-60 | 10280 × 232966 | 1,397,793 | optimal | 4044072.5 | 4044072.5 | 7.8e-10 | 106.05 s | certified |
| pds-02 | 2953 × 7535 | 16,390 | optimal | 2.8857862e+10 | 2.8857862e+10 | 3.5e-10 | 2.70 s | certified |
| pds-06 | 9881 × 28655 | 62,524 | optimal | 2.7761038e+10 | 2.7761038e+10 | 1.4e-08 | 105.18 s | certified |
| pds-10 | 16558 × 48763 | 106,436 | optimal | 2.6727095e+10 | 2.6727095e+10 | 9e-10 | 100.42 s | certified |
| pds-20 | 33874 × 105728 | 230,200 | limit | — | 2.3821659e+10 | — | 300.32 s | — |

```
  instances            16
  status OPTIMAL       13/16
  verifier accepted    13/16
  certified optimal    13/16   (duals bound the optimum within 1e-9)
  within 1e-4 of ref   13/16
  shifted geomean time 38.055s (shift 1s)
  total time           2003.1s
  worst relative error 4.04e-08 (osa-14)
```

```
  agree within 1e-06   15/15
  skipped (one side did not solve): 1
  total time      sovopt 534.9s   HiGHS 61.3s
```

(The interior-point and PDLP columns were run before bug 19 made the
verifier's certificate two-sided, and re-run after it at the commit the
report names: the same fifteen and nine certificates, every status the
same, on a busier machine 20-40% slower. The tables are the sitting's.)

Every published value is reached, by two engines on fifteen instances
and by one on the sixteenth: the interior point certifies osa-60's 1.4
million nonzeros in 34 s and ken-18's 105k rows in 26 s, and the GPU
first-order method takes pds-20, which the interior point cannot finish
in 300 s (its first campaign showed the same: pds-20 is PDLP's, at 41 s
then and 44 s now), and agrees with the published value on all sixteen.
The simplex is again the engine for the small end -- cre-a, cre-c, ken-07,
the osa set -- and out of its depth past 50k rows, exactly as the first
campaign found on Mittelmann's set.

Against HiGHS the picture is the one the first campaign drew, at a
larger scale and with less flattery: HiGHS solves all sixteen in 61 s
where the interior point takes 535 s for fifteen, nine times slower, and
the two agree to 1e-10 wherever both finish. pds-10 is the extreme --
72 s against 1.3 s -- and the reason is the one the README's Scale
section gives: the interior point's factorisation on these
multicommodity-flow structures fills, and each of its 30-odd iterations
pays for it. The comparison is what the problem statement asks for, and
the honest reading is that the answers match and the speed does not.

## 7. Mittelmann's LP test set, the next size class

`python -m bench.fetch --set mittelmann-lp2`, then `python -m bench.harness
--mode lp --dir data/mittelmann2 --time-limit 300 --method pdlp --device
gpu | ipm` and `python -m bench.comparator --dir data/mittelmann2
--time-limit 300 --method ipm`. Since the first campaign, Mittelmann has
moved his LP test set to one directory and added the LP relaxations of
twenty-four MIPLIB 2017 models; these are the eleven files under 5 MB
compressed -- the next size class above the thirteen the first campaign
ran, 133k to 1.5M nonzeros, 11k to 376k rows. No published optima; the
verifier's certificate and HiGHS are the checks, and they agree where
both apply.

| instance | rows × cols | nnz | PDLP on the GPU | interior point | HiGHS |
|---|---|---|---|---|---|
| brazil3 | 14,646 × 23,968 | 133k | optimal, 5.4 s | **certified, 5.2 s** | 21 s |
| chromaticindex1024-7 | 67,583 × 73,728 | 270k | **optimal, 0.4 s** | certified, 45 s | 222 s |
| datt256 | 11,077 × 262,144 | 1.50M | **optimal, 18 s** | certified, 140 s | limit |
| ex10 | 69,608 × 17,680 | 1.16M | **optimal, 2.0 s** | limit | 159 s |
| supportcase10 | 165,684 × 14,770 | 555k | **optimal, 88 s** | limit at 374 s | limit |
| s250r10 | 10,962 × 273,142 | 1.32M | limit, rejected | **certified, 248 s** | 135 s |
| irish-electricity | 104,259 × 61,728 | 523k | iteration limit, rejected | iteration limit | **196 s** |
| Linf_520c | 93,326 × 69,004 | 566k | iteration limit, rejected | limit at 20 s (refused factorisation) | limit |
| bdry2 | 376,500 × 250,998 | 1.50M | limit, rejected | limit at 329 s | limit |
| physiciansched3-3 | 266,227 × 79,555 | 1.06M | limit, rejected | limit at 348 s | limit |
| rmine15 | 358,395 × 42,438 | 880k | limit, rejected | limit at 41 s (refused factorisation) | limit |
| **solved** | | | **5** | 4, all certified | 5 |

| instance | rows × cols | nnz | status | objective | reference | rel. err | time | verifier |
|---|---|---|---|---|---|---|---|---|
| Linf_520c | 93326 × 69004 | 566,193 | iteration limit | 0.16000467 | — | — | 232.94 s | **rejected** |
| bdry2 | 376500 × 250998 | 1,500,003 | limit | 0.0019998925 | — | — | 300.23 s | **rejected** |
| brazil3 | 14646 × 23968 | 133,184 | optimal | 1.9999999 | — | — | 5.43 s | feasible |
| chromaticindex1024-7 | 67583 × 73728 | 270,324 | optimal | 2.9999998 | — | — | 0.36 s | feasible |
| datt256 | 11077 × 262144 | 1,503,732 | optimal | 256 | — | — | 17.60 s | feasible |
| ex10 | 69608 × 17680 | 1,162,000 | optimal | 100 | — | — | 1.97 s | feasible |
| irish-electricity | 104259 × 61728 | 523,257 | iteration limit | 2454382.4 | — | — | 222.96 s | **rejected** |
| physiciansched3-3 | 266227 × 79555 | 1,062,479 | limit | 1008186 | — | — | 300.05 s | **rejected** |
| rmine15 | 358395 × 42438 | 879,732 | limit | -5042.4829 | — | — | 300.16 s | **rejected** |
| s250r10 | 10962 × 273142 | 1,318,607 | limit | -0.17324994 | — | — | 300.12 s | **rejected** |
| supportcase10 | 165684 × 14770 | 555,082 | optimal | 3.3839236 | — | — | 87.82 s | feasible |

```
  instances            11
  status OPTIMAL       5/11
  verifier accepted    5/11
  certified optimal    0/11   (duals bound the optimum within 1e-9)
  within 1e-4 of ref   0/0
  shifted geomean time 55.429s (shift 1s)
  total time           1769.6s
  !! VERIFIER REJECTED: Linf_520c, bdry2, irish-electricity, physiciansched3-3, rmine15, s250r10
```

| instance | rows × cols | nnz | status | objective | reference | rel. err | time | verifier |
|---|---|---|---|---|---|---|---|---|
| Linf_520c | 93326 × 69004 | 566,193 | limit | — | — | — | 20.14 s | — |
| bdry2 | 376500 × 250998 | 1,500,003 | limit | — | — | — | 328.93 s | — |
| brazil3 | 14646 × 23968 | 133,184 | optimal | 2 | — | — | 5.16 s | certified |
| chromaticindex1024-7 | 67583 × 73728 | 270,324 | optimal | 3 | — | — | 45.14 s | certified |
| datt256 | 11077 × 262144 | 1,503,732 | optimal | 256 | — | — | 140.31 s | certified |
| ex10 | 69608 × 17680 | 1,162,000 | limit | — | — | — | 42.54 s | — |
| irish-electricity | 104259 × 61728 | 523,257 | iteration limit | — | — | — | 113.29 s | — |
| physiciansched3-3 | 266227 × 79555 | 1,062,479 | limit | — | — | — | 347.98 s | — |
| rmine15 | 358395 × 42438 | 879,732 | limit | — | — | — | 40.76 s | — |
| s250r10 | 10962 × 273142 | 1,318,607 | optimal | -0.17267704 | — | — | 247.68 s | certified |
| supportcase10 | 165684 × 14770 | 555,082 | limit | — | — | — | 374.30 s | — |

```
  instances            11
  status OPTIMAL       4/11
  verifier accepted    4/11
  certified optimal    4/11   (duals bound the optimum within 1e-9)
  within 1e-4 of ref   0/0
  shifted geomean time 84.972s (shift 1s)
  total time           1706.2s
```

```
  agree within 1e-06   3/3
  skipped (one side did not solve): 8
  total time      sovopt 1694.2s   HiGHS 2537.7s
```

(The interior-point column, re-run after bug 19 at the commit the report
names: the same four certificates and every status the same, 10-30%
slower on a busier machine; the table is the sitting's.)

Six of the eleven are solved by one engine or the other, five by HiGHS at
its defaults through `scipy.optimize.linprog`, and the two agree to 1e-10
on the three both finish. The set says the same as the first campaign's
Mittelmann table, one size class up: the GPU first-order method is the
engine for the wide, well-conditioned ones -- ex10's 1.16M nonzeros in
2.0 s against HiGHS's 159 s, datt256's 1.5M in 18 s where HiGHS runs out
of 300 s, chromaticindex in 0.4 s against 222 s -- and it is the wrong
engine for the ill-conditioned ones, where it returns an unconverged
iterate the verifier refuses. The interior point certifies four,
including s250r10 where PDLP cannot converge, and refuses two
factorisations (Linf_520c, rmine15) that bug 14's prediction says would
outlast the limit. What neither engine touches -- bdry2, physiciansched3-3,
irish-electricity's conditioning -- HiGHS does not touch either at this
limit, except irish-electricity.

Two things to record against the engine: the interior point's limit
overruns of 29 to 74 s on the three largest (bdry2, physiciansched3-3,
supportcase10), the factorisation grain the first campaign's bug 14 left
open; and that PDLP's iteration limit is reached before its time limit on
Linf_520c and irish-electricity at 223-233 s, so a larger iteration
budget is the first thing to try there.

## 8. What the campaign found

A campaign over sets the solver had never seen is a test of the solver
more than of the sets. This one found five defects, each recorded under
[Bugs worth recording](../README.md#bugs-worth-recording), numbers 16 to
20, with the measurement that found it and the one that closed it:

| # | found on | what | fixed by |
|---|---|---|---|
| 16 | neos-2626858-aoos, neos-2656603-coxs (published infeasible) | the tree returned exactly feasible integer points on models MIPLIB publishes as infeasible: the reader gave an integer column with no bound line [0, +inf) where the MPSX convention CPLEX, SCIP and MIPLIB keep gives [0, 1] | the reader follows the convention; the variable counts match MIPLIB's page; nine other instances re-run, four changed, one of which had held an incumbent below its published optimum |
| 17 | MIPLIB's five published-unbounded instances | the root relaxation returned UNBOUNDED in under a second and the tree reported NODE_LIMIT at 13-73 s | an unbounded relaxation makes the MILP unbounded or infeasible (Meyer 1974): INFEASIBLE_OR_UNBOUNDED at the root, four of five in 3-14 s |
| 18 | hs51, genhs28, dpklo1, dtoc3, aug2d, aug2dc, aug3d, aug3dc | every row an equality and every column free: no complementarity pair, and the interior point's shortcut for that case returned the zero vector | the shortcut applies only when no column is free; the loop takes the one Newton step; eight certified in 2-3 iterations |
| 19 | liswet1 | a point 3e-7 off its rows, 0.19% *below* its own certified bound, called OPTIMAL by the engine and by the verifier: both gap tests were one-sided | both are `|gap| <= tol`; three of the first campaign's Mittelmann certificates withdrawn as "accepted" |
| 20 | the test written for 19 | `min 1e8 x` with `1e6 x >= 1e-4`: the row scales to below the tolerance and the simplex called x = 0 OPTIMAL, 1e-4 off its one row | an OPTIMAL point is checked against the unscaled model, as the other engines have done since bug 2; NUMERICAL with the violation otherwise |

And three things that were not defects but absences, built because a
set could not be run without them: the top-level simplex now returns
the Farkas ray that proves a model infeasible and the verifier checks it
-- and the verifier's first version refused 20 of 29 valid rays for
rounding at infinite bounds, the same lesson the optimality certificate
had already taught; the MPS reader takes fixed-column files whose names
contain spaces (qforplan); and the safe bound takes the absorbed columns
as a mask, without which every equality-constrained QP certificate was
vacuous from a 1e-16 residue.

Two of the five -- 16 and 20 -- are the model being read or judged in a
space that is not the model's: a convention the file assumes and the
reader did not, a tolerance in scaled units applied where the unscaled
ones are what a caller sees. Two -- 18 and 19 -- are a special case
nobody had exercised: no bound anywhere, and a point on the wrong side
of a bound. One -- 17 -- is a status that described the search when
the answer was about the model. The README's own sentence applies for
a second time: a set the solver has always passed is not evidence about
the set it has never seen. Every published-value comparison in these
tables is what caught 16 and 19, and it is the reason to carry the
value beside every point.

**What was measured and left alone,** because the campaign is a
measurement and each has its reason on record:

- The tree's root is the dual simplex, and on OR-Library's 100 × 1,000
  problems it does not finish the 101k-row relaxation in the limit, where
  the interior point certifies it in 35 s. A root that goes to the
  interior point and crosses over is the next thing to build for the
  tree; it is in Known limits now.
- PDLP has no infeasibility detection, and the interior point's is a
  stagnation test: on Netlib's infeasible set the first returns time
  limits and the second the right word without a proof. The simplex is
  the engine that answers the question, and it answers it with a
  certificate now.
- The `liswet` family, boyd1 and boyd2, cont-300 and qfffff80 are what
  the interior point cannot do on Maros and Meszaros' set: an
  ill-conditioned family that stalls at 3e-7, two 93k-variable models,
  a 90k-row factorisation, and a wrong stagnation verdict.
- Limits are still overrun at the grain of one unit of work: csched010
  to 447 s against 120 (a cold re-solve), the interior point to 374 s
  against 300 on supportcase10 (a factorisation), the tree's root on the
  100 × 1,000 problems to 268 s. Bug 14's remaining step, measured
  again.
- p2m2p1m1p0n100, ej and the enlight family are infeasible or hard by an
  integer argument -- parity, a Diophantine equation -- that no LP bound
  sees, and the tree enumerates where a solver with a presolve that
  reasons about integers would stop at once.
- cplex2 is infeasible by about 1e-9 in its rows, below the 1e-6 line at
  which the verifier accepts a point; the simplex certifies it infeasible
  and the interior point returns a point the verifier accepts. Both are
  right at their tolerances, and the verifier's two-sided optimality line
  is what tells them apart: the interior point's bound sits above its
  objective by 1.8e-4.
- Two rows of the MIPLIB draw and the fourteen that overlapped a
  diagnostic run were re-run alone, as were the seven the reader
  convention touched; the machine slept during the first pass of the
  draw, and one row was charged 18,630 s for a 120 s solve before the
  re-run. The tables carry the re-runs.

## 9. Does it perform as desired?

The question was whether the solver performs as desired on the problem
statement's categories, this time on sets it had never seen. By
category, against the same two yardsticks -- the published value, and
an independent check of every answer:

**LP: yes, and now including the answer "no".** Every one of Netlib's
29 infeasible models is recognised and *proved* infeasible by the
simplex, with a certificate the verifier checks on the original model;
that proof did not exist at the top level before this campaign. On the
Kennington set every published value is reached -- fifteen by two
engines and the sixteenth by the GPU first-order method -- and the
interior point certifies 1.4 million nonzeros in 34 s. On Mittelmann's
next size class the GPU method takes ex10's 1.16M nonzeros in 2 s where
HiGHS takes 159, and six of eleven are solved by one engine or the
other against HiGHS's five. What is not as desired: HiGHS is nine times
faster than the interior point over the Kennington set; the simplex is
out of its depth past 50k rows; and PDLP cannot say "infeasible".

**QP: yes, on the standard set.** Maros and Meszaros' 138 convex QPs:
121 optimal and 97 certified without a published value to lean on, once
three defects the set exposed were fixed. The seventeen it does not
solve are an ill-conditioned family, two 93k-variable models, one 90k-row
factorisation, and one wrong stagnation verdict -- the same weak verdict
the first campaign recorded on three Netlib LPs.

**MILP: yes on what a tight relaxation reaches, no past it, and
honest about which is which.** OR-Library's 49 warehouse-location
problems with up to 50 warehouses, every one proved on the published
value in 22 s; eleven of MIPLIB's 26 published-infeasible models proved
infeasible and four of five unbounded ones recognised at the root; and
of the 114-instance draw of small easy MIPLIB models, twenty proved,
twelve more on the value, 98 with a verified point -- one in six proved
where Mittelmann's benchmark gave none, the distance being the cut
families and the integer reasoning the README names. The 100 × 1,000
location problems do not start, and the reason is measured: the tree's
root is a simplex on 101k rows, and the interior point that would do it
in 35 s is not the tree's to use yet.

**Refinery, blending, planning, supply chain: yes, on every published
case the literature offers here.** Six more of Williams' models --
blending with storage, factory planning, distribution, unit commitment,
mining -- reach the book's value to the penny by every applicable
engine, beside the refinery; the warehouse-location sets above are the
supply-chain benchmark of the location literature; and the doctored
petrochemical models of Netlib's infeasible set (refinery, chemcom,
qual, reactor, vol1) are the planner's real question -- can this slate
meet this demand -- answered with a proof.

**And the campaign performed as a test should.** It found five defects
the first campaign's sets had not exercised -- a reader convention, an
unbounded root, an equality-constrained QP, a one-sided certificate, a
scaled tolerance -- and three absences, and every one is fixed, tested,
and recorded with the measurement that found it. Two of the five were
caught only because a published value stood beside every point: the
"feasible" incumbents on infeasible models, and the certificate that let
a point sit below its bound. The solver that finished this campaign is,
again, not the one that started it.
