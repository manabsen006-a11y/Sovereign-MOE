# Test_Data — two years of hourly gasoline-blending data

The motor-spirit blending area of a 15 MMTPA coastal refinery, hour by hour
from **1-Oct-2026 00:00 to 30-Sep-2028 23:00 (17,544 hours)**, written as the
six tables `sovopt blend` reads.

| File | Rows | Cells | Numbers |
|---|---:|---:|---:|
| components.csv | 30 | 660 | 570 |
| products.csv | 4 | 84 | 60 |
| price.csv | 17,544 | 543,864 | 526,320 |
| demand.csv | 70,176 | 350,880 | 210,528 |
| capacity.csv | 315,792 | 947,376 | 315,792 |
| pools.csv | 7 | 21 | 7 |
| **total** | | **1,842,885** | **1,053,277** |

**These are representative values, not any refinery's records.** The
product limits are the published standards; stream properties are typical
values for each stream type from the refining literature; prices, rates,
campaigns and outages are realistic in level and pattern but are generated.
`generate.py` makes them, deterministically (seed 26119):

```bash
python Test_Data/generate.py
```

## What is in it

**30 streams** (`components.csv`), each with its blend header, hourly
rundown, tankage and 13 lab qualities:

| Unit | Streams |
|---|---|
| 3 crude units | light straight-run naphtha (hydrotreated) ×3, heavy straight-run naphtha (untreated) ×3 |
| 2 hydrocrackers | light naphtha ×2, heavy naphtha surplus |
| 2 isomerisation units | isomerate (with recycle, and once-through) |
| 2 CCR reformers + splitter + BenSat | heavy reformate ×2, benzene-saturated light reformate |
| aromatics complex | raffinate, C9 aromatics, toluene |
| FCC and petrochemical FCC | light, mid and heavy cracked naphtha from each, after selective hydrotreating |
| DHDT | stabilised wild naphtha |
| gas plant | n-butane, i-butane |
| imports | alkylate, MTBE, reformate, natural gasoline |

`cost` is each stream's opportunity value on 1-Oct-2026 (its alternative
disposition: naphtha export, the aromatics complex, LPG, import parity);
`available` is its rundown in kL/h; `direct` is `*` for streams with their
own tank, blank for the ones that share a rundown tank (`pools.csv`).

**4 products** (`products.csv`):

| Product | Specification |
|---|---|
| MS91_BSVI | IS 2796:2017 BS-VI regular: RON ≥ 91, MON ≥ 81, density 720–775, DVPE ≤ 60 kPa, S ≤ 10, benzene ≤ 1, aromatics ≤ 35, olefins ≤ 21, O ≤ 2.7, E70 10–45, E100 40–70, E150 ≥ 75 |
| MS95_BSVI | the same, RON ≥ 95, MON ≥ 85, olefins ≤ 18 |
| EXP_EUR95 | EN 228 summer class A (E5): as MS95 with DVPE 45–60 kPa, E70 20–48, E100 46–71 |
| NAPHTHA_EXP | representative open-spec naphtha terms: density 665–740, paraffins ≥ 65, olefins ≤ 1, S ≤ 650, RVP ≤ 83 kPa |

`demand_min` is blank on purpose (the pooling model refuses it); the
hourly commitments are in `demand.csv`.

**Prices** (`price.csv`, `demand.csv`) are daily, carried through each hour,
and follow a crude-linked market: Brent mean-reverting around $70/bbl, the
rupee weakening ~2.5 % a year from ₹88.3/$, a gasoline crack peaking in
May, naphtha firm in winter, butane following the LPG contract price's
winter rise, MTBE and alkylate at import parity. Saturday and Sunday carry
Friday's value, as price assessments do.

**Demand** (`demand.csv`) follows two blenders:

- **BL-1** blends MS91 continuously, with monthly seasonality (the Oct–Nov
  festive peak, the Apr–May summer peak, the monsoon dip), a daytime peak
  and a Sunday dip; four planned 16-hour stops.
- **BL-2** blends export cargoes (two a month, 120 h and 96 h campaigns),
  and MS95 campaigns (Tuesday 06:00–Wednesday 06:00, Saturday 06:00–22:00)
  when it is not on export.
- Naphtha export runs continuously.

A grade off campaign has `demand_min = demand_max = 0` for that hour.

**Header capacity** (`capacity.csv`) is every blend header's pumping limit
for every hour: four planned maintenance jobs per header over the two years
(two-pump headers drop to half, the rest to zero; never during an export
campaign) and random pump trips that halve a header for one to three hours.
0.34 % of header-hours are affected.

**Pools** (`pools.csv`) are the shared rundown tanks: the three crude units'
light naphtha, their heavy naphtha, the two hydrocrackers' light naphtha,
the two CCRs' heavy reformate, and the two FCCs' LCN, MCN and HCN each.

## A day and a month

Two windows cut from the full set by `generate.py`, each with the same six
files, sized for the page:

| Folder | Hours | What happens in it |
|---|---:|---|
| `One_Day_2026-10-13/` | 24 (Tuesday 13-Oct-2026) | MS91 all day; the MS95 campaign starts at 06:00; the alkylate header is down 8 h for planned work and the FCC mid-cut header derated 3 h; naphtha export all day |
| `One_Month_2026-10/` | 744 (October 2026) | two export cargo campaigns (4–9 and 18–22 Oct), nine MS95 campaigns, seven header outages, the festive-season demand peak |

Each is an exact slice of the full set; `cost` and product `price` in its
`components.csv` and `products.csv` are its first day's, which is what its
one-period and pooling runs price. On the page (`python -m ui.server`,
*blend from CSV tables*, the files in their slots), measured on the
development laptop:

| Folder | Files | Result |
|---|---|---|
| day | components + products | OPTIMAL, margin ₹1,628.03k, 0.3 s, ACCEPTED |
| day | + price, demand, capacity | OPTIMAL, ₹38,925.42k over the day, 14 s on the CPU, ACCEPTED -- inside the page's default 30 s. The GPU column runs PDLP, a first-order method, and stops about 0.04% short (time or iteration limit); the plan and its check are the CPU's |
| day | + pools | time limit at 300 s with ₹1,627.79k, within 0.015% of the proven bound, ACCEPTED -- set the page's time limit to 300 |
| month | components + products | OPTIMAL, ₹1,495.24k, ACCEPTED |
| month | + pools | the full set's one-hour model (same first-day prices): 300 s, within 0.04% of the bound, ACCEPTED |
| month | + price, demand, capacity | `GAP_LIMIT`, ₹1,453,199.28k, 14 min on the CPU, independent check **FEASIBLE** (feasible to 1e-06, optimal to 7.3e-09). Set the device to **cpu** and the time limit to **1800**: at the default 30 s it stops at the limit and the page says "no plan returned (TIME_LIMIT)", and "compare cpu & gpu" runs PDLP on the GPU after it, which does not converge here |

The month's plan on the command line:

```bash
python -m sovopt.cli blend Test_Data/One_Month_2026-10/components.csv \
    Test_Data/One_Month_2026-10/products.csv \
    --prices Test_Data/One_Month_2026-10/price.csv \
    --demand Test_Data/One_Month_2026-10/demand.csv \
    --capacity Test_Data/One_Month_2026-10/capacity.csv \
    --time-limit 1800 --plan month_plan.csv --out month_plan.json
```

At 1.4M nonzeros `auto` takes the interior point (it took PDLP until this
was measured; README, "--method auto"). It returns `GAP_LIMIT` in about 13
minutes: margin ₹1,453,199.28k, every row met to 5.4e-10, within 7.3e-9 of
the proven bound (README, bug 22), and says so:

    independent check: FEASIBLE (feasible to 1e-06, and optimal to 7.3e-09 -- short of the 1e-09 certificate)

`ACCEPTED` means certified optimal, `FEASIBLE` a plan that meets every
constraint with its optimality gap stated, `REJECTED` one that breaks the
model, with the check it failed.

## Units

| Quantity | Unit |
|---|---|
| rates: available, demand, header capacity, pool capacity | kL per hour |
| stock: storage_max, opening_stock, closing_stock | kL |
| prices and costs | ₹ per litre |
| storage cost | ₹ per litre per hour held |
| margin reported | ₹ thousand (₹/L × kL) |
| density | kg/m³ at 15 °C |
| sulfur | mg/kg |
| benzene, aromatics, olefins, paraffins, e70/e100/e150 | % v/v |
| oxygen | % m/m |
| ron, mon | blending octane numbers |
| rvp_index | RVP^1.25, RVP in kPa |

## Running it

```bash
# one hour at nameplate rates: the in-line blenders' optimal recipe
python -m sovopt.cli blend Test_Data/components.csv Test_Data/products.csv

# the same hour through the shared rundown tanks
python -m sovopt.cli blend Test_Data/components.csv Test_Data/products.csv \
    --pools Test_Data/pools.csv --time-limit 300

# the hourly plan over the horizon (see below before running it whole)
python -m sovopt.cli blend Test_Data/components.csv Test_Data/products.csv \
    --prices Test_Data/price.csv --demand Test_Data/demand.csv \
    --capacity Test_Data/capacity.csv
```

Measured on the development laptop (RTX 3050 4 GB, 16 GB RAM):

| Run | Model | Result |
|---|---|---|
| one hour, blending | 86 × 124 | OPTIMAL, ₹1,495k margin, 0.3 s, independent check ACCEPTED |
| one hour, pooling | 144 × 1,198, 840 bilinear terms | time limit at 300 s: plan ₹1,494.66k against a proven bound of ₹1,495.24k (gap 0.04 %), ACCEPTED |
| plan, first 24 h | 2,526 × 4,416, 46k nnz | default engine OPTIMAL, 27 s (13,165 pivots); IPM OPTIMAL, 3.5 s; both ACCEPTED |
| plan, first week | 17,502 × 30,912, 321k nnz | IPM OPTIMAL, 110 s, 0.65 GB, ACCEPTED |
| plan, first month | 77,406 × 136,896, 1.4 M nnz | IPM `GAP_LIMIT`, ~820 s, 1.72 GB: feasible (rows to 5.4e-10), margin within 7.3e-9 of the proven bound, not certified at 1e-9 |
| plan, full two years | ~1.8 M × 3.2 M, ~34 M nnz | not run: extrapolates past this machine's memory and to days of IPM time |

Three things to know:

- **This data found a dual simplex bug, now fixed** (README, bug 21):
  on the hourly plans the dual loop lost dual feasibility and never
  noticed, so `--prices` runs never finished. It now detects the loss and
  the primal completes the solve; every window from 2 to 24 hours solves
  on the default engine to the interior point's value.
- **The month window is optimal to 7 parts in a billion, and says so.**
  The interior point stalls there at a gap of 5.4e-9, short of the 1e-9 the
  independent check certifies at; it used to report that as `OPTIMAL`
  and the check refused it (README, bug 22). It now reports `GAP_LIMIT`
  with the certified bound and gap: the plan is feasible, and the margin
  is pinned between ₹1,453,199.281k and ₹1,453,199.292k.
- **The whole horizon is a stress case, not a planning run.** No refinery
  solves two years hourly as one LP; schedulers run days to weeks hourly,
  planners run months. The plan's size is streams × grades × hours, and at
  17,544 hours that is far past what this laptop holds.

## Blending assumptions

Qualities blend linearly by volume: exact for density and the %v/v
properties including the PIONA-type paraffins; through blending numbers
for octane and the RVP^1.25 index for vapour pressure; approximately for
the E-points; and the usual approximation for the mass-based sulfur and
oxygen. Distillation temperatures (T10/T50/T90/FBP) are not modelled
because they do not blend linearly.

Per-hour rundown is constant for each stream (the table has one
`available`), so unit turnarounds cannot be expressed; header outages
can, and are.

## Sources

- Bureau of Indian Standards, IS 2796:2017, *Motor Gasoline — Specification* (BS-VI).
- CEN, EN 228:2012+A1:2017, *Automotive fuels — Unleaded petrol*.
- Gary, Handwerk & Kaiser, *Petroleum Refining: Technology and Economics*,
  5th ed., CRC Press (2007), ch. 14 — blend-stock properties, blending
  octane numbers, the RVP blending index.
- Jones & Pujadó (eds.), *Handbook of Petroleum Processing*, Springer
  (2006) — reformer, isomerisation, FCC, alkylation and aromatics-complex
  product properties.
