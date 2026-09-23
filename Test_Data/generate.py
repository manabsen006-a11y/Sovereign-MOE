"""Generate Test_Data: two years of hourly gasoline-blending data for a
15 MMTPA refinery, written as the six tables `sovopt blend` reads.

    python Test_Data/generate.py

Deterministic (seed 26119, the problem statement's number): the same files
every run. The values are representative -- published specifications,
typical stream properties from the refining literature, prices and rates
realistic in level and pattern -- not any refinery's records. See README.md.
"""

import csv
import math
import os
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(26119)

START = datetime(2026, 10, 1)
END = datetime(2028, 10, 1)
HOURS = int((END - START).total_seconds()) // 3600          # 17,544
DAYS = HOURS // 24                                          # 731
DAY0 = [START + timedelta(days=d) for d in range(DAYS)]
LABEL = [(START + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M") for h in range(HOURS)]

QUALITIES = ["ron", "mon", "density", "rvp_index", "sulfur", "benzene", "aromatics",
             "olefins", "oxygen", "e70", "e100", "e150", "paraffins"]


def rvp_index(kpa):
    """RVP blending index, RVP^1.25 (Gary, Handwerk & Kaiser, ch. 14)."""
    return round(kpa ** 1.25, 1)


# --------------------------------------------------------------------------- #
# streams: (name, header, rundown kL/h, tankage kL, price basis, adder Rs/L,   #
#           direct, ron, mon, density, rvp kPa, sulfur, benzene, aromatics,    #
#           olefins, oxygen, e70, e100, e150, paraffins)                      #
# --------------------------------------------------------------------------- #
STREAMS = [
    ("CDU1_LSRN", "HDR_LSRN", 16, 12000, "NAPHTHA", 0.20, "", 68.5, 66.4, 656, 80, 0.8, 1.3, 3.1, 0.3, 0, 76, 97, 100, 88),
    ("CDU2_LSRN", "HDR_LSRN", 13, 10000, "NAPHTHA", 0.15, "", 67.1, 65.2, 659, 78, 0.9, 1.5, 3.6, 0.3, 0, 73, 96, 100, 86),
    ("CDU3_LSRN", "HDR_LSRN", 19, 14000, "NAPHTHA", 0.25, "", 70.2, 68.0, 652, 83, 0.6, 1.1, 2.7, 0.2, 0, 79, 98, 100, 89),
    ("CDU1_HSRN", "HDR_HSRN", 11, 8000, "NAPHTHA", -0.40, "", 47.0, 45.5, 748, 9, 380, 0.4, 13, 0.2, 0, 1, 12, 72, 58),
    ("CDU2_HSRN", "HDR_HSRN", 9, 8000, "NAPHTHA", -0.45, "", 44.0, 43.0, 752, 8, 460, 0.3, 12, 0.2, 0, 1, 10, 69, 60),
    ("CDU3_HSRN", "HDR_HSRN", 13, 8000, "NAPHTHA", -0.35, "", 49.0, 47.5, 744, 10, 290, 0.4, 14, 0.2, 0, 2, 14, 74, 56),
    ("HCU1_LN", "HDR_HCU_NAP", 22, 15000, "NAPHTHA", 0.50, "", 77.5, 75.8, 666, 77, 0.5, 0.8, 3.2, 0.1, 0, 72, 97, 100, 84),
    ("HCU2_LN", "HDR_HCU_NAP", 18, 12000, "NAPHTHA", 0.45, "", 75.6, 74.1, 670, 74, 0.5, 0.9, 3.8, 0.1, 0, 69, 96, 100, 83),
    ("HCU1_HN", "HDR_HCU_NAP", 10, 8000, "NAPHTHA", 0.30, "*", 58.0, 56.0, 742, 7, 0.6, 0.5, 14, 0.1, 0, 1, 14, 78, 52),
    ("ISOM1", "HDR_ISOM", 62, 30000, "NAPHTHA", 3.40, "*", 87.6, 85.9, 651, 92, 0.3, 0.05, 0.4, 0.1, 0, 91, 100, 100, 97),
    ("ISOM2", "HDR_ISOM", 38, 20000, "NAPHTHA", 2.60, "*", 82.4, 80.6, 648, 97, 0.3, 0.05, 0.3, 0.1, 0, 93, 100, 100, 97),
    ("CCR1_HREF", "HDR_HREF", 56, 40000, "GASOLINE95", 1.00, "", 105.8, 94.6, 862, 6, 0.2, 0.3, 88, 0.5, 0, 1, 6, 58, 10),
    ("CCR2_HREF", "HDR_HREF", 45, 32000, "GASOLINE95", 0.90, "", 104.9, 93.8, 858, 7, 0.2, 0.4, 86, 0.6, 0, 1, 8, 61, 12),
    ("BENSAT_LREF", "HDR_BENSAT", 34, 18000, "NAPHTHA", 1.80, "*", 72.6, 70.8, 688, 68, 0.2, 0.3, 0.8, 0.1, 0, 64, 98, 100, 74),
    ("RAFFINATE", "HDR_RAFF", 40, 20000, "NAPHTHA", 0.10, "*", 62.0, 60.5, 684, 42, 0.3, 0.5, 1.5, 0.5, 0, 34, 86, 100, 91),
    ("C9_AROMATICS", "HDR_C9ARO", 12, 8000, "GASOLINE95", 1.50, "*", 112.0, 100.5, 879, 1, 1.0, 0.0, 98, 0.2, 0, 0, 0, 6, 1),
    ("TOLUENE", "HDR_TOLUENE", 10, 6000, "TOLUENE", 0.00, "*", 120.0, 107.0, 871, 3.5, 0.5, 0.1, 99.8, 0.0, 0, 0, 2, 100, 0),
    ("FCC1_LCN", "HDR_FCC_LCN", 58, 25000, "GASOLINE92", -1.60, "", 92.6, 80.4, 676, 74, 7.0, 0.9, 10, 42, 0, 66, 91, 100, 38),
    ("PFCC_LCN", "HDR_FCC_LCN", 34, 16000, "GASOLINE92", -1.40, "", 94.8, 81.6, 682, 80, 6.0, 1.2, 14, 48, 0, 70, 93, 100, 30),
    ("FCC1_MCN", "HDR_FCC_MCN", 24, 12000, "GASOLINE92", -2.20, "", 88.4, 78.2, 752, 30, 8.0, 0.6, 32, 22, 0, 12, 55, 98, 30),
    ("PFCC_MCN", "HDR_FCC_MCN", 16, 9000, "GASOLINE92", -2.00, "", 90.6, 79.4, 760, 28, 7.5, 0.8, 38, 26, 0, 10, 52, 97, 24),
    ("FCC1_HCN", "HDR_FCC_HCN", 42, 22000, "GASOLINE92", -3.00, "", 87.1, 77.9, 818, 10, 9.0, 0.3, 49, 10, 0, 0, 9, 54, 26),
    ("PFCC_HCN", "HDR_FCC_HCN", 28, 15000, "GASOLINE92", -2.80, "", 90.2, 79.8, 835, 8, 8.5, 0.6, 62, 14, 0, 0, 7, 50, 16),
    ("ALKYLATE_IMP", "HDR_ALKY", 20, 20000, "GASOLINE95", 3.20, "*", 95.6, 93.1, 697, 34, 3.0, 0.0, 0, 0.4, 0, 15, 46, 93, 99),
    ("N_BUTANE", "HDR_C4", 18, 3500, "LPG", 0.00, "*", 94.0, 90.0, 584, 356, 2.0, 0.0, 0, 0.5, 0, 100, 100, 100, 99),
    ("ISO_BUTANE", "HDR_C4", 8, 2000, "LPG", 1.20, "*", 101.0, 97.5, 563, 496, 2.0, 0.0, 0, 0.3, 0, 100, 100, 100, 99.7),
    ("MTBE_IMP", "HDR_OXY", 25, 18000, "MTBE", 0.00, "*", 116.0, 101.0, 745, 58, 5.0, 0.0, 0, 0.0, 18.2, 100, 100, 100, 0),
    ("REFORMATE_IMP", "HDR_IMP_REF", 30, 25000, "GASOLINE95", 0.40, "*", 98.2, 88.1, 808, 30, 0.5, 2.8, 68, 0.8, 0, 12, 31, 86, 28),
    ("NATGASOLINE_IMP", "HDR_NATGAS", 20, 15000, "NAPHTHA", 0.60, "*", 70.5, 68.4, 664, 88, 15.0, 0.9, 4, 0.2, 0, 70, 92, 99, 90),
    ("DHDT_NAPHTHA", "HDR_DHDT", 14, 6000, "NAPHTHA", -0.20, "*", 61.0, 59.0, 734, 45, 2.0, 0.5, 10, 1.0, 0, 30, 60, 95, 50),
]

# blend headers: nominal rate kL/h; the two-pump headers drop to half, not
# zero, for planned work
HEADERS = {
    "HDR_LSRN": 65, "HDR_HSRN": 45, "HDR_HCU_NAP": 70, "HDR_ISOM": 135, "HDR_HREF": 135,
    "HDR_BENSAT": 45, "HDR_RAFF": 55, "HDR_C9ARO": 16, "HDR_TOLUENE": 14, "HDR_FCC_LCN": 120,
    "HDR_FCC_MCN": 52, "HDR_FCC_HCN": 92, "HDR_ALKY": 30, "HDR_C4": 30, "HDR_OXY": 32,
    "HDR_IMP_REF": 40, "HDR_NATGAS": 28, "HDR_DHDT": 20,
}
TWO_PUMP = {"HDR_LSRN", "HDR_ISOM", "HDR_HREF", "HDR_FCC_LCN", "HDR_FCC_HCN"}

# shared rundown tanks (the pooling model); every other stream has its own tank
POOLS = [
    ("TK_LSRN_COMMON", 60, ["CDU1_LSRN", "CDU2_LSRN", "CDU3_LSRN"]),
    ("TK_HSRN_COMMON", 45, ["CDU1_HSRN", "CDU2_HSRN", "CDU3_HSRN"]),
    ("TK_HCU_LN", 50, ["HCU1_LN", "HCU2_LN"]),
    ("TK_HREF_COMMON", 110, ["CCR1_HREF", "CCR2_HREF"]),
    ("TK_FCC_LCN", 100, ["FCC1_LCN", "PFCC_LCN"]),
    ("TK_FCC_MCN", 45, ["FCC1_MCN", "PFCC_MCN"]),
    ("TK_FCC_HCN", 80, ["FCC1_HCN", "PFCC_HCN"]),
]

# products: BIS IS 2796:2017 (BS-VI) for the domestic grades, EN 228 summer
# class A (E5) for the export grade, representative open-spec terms for naphtha
PRODUCTS = [
    ("MS91_BSVI", 280, dict(ron_min=91, mon_min=81, density_min=720, density_max=775,
                            rvp_index_max=rvp_index(60), sulfur_max=10, benzene_max=1.0,
                            aromatics_max=35, olefins_max=21, oxygen_max=2.7, e70_min=10,
                            e70_max=45, e100_min=40, e100_max=70, e150_min=75)),
    ("MS95_BSVI", 120, dict(ron_min=95, mon_min=85, density_min=720, density_max=775,
                            rvp_index_max=rvp_index(60), sulfur_max=10, benzene_max=1.0,
                            aromatics_max=35, olefins_max=18, oxygen_max=2.7, e70_min=10,
                            e70_max=45, e100_min=40, e100_max=70, e150_min=75)),
    ("EXP_EUR95", 300, dict(ron_min=95, mon_min=85, density_min=720, density_max=775,
                            rvp_index_min=rvp_index(45), rvp_index_max=rvp_index(60),
                            sulfur_max=10, benzene_max=1.0, aromatics_max=35, olefins_max=18,
                            oxygen_max=2.7, e70_min=20, e70_max=48, e100_min=46, e100_max=71,
                            e150_min=75)),
    ("NAPHTHA_EXP", 190, dict(density_min=665, density_max=740, rvp_index_max=rvp_index(83),
                              sulfur_max=650, olefins_max=1.0, paraffins_min=65)),
]
SPEC_COLS = ["ron_min", "mon_min", "density_min", "density_max", "rvp_index_min",
             "rvp_index_max", "sulfur_max", "benzene_max", "aromatics_max", "olefins_max",
             "oxygen_max", "e70_min", "e70_max", "e100_min", "e100_max", "e150_min",
             "paraffins_min"]


# --------------------------------------------------------------------------- #
# daily markets                                                               #
# --------------------------------------------------------------------------- #
def ou(n, mean, sd, kappa, x0=None):
    """Mean-reverting daily path with stationary sd ``sd``."""
    step = sd * math.sqrt(2 * kappa)
    x = np.empty(n)
    x[0] = mean if x0 is None else x0
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (mean - x[t - 1]) + step * rng.standard_normal()
    return x


def weekend_carry(a):
    """No assessment on Saturday or Sunday: Friday's value stands."""
    a = a.copy()
    for d, day in enumerate(DAY0):
        if day.weekday() >= 5 and d > 0:
            a[d] = a[d - 1]
    return a


doy = np.array([d.timetuple().tm_yday for d in DAY0], dtype=float)
t_yr = np.arange(DAYS) / 365.0
brent = ou(DAYS, 70.0, 6.0, 0.02, x0=69.0)                                   # $/bbl
fx = 88.3 * np.exp(0.025 * t_yr + np.cumsum(rng.normal(0, 0.0012, DAYS)))   # Rs/$
gas_crack = 9.0 + 2.5 * np.sin(2 * np.pi * (doy - 45) / 365) + ou(DAYS, 0, 1.5, 0.05)
prem95 = ou(DAYS, 3.0, 0.4, 0.05)
naph_crack = -4.5 + 1.5 * np.cos(2 * np.pi * (doy - 15) / 365) + ou(DAYS, 0, 1.2, 0.05)
lpg_cp = 590 + 55 * np.cos(2 * np.pi * (doy - 15) / 365) + ou(DAYS, 0, 25, 0.03)   # $/t
toluene_t = 860 + 6 * (brent - 70) + ou(DAYS, 0, 40, 0.03)                      # $/t

per_bbl = fx / 158.987
SERIES = {
    "NAPHTHA": (brent + naph_crack) * per_bbl,
    "GASOLINE92": (brent + gas_crack) * per_bbl,
    "GASOLINE95": (brent + gas_crack + prem95) * per_bbl,
    "LPG": lpg_cp * 0.584 * fx / 1000.0,
    "TOLUENE": toluene_t * 0.871 * fx / 1000.0,
}
SERIES["MTBE"] = 1.25 * SERIES["GASOLINE95"] + ou(DAYS, 0, 0.8, 0.05)
SERIES = {k: weekend_carry(v) for k, v in SERIES.items()}

comp_daily = {}
for s in STREAMS:
    name, basis, adder = s[0], s[4], s[5]
    comp_daily[name] = np.round(weekend_carry(SERIES[basis] + adder + ou(DAYS, 0, 0.06, 0.1)), 2)

ms91_px = np.round(SERIES["GASOLINE92"] + 0.60, 2)
PROD_DAILY = {
    "MS91_BSVI": ms91_px,
    "MS95_BSVI": np.round(ms91_px + 2.80 + weekend_carry(ou(DAYS, 0, 0.10, 0.1)), 2),
    "EXP_EUR95": np.round(SERIES["GASOLINE95"] - 0.45, 2),
    "NAPHTHA_EXP": np.round(SERIES["NAPHTHA"] - 0.10, 2),
}


# --------------------------------------------------------------------------- #
# blending campaigns                                                          #
# --------------------------------------------------------------------------- #
def hour_of(dt):
    return int((dt - START).total_seconds()) // 3600


# BL-2: export cargoes take precedence (two a month), MS95 campaigns in the gaps
bl2 = np.zeros(HOURS, dtype=np.int8)                    # 0 idle, 1 MS95, 2 export
m = START
while m < END:
    for first_day, dur in ((3 + int(rng.integers(0, 5)), 120), (17 + int(rng.integers(0, 5)), 96)):
        s = hour_of(m.replace(day=first_day, hour=6))
        bl2[s:min(s + dur, HOURS)] = 2
    m = (m.replace(day=28) + timedelta(days=4)).replace(day=1)
for d, day in enumerate(DAY0):
    if day.weekday() == 1:                              # Tuesday 06:00 to Wednesday 06:00
        s, e = d * 24 + 6, d * 24 + 30
    elif day.weekday() == 5:                            # Saturday 06:00 to 22:00
        s, e = d * 24 + 6, d * 24 + 22
    else:
        continue
    e = min(e, HOURS)
    if not bl2[s:e].any():
        bl2[s:e] = 1

# BL-1: MS91 continuously, but for four planned 16-hour stops
bl1_stop = np.zeros(HOURS, dtype=bool)
for d in rng.choice(np.arange(20, DAYS - 20), size=4, replace=False):
    bl1_stop[d * 24 + 6:d * 24 + 22] = True

SEASON = {10: 1.07, 11: 1.08, 12: 1.00, 1: 0.97, 2: 0.96, 3: 1.02,
          4: 1.05, 5: 1.06, 6: 1.00, 7: 0.93, 8: 0.92, 9: 0.97}
wobble = ou(HOURS, 0, 0.02, 0.05)


def demand_rows():
    for h in range(HOURS):
        dt = START + timedelta(hours=h)
        d = h // 24
        season = SEASON[dt.month]
        diurnal = 1 + 0.06 * math.sin(2 * math.pi * (dt.hour - 8) / 24)
        weekly = 0.95 if dt.weekday() == 6 else 1.0
        r = 300 * season * diurnal * weekly * (1 + wobble[h])
        if bl1_stop[h]:
            ms91 = (0.0, 0.0)
        else:
            ms91 = (round(0.70 * r, 1), round(1.15 * r, 1))
        ms95 = (round(80 * season ** 0.5, 1), round(120 * season ** 0.5, 1)) if bl2[h] == 1 else (0.0, 0.0)
        exp = (200.0, 300.0) if bl2[h] == 2 else (0.0, 0.0)
        naph = (40.0, 190.0)
        for prod, (lo, hi) in (("MS91_BSVI", ms91), ("MS95_BSVI", ms95),
                               ("EXP_EUR95", exp), ("NAPHTHA_EXP", naph)):
            yield [LABEL[h], prod, f"{PROD_DAILY[prod][d]:.2f}", f"{lo:g}", f"{hi:g}"]


# --------------------------------------------------------------------------- #
# header capacity: planned work (four jobs a header, off the export           #
# campaigns) and pump trips                                                   #
# --------------------------------------------------------------------------- #
cap_factor = {line: np.ones(HOURS) for line in HEADERS}
for line in HEADERS:
    placed = 0
    while placed < 4:
        d = int(rng.integers(5, DAYS - 5))
        s, e = d * 24 + 8, d * 24 + 8 + int(rng.integers(8, 13))
        if (bl2[s:e] == 2).any():
            continue
        cap_factor[line][s:e] = 0.5 if line in TWO_PUMP else 0.0
        placed += 1
    for s in rng.choice(HOURS - 4, size=rng.poisson(HOURS * 0.0006), replace=False):
        cap_factor[line][s:s + int(rng.integers(1, 4))] *= 0.5


def capacity_rows():
    for h in range(HOURS):
        for line, nominal in HEADERS.items():
            yield [line, f"{nominal * cap_factor[line][h]:.1f}", LABEL[h]]


# --------------------------------------------------------------------------- #
# write                                                                       #
# --------------------------------------------------------------------------- #
def write(name, header, rows):
    path = os.path.join(HERE, name)
    cells = numeric = n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for row in rows:
            w.writerow(row)
            n += 1
            cells += len(row)
            numeric += sum(1 for c in row if _is_num(c))
    print(f"  {name:15s} {n:>9,} rows {cells:>11,} cells {numeric:>11,} numbers "
          f"{os.path.getsize(path) / 1e6:7.1f} MB")
    return cells, numeric


def _is_num(c):
    try:
        float(c)
        return True
    except ValueError:
        return False


def main():
    print(f"{HOURS:,} hours, {START:%d-%b-%Y} to {END - timedelta(hours=1):%d-%b-%Y %H:%M}")
    totals = []
    comp_header = ["name", "cost", "available", "line", "storage_max", "storage_cost",
                   "opening_stock", "closing_stock", "direct"] + QUALITIES
    comp_rows = []
    for s in STREAMS:
        (name, line, avail, tank, _basis, _adder, direct,
         ron, mon, dens, rvp, sul, benz, aro, ole, oxy, e70, e100, e150, par) = s
        opening = round(0.45 * tank, -2)
        hold = "0.00016" if line == "HDR_C4" else "0.00007"   # Rs/L per hour held
        comp_rows.append([name, f"{comp_daily[name][0]:.2f}", avail, line, tank, hold,
                          f"{opening:g}", f"{opening:g}", direct,
                          ron, mon, dens, rvp_index(rvp), sul, benz, aro, ole, oxy,
                          e70, e100, e150, par])
    totals.append(write("components.csv", comp_header, comp_rows))

    prod_rows = []
    for name, rate, spec in PRODUCTS:
        prod_rows.append([name, f"{PROD_DAILY[name][0]:.2f}", "", rate]
                         + ["" if spec.get(c) is None else f"{spec[c]:g}" for c in SPEC_COLS])
    totals.append(write("products.csv", ["name", "price", "demand_min", "demand_max"] + SPEC_COLS,
                        prod_rows))

    names = [s[0] for s in STREAMS]
    totals.append(write("price.csv", ["period"] + names,
                        ([LABEL[h]] + [f"{comp_daily[n][h // 24]:.2f}" for n in names]
                         for h in range(HOURS))))
    totals.append(write("demand.csv", ["period", "product", "price", "demand_min", "demand_max"],
                        demand_rows()))
    totals.append(write("capacity.csv", ["line", "capacity", "period"], capacity_rows()))
    totals.append(write("pools.csv", ["name", "capacity", "inputs"],
                        ([n, c, ";".join(i)] for n, c, i in POOLS)))
    print(f"  {'total':15s} {'':>14} {sum(c for c, _ in totals):>11,} cells "
          f"{sum(n for _, n in totals):>11,} numbers")


if __name__ == "__main__":
    main()
