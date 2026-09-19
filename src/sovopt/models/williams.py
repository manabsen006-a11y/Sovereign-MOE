"""Williams' textbook models with published optima -- blending, production
planning, distribution, unit commitment, mining.

*Model Building in Mathematical Programming* prints every coefficient of
its problems and the optimal value of each, which is what makes them
usable as answer keys: a model transcribed from the book that reaches the
book's value to the penny is, with overwhelming likelihood, the book's
model. The refinery (12.6) is in :mod:`refinery`; these are the others
whose subject is the problem statement's -- crude and oil blending with
inventory, multi-period production planning under machine maintenance,
a two-tier distribution network, a day of power generation, a mine
schedule under blending constraints.

``food_manufacture_1`` (12.1, LP)
    Five oils bought on a six-month forward market, refined (two lines
    with monthly capacities) and blended into a product sold at a fixed
    price, whose hardness must lie in a band; oils can be stored at a
    cost. Published optimum: profit 107,842.59.

``food_manufacture_2`` (12.2, MILP)
    The same with three logical conditions: at most three oils a month,
    at least 20 tons of any oil used, and either vegetable oil forces the
    third non-vegetable oil. Published optimum: 100,278.71.

``factory_planning_1`` (12.3, LP)
    Seven products over six months on five machine types, each product
    with monthly market limits, machines down for maintenance on a fixed
    schedule, stock at a holding cost with a closing requirement.
    Published optimum: 93,715.18.

``distribution_1`` (12.19, LP)
    Two factories, four depots, six customers, a cost per ton on every
    permitted link, throughput limits on depots and capacities on
    factories. Published optimum: cost 198,500.

``tariff_rates`` (12.15, MILP)
    A day in five load periods, three generator types with minimum and
    maximum output, a running cost, a per-MWh cost above the minimum and
    a start-up cost, a 15% reserve to be met by running generators.
    Published optimum: cost 988,540.

``mining`` (12.7, MILP)
    Four mines over five years: a royalty for every year a mine is kept
    open, a mine once closed stays closed, at most three mines worked a
    year, ore qualities that must blend to a yearly target, revenue and
    royalties discounted at 10% a year. Published optimum: 146.862
    (million pounds).

Every builder returns a :class:`Problem` with column and row names, and
``PUBLISHED`` maps the builder's name to the book's optimal value.

References
----------
Williams, H.P., "Model Building in Mathematical Programming", 5th ed.,
  Wiley (2013): problems 12.1, 12.2, 12.3, 12.7, 12.15, 12.19 and their
  solutions in chapter 13.
"""

from __future__ import annotations

import numpy as np

from ..core.problem import ObjSense, Problem
from ._builder import Builder as _Builder

__all__ = ["food_manufacture_1", "food_manufacture_2", "factory_planning_1",
           "distribution_1", "tariff_rates", "mining", "PUBLISHED", "MODELS"]


# --------------------------------------------------------------------------- #
# 12.1 / 12.2  Food manufacture                                               #
# --------------------------------------------------------------------------- #

_OILS = ["VEG1", "VEG2", "OIL1", "OIL2", "OIL3"]
_VEG = ["VEG1", "VEG2"]
_NONVEG = ["OIL1", "OIL2", "OIL3"]
_HARDNESS = {"VEG1": 8.8, "VEG2": 6.1, "OIL1": 2.0, "OIL2": 4.2, "OIL3": 5.0}
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
# buying prices, pounds per ton, by month then oil
_PRICES = [
    [110, 120, 130, 110, 115],
    [130, 130, 110, 90, 115],
    [110, 140, 130, 100, 95],
    [120, 110, 120, 120, 125],
    [100, 120, 150, 110, 105],
    [90, 100, 140, 80, 135],
]


def _food(logical: bool) -> Problem:
    b = _Builder("williams_food_manufacture_2" if logical else "williams_food_manufacture_1",
                 ObjSense.MAXIMISE)
    buy, use, store, prod = {}, {}, {}, {}
    delta = {}
    for t, mon in enumerate(_MONTHS):
        for k, o in enumerate(_OILS):
            buy[o, t] = b.col(f"BUY_{o}_{mon}", c=-float(_PRICES[t][k]))
            use[o, t] = b.col(f"USE_{o}_{mon}")
            store[o, t] = b.col(f"STORE_{o}_{mon}", hi=1000.0, c=-5.0)
            if logical:
                delta[o, t] = b.binary(f"D_{o}_{mon}")
        prod[t] = b.col(f"PROD_{mon}", c=150.0)
    for t, mon in enumerate(_MONTHS):
        for o in _OILS:
            # opening stock 500, closing stock 500
            terms = {buy[o, t]: 1.0, use[o, t]: -1.0, store[o, t]: -1.0}
            if t > 0:
                terms[store[o, t - 1]] = 1.0
            b.eq(f"STOCK_{o}_{mon}", terms, 0.0 if t > 0 else -500.0)
            if t == len(_MONTHS) - 1:
                b.eq(f"CLOSE_{o}", {store[o, t]: 1.0}, 500.0)
        b.le(f"REFINE_VEG_{mon}", {use[o, t]: 1.0 for o in _VEG}, 200.0)
        b.le(f"REFINE_NONVEG_{mon}", {use[o, t]: 1.0 for o in _NONVEG}, 250.0)
        b.eq(f"BLEND_{mon}", {**{use[o, t]: 1.0 for o in _OILS}, prod[t]: -1.0})
        b.le(f"HARD_HI_{mon}", {**{use[o, t]: _HARDNESS[o] for o in _OILS}, prod[t]: -6.0}, 0.0)
        b.ge(f"HARD_LO_{mon}", {**{use[o, t]: _HARDNESS[o] for o in _OILS}, prod[t]: -3.0}, 0.0)
        if logical:
            for o in _OILS:
                cap = 200.0 if o in _VEG else 250.0
                b.le(f"USE_IF_{o}_{mon}", {use[o, t]: 1.0, delta[o, t]: -cap}, 0.0)
                b.ge(f"MIN20_{o}_{mon}", {use[o, t]: 1.0, delta[o, t]: -20.0}, 0.0)
            b.le(f"THREE_{mon}", {delta[o, t]: 1.0 for o in _OILS}, 3.0)
            b.le(f"VEG1_OIL3_{mon}", {delta["VEG1", t]: 1.0, delta["OIL3", t]: -1.0}, 0.0)
            b.le(f"VEG2_OIL3_{mon}", {delta["VEG2", t]: 1.0, delta["OIL3", t]: -1.0}, 0.0)
    return b.problem()


def food_manufacture_1() -> Problem:
    """Williams 12.1: the blending-with-storage LP. Optimum 107,842.59."""
    return _food(False)


def food_manufacture_2() -> Problem:
    """Williams 12.2: 12.1 with the three logical conditions. Optimum
    100,278.71."""
    return _food(True)


# --------------------------------------------------------------------------- #
# 12.3  Factory planning 1                                                    #
# --------------------------------------------------------------------------- #

_PRODUCTS = [f"PROD{k}" for k in range(1, 8)]
_PROFIT = [10, 6, 8, 4, 11, 9, 3]
_MACHINES = ["GRINDER", "VDRILL", "HDRILL", "BORER", "PLANER"]
_COUNT = {"GRINDER": 4, "VDRILL": 2, "HDRILL": 3, "BORER": 1, "PLANER": 1}
# hours per unit of product on each machine type
_TIME = {
    "GRINDER": [0.5, 0.7, 0.0, 0.0, 0.3, 0.2, 0.5],
    "VDRILL": [0.1, 0.2, 0.0, 0.3, 0.0, 0.6, 0.0],
    "HDRILL": [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.6],
    "BORER": [0.05, 0.03, 0.0, 0.07, 0.1, 0.0, 0.08],
    "PLANER": [0.0, 0.0, 0.01, 0.0, 0.05, 0.0, 0.05],
}
# machines down for maintenance, by month
_DOWN = [
    {"GRINDER": 1},
    {"HDRILL": 2},
    {"BORER": 1},
    {"VDRILL": 1},
    {"GRINDER": 1, "VDRILL": 1},
    {"PLANER": 1, "HDRILL": 1},
]
# market limits by month, per product
_MARKET = [
    [500, 1000, 300, 300, 800, 200, 100],
    [600, 500, 200, 0, 400, 300, 150],
    [300, 600, 0, 0, 500, 400, 100],
    [200, 300, 400, 500, 200, 0, 100],
    [0, 100, 500, 100, 1000, 300, 0],
    [500, 500, 100, 300, 1100, 500, 60],
]
_HOURS = 24 * 16.0          # two eight-hour shifts, 24 working days


def factory_planning_1() -> Problem:
    """Williams 12.3. Optimum 93,715.18."""
    b = _Builder("williams_factory_planning_1", ObjSense.MAXIMISE)
    make, sell, hold = {}, {}, {}
    for t, mon in enumerate(_MONTHS):
        for k, p in enumerate(_PRODUCTS):
            make[p, t] = b.col(f"MAKE_{p}_{mon}")
            sell[p, t] = b.col(f"SELL_{p}_{mon}", hi=float(_MARKET[t][k]), c=float(_PROFIT[k]))
            hold[p, t] = b.col(f"HOLD_{p}_{mon}", hi=100.0, c=-0.5)
    for t, mon in enumerate(_MONTHS):
        for k, p in enumerate(_PRODUCTS):
            terms = {make[p, t]: 1.0, sell[p, t]: -1.0, hold[p, t]: -1.0}
            if t > 0:
                terms[hold[p, t - 1]] = 1.0
            b.eq(f"STOCK_{p}_{mon}", terms, 0.0)
            if t == len(_MONTHS) - 1:
                b.eq(f"CLOSE_{p}", {hold[p, t]: 1.0}, 50.0)
        for m in _MACHINES:
            avail = _COUNT[m] - _DOWN[t].get(m, 0)
            b.le(f"CAP_{m}_{mon}",
                 {make[p, t]: _TIME[m][k] for k, p in enumerate(_PRODUCTS) if _TIME[m][k]},
                 _HOURS * avail)
    return b.problem()


# --------------------------------------------------------------------------- #
# 12.19  Distribution 1                                                       #
# --------------------------------------------------------------------------- #

_FACTORIES = {"LIVERPOOL": 150000.0, "BRIGHTON": 200000.0}
_DEPOTS = {"NEWCASTLE": 70000.0, "BIRMINGHAM": 50000.0, "LONDON": 100000.0, "EXETER": 40000.0}
_CUSTOMERS = {"C1": 50000.0, "C2": 10000.0, "C3": 40000.0, "C4": 35000.0,
              "C5": 60000.0, "C6": 20000.0}
# cost per ton on every permitted link
_LINKS = {
    ("LIVERPOOL", "NEWCASTLE"): 0.5, ("LIVERPOOL", "BIRMINGHAM"): 0.5,
    ("LIVERPOOL", "LONDON"): 1.0, ("LIVERPOOL", "EXETER"): 0.2,
    ("BRIGHTON", "BIRMINGHAM"): 0.3, ("BRIGHTON", "LONDON"): 0.5,
    ("BRIGHTON", "EXETER"): 0.2,
    ("LIVERPOOL", "C1"): 1.0, ("LIVERPOOL", "C3"): 1.5, ("LIVERPOOL", "C4"): 2.0,
    ("LIVERPOOL", "C6"): 1.0, ("BRIGHTON", "C1"): 2.0,
    ("NEWCASTLE", "C2"): 1.5, ("NEWCASTLE", "C3"): 0.5, ("NEWCASTLE", "C4"): 1.5,
    ("NEWCASTLE", "C6"): 1.0,
    ("BIRMINGHAM", "C1"): 1.0, ("BIRMINGHAM", "C2"): 0.5, ("BIRMINGHAM", "C3"): 0.5,
    ("BIRMINGHAM", "C4"): 1.0, ("BIRMINGHAM", "C5"): 0.5,
    ("LONDON", "C2"): 1.5, ("LONDON", "C3"): 2.0, ("LONDON", "C5"): 0.5,
    ("LONDON", "C6"): 1.5,
    ("EXETER", "C3"): 0.2, ("EXETER", "C4"): 1.5, ("EXETER", "C5"): 0.5,
    ("EXETER", "C6"): 1.5,
}


def distribution_1() -> Problem:
    """Williams 12.19. Optimum 198,500."""
    b = _Builder("williams_distribution_1", ObjSense.MINIMISE)
    flow = {(s, d): b.col(f"{s}_{d}", c=c) for (s, d), c in _LINKS.items()}
    for f, cap in _FACTORIES.items():
        b.le(f"CAP_{f}", {v: 1.0 for (s, d), v in flow.items() if s == f}, cap)
    for dp, thru in _DEPOTS.items():
        inbound = {v: 1.0 for (s, d), v in flow.items() if d == dp}
        outbound = {v: 1.0 for (s, d), v in flow.items() if s == dp}
        b.le(f"THRU_{dp}", inbound, thru)
        b.eq(f"BAL_{dp}", {**inbound, **{v: -1.0 for v in outbound}}, 0.0)
    for cu, dem in _CUSTOMERS.items():
        b.eq(f"DEMAND_{cu}", {v: 1.0 for (s, d), v in flow.items() if d == cu}, dem)
    return b.problem()


# --------------------------------------------------------------------------- #
# 12.15  Tariff rates (unit commitment)                                       #
# --------------------------------------------------------------------------- #

_PERIODS = [("0-6", 6.0, 15000.0), ("6-9", 3.0, 30000.0), ("9-15", 6.0, 25000.0),
            ("15-18", 3.0, 40000.0), ("18-24", 6.0, 27000.0)]
# type: (available, min MW, max MW, cost per hour at minimum, cost per MWh
# above minimum, start-up cost)
_GENERATORS = {
    "T1": (12, 850.0, 2000.0, 1000.0, 2.0, 2000.0),
    "T2": (10, 1250.0, 1750.0, 2600.0, 1.3, 1000.0),
    "T3": (5, 1500.0, 4000.0, 3000.0, 3.0, 500.0),
}


def tariff_rates() -> Problem:
    """Williams 12.15. Optimum 988,540 per day."""
    b = _Builder("williams_tariff_rates", ObjSense.MINIMISE)
    n_on, started, out = {}, {}, {}
    for t, (per, hours, demand) in enumerate(_PERIODS):
        for g, (avail, lo, hi, c_min, c_mwh, c_start) in _GENERATORS.items():
            n_on[g, t] = b.integer(f"ON_{g}_{per}", avail, c=hours * (c_min - c_mwh * lo))
            started[g, t] = b.integer(f"START_{g}_{per}", avail, c=c_start)
            out[g, t] = b.col(f"MW_{g}_{per}", c=hours * c_mwh)
    for t, (per, hours, demand) in enumerate(_PERIODS):
        b.eq(f"DEMAND_{per}", {out[g, t]: 1.0 for g in _GENERATORS}, demand)
        b.ge(f"RESERVE_{per}", {n_on[g, t]: _GENERATORS[g][2] for g in _GENERATORS},
             1.15 * demand)
        for g, (avail, lo, hi, *_r) in _GENERATORS.items():
            b.ge(f"MIN_{g}_{per}", {out[g, t]: 1.0, n_on[g, t]: -lo}, 0.0)
            b.le(f"MAX_{g}_{per}", {out[g, t]: 1.0, n_on[g, t]: -hi}, 0.0)
            prev = n_on[g, (t - 1) % len(_PERIODS)]        # the day is cyclic
            b.ge(f"STARTS_{g}_{per}", {started[g, t]: 1.0, n_on[g, t]: -1.0, prev: 1.0}, 0.0)
    return b.problem()


# --------------------------------------------------------------------------- #
# 12.7  Mining                                                                #
# --------------------------------------------------------------------------- #

_MINES = ["M1", "M2", "M3", "M4"]
_ROYALTY = [5.0, 4.0, 4.0, 5.0]           # million pounds a year, while open
_EXTRACT = [2.0, 2.5, 1.3, 3.0]           # million tons a year
_QUALITY = [1.0, 0.7, 1.5, 0.5]
_TARGET = [0.9, 0.8, 1.2, 0.6, 1.0]       # blended quality required, years 1-5
_PRICE = 10.0                             # pounds per ton
_DISCOUNT = 0.10


def mining() -> Problem:
    """Williams 12.7, in millions of pounds. Optimum 146.862."""
    b = _Builder("williams_mining", ObjSense.MAXIMISE)
    years = range(len(_TARGET))
    disc = [1.0 / (1.0 + _DISCOUNT) ** t for t in years]
    is_open, worked, ore, blend = {}, {}, {}, {}
    for t in years:
        for k, m in enumerate(_MINES):
            is_open[m, t] = b.binary(f"OPEN_{m}_Y{t + 1}", c=-disc[t] * _ROYALTY[k])
            worked[m, t] = b.binary(f"WORK_{m}_Y{t + 1}")
            ore[m, t] = b.col(f"ORE_{m}_Y{t + 1}", hi=_EXTRACT[k])
        blend[t] = b.col(f"BLEND_Y{t + 1}", c=disc[t] * _PRICE)
    for t in years:
        for k, m in enumerate(_MINES):
            b.le(f"EXTRACT_{m}_Y{t + 1}", {ore[m, t]: 1.0, worked[m, t]: -_EXTRACT[k]}, 0.0)
            b.le(f"WORK_IF_OPEN_{m}_Y{t + 1}", {worked[m, t]: 1.0, is_open[m, t]: -1.0}, 0.0)
            if t > 0:
                b.le(f"STAY_CLOSED_{m}_Y{t + 1}", {is_open[m, t]: 1.0, is_open[m, t - 1]: -1.0}, 0.0)
        b.le(f"THREE_Y{t + 1}", {worked[m, t]: 1.0 for m in _MINES}, 3.0)
        b.eq(f"BLEND_Y{t + 1}", {**{ore[m, t]: 1.0 for m in _MINES}, blend[t]: -1.0}, 0.0)
        b.eq(f"QUALITY_Y{t + 1}",
             {**{ore[m, t]: _QUALITY[k] for k, m in enumerate(_MINES)}, blend[t]: -_TARGET[t]}, 0.0)
    return b.problem()


PUBLISHED = {
    "food_manufacture_1": 107842.59,
    "food_manufacture_2": 100278.71,
    "factory_planning_1": 93715.18,
    "distribution_1": 198500.0,
    "tariff_rates": 988540.0,
    "mining": 146.862,
}
"""The book's optimal values, chapter 13; ``mining`` in millions of pounds."""

MODELS = {
    "food_manufacture_1": food_manufacture_1,
    "food_manufacture_2": food_manufacture_2,
    "factory_planning_1": factory_planning_1,
    "distribution_1": distribution_1,
    "tariff_rates": tariff_rates,
    "mining": mining,
}
