"""What the data-dependent tests need, fetched once on first use.

A fresh clone carries no instances -- ``data/`` is gitignored, and the
benches fetch the sets they run -- so the tests that read an instance
skipped there: 33 of them, and a green suite on a clone said less than
the README's count. This fetches, before collection, exactly the files
those tests name, through the same fetchers the benches use, so every
file carries its machine-read reference value in its header:

- five MIPLIB instances (0.8 MB): the LP values, the MILP optima, the
  node-LP and crossover checks, gt2's KKT, the UI's pasted model;
- ten Netlib problems (90 KB): the expander against the readme's optima;
- five QPLIB instances with their solution files and page records
  (1.4 MB), one of each class the reader handles and 9002, which
  publishes no solution;
- one fixed-charge transportation model from plato (3 KB), for the
  provenance-header check.

It runs before collection because two tests parametrize over what is on
disk. ``SOVOPT_FETCH=0`` disables it, and a fetch that fails -- no
network, a source down -- is reported once and leaves those tests
skipping exactly as they did, with their own reasons.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

MIPLIB = ["flugpl", "gt2", "p0201", "khb05250", "mod010"]
NETLIB = ["afiro", "adlittle", "blend", "share2b", "sc50a", "beaconfd",
          "bandm", "boeing2", "e226", "bore3d"]
QPLIB = ["0018", "0031", "10050", "8845", "9002"]
FCTP = ["bk4x3"]


def enabled() -> bool:
    return os.environ.get("SOVOPT_FETCH", "1").lower() not in ("0", "no", "false", "off")


def _report(what: str, err: Exception) -> None:
    print(f"  tests/_data: could not fetch {what}: {type(err).__name__}: "
          f"{str(err)[:80]} -- the tests that need it will skip",
          file=sys.stderr)


def ensure() -> None:
    """Fetch whatever the data tests need and is not on disk. Quiet when
    everything is present; one line per fetched set otherwise."""
    if not enabled():
        return
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    inst = os.path.join(DATA, "instances")
    missing = [n for n in MIPLIB if not os.path.exists(os.path.join(inst, f"{n}.mps"))]
    if missing:
        print(f"  tests/_data: fetching {len(missing)} MIPLIB instance(s) into data/instances")
        try:
            from bench.fetch import fetch_one
            for name in missing:
                fetch_one(name, inst)
        except Exception as e:                            # noqa: BLE001
            _report("MIPLIB instances", e)

    netlib = os.path.join(DATA, "netlib")
    missing = [n for n in NETLIB if not os.path.exists(os.path.join(netlib, n))]
    if missing:
        print(f"  tests/_data: fetching {len(missing)} Netlib problem(s) into data/netlib")
        try:
            from bench.netlib import fetch
            fetch(netlib, only=missing)
        except Exception as e:                            # noqa: BLE001
            _report("Netlib problems", e)

    qplib = os.path.join(DATA, "qplib")
    missing = [n for n in QPLIB
               if not os.path.exists(os.path.join(qplib, f"QPLIB_{n}.qplib"))]
    if missing or not os.path.exists(os.path.join(qplib, "index.json")):
        print(f"  tests/_data: fetching {len(missing)} QPLIB instance(s) into data/qplib")
        try:
            from bench.qplib import fetch
            fetch(qplib, only=QPLIB)
        except Exception as e:                            # noqa: BLE001
            _report("QPLIB instances", e)

    fctp = os.path.join(DATA, "fctp")
    missing = [n for n in FCTP if not os.path.exists(os.path.join(fctp, f"{n}.mps"))]
    if missing:
        print(f"  tests/_data: fetching {len(missing)} fctp instance(s) into data/fctp")
        try:
            from bench.fetch import fetch_one
            for name in missing:
                fetch_one(name, fctp, source="plato")
        except Exception as e:                            # noqa: BLE001
            _report("fctp instances", e)
