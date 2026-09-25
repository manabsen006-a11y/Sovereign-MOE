"""The local web interface, driven in-process.

``ui/server.py`` is one page and two endpoints over the same engines the CLI
uses. It had never been exercised by a test, so nothing guarded it against a
change in the solver's return shape -- the page reads ``sol.log`` rows and
``sol.dual_bound`` -- and the paste path writes a file next to the module.
These run the app through FastAPI's test client without a socket.
"""

import os

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
try:
    # starlette >= 1.6 drives its client with httpx2 and warns on httpx;
    # neither installed is a RuntimeError, which importorskip would not catch
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError) as e:                          # noqa: BLE001
    pytest.skip(f"fastapi's test client is unusable: {e}", allow_module_level=True)

import ui.server as server  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(server.app)


def test_the_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>SOVOPT</title>" in r.text
    assert "/api/solve" in r.text and "/api/hw" in r.text


def test_hardware_report_names_the_thread_count(client):
    r = client.get("/api/hw")
    assert r.status_code == 200
    body = r.json()
    assert body["threads"] >= 1
    assert isinstance(body["gpu"], bool)
    if not body["gpu"]:
        assert body["error"]


@pytest.mark.parametrize("template", ["blending", "production_planning",
                                      "unit_scheduling"])
def test_a_template_solves_on_the_cpu_and_reports_a_checked_point(client, template):
    r = client.post("/api/solve", json={"template": template, "size": 6,
                                        "seed": 1, "device": "cpu",
                                        "time_limit": 30})
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body, body.get("error")
    assert body["summary"]
    assert body["scaling"]
    (res,) = body["results"]
    assert res["device"] == "cpu"
    assert res["status"] in ("OPTIMAL", "TIME_LIMIT")
    assert res["objective"] is not None
    assert res["viol_row"] <= 1e-6 and res["viol_bound"] <= 1e-6
    assert res["viol_int"] <= 1e-6
    assert res["time"] > 0
    assert res["method"]


def test_pasted_mps_text_is_read_and_solved(client):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "instances", "flugpl.mps")
    if not os.path.exists(path):
        pytest.skip("instance not fetched")
    text = open(path, encoding="utf-8").read()
    r = client.post("/api/solve", json={"source": "mps", "mps": text,
                                        "device": "cpu", "time_limit": 30})
    body = r.json()
    assert "error" not in body, body.get("error")
    (res,) = body["results"]
    assert res["status"] == "OPTIMAL"
    assert abs(res["objective"] - 1201500.0) <= 1e-6 * 1201500.0
    assert res["nodes"] >= 1


def test_pasted_lp_text_is_recognised_without_a_filename(client):
    text = """Maximize
 obj: 3 x + 2 y
Subject To
 c1: x + y <= 4
 c2: x + 3 y <= 6
Bounds
 0 <= x <= 3
End
"""
    r = client.post("/api/solve", json={"source": "mps", "mps": text,
                                        "device": "cpu", "time_limit": 10})
    body = r.json()
    assert "error" not in body, body.get("error")
    (res,) = body["results"]
    assert res["status"] == "OPTIMAL"
    assert abs(res["objective"] - 11.0) <= 1e-6


def test_an_empty_paste_is_an_error_not_a_crash(client):
    r = client.post("/api/solve", json={"source": "mps", "mps": "   "})
    body = r.json()
    assert "error" in body
    assert "no model text" in body["error"]


def test_a_gpu_request_without_a_gpu_falls_back_to_the_cpu(client):
    from sovopt.core.backend import gpu_selftest
    if gpu_selftest()[0]:
        pytest.skip("a GPU is present")
    r = client.post("/api/solve", json={"template": "blending", "size": 4,
                                        "device": "gpu", "time_limit": 10})
    body = r.json()
    assert "error" not in body, body.get("error")
    assert [x["device"] for x in body["results"]] == ["cpu"]


# --------------------------------------------------------------------------- #
# the user's own data: a model file, or the planner's two tables              #
# --------------------------------------------------------------------------- #

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples", "blending")

TINY_MPS = """NAME          TINY
ROWS
 N  COST
 L  R1
 G  R2
COLUMNS
    X         COST         1.0   R1           1.0
    X         R2           1.0
    Y         COST         2.0   R1           1.0
    Y         R2           3.0
RHS
    RHS       R1           4.0   R2           3.0
ENDATA
"""


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


def test_an_uploaded_model_file_is_read_by_its_suffix(client):
    """The page sends the file's bytes and its name; the suffix picks the
    reader, and a compression suffix is looked through as on the command
    line. min x + 2y with x + y <= 4, x + 3y >= 3: y = 1, cost 2."""
    r = client.post("/api/solve", json={"source": "file", "filename": "tiny.mps",
                                        "data_b64": _b64(TINY_MPS.encode()),
                                        "device": "cpu", "time_limit": 30})
    body = r.json()
    assert "error" not in body, body.get("error")
    (res,) = body["results"]
    assert res["status"] == "OPTIMAL" and abs(res["objective"] - 2.0) < 1e-9


def test_an_uploaded_gzipped_model_is_decompressed(client):
    import gzip
    r = client.post("/api/solve", json={"source": "file", "filename": "tiny.mps.gz",
                                        "data_b64": _b64(gzip.compress(TINY_MPS.encode())),
                                        "device": "cpu", "time_limit": 30})
    body = r.json()
    assert "error" not in body, body.get("error")
    assert abs(body["results"][0]["objective"] - 2.0) < 1e-9


def test_an_upload_with_an_unknown_suffix_is_refused_with_a_reason(client):
    r = client.post("/api/solve", json={"source": "file", "filename": "model.xlsx",
                                        "data_b64": _b64(b"PK..."), "device": "cpu"})
    body = r.json()
    assert "error" in body and "not a .mps, .lp or .qps" in body["error"]


def test_the_two_csv_tables_give_a_plan_in_the_planners_terms(client):
    """examples/blending through the page: the plan names products,
    components, recipe rows and shadow prices, and the independent check
    on the model built from the tables accepts it."""
    comps = open(os.path.join(EXAMPLES, "components.csv"), encoding="utf-8").read()
    prods = open(os.path.join(EXAMPLES, "products.csv"), encoding="utf-8").read()
    r = client.post("/api/solve", json={"source": "blend", "components_csv": comps,
                                        "products_csv": prods, "device": "cpu",
                                        "time_limit": 30})
    body = r.json()
    assert "error" not in body, body.get("error")
    assert body["results"][0]["status"] == "OPTIMAL"
    plan = body["plan"]
    assert plan["status"] == "OPTIMAL"
    assert abs(plan["objective"] - 802000.0) < 1e-6 * 802000.0
    assert plan["verifier_verdict"] == "ACCEPTED"
    assert {p["name"] for p in plan["products"]} == {"Premium", "Regular", "Export"}
    assert {c["name"] for c in plan["components"]} == {
        "LightNaphtha", "HeavyNaphtha", "Reformate", "FCCGasoline", "Alkylate", "Isomerate"}
    assert any(r["product"] == "Premium" and r["component"] == "Alkylate" for r in plan["recipe"])
    light = next(c for c in plan["components"] if c["name"] == "LightNaphtha")
    assert light["at_limit"] and abs(light["shadow_price"] - 191.6) < 1e-6
    assert any(s["binding"] for s in plan["specs"])


def test_a_table_the_model_cannot_be_built_from_says_why(client):
    r = client.post("/api/solve", json={"source": "blend",
                                        "components_csv": "name,cost,sulfur\nA,1,0.1\n",
                                        "products_csv": "name,price,octane_min\nP,2,90\n",
                                        "device": "cpu"})
    body = r.json()
    assert "error" in body and "no component has a 'octane' column" in body["error"]


def test_a_solve_that_ends_without_a_point_is_reported_not_crashed(client, monkeypatch):
    """The interior point withholds a point that fails its feasibility cap
    at a time limit -- a month of hourly blending at the page's default 30 s
    does exactly that. The page used to put NaN into its JSON for the
    missing point, which JSON cannot carry, and showed a traceback where
    the status belonged."""
    import sovopt.cli as cli
    from sovopt.core.problem import Solution, Status
    monkeypatch.setattr(cli, "solve", lambda prob, **kw: Solution(
        status=Status.TIME_LIMIT, iterations=3, time=0.1, method="ipm"))
    comps = open(os.path.join(EXAMPLES, "components.csv"), encoding="utf-8").read()
    prods = open(os.path.join(EXAMPLES, "products.csv"), encoding="utf-8").read()
    body = client.post("/api/solve", json={"source": "blend", "components_csv": comps,
                                           "products_csv": prods, "device": "cpu",
                                           "time_limit": 5}).json()
    assert "error" not in body, body.get("error")
    res = body["results"][0]
    assert res["status"] == "TIME_LIMIT"
    assert res["objective"] is None and res["viol_row"] is None
