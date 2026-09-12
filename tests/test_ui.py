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
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

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
