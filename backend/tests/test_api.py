# backend/tests/test_api.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def known_customer():
    with TestClient(app) as c:
        r = c.get("/api/customers", params={"limit": 1})
        assert r.status_code == 200
        yield r.json()["items"][0]["customer_id"]


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_kpi_summary():
    r = client.get("/api/kpi/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total_customers"] == 2500
    assert body["customers_with_orders"] > 0
    assert 0 <= body["avg_churn_probability"] <= 1
    assert len(body["segment_distribution"]) == 4


def test_kpi_distributions():
    r = client.get("/api/kpi/distributions")
    assert r.status_code == 200
    body = r.json()
    for key in ("churn_proba", "monetary", "frequency", "on_time_ratio"):
        assert key in body
        assert sum(b["count"] for b in body[key]) > 0


def test_customers_pagination_and_filters():
    r = client.get("/api/customers", params={"limit": 5, "offset": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2500
    assert len(body["items"]) == 5

    seg = body["items"][0]["segment_name"]
    r2 = client.get("/api/customers", params={"segment": seg, "limit": 3})
    assert r2.status_code == 200
    assert all(item["segment_name"] == seg for item in r2.json()["items"])

    r3 = client.get("/api/customers", params={"min_frequency": 5, "limit": 3})
    assert all(item["frequency"] >= 5 for item in r3.json()["items"])


def test_customer_detail(known_customer):
    r = client.get(f"/api/customers/{known_customer}")
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == known_customer
    assert 0 <= body["churn_proba"] <= 1
    assert body["recommended_action"]


def test_customer_not_found():
    r = client.get("/api/customers/1")
    assert r.status_code == 404


def test_segments():
    r = client.get("/api/segments")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 4
    assert {s["segment_name"] for s in body} == {
        "High-Value Champions", "Promising Customers", "Needs Attention", "At-Risk"
    }


def test_top_risk_sorted():
    r = client.get("/api/churn/top-risk", params={"limit": 20})
    assert r.status_code == 200
    probs = [c["churn_proba"] for c in r.json()]
    assert probs == sorted(probs, reverse=True)


def test_churn_metrics():
    r = client.get("/api/churn/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "roc_auc_score" in body
    assert body["roc_auc_score"] < 0.99


def test_retention_actions():
    r = client.get("/api/retention/actions")
    assert r.status_code == 200
    assert len(r.json()) == 4


def test_retention_recommend(known_customer):
    r = client.get(f"/api/retention/recommend/{known_customer}")
    assert r.status_code == 200
    body = r.json()
    assert body["recommended_action_id"] in (0, 1, 2, 3)
    assert len(body["q_values"]) == 4
    best = max(body["q_values"], key=body["q_values"].get)
    assert best == body["recommended_action"]


def test_retention_evaluation():
    r = client.get("/api/retention/evaluation")
    assert r.status_code == 200
    body = r.json()
    for policy in ("never_act", "always_discount", "rule_based", "dqn"):
        assert policy in body
        assert "mean_net_profit" in body[policy]
