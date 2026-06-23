import pytest
from fastapi.testclient import TestClient

from app.deps import reset_engine
from app.main import app


@pytest.fixture(autouse=True)
def fresh_engine():
    # Each test starts with a clean, empty index.
    reset_engine()
    yield
    reset_engine()


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_upload_and_ask_end_to_end(client):
    content = (
        b"Acme Corp gives every full-time employee 24 days of paid annual leave per year. "
        b"Sick leave is 12 days. Expenses are reimbursed within 30 days of a valid receipt."
    )
    res = client.post(
        "/api/documents",
        files={"file": ("handbook.txt", content, "text/plain")},
    )
    assert res.status_code == 201
    body = res.json()
    assert body["chunks"] >= 1

    res = client.post("/api/ask", json={"question": "How many days of annual leave?"})
    assert res.status_code == 200
    data = res.json()
    assert "24" in data["answer"]
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["doc_name"] == "handbook.txt"


def test_unsupported_file_type_rejected(client):
    res = client.post(
        "/api/documents",
        files={"file": ("image.png", b"\x89PNG\r\n", "image/png")},
    )
    assert res.status_code == 415


def test_empty_question_rejected(client):
    res = client.post("/api/ask", json={"question": ""})
    assert res.status_code == 422


def test_stats_reports_provider(client):
    res = client.get("/api/stats")
    assert res.status_code == 200
    assert res.json()["provider"] in {"offline", "openai", "gemini"}
