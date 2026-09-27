from fastapi.testclient import TestClient
from grounded_qa.api.main import create_app
from grounded_qa.config import Settings
from grounded_qa.generation.client import ModelError, Generator
from grounded_qa.ingestion.pdf import read_pdf
import httpx
import pytest


class Stub:
    def __init__(self, p):
        self.settings = Settings(data_dir=p)

    def query(self, q, ids):
        if q == "timeout":
            raise ModelError("Model timed out")
        return {"question": q, "ids": ids}

    def ingest(self, p, title):
        read_pdf(p)


def test_http_validation_and_errors(tmp_path):
    with TestClient(create_app(Stub(tmp_path))) as c:
        assert c.get("/health").status_code == 200
        assert c.post("/query", json={"question": "x"}).status_code == 422
        assert c.post("/query", json={"question": "   "}).status_code == 422
        assert c.post("/query", json={"question": "timeout"}).status_code == 503
        assert (
            c.post("/query", json={"question": "privacy?"}).json()["question"]
            == "privacy?"
        )
        assert (
            c.post(
                "/documents", files={"file": ("x.pdf", b"bad", "application/pdf")}
            ).status_code
            == 422
        )
        assert c.get("/documents/not-an-id/pdf").status_code == 404
        assert (
            c.post(
                "/documents", files={"file": ("x.pdf", b"x" * (21 * 1024 * 1024))}
            ).status_code
            == 413
        )


def test_timeout_does_not_call_paid_provider(monkeypatch):
    calls = []

    def post(self, url, **kwargs):
        calls.append(url)
        raise httpx.ReadTimeout("timeout")

    monkeypatch.setattr(httpx.Client, "post", post)
    with pytest.raises(ModelError):
        Generator(Settings()).generate("hello", {})
    assert len(calls) == 1 and "11434" in calls[0]


def test_explicit_provider_requires_key():
    with pytest.raises(ValueError):
        Settings(llm_provider="openai", api_key="")
