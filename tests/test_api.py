import httpx
import pytest
from fastapi.testclient import TestClient

from grounded_qa.api.main import create_app
from grounded_qa.config import Settings
from grounded_qa.generation.client import Generator, ModelError
from grounded_qa.ingestion.pdf import read_pdf


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


def test_rejects_foreign_host_and_cross_origin_posts(tmp_path):
    with TestClient(create_app(Stub(tmp_path))) as c:
        assert c.get("/health", headers={"host": "evil.example"}).status_code == 400
        post = c.post(
            "/query",
            json={"question": "privacy?"},
            headers={"origin": "https://evil.example"},
        )
        assert post.status_code == 403
        ok = c.post(
            "/query",
            json={"question": "privacy?"},
            headers={"origin": "http://localhost:8501"},
        )
        assert ok.status_code == 200


def test_health_reports_missing_model_without_failing(tmp_path, monkeypatch):
    real_get = httpx.Client.get

    def fake_get(self, url, *args, **kwargs):
        # Only the Ollama probe fails; TestClient is also an httpx.Client.
        if str(url).startswith("http://localhost:11434"):
            raise httpx.ConnectError("down")
        return real_get(self, url, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    with TestClient(create_app(Stub(tmp_path))) as c:
        body = c.get("/health").json()
        assert body["status"] == "ready" and body["model_available"] is False


def test_extract_json_handles_fences_and_chatter():
    from grounded_qa.generation.client import extract_json

    assert extract_json('```json\n{"claims": []}\n```') == '{"claims": []}'
    assert extract_json('Sure! {"claims": []} Hope that helps') == '{"claims": []}'
    assert extract_json('{"claims": []}') == '{"claims": []}'


def test_e5_prefix_detection_is_not_a_substring_match():
    from grounded_qa.retrieval.embeddings import needs_e5_prefixes

    assert needs_e5_prefixes("intfloat/e5-small-v2")
    assert needs_e5_prefixes("intfloat/multilingual-e5-base")
    assert not needs_e5_prefixes("sentence-transformers/all-MiniLM-L6-v2")
    assert not needs_e5_prefixes("BAAI/bge-small-en-v1.5")
    assert not needs_e5_prefixes("someorg/lite5-embed")


def test_manifest_path_resolves_outside_repo_root(tmp_path, monkeypatch):
    from grounded_qa.config import manifest_path

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CORPUS_MANIFEST", raising=False)
    assert manifest_path().name == "manifest.json"  # found next to the package
    custom = tmp_path / "m.json"
    custom.write_text("{}")
    monkeypatch.setenv("CORPUS_MANIFEST", str(custom))
    assert manifest_path() == custom


def test_middleware_is_registered_exactly_once(tmp_path):
    """Regression: a bad merge once registered host/origin checks twice."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    app = create_app(Stub(tmp_path))
    kinds = [m.cls for m in app.user_middleware]
    assert kinds.count(TrustedHostMiddleware) == 1
    assert kinds.count(BaseHTTPMiddleware) == 1


def test_empty_allowed_hosts_env_falls_back_to_settings(monkeypatch):
    from grounded_qa.api.main import allowed_hosts

    monkeypatch.setenv("ALLOWED_HOSTS", "   ")
    assert "localhost" in allowed_hosts()
    monkeypatch.setenv("ALLOWED_HOSTS", "a.example, b.example")
    assert allowed_hosts() == ["a.example", "b.example"]
