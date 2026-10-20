"""Additional unit tests for grounded_qa.api.main — uncovered API paths."""

from fastapi.testclient import TestClient

from grounded_qa.api.main import _split, allowed_hosts, create_app
from grounded_qa.config import Settings
from grounded_qa.ingestion.pdf import PDFError

# ---------------------------------------------------------------------------
# _split() helper
# ---------------------------------------------------------------------------


def test_split_basic():
    assert _split("a,b,c") == ["a", "b", "c"]


def test_split_strips_whitespace():
    assert _split("  a , b , c  ") == ["a", "b", "c"]


def test_split_ignores_empty_segments():
    assert _split(",a,,b,") == ["a", "b"]


def test_split_empty_string():
    assert _split("") == []


def test_split_whitespace_only():
    assert _split("   ,   ") == []


# ---------------------------------------------------------------------------
# allowed_hosts()
# ---------------------------------------------------------------------------


def test_allowed_hosts_default_contains_localhost(monkeypatch):
    monkeypatch.delenv("ALLOWED_HOSTS", raising=False)
    hosts = allowed_hosts()
    assert "localhost" in hosts
    assert "127.0.0.1" in hosts


def test_allowed_hosts_env_overrides(monkeypatch):
    monkeypatch.setenv("ALLOWED_HOSTS", "myhost.local,192.168.1.1")
    hosts = allowed_hosts()
    assert hosts == ["myhost.local", "192.168.1.1"]


def test_allowed_hosts_blank_env_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("ALLOWED_HOSTS", "   ")
    hosts = allowed_hosts()
    assert "localhost" in hosts


def test_allowed_hosts_with_explicit_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOWED_HOSTS", raising=False)
    s = Settings(data_dir=tmp_path, allowed_hosts="custom.host")
    # allowed_hosts() reads from env first; pass settings explicitly
    # (the function signature accepts an optional settings arg)
    hosts = allowed_hosts(s)
    assert "custom.host" in hosts


# ---------------------------------------------------------------------------
# Stub service for API tests
# ---------------------------------------------------------------------------


class StubService:
    def __init__(self, tmp_path):
        self.settings = Settings(data_dir=tmp_path)

    def query(self, question, document_ids):
        return {"question": question, "status": "answered", "claims": []}

    def ingest(self, path, title):
        return {
            "document_id": "abc" * 21 + "ab",
            "title": title,
            "pages": 1,
            "chunks": 1,
        }

    def list_documents(self):
        return []


class StubServicePDFError:
    def __init__(self, tmp_path):
        self.settings = Settings(data_dir=tmp_path)

    def query(self, question, document_ids):
        return {}

    def ingest(self, path, title):
        raise PDFError("image-only PDF")

    def list_documents(self):
        return []


# ---------------------------------------------------------------------------
# /documents endpoint — list and PDF serve
# ---------------------------------------------------------------------------


def test_documents_list_empty(tmp_path):
    with TestClient(create_app(StubService(tmp_path))) as c:
        resp = c.get("/documents")
        assert resp.status_code == 200
        assert resp.json() == []


def test_pdf_endpoint_rejects_non_hex_id(tmp_path):
    with TestClient(create_app(StubService(tmp_path))) as c:
        # "not-an-id" is not 64 hex chars
        assert c.get("/documents/not-an-id/pdf").status_code == 404


def test_pdf_endpoint_rejects_short_hex(tmp_path):
    with TestClient(create_app(StubService(tmp_path))) as c:
        short_hex = "a" * 32  # only 32 chars, need 64
        assert c.get(f"/documents/{short_hex}/pdf").status_code == 404


def test_pdf_endpoint_missing_file_returns_404(tmp_path):
    with TestClient(create_app(StubService(tmp_path))) as c:
        valid_hex = "a" * 64  # right format, file doesn't exist
        assert c.get(f"/documents/{valid_hex}/pdf").status_code == 404


def test_pdf_endpoint_serves_existing_file(tmp_path):
    doc_id = "a" * 64
    doc_dir = tmp_path / "documents" / doc_id
    doc_dir.mkdir(parents=True)
    pdf_path = doc_dir / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test content")

    with TestClient(create_app(StubService(tmp_path))) as c:
        resp = c.get(f"/documents/{doc_id}/pdf")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"


# ---------------------------------------------------------------------------
# Upload — PDFError from service → 422
# ---------------------------------------------------------------------------


def test_upload_pdf_error_returns_422(tmp_path):
    with TestClient(create_app(StubServicePDFError(tmp_path))) as c:
        resp = c.post(
            "/documents",
            files={
                "file": ("test.pdf", b"%PDF-1.4 minimal content", "application/pdf")
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DEMO_READ_ONLY flag
# ---------------------------------------------------------------------------


def test_upload_blocked_in_demo_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_READ_ONLY", "1")
    with TestClient(create_app(StubService(tmp_path))) as c:
        resp = c.post(
            "/documents",
            files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
        )
        assert resp.status_code == 403
        assert "demo" in resp.json()["detail"].lower()


def test_upload_allowed_when_demo_mode_off(tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_READ_ONLY", raising=False)
    with TestClient(create_app(StubService(tmp_path))) as c:
        resp = c.post(
            "/documents",
            files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
        )
        # StubService.ingest returns metadata dict → 200
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /health with non-Ollama provider (model_available stays None)
# ---------------------------------------------------------------------------


def test_health_non_ollama_provider_no_model_check(tmp_path):
    """For extractive provider, /health should not attempt to probe Ollama."""

    class ExtractiveStub:
        settings = Settings(data_dir=tmp_path, llm_provider="extractive")

        def list_documents(self):
            return []

    with TestClient(create_app(ExtractiveStub())) as c:
        body = c.get("/health").json()
        assert body["status"] == "ready"
        assert body["model_available"] is None
        assert body["provider"] == "extractive"


# ---------------------------------------------------------------------------
# Cross-origin: same-origin requests pass through
# ---------------------------------------------------------------------------


def test_same_origin_post_is_allowed(tmp_path):
    with TestClient(create_app(StubService(tmp_path))) as c:
        resp = c.post(
            "/query",
            json={"question": "What is risk?"},
            headers={"origin": "http://localhost:8501"},
        )
        assert resp.status_code == 200


def test_no_origin_header_post_is_allowed(tmp_path):
    """Programmatic clients (CLI, scripts) don't send Origin."""
    with TestClient(create_app(StubService(tmp_path))) as c:
        resp = c.post("/query", json={"question": "What is risk?"})
        assert resp.status_code == 200
