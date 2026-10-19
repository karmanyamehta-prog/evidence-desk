"""Local single-user HTTP API. No authentication: bind only to loopback."""

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from grounded_qa.config import Settings
from grounded_qa.generation.client import ModelError
from grounded_qa.ingestion.pdf import PDFError
from grounded_qa.models import Query
from grounded_qa.service import Service


def _split(raw: str) -> list[str]:
    return [h.strip() for h in raw.split(",") if h.strip()]


def allowed_hosts(settings: Settings | None = None) -> list[str]:
    """Hosts the API answers to (ALLOWED_HOSTS); blank falls back to the defaults."""
    hosts = _split((settings or Settings()).allowed_hosts)
    return hosts or _split(Settings.model_fields["allowed_hosts"].default)


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service or await run_in_threadpool(Service, Settings())
        yield

    app = FastAPI(title="Grounded QA", version="0.1.0", lifespan=lifespan)
    hosts = allowed_hosts()
    # Unauthenticated local API: reject DNS-rebinding hosts and cross-site browser POSTs.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    @app.middleware("http")
    async def reject_foreign_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and request.method not in {"GET", "HEAD", "OPTIONS"}:
            host = urlsplit(origin).hostname or ""
            if host not in hosts and f"[{host}]" not in hosts:
                return JSONResponse(
                    {"detail": "Cross-origin request refused"}, status_code=403
                )
        return await call_next(request)

    @app.get("/health")
    def health():
        """Liveness plus a best-effort model check; never fails because Ollama is down."""
        s = app.state.service.settings
        model_available = None
        if s.llm_provider == "ollama":
            try:
                with httpx.Client(trust_env=False, timeout=2) as client:
                    tags = client.get(s.ollama_url.rstrip("/") + "/api/tags").json()
                model_available = any(
                    m.get("name") == s.llm_model for m in tags.get("models", [])
                )
            except (httpx.HTTPError, ValueError):
                model_available = False
        return {
            "status": "ready",
            "provider": s.llm_provider,
            "model_available": model_available,
        }

    @app.post("/query")
    def query(body: Query):
        try:
            return app.state.service.query(body.question.strip(), body.document_ids)
        except ModelError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/documents")
    async def upload(file: UploadFile):
        if os.getenv("DEMO_READ_ONLY") == "1":
            raise HTTPException(403, "Hosted demo uses the fixed corpus only")
        settings = app.state.service.settings
        temp_path = None
        try:
            with NamedTemporaryFile(suffix=".pdf", delete=False) as target:
                temp_path = Path(target.name)
                size = 0
                while block := await file.read(1024 * 1024):
                    size += len(block)
                    if size > settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, "PDF exceeds upload limit")
                    target.write(block)
            title = Path((file.filename or "upload.pdf").replace("\\", "/")).name
            return await run_in_threadpool(app.state.service.ingest, temp_path, title)
        except PDFError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            await file.close()
            if temp_path:
                temp_path.unlink(missing_ok=True)

    @app.get("/documents")
    def documents():
        return app.state.service.list_documents()

    @app.get("/documents/{document_id}/pdf")
    def pdf(document_id: str):
        if not re.fullmatch(r"[a-f0-9]{64}", document_id):
            raise HTTPException(404, "Document not found")
        path = (
            app.state.service.settings.data_dir
            / "documents"
            / document_id
            / "source.pdf"
        )
        if not path.is_file():
            raise HTTPException(404, "Document not found")
        return FileResponse(path, media_type="application/pdf")

    return app


app = create_app()
