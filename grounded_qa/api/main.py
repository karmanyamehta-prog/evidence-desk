"""Local single-user HTTP API. No authentication: bind only to loopback."""
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
import re
import os
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from grounded_qa.config import Settings
from grounded_qa.models import Query
from grounded_qa.service import Service
from grounded_qa.ingestion.pdf import PDFError
from grounded_qa.generation.client import ModelError

def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service or await run_in_threadpool(Service, Settings())
        yield
    app = FastAPI(title="Grounded QA", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ready", "provider": app.state.service.settings.llm_provider}

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
                while block := await file.read(1024*1024):
                    size += len(block)
                    if size > settings.max_upload_mb * 1024*1024:
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
        import json
        return [json.loads(p.read_text()) for p in
                (app.state.service.settings.data_dir / "documents").glob("*/metadata.json")]

    @app.get("/documents/{document_id}/pdf")
    def pdf(document_id: str):
        if not re.fullmatch(r"[a-f0-9]{64}", document_id):
            raise HTTPException(404, "Document not found")
        path = app.state.service.settings.data_dir / "documents" / document_id / "source.pdf"
        if not path.is_file():
            raise HTTPException(404, "Document not found")
        return FileResponse(path, media_type="application/pdf")
    return app

app = create_app()
