"""Offline end-to-end test: real PDF parsing, chunking, Chroma and HTTP layer.

Only the embedding model and the LLM are replaced (a hashed bag-of-words embedder
and a scripted generator), so this runs in CI without torch weights or Ollama.
It does not test E5 retrieval quality or Ollama output.
"""

import hashlib
import re

import matplotlib
import pytest
from fastapi.testclient import TestClient

from grounded_qa.api.main import create_app
from grounded_qa.config import Settings
from grounded_qa.models import Claim, Draft, Evidence
from grounded_qa.retrieval.store import VectorStore
from grounded_qa.service import Service

matplotlib.use("Agg")  # non-interactive backend; avoids Tk/display dependency in CI
plt = pytest.importorskip("matplotlib.pyplot")

PAGES = [
    "Privacy controls limit how personal data is collected and retained by the system.",
    "Human oversight is important for managing risk in high impact deployments.",
    "Incident reporting procedures require documented escalation to the safety board.",
]


class Tok:
    def encode(self, text, add_special_tokens=False):
        return re.findall(r"\w+|[^\w\s]", text)


class HashEmbedder:
    tokenizer = Tok()

    def _vec(self, text):
        v = [0.0] * 64
        for w in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def passages(self, texts):
        return [self._vec(t) for t in texts]

    queries = passages


class ScriptedGenerator:
    """Quotes the top source, as a well-behaved model would."""

    last_raw = "scripted"

    def generate(self, question, sources, correction="", attempt=0):
        sid, src = next(iter(sources.items()))
        quote = " ".join(src["text"].split()[:8])
        return Draft(
            claims=[
                Claim(text="Summary.", evidence=[Evidence(source_id=sid, quote=quote)])
            ]
        )


@pytest.fixture
def pdf_path(tmp_path):
    plt.rcParams["pdf.fonttype"] = 42  # embed real fonts so text is extractable
    from matplotlib.backends.backend_pdf import PdfPages

    path = tmp_path / "report.pdf"
    with PdfPages(path) as pdf:
        for text in PAGES:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.1, 0.8, "\n".join(re.findall(r".{1,60}(?:\s|$)", text)))
            pdf.savefig(fig)
            plt.close(fig)
    return path


def make_service(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", llm_provider="ollama")
    embedder = HashEmbedder()
    store = VectorStore(settings.data_dir / "chroma", "test", embedder)
    return Service(settings, embedder, store, ScriptedGenerator())


def test_upload_query_and_serve_pdf(tmp_path, pdf_path):
    service = make_service(tmp_path)
    with TestClient(create_app(service)) as client:
        upload = client.post(
            "/documents", files={"file": ("report.pdf", pdf_path.read_bytes())}
        )
        assert upload.status_code == 200, upload.text
        doc = upload.json()
        assert doc["pages"] == 3 and doc["chunks"] >= 3
        assert [d["document_id"] for d in client.get("/documents").json()] == [
            doc["document_id"]
        ]

        result = client.post(
            "/query", json={"question": "How is incident escalation documented?"}
        ).json()
        assert result["status"] == "answered" and not result["abstain"]
        assert result["attempts"] and result["attempts"][0]["error"] == ""
        source = result["sources"]["S1"]
        assert source["page"] == 3 and "escalation" in source["text"]

        pdf = client.get(f"/documents/{source['document_id']}/pdf")
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")


def test_reupload_is_idempotent_and_survives_restart(tmp_path, pdf_path):
    first = make_service(tmp_path).ingest(pdf_path, "report")
    again = make_service(tmp_path).ingest(pdf_path, "report")  # new process, same data
    assert first == again
    assert len(make_service(tmp_path).list_documents()) == 1


def test_failed_index_leaves_no_files(tmp_path, pdf_path):
    service = make_service(tmp_path)

    def boom(chunks):
        raise RuntimeError("embedding failed")

    service.store.add = boom
    with pytest.raises(RuntimeError):
        service.ingest(pdf_path, "report")
    assert not list((service.settings.data_dir / "documents").glob("*"))
    assert service.list_documents() == []
