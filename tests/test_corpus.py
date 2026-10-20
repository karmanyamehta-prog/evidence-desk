"""Unit tests for the multi-document corpus: manifest, fetch, and CLI ingest.

The corpus manifest now lists documents rather than describing a single PDF.
These tests cover the manifest reader, the checksum-verified fetch helper and
the action-ID scheme of the second corpus document (the Generative AI Profile,
which numbers its appendix actions "A.1.1" rather than "GOVERN 1.6").
"""

import hashlib
import json

import pytest

from grounded_qa.config import corpus_documents
from grounded_qa.ingestion.chunking import chunk_pages

# ---------------------------------------------------------------------------
# corpus_documents
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path, payload):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_corpus_documents_reads_the_documents_list(monkeypatch, tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {
            "corpus": "test",
            "documents": [
                {"publication": "A", "sha256": "aa"},
                {"publication": "B", "sha256": "bb"},
            ],
        },
    )
    monkeypatch.setenv("CORPUS_MANIFEST", str(manifest))
    docs = corpus_documents()
    assert [d["publication"] for d in docs] == ["A", "B"]


def test_corpus_documents_accepts_a_bare_list(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path, [{"publication": "A", "sha256": "aa"}])
    monkeypatch.setenv("CORPUS_MANIFEST", str(manifest))
    assert len(corpus_documents()) == 1


def test_corpus_documents_accepts_a_single_document_manifest(monkeypatch, tmp_path):
    """An older one-document manifest keeps working."""
    manifest = _write_manifest(tmp_path, {"title": "old", "sha256": "aa"})
    monkeypatch.setenv("CORPUS_MANIFEST", str(manifest))
    docs = corpus_documents()
    assert len(docs) == 1
    assert docs[0]["sha256"] == "aa"


def test_corpus_documents_rejects_an_empty_document_list(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path, {"documents": []})
    monkeypatch.setenv("CORPUS_MANIFEST", str(manifest))
    with pytest.raises(ValueError, match="no documents"):
        corpus_documents()


def test_corpus_documents_rejects_a_manifest_without_documents(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path, {"corpus": "test"})
    monkeypatch.setenv("CORPUS_MANIFEST", str(manifest))
    with pytest.raises(ValueError, match="no documents"):
        corpus_documents()


def test_checked_in_manifest_lists_two_documents():
    """The repository corpus ships the RMF and the Generative AI Profile."""
    docs = corpus_documents()
    publications = {d["publication"] for d in docs}
    assert publications == {"NIST AI 100-1", "NIST AI 600-1"}
    for doc in docs:
        assert len(doc["sha256"]) == 64
        assert doc["source_url"].startswith("https://")
        assert doc["pages"] > 0


# ---------------------------------------------------------------------------
# fetch_document: checksum enforcement
# ---------------------------------------------------------------------------


def test_fetch_document_verifies_an_existing_file(tmp_path, monkeypatch):
    from grounded_qa import cli

    payload = b"%PDF-1.7 fake body"
    digest = hashlib.sha256(payload).hexdigest()
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "doc.pdf").write_bytes(payload)

    def fail(*args, **kwargs):
        raise AssertionError("must not download when the checksum already matches")

    monkeypatch.setattr(cli.urllib.request, "urlopen", fail)
    path = cli.fetch_document(
        {
            "publication": "X",
            "file": "doc.pdf",
            "sha256": digest,
            "source_url": "https://example/x.pdf",
        },
        sources,
    )
    assert path.read_bytes() == payload


def test_fetch_document_rejects_a_checksum_mismatch(tmp_path, monkeypatch):
    from grounded_qa import cli

    payload = b"%PDF-1.7 fake body"
    sources = tmp_path / "sources"
    sources.mkdir()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return payload

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(ValueError, match="checksum changed"):
        cli.fetch_document(
            {
                "publication": "X",
                "sha256": "0" * 64,
                "source_url": "https://example/x.pdf",
            },
            sources,
        )


def test_fetch_document_rejects_a_non_pdf(tmp_path, monkeypatch):
    from grounded_qa import cli

    payload = b"<html>not a pdf</html>"
    sources = tmp_path / "sources"
    sources.mkdir()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return payload

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(ValueError, match="checksum changed"):
        cli.fetch_document(
            {
                "publication": "X",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_url": "https://example/x.pdf",
            },
            sources,
        )


def test_local_path_uses_the_manifest_filename(tmp_path):
    from grounded_qa import cli

    path = cli._local_path(
        tmp_path, {"publication": "NIST AI 100-1", "file": "rmf.pdf", "sha256": "x"}
    )
    assert path.name == "rmf.pdf"


def test_local_path_falls_back_to_the_publication_name(tmp_path):
    from grounded_qa import cli

    path = cli._local_path(tmp_path, {"publication": "NIST AI 100-1", "sha256": "x"})
    assert path.name == "NIST-AI-100-1.pdf"


# ---------------------------------------------------------------------------
# Action-ID schemes
# ---------------------------------------------------------------------------


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_rmf_action_id_is_still_recognized():
    pages = [{"page": 1, "text": "GOVERN 1.6: Inventory all AI systems in use."}]
    chunks = chunk_pages(pages, "doc", "Title", Tokenizer(), 32, 4)
    assert chunks[0].embed_prefix == "GOVERN 1.6: "


def test_genai_profile_action_id_is_recognized():
    """The Generative AI Profile numbers appendix actions "A.1.1"."""
    pages = [
        {
            "page": 51,
            "text": (
                "A.1.1. Overview\nOrganizations may choose to use a set of "
                "governance practices for generative AI risks."
            ),
        }
    ]
    chunks = chunk_pages(pages, "doc", "Title", Tokenizer(), 32, 4)
    assert chunks[0].embed_prefix == "A.1.1: "
    # The stored text stays the verbatim source passage.
    assert chunks[0].text.startswith("A.1.1. Overview")


def test_action_ids_split_sections():
    pages = [
        {
            "page": 51,
            "text": (
                "A.1.1. Overview\nFirst section body text for the splitter.\n"
                "A.1.2. Third-Party Considerations\nSecond section body text here."
            ),
        }
    ]
    chunks = chunk_pages(pages, "doc", "Title", Tokenizer(), 20, 4)
    prefixes = {c.embed_prefix for c in chunks}
    assert "A.1.1: " in prefixes
    assert "A.1.2: " in prefixes


def test_genai_action_id_is_caught_by_the_citation_check():
    """A claim naming "A.1.1" must be checked against its quote."""
    from grounded_qa.citations import filter_claims
    from grounded_qa.models import Claim, Draft, Evidence

    sources = {
        "S1": {"text": "A.1.1. Overview Organizations should document decisions."},
        "S2": {"text": "A.1.2. Overview Organizations should document decisions."},
    }
    # The quote comes from S2 but the claim names A.1.1 — a mismatch.
    draft = Draft(
        claims=[
            Claim(
                text="A.1.1 requires documentation.",
                evidence=[
                    Evidence(
                        source_id="S2",
                        quote="A.1.2. Overview Organizations should document",
                    )
                ],
            )
        ]
    )
    valid, errors = filter_claims(draft, sources)
    assert valid == []
    assert "A.1.1" in errors[0]


def test_genai_action_id_matches_its_own_quote():
    from grounded_qa.citations import filter_claims
    from grounded_qa.models import Claim, Draft, Evidence

    sources = {
        "S1": {"text": "A.1.1. Overview Organizations should document decisions."},
    }
    draft = Draft(
        claims=[
            Claim(
                text="A.1.1 requires documentation.",
                evidence=[
                    Evidence(
                        source_id="S1",
                        quote="A.1.1. Overview Organizations should document",
                    )
                ],
            )
        ]
    )
    valid, errors = filter_claims(draft, sources)
    assert len(valid) == 1
    assert errors == []
