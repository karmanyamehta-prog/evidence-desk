"""Reproducible commands: fetch, ingest, doctor."""

import argparse
import hashlib
import json
import logging
import platform
import shutil
import urllib.request
from pathlib import Path

from grounded_qa.config import Settings, corpus_documents

logger = logging.getLogger(__name__)

USER_AGENT = "evidence-desk/0.1 (corpus fetch; checksum-verified)"


def _local_path(sources_dir: Path, document: dict) -> Path:
    """A stable local filename per document, derived from its checksum."""
    name = document.get("file") or f"{document['publication'].replace(' ', '-')}.pdf"
    return sources_dir / name


def fetch_document(document: dict, sources_dir: Path) -> Path:
    """Download one manifest document and verify it against its checksum."""
    sources_dir.mkdir(parents=True, exist_ok=True)
    path = _local_path(sources_dir, document)
    expected = document["sha256"]
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        logger.info("Verified existing %s", path)
        print(f"Verified existing {path}")
        return path
    request = urllib.request.Request(
        document["source_url"], headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=120) as r:
        content = r.read(21 * 1024 * 1024)
    if (
        not content.startswith(b"%PDF-")
        or hashlib.sha256(content).hexdigest() != expected
    ):
        raise ValueError(
            f"Source checksum changed for {document['publication']}; "
            "review the corpus before using it"
        )
    path.write_bytes(content)
    logger.info("Downloaded and verified %s", path)
    print(f"Downloaded and verified {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fetch", "ingest", "doctor"])
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    s = Settings()
    if args.command == "doctor":
        logger.info(
            "Python %s OS %s free_disk_gb %.1f model %s provider %s",
            platform.python_version(),
            platform.platform(),
            round(shutil.disk_usage(".").free / 1e9, 1),
            s.llm_model,
            s.llm_provider,
        )
        print(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "os": platform.platform(),
                    "free_disk_gb": round(shutil.disk_usage(".").free / 1e9, 1),
                    "model": s.llm_model,
                    "provider": s.llm_provider,
                },
                indent=2,
            )
        )
        return
    sources_dir = s.data_dir / "sources"
    if args.command == "fetch":
        for document in corpus_documents():
            fetch_document(document, sources_dir)
        return
    from grounded_qa.service import Service

    if args.pdf:
        # An explicit PDF is ingested on its own, with the filename as the title.
        print(
            json.dumps(
                Service(s).ingest(args.pdf, args.pdf.name, ""),
                indent=2,
            )
        )
        return
    # Ingest every document in the manifest, in manifest order.
    service = Service(s)
    results = []
    for document in corpus_documents():
        path = fetch_document(document, sources_dir)
        results.append(
            service.ingest(
                path,
                document["title"],
                document.get("source_url", ""),
            )
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
