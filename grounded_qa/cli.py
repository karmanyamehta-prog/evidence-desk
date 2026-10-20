"""Reproducible commands: fetch, ingest, doctor."""

import argparse
import hashlib
import json
import logging
import platform
import shutil
import urllib.request
from pathlib import Path

from grounded_qa.config import Settings, manifest_path

RMF_URL = "https://www.govinfo.gov/content/pkg/GOVPUB-C13-ef6882c6bd970bc733de394f469ea3f3/pdf/GOVPUB-C13-ef6882c6bd970bc733de394f469ea3f3.pdf"

logger = logging.getLogger(__name__)


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
    path = args.pdf or s.data_dir / "sources" / "rmf.pdf"
    if args.command == "fetch":
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = json.loads(manifest_path().read_text())["sha256"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            logger.info("Verified existing %s", path)
            print(f"Verified existing {path}")
            return
        request = urllib.request.Request(
            RMF_URL,
            headers={
                "User-Agent": "evidence-desk/0.1 (corpus fetch; checksum-verified)"
            },
        )
        with urllib.request.urlopen(request, timeout=60) as r:
            content = r.read(21 * 1024 * 1024)
        if (
            not content.startswith(b"%PDF-")
            or hashlib.sha256(content).hexdigest() != expected
        ):
            raise ValueError(
                "Source checksum changed; review the corpus before using it"
            )
        path.write_bytes(content)
        logger.info("Downloaded and verified %s", path)
        print(f"Downloaded and verified {path}")
        return
    from grounded_qa.service import Service

    print(
        json.dumps(
            Service(s).ingest(
                path,
                path.name if args.pdf else "NIST AI RMF 1.0 (2023)",
                "" if args.pdf else RMF_URL,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
