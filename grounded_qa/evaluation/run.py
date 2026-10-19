"""Run frozen queries. Raw outputs survive failures; no invented judge labels."""

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

from grounded_qa.config import Settings, manifest_path
from grounded_qa.generation.client import SYSTEM, ModelError
from grounded_qa.service import Service


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=Path, default=Path("evaluation/queries.json"))
    p.add_argument("--out", type=Path, default=Path("evaluation/runs/local.json"))
    p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()
    s = Settings()
    t = time.perf_counter()
    service = Service(s)
    queries = json.loads(args.queries.read_text())
    rows = []
    run = {
        "timestamp": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "settings": s.model_dump(mode="json", exclude={"api_key"}),
        "query_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        "initialization_seconds": time.perf_counter() - t,
        "rows": rows,
        "prompt": SYSTEM,
        "seed": 42,
        "temperature": 0,
        "corpus": json.loads(manifest_path().read_text()),
        "index_chunk_count": service.store.collection.count(),
    }
    if s.llm_provider == "ollama":
        import httpx

        with httpx.Client(trust_env=False, timeout=20) as c:
            response = c.get(s.ollama_url + "/api/tags")
            response.raise_for_status()
            run["ollama_models"] = response.json()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for q in queries:
        if args.split != "all" and q["split"] != args.split:
            continue
        for repetition in range(args.repeats):
            t = time.perf_counter()
            try:
                result = service.query(q["question"])
                rows.append(
                    {
                        "query_id": q["id"],
                        "split": q["split"],
                        "repetition": repetition,
                        "result": result,
                    }
                )
                print(
                    q["id"],
                    result["abstain"],
                    round(result["timing_ms"]["total"] / 1000, 2),
                    flush=True,
                )
            except (ModelError, ValueError) as exc:
                rows.append(
                    {
                        "query_id": q["id"],
                        "split": q["split"],
                        "repetition": repetition,
                        "error": str(exc),
                        "elapsed_seconds": time.perf_counter() - t,
                    }
                )
                print(q["id"], str(exc), flush=True)
            args.out.write_text(
                json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
