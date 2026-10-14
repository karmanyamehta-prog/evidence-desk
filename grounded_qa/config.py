"""Validated environment-only configuration; no credentials in source."""

import os
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def manifest_path() -> Path:
    """Corpus manifest: CORPUS_MANIFEST, then ./corpus, then next to the package."""
    candidates = [
        Path(os.environ["CORPUS_MANIFEST"])
        if os.environ.get("CORPUS_MANIFEST")
        else None,
        Path.cwd() / "corpus" / "manifest.json",
        Path(__file__).resolve().parents[1] / "corpus" / "manifest.json",
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    raise FileNotFoundError(
        "corpus/manifest.json not found; run from the repo root or set CORPUS_MANIFEST"
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    data_dir: Path = Path("data")
    embedding_model: str = "intfloat/e5-small-v2"
    embedding_prefixes: bool | None = None  # None = infer from model name
    allowed_hosts: str = "localhost,127.0.0.1,[::1],testserver,api"
    embedding_revision: str = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
    chunk_tokens: int = Field(350, ge=32, le=450)
    chunk_overlap: int = Field(50, ge=0)
    top_k: int = Field(5, ge=1, le=10)
    llm_provider: Literal["ollama", "openai", "anthropic", "extractive"] = "ollama"
    llm_model: str = "qwen2.5:3b-instruct-q4_K_M"
    ollama_url: str = "http://localhost:11434"
    api_key: str = ""
    provider_url: str = ""
    model_timeout: float = Field(180, gt=0)
    context_tokens: int = Field(4096, ge=2048)
    output_tokens: int = Field(384, ge=64, le=2048)
    context_chars: int = Field(4500, ge=1000, le=16000)
    cpu_threads: int = Field(4, ge=1)
    max_upload_mb: int = Field(20, ge=1, le=100)
    max_pages: int = Field(200, ge=1)

    @model_validator(mode="after")
    def validate_combination(self):
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_TOKENS")
        if self.llm_provider in {"openai", "anthropic"} and (
            not self.api_key or not self.provider_url
        ):
            raise ValueError("Paid provider requires explicit API_KEY and PROVIDER_URL")
        return self
