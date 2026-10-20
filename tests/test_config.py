"""Unit tests for grounded_qa.config — Settings validation and manifest_path."""

import json

import pytest

from grounded_qa.config import Settings, manifest_path

# ---------------------------------------------------------------------------
# Settings field defaults
# ---------------------------------------------------------------------------


def test_settings_defaults(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.llm_provider == "ollama"
    assert s.top_k == 5
    assert s.chunk_tokens == 350
    assert s.chunk_overlap == 50
    assert s.max_upload_mb == 20


# ---------------------------------------------------------------------------
# model_validator: overlap vs size
# ---------------------------------------------------------------------------


def test_overlap_equal_to_size_raises(tmp_path):
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(data_dir=tmp_path, chunk_tokens=100, chunk_overlap=100)


def test_overlap_greater_than_size_raises(tmp_path):
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(data_dir=tmp_path, chunk_tokens=100, chunk_overlap=150)


def test_overlap_less_than_size_passes(tmp_path):
    s = Settings(data_dir=tmp_path, chunk_tokens=100, chunk_overlap=99)
    assert s.chunk_overlap == 99


def test_zero_overlap_passes(tmp_path):
    s = Settings(data_dir=tmp_path, chunk_overlap=0)
    assert s.chunk_overlap == 0


# ---------------------------------------------------------------------------
# model_validator: paid provider requires key + url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_paid_provider_without_key_raises(tmp_path, provider):
    with pytest.raises(ValueError, match="API_KEY"):
        Settings(
            data_dir=tmp_path,
            llm_provider=provider,
            api_key="",
            provider_url="https://api.example.com",
        )


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_paid_provider_without_url_raises(tmp_path, provider):
    with pytest.raises(ValueError, match="API_KEY"):
        Settings(
            data_dir=tmp_path, llm_provider=provider, api_key="sk-abc", provider_url=""
        )


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_paid_provider_with_key_and_url_passes(tmp_path, provider):
    s = Settings(
        data_dir=tmp_path,
        llm_provider=provider,
        api_key="sk-abc123",
        provider_url="https://api.example.com/v1",
    )
    assert s.llm_provider == provider


def test_extractive_provider_needs_no_key(tmp_path):
    s = Settings(data_dir=tmp_path, llm_provider="extractive")
    assert s.llm_provider == "extractive"


def test_ollama_needs_no_key(tmp_path):
    s = Settings(data_dir=tmp_path, llm_provider="ollama")
    assert s.llm_provider == "ollama"


# ---------------------------------------------------------------------------
# Field-level constraints
# ---------------------------------------------------------------------------


def test_chunk_tokens_min_bound(tmp_path):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, chunk_tokens=31, chunk_overlap=0)


def test_top_k_min_bound(tmp_path):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, top_k=0)


def test_top_k_max_bound(tmp_path):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, top_k=11)


def test_model_timeout_must_be_positive(tmp_path):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, model_timeout=0)


# ---------------------------------------------------------------------------
# manifest_path
# ---------------------------------------------------------------------------


def test_manifest_path_found_next_to_package(monkeypatch, tmp_path):
    """Falls through to the package-relative candidate when cwd has no corpus."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CORPUS_MANIFEST", raising=False)
    p = manifest_path()
    assert p.name == "manifest.json"
    assert p.is_file()


def test_manifest_path_env_var_wins(monkeypatch, tmp_path):
    custom = tmp_path / "custom_manifest.json"
    custom.write_text(json.dumps({"sha256": "abc"}))
    monkeypatch.setenv("CORPUS_MANIFEST", str(custom))
    assert manifest_path() == custom


def test_manifest_path_raises_when_all_candidates_missing(monkeypatch, tmp_path):
    """FileNotFoundError is raised only when every candidate path is absent.

    The package-relative corpus/manifest.json always exists in the repo, so we
    can only exercise this branch by pointing CORPUS_MANIFEST at a nonexistent
    path AND patching Path.is_file to pretend the package-relative one is also
    gone.
    """
    from pathlib import Path

    from grounded_qa import config as config_module

    nonexistent = tmp_path / "no_such_file.json"
    monkeypatch.setenv("CORPUS_MANIFEST", str(nonexistent))
    monkeypatch.chdir(tmp_path)

    def fake_is_file(self):
        # Pretend every path is absent
        return False

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    with pytest.raises(FileNotFoundError):
        config_module.manifest_path()


def test_manifest_path_cwd_corpus_wins(monkeypatch, tmp_path):
    """If ./corpus/manifest.json exists in cwd, that takes priority over the package."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    manifest = corpus_dir / "manifest.json"
    manifest.write_text(json.dumps({"sha256": "cwd-sha"}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CORPUS_MANIFEST", raising=False)
    assert manifest_path() == manifest
