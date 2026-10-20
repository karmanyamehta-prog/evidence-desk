"""Unit tests for grounded_qa.generation.client — Generator and extract_json."""

import json

import httpx
import pytest

from grounded_qa.config import Settings
from grounded_qa.generation.client import Generator, ModelError, extract_json
from grounded_qa.models import Draft

# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


def test_extract_json_plain_object():
    assert extract_json('{"claims": []}') == '{"claims": []}'


def test_extract_json_strips_whitespace():
    assert extract_json('  {"claims": []}  ') == '{"claims": []}'


def test_extract_json_fenced_with_json_tag():
    result = extract_json('```json\n{"claims": []}\n```')
    assert result == '{"claims": []}'


def test_extract_json_fenced_without_tag():
    result = extract_json('```\n{"claims": []}\n```')
    assert result == '{"claims": []}'


def test_extract_json_chatter_around_object():
    result = extract_json('Sure! {"claims": []} Hope that helps')
    assert result == '{"claims": []}'


def test_extract_json_returns_input_when_no_braces():
    # No JSON object — returns the input as-is so the caller gets a clear parse error
    text = "No JSON here at all"
    assert extract_json(text) == text


def test_extract_json_multiline_object():
    text = '```json\n{\n  "claims": []\n}\n```'
    result = extract_json(text)
    parsed = json.loads(result)
    assert parsed == {"claims": []}


# ---------------------------------------------------------------------------
# Generator — extractive mode (no HTTP calls)
# ---------------------------------------------------------------------------


def test_extractive_mode_returns_draft_from_sources(tmp_path):
    settings = Settings(data_dir=tmp_path, llm_provider="extractive")
    gen = Generator(settings)
    sources = {
        "S1": {"text": "AI systems should be transparent and accountable."},
        "S2": {"text": "Risk management frameworks reduce organisational exposure."},
        "S3": {"text": "This third source should be ignored due to the two-claim cap."},
    }
    draft = gen.generate("What is important for AI?", sources)
    assert isinstance(draft, Draft)
    # Extractive mode caps at 2 claims
    assert len(draft.claims) <= 2
    # Each claim's evidence quote is the chunk text
    for claim in draft.claims:
        assert len(claim.evidence) == 1
        assert claim.evidence[0].quote in (s["text"] for s in sources.values())


def test_extractive_mode_empty_sources_returns_empty_draft(tmp_path):
    settings = Settings(data_dir=tmp_path, llm_provider="extractive")
    gen = Generator(settings)
    draft = gen.generate("question", {})
    assert draft.abstain


def test_extractive_mode_last_raw_stays_none(tmp_path):
    settings = Settings(data_dir=tmp_path, llm_provider="extractive")
    gen = Generator(settings)
    gen.generate(
        "q", {"S1": {"text": "Some text long enough to be a valid quote here."}}
    )
    assert gen.last_raw is None


# ---------------------------------------------------------------------------
# Generator — HTTP providers (monkeypatched)
# ---------------------------------------------------------------------------


def _make_response(body: dict, status: int = 200):
    """Return an httpx.Response with a request attached (required for raise_for_status)."""
    content = json.dumps(body).encode()
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return httpx.Response(status_code=status, content=content, request=request)


def test_ollama_provider_parses_response(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, llm_provider="ollama")
    payload = {"claims": []}

    def fake_post(self, url, **kwargs):
        return _make_response({"message": {"content": json.dumps(payload)}})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    gen = Generator(settings)
    draft = gen.generate("question", {"S1": {"text": "evidence text here"}})
    assert isinstance(draft, Draft)
    assert gen.last_raw == json.dumps(payload)


def test_openai_provider_parses_response(tmp_path, monkeypatch):
    settings = Settings(
        data_dir=tmp_path,
        llm_provider="openai",
        api_key="sk-test",
        provider_url="https://api.openai.com/v1/chat/completions",
    )
    payload = {"claims": []}

    def fake_post(self, url, **kwargs):
        return _make_response(
            {"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    gen = Generator(settings)
    draft = gen.generate("question", {"S1": {"text": "evidence text here"}})
    assert isinstance(draft, Draft)


def test_anthropic_provider_parses_response(tmp_path, monkeypatch):
    settings = Settings(
        data_dir=tmp_path,
        llm_provider="anthropic",
        api_key="sk-ant-test",
        provider_url="https://api.anthropic.com/v1/messages",
    )
    payload = {"claims": []}

    def fake_post(self, url, **kwargs):
        return _make_response(
            {"content": [{"type": "text", "text": json.dumps(payload)}]}
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    gen = Generator(settings)
    draft = gen.generate("question", {"S1": {"text": "evidence text here"}})
    assert isinstance(draft, Draft)


def test_timeout_raises_model_error(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, llm_provider="ollama")

    def fake_post(self, url, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    gen = Generator(settings)
    with pytest.raises(ModelError, match="timed out"):
        gen.generate("question", {"S1": {"text": "text"}})


def test_http_error_raises_model_error(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, llm_provider="ollama")

    def fake_post(self, url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    gen = Generator(settings)
    with pytest.raises(ModelError, match="unavailable"):
        gen.generate("question", {"S1": {"text": "text"}})


def test_bad_json_response_raises_model_error(tmp_path, monkeypatch):
    """A response that is not valid JSON for Draft should raise ModelError."""
    settings = Settings(data_dir=tmp_path, llm_provider="ollama")

    def fake_post(self, url, **kwargs):
        return _make_response({"message": {"content": "not valid json at all"}})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    gen = Generator(settings)
    with pytest.raises(
        (ModelError, ValueError)
    ):  # pydantic ValidationError or ModelError
        gen.generate("question", {"S1": {"text": "text"}})


def test_missing_key_in_response_raises_model_error(tmp_path, monkeypatch):
    """Ollama response missing 'message' key should raise ModelError."""
    settings = Settings(data_dir=tmp_path, llm_provider="ollama")

    def fake_post(self, url, **kwargs):
        # Missing 'message' key
        return _make_response({"unexpected": "shape"})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    gen = Generator(settings)
    with pytest.raises(ModelError):
        gen.generate("question", {"S1": {"text": "text"}})


def test_http_4xx_raises_model_error(tmp_path, monkeypatch):
    """A 4xx HTTP status from the provider should raise ModelError."""
    settings = Settings(data_dir=tmp_path, llm_provider="ollama")

    def fake_post(self, url, **kwargs):
        return _make_response({"error": "not found"}, status=404)

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    gen = Generator(settings)
    with pytest.raises(ModelError):
        gen.generate("question", {"S1": {"text": "text"}})


def test_attempt_seeds_differ(tmp_path, monkeypatch):
    """Seed changes between attempts so retry gets a different sample."""
    seeds_seen = []

    def fake_post(self, url, json=None, **kwargs):
        if json and "options" in json:
            seeds_seen.append(json["options"]["seed"])
        return _make_response({"message": {"content": '{"claims": []}'}})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    settings = Settings(data_dir=tmp_path, llm_provider="ollama")
    gen = Generator(settings)
    gen.generate("q", {"S1": {"text": "t"}}, attempt=0)
    gen.generate("q", {"S1": {"text": "t"}}, attempt=1)
    assert seeds_seen[0] != seeds_seen[1]
