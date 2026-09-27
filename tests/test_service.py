from grounded_qa.config import Settings
from grounded_qa.models import Draft, Claim, Evidence, Chunk, Hit
from grounded_qa.service import Service


class Store:
    def __init__(self, hits):
        self.hits = hits

    def search(self, *args):
        return self.hits


class Generator:
    def __init__(self, drafts):
        self.drafts = iter(drafts)
        self.calls = 0

    def generate(self, *args):
        self.calls += 1
        return next(self.drafts)


def test_empty_search_never_generates(tmp_path):
    generator = Generator([])
    service = Service(Settings(data_dir=tmp_path), object(), Store([]), generator)
    assert service.query("a question")["abstain"]
    assert generator.calls == 0


def test_invalid_quote_retries_then_abstains(tmp_path):
    bad = Draft(
        claims=[
            Claim(
                text="An unsupported answer.",
                evidence=[
                    Evidence(source_id="S1", quote="A fabricated source quotation")
                ],
            )
        ]
    )
    hits = [
        Hit(Chunk("a", "The actual source is a different sentence.", {"page": 1}), 0.1)
    ]
    generator = Generator([bad, bad])
    service = Service(Settings(data_dir=tmp_path), object(), Store(hits), generator)
    result = service.query("a question")
    assert result["abstain"] and not result["claims"] and generator.calls == 2
