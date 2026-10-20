from grounded_qa.config import Settings
from grounded_qa.models import Chunk, Claim, Draft, Evidence, Hit
from grounded_qa.service import Service


class Store:
    def __init__(self, hits):
        self.hits = hits

    def search(self, *args, **kwargs):
        return self.hits

    def all_chunks(self, document_ids=None):
        return [h.chunk for h in self.hits]


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


def _claim(text, quote):
    return Draft(
        claims=[Claim(text=text, evidence=[Evidence(source_id="S1", quote=quote)])]
    )


HITS = [
    Hit(
        Chunk(
            "a",
            "MEASURE 2.5: The AI system to be deployed is demonstrated valid.",
            {"page": 1},
        ),
        0.1,
    )
]


def test_status_answered_abstained_rejected(tmp_path):
    def run(drafts):
        generator = Generator(drafts)
        service = Service(Settings(data_dir=tmp_path), object(), Store(HITS), generator)
        return service.query("a question"), generator

    good = _claim("MEASURE 2.5 holds.", "MEASURE 2.5: The AI system to be deployed")
    result, generator = run([good])
    assert result["status"] == "answered" and generator.calls == 1

    result, _ = run([Draft()])  # model returned no claims
    assert (
        result["status"] == "abstained" and result["reason"] == "Insufficient evidence"
    )

    wrong_id = _claim("MANAGE 2.5 holds.", "MEASURE 2.5: The AI system to be deployed")
    result, generator = run([wrong_id, wrong_id])
    # Every claim failed validation -> rejected (not "abstained": the model did
    # return claims, they were just not provable).
    assert result["status"] == "rejected" and generator.calls == 2
    assert [a["error"] != "" for a in result["attempts"]] == [True, True]


def test_retry_can_rescue_and_is_logged(tmp_path):
    bad = _claim("Wrong.", "A fabricated source quotation here")
    good = _claim("MEASURE 2.5 holds.", "MEASURE 2.5: The AI system to be deployed")
    service = Service(
        Settings(data_dir=tmp_path), object(), Store(HITS), Generator([bad, good])
    )
    result = service.query("a question")
    assert result["status"] == "answered"
    assert [bool(a["error"]) for a in result["attempts"]] == [True, False]
