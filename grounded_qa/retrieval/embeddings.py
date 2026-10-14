"""E5 prefixes and input-limit checks prevent silent truncation."""

import re


def needs_e5_prefixes(model: str) -> bool:
    """True for E5-family names (e5-small-v2, multilingual-e5-base), not any 'e5' substring."""
    return re.search(r"(^|[/_-])e5($|[_-])", model.lower()) is not None


class LocalEmbeddings:
    def __init__(self, model, revision, threads=4, prefixes=None):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        self.model = SentenceTransformer(
            model, revision=revision, device="cpu", trust_remote_code=False
        )
        self.tokenizer = self.model.tokenizer
        self.prefixes = needs_e5_prefixes(model) if prefixes is None else prefixes

    def _encode(self, texts, prefix):
        values = [(prefix + t) if self.prefixes else t for t in texts]
        if any(
            len(self.tokenizer.encode(t)) > self.model.max_seq_length for t in values
        ):
            raise ValueError(
                "Embedding input exceeds model token limit; reduce chunk or question length"
            )
        return self.model.encode(
            values, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def passages(self, texts):
        return self._encode(texts, "passage: ")

    def queries(self, texts):
        return self._encode(texts, "query: ")
