"""BM25 keyword index for hybrid retrieval alongside cosine embeddings.

Uses reciprocal rank fusion (RRF) to merge BM25 and cosine ranked lists.
The BM25 index is rebuilt in memory at search time from the stored chunk texts;
for corpora of this size (~200 chunks) this is fast and avoids a second
persistence layer.
"""

from __future__ import annotations

from grounded_qa.models import Chunk, Hit


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace-split tokeniser — fast, no extra dependencies."""
    return text.lower().split()


def rrf_fuse(
    ranked_lists: list[list[str]],
    all_hits: dict[str, Hit],
    k: int = 60,
    top_n: int = 5,
) -> list[Hit]:
    """Reciprocal Rank Fusion over multiple ranked ID lists.

    k=60 is the standard RRF constant (Cormack et al. 2009).
    Returns the top_n hits sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_n]
    return [all_hits[cid] for cid in fused if cid in all_hits]


class BM25Index:
    """In-memory BM25 index over a fixed set of chunks.

    Rebuilt from the vector store's stored texts so there is no secondary
    persistence and no sync issue between the two indexes.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "rank-bm25 is required for hybrid search: pip install rank-bm25"
            ) from exc

        self._chunks = chunks
        if not chunks:
            # rank_bm25 divides by corpus size; an empty corpus is degenerate.
            self._bm25 = None
            return
        corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Return up to k hits ranked by BM25 score (descending)."""
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Hit(self._chunks[i], float(1.0 - min(scores[i], 1.0)))
            for i in ranked
            if scores[i] > 0
        ]

    @property
    def chunk_ids(self) -> list[str]:
        return [c.id for c in self._chunks]


def hybrid_search(
    question: str,
    cosine_hits: list[Hit],
    bm25_index: BM25Index,
    top_k: int = 5,
) -> list[Hit]:
    """Fuse cosine and BM25 results via RRF.

    cosine_hits: already-ranked list from the vector store.
    bm25_index: BM25 index over the same chunk set.
    Returns top_k fused hits.
    """
    bm25_hits = bm25_index.search(question, k=top_k * 2)

    # Build a lookup of all hits from both sources by chunk ID
    all_hits: dict[str, Hit] = {}
    for hit in cosine_hits:
        all_hits[hit.chunk.id] = hit
    for hit in bm25_hits:
        # If a chunk appears in both, keep the cosine hit (has real distance)
        if hit.chunk.id not in all_hits:
            all_hits[hit.chunk.id] = hit

    cosine_ranked = [h.chunk.id for h in cosine_hits]
    bm25_ranked = [h.chunk.id for h in bm25_hits]

    return rrf_fuse([cosine_ranked, bm25_ranked], all_hits, top_n=top_k)
