from app.core.embeddings import HashingEmbedder
from app.core.vector_store import StoredChunk, VectorStore


def _chunk(text: str, index: int) -> StoredChunk:
    return StoredChunk(chunk_id=-1, doc_id="d1", doc_name="doc.txt", chunk_index=index, text=text)


def test_search_returns_most_relevant_chunk_first():
    emb = HashingEmbedder(dim=512)
    texts = [
        "The leave policy grants employees annual paid time off.",
        "Reimbursement for business expenses is processed in thirty days.",
        "Performance reviews happen twice a year in June and December.",
    ]
    store = VectorStore(dim=emb.dim)
    store.add(emb.embed(texts), [_chunk(t, i) for i, t in enumerate(texts)])

    query = emb.embed(["how many annual leave days do employees get"])[0]
    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert "leave policy" in results[0].chunk.text
    assert results[0].score >= results[1].score


def test_empty_store_returns_no_results():
    store = VectorStore(dim=64)
    emb = HashingEmbedder(dim=64)
    assert store.search(emb.embed(["anything"])[0]) == []


def test_chunk_ids_are_assigned_incrementally():
    emb = HashingEmbedder(dim=64)
    store = VectorStore(dim=64)
    texts = ["alpha", "beta", "gamma"]
    store.add(emb.embed(texts), [_chunk(t, i) for i, t in enumerate(texts)])
    assert store.size == 3
    assert sorted(c.chunk_id for c in store._chunks) == [0, 1, 2]
