import numpy as np

from app.core.embeddings import HashingEmbedder


def test_embeddings_are_normalised():
    emb = HashingEmbedder(dim=256)
    vecs = emb.embed(["hello world", "machine learning is fun"])
    assert vecs.shape == (2, 256)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_similar_text_scores_higher_than_unrelated():
    emb = HashingEmbedder(dim=512)
    query = emb.embed(["annual leave policy for employees"])[0]
    related = emb.embed(["employees get annual leave days every year"])[0]
    unrelated = emb.embed(["the spaceship launched into orbit at dawn"])[0]
    assert float(query @ related) > float(query @ unrelated)


def test_empty_input_returns_empty_matrix():
    emb = HashingEmbedder(dim=128)
    vecs = emb.embed([])
    assert vecs.shape == (0, 128)


def test_long_text_with_sign_collisions_does_not_crash():
    # A small dimension forces many bucket collisions, some of which cancel to
    # zero. This used to raise "math domain error" via math.log(0).
    emb = HashingEmbedder(dim=16)
    long_text = " ".join(f"word{i}" for i in range(500))
    vecs = emb.embed([long_text])
    assert vecs.shape == (1, 16)
    assert np.isfinite(vecs).all()
