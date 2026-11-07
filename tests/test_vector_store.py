from types import SimpleNamespace

import pytest

from src.rag.chunking import chunk_documents
from src.rag.embeddings import EMBEDDING_MODEL, GeminiEmbedder
from src.rag.vector_store import FaissVectorStore


class FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents, config):
        self.calls.append((model, list(contents), config.task_type))
        vectors = []
        for text in contents:
            if "bees" in text.lower():
                values = [1.0, 0.0]
            elif "ocean" in text.lower():
                values = [0.0, 1.0]
            else:
                values = [0.8, 0.2]
            vectors.append(SimpleNamespace(values=values))
        return SimpleNamespace(embeddings=vectors)


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_gemini_embedder_uses_model_and_document_query_tasks():
    client = FakeClient()
    embedder = GeminiEmbedder(client=client)

    document_vectors = embedder.embed_documents(["Bees pollinate crops", "Ocean life"])
    query_vector = embedder.embed_query("How do bees help crops?")

    assert len(document_vectors) == 2
    assert query_vector == [1.0, 0.0]
    assert client.models.calls == [
        (EMBEDDING_MODEL, ["Bees pollinate crops", "Ocean life"], "RETRIEVAL_DOCUMENT"),
        (EMBEDDING_MODEL, ["How do bees help crops?"], "RETRIEVAL_QUERY"),
    ]


def test_gemini_embedder_batches_more_than_one_api_request():
    client = FakeClient()
    embedder = GeminiEmbedder(client=client)

    vectors = embedder.embed_documents([f"document {index}" for index in range(101)])

    assert len(vectors) == 101
    assert [len(call[1]) for call in client.models.calls] == [100, 1]


def test_indexes_all_chunks_deduplicates_and_reloads_for_similarity_search(tmp_path):
    documents = [
        {
            "name": "pollination.txt",
            "content": "Bees pollinate flowers and support crops.",
            "metadata": {"category": "agriculture"},
        },
        {
            "name": "ocean.txt",
            "content": "Ocean water contains many kinds of fish.",
        },
    ]
    chunks = chunk_documents(documents)
    client = FakeClient()
    embedder = GeminiEmbedder(client=client)
    store = FaissVectorStore(tmp_path)

    added_count = store.add_chunks(
        [*chunks, {"text": "   ", "source": "empty.txt", "chunk_index": 0}],
        embedder,
    )

    assert added_count == 2
    assert store.count == len(chunks)
    assert store.add_chunks(chunks, embedder) == 0
    assert len(client.models.calls) == 1

    reloaded = FaissVectorStore(tmp_path)
    results = reloaded.search("How do bees help crops?", embedder)

    assert reloaded.count == 2
    assert len(results) == 2
    assert results[0]["text"] == chunks[0]["text"]
    assert results[0]["metadata"]["source"] == "pollination.txt"
    assert results[0]["metadata"]["chunk_index"] == 0
    assert results[0]["metadata"]["metadata"] == {"category": "agriculture"}
    assert results[0]["score"] > results[1]["score"]


def test_identical_text_from_different_sources_is_embedded_once(tmp_path):
    chunks = chunk_documents(
        [
            {"name": "first.txt", "content": "The same source text."},
            {"name": "second.txt", "content": "The same source text."},
        ]
    )
    client = FakeClient()
    store = FaissVectorStore(tmp_path)

    assert store.add_chunks(chunks, GeminiEmbedder(client=client)) == 2
    assert store.count == 2
    assert client.models.calls[0][1] == ["The same source text."]


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        GeminiEmbedder(client=FakeClient()).embed_query(" ")
