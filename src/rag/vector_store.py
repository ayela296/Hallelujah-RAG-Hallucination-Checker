import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict

from src.rag.embeddings import EMBEDDING_MODEL, GeminiEmbedder


logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    text: str
    score: float
    metadata: dict[str, object]


class _Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class FaissVectorStore:
    """Persist normalized Gemini vectors and their source chunks in FAISS."""

    def __init__(self, directory: str | Path = "vector_db") -> None:
        self.directory = Path(directory)
        self.index_path = self.directory / "index.faiss"
        self.metadata_path = self.directory / "metadata.json"
        self._index = None
        self._records: list[dict[str, object]] = []
        self._load()

    @property
    def count(self) -> int:
        return len(self._records)

    def add_chunks(
        self, chunks: Iterable[Mapping[str, object]], embedder: _Embedder
    ) -> int:
        """Embed and store only new, non-empty chunks; return the number added."""
        existing_ids = {str(record["chunk_id"]) for record in self._records}
        pending: list[tuple[str, dict[str, object], str]] = []

        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise TypeError("each chunk must be a mapping")
            text = chunk.get("text")
            if not isinstance(text, str):
                raise ValueError("each chunk must include text as a string")
            if not text.strip():
                logger.warning("Skipping empty chunk")
                continue

            metadata = {key: value for key, value in chunk.items() if key != "text"}
            chunk_id = metadata.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                chunk_id = self._make_chunk_id(text, metadata)
                metadata["chunk_id"] = chunk_id

            try:
                json.dumps(metadata, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise TypeError("chunk metadata must be JSON serializable") from error

            if chunk_id in existing_ids:
                logger.info("Skipping already indexed chunk %s", chunk_id)
                continue
            existing_ids.add(chunk_id)
            pending.append((text, metadata, chunk_id))

        if not pending:
            return 0

        unique_texts: list[str] = []
        text_positions: dict[str, int] = {}
        for text, _, _ in pending:
            if text not in text_positions:
                text_positions[text] = len(unique_texts)
                unique_texts.append(text)

        unique_vectors = embedder.embed_documents(unique_texts)
        if len(unique_vectors) != len(unique_texts):
            raise RuntimeError("The embedder returned a mismatched vector count.")
        vectors = [unique_vectors[text_positions[text]] for text, _, _ in pending]
        self._add_vectors(pending, vectors)
        self.save()
        return len(pending)

    def search(
        self,
        query: str,
        embedder: _Embedder | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return the nearest chunks by cosine similarity, with their metadata."""
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self._index is None or not self._records:
            raise ValueError("The vector index is empty; index documents first.")

        if embedder is None:
            embedder = GeminiEmbedder()
        vector = embedder.embed_query(query)
        import faiss
        import numpy as np

        query_vector = np.asarray([vector], dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[1] != self._index.d:
            raise ValueError("The query embedding dimension does not match the index.")
        if not np.isfinite(query_vector).all():
            raise ValueError("The query embedding contains non-finite values.")
        faiss.normalize_L2(query_vector)
        if not np.linalg.norm(query_vector):
            raise ValueError("The query embedding must not be a zero vector.")

        similarities, indices = self._index.search(
            query_vector, min(top_k, len(self._records))
        )
        results: list[SearchResult] = []
        for score, index_id in zip(similarities[0], indices[0]):
            if index_id < 0:
                continue
            record = self._records[int(index_id)]
            record_metadata = record.get("metadata")
            if not isinstance(record_metadata, dict):
                raise RuntimeError("Stored chunk metadata is invalid.")
            results.append(
                {
                    "text": str(record["text"]),
                    "score": float(score),
                    "metadata": dict(record_metadata),
                }
            )
        return results

    def save(self) -> None:
        if self._index is None:
            return

        import faiss

        self.directory.mkdir(parents=True, exist_ok=True)
        temporary_index = self.index_path.with_suffix(".faiss.tmp")
        temporary_metadata = self.metadata_path.with_suffix(".json.tmp")
        try:
            faiss.write_index(self._index, str(temporary_index))
            with temporary_metadata.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "model": EMBEDDING_MODEL,
                        "dimension": self._index.d,
                        "records": self._records,
                    },
                    file,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            os.replace(temporary_index, self.index_path)
            os.replace(temporary_metadata, self.metadata_path)
        finally:
            temporary_index.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)

    def _add_vectors(
        self,
        pending: Sequence[tuple[str, dict[str, object], str]],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        import faiss
        import numpy as np

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(pending):
            raise ValueError("Embeddings must be a two-dimensional vector matrix.")
        if not matrix.shape[1]:
            raise ValueError("Embeddings must not be empty vectors.")
        if not np.isfinite(matrix).all():
            raise ValueError("Embeddings contain non-finite values.")

        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0):
            raise ValueError("Embeddings must not be zero vectors.")
        faiss.normalize_L2(matrix)

        if self._index is None:
            self._index = faiss.IndexFlatIP(matrix.shape[1])
        elif self._index.d != matrix.shape[1]:
            raise ValueError("Embedding dimension does not match the existing index.")

        self._index.add(matrix)
        self._records.extend(
            {"text": text, "metadata": metadata, "chunk_id": chunk_id}
            for text, metadata, chunk_id in pending
        )

    def _load(self) -> None:
        index_exists = self.index_path.exists()
        metadata_exists = self.metadata_path.exists()
        if index_exists != metadata_exists:
            raise ValueError(
                "The FAISS index and metadata files are incomplete or out of sync."
            )
        if not index_exists:
            return

        import faiss

        with self.metadata_path.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        if saved.get("model") != EMBEDDING_MODEL:
            raise ValueError(
                f"Stored vectors were created with an unsupported model: "
                f"{saved.get('model')!r}."
            )
        self._index = faiss.read_index(str(self.index_path))
        self._records = saved.get("records", [])
        if self._index.ntotal != len(self._records):
            raise ValueError("FAISS vector count does not match stored metadata.")
        if self._index.d != saved.get("dimension"):
            raise ValueError("FAISS dimension does not match stored metadata.")
        if self._index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("FAISS index must use inner-product similarity.")

    @staticmethod
    def _make_chunk_id(text: str, metadata: Mapping[str, object]) -> str:
        identity = json.dumps(
            {"text": text, "metadata": metadata},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        import hashlib

        return hashlib.sha256(identity.encode("utf-8")).hexdigest()
