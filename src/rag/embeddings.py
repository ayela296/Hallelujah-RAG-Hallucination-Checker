import logging
import os
from collections.abc import Sequence
from typing import Protocol


logger = logging.getLogger(__name__)
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_BATCH_SIZE = 100


class _EmbeddingResponse(Protocol):
    embeddings: Sequence[object]


class _EmbeddingModels(Protocol):
    def embed_content(
        self, *, model: str, contents: Sequence[str], config: object
    ) -> _EmbeddingResponse: ...


class _GenAIClient(Protocol):
    models: _EmbeddingModels


class GeminiEmbedder:
    """Generate document and query embeddings with Google's Gen AI SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        client: _GenAIClient | None = None,
    ) -> None:
        if client is None:
            if api_key is None:
                from dotenv import load_dotenv

                load_dotenv()
            resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not resolved_api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required to generate Gemini embeddings."
                )
            from google import genai

            client = genai.Client(api_key=resolved_api_key)
        self._client = client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed all non-empty document texts, using the document retrieval task."""
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> list[float]:
        """Embed one user query with the matching query retrieval task."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        return self._embed([query], task_type="RETRIEVAL_QUERY")[0]

    def _embed(
        self, texts: Sequence[str], *, task_type: str
    ) -> list[list[float]]:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("all texts to embed must be strings")
        if any(not text.strip() for text in texts):
            raise ValueError("texts to embed must not contain empty strings")
        if not texts:
            return []

        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            logger.info(
                "Requesting %s Gemini embeddings (%s task)",
                len(batch),
                task_type,
            )
            response = self._client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            embeddings = getattr(response, "embeddings", None)
            if embeddings is None:
                raise RuntimeError("Gemini returned a response without embeddings.")
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    "Gemini returned a different number of embeddings than inputs."
                )
            for embedding in embeddings:
                values = getattr(embedding, "values", None)
                if values is None:
                    raise RuntimeError("Gemini returned an embedding without values.")
                vectors.append([float(value) for value in values])
        return vectors
