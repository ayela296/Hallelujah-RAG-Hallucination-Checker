import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import NotRequired, TypedDict


class TextChunk(TypedDict):
    text: str
    source: str
    chunk_index: int
    chunk_id: str
    metadata: NotRequired[dict[str, object]]


def _chunk_id(
    text: str, source: str, chunk_index: int, metadata: Mapping[str, object]
) -> str:
    identity = json.dumps(
        {
            "text": text,
            "source": source,
            "chunk_index": chunk_index,
            "metadata": metadata,
        },
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _validate_chunk_parameters(chunk_size: int, overlap: int) -> None:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise TypeError("overlap must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")


def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 250,
    overlap: int = 40,
) -> list[TextChunk]:
    """Split document text into word-based passages with source metadata."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    _validate_chunk_parameters(chunk_size, overlap)

    words = text.split()
    chunks: list[TextChunk] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(
            {
                "text": " ".join(words[start:end]),
                "source": source,
                "chunk_index": len(chunks),
                "chunk_id": _chunk_id(
                    " ".join(words[start:end]), source, len(chunks), {}
                ),
            }
        )
        if end == len(words):
            break
        start = end - overlap

    return chunks


def chunk_documents(
    documents: Iterable[Mapping[str, object]],
    chunk_size: int = 250,
    overlap: int = 40,
) -> list[TextChunk]:
    """Chunk each named source document independently."""
    _validate_chunk_parameters(chunk_size, overlap)
    if not isinstance(documents, Iterable) or isinstance(
        documents, (str, bytes, Mapping)
    ):
        raise TypeError("documents must be an iterable of document mappings")

    chunks: list[TextChunk] = []
    for document in documents:
        if not isinstance(document, Mapping):
            raise TypeError("each document must be a mapping with name and content")
        if "name" not in document or "content" not in document:
            raise ValueError("each document must include name and content")
        if not isinstance(document["name"], str):
            raise TypeError("document name must be a string")
        if not isinstance(document["content"], str):
            raise TypeError("document content must be a string")

        supplied_metadata = document.get("metadata", {})
        if not isinstance(supplied_metadata, Mapping):
            raise TypeError("document metadata must be a mapping")
        document_metadata = dict(supplied_metadata)
        document_metadata.update(
            {
                key: value
                for key, value in document.items()
                if key not in {"name", "content", "metadata"}
            }
        )
        try:
            json.dumps(
                document_metadata,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise TypeError("document metadata must be JSON serializable") from error

        document_chunks = chunk_document(
            text=document["content"],
            source=document["name"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk in document_chunks:
            if document_metadata:
                chunk["metadata"] = document_metadata
            chunk["chunk_id"] = _chunk_id(
                chunk["text"],
                chunk["source"],
                chunk["chunk_index"],
                document_metadata,
            )
        chunks.extend(document_chunks)
    return chunks