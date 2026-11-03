import math
import re
from collections.abc import Iterable, Mapping
from typing import TypedDict


class RetrievedPassage(TypedDict):
    text: str
    source: str
    chunk_index: int
    relevance_score: float


class ClaimEvidence(TypedDict):
    claim: str
    evidence: list[RetrievedPassage]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[\w'-]+\b", text.lower()))


def retrieve_evidence(
    claims: Iterable[str],
    chunks: Iterable[Mapping[str, object]],
    top_k: int = 3,
) -> list[ClaimEvidence]:
    """Return each claim with its highest-scoring source passages."""
    if isinstance(claims, (str, bytes, Mapping)) or not isinstance(claims, Iterable):
        raise TypeError("claims must be an iterable of strings")
    if isinstance(chunks, (str, bytes, Mapping)) or not isinstance(chunks, Iterable):
        raise TypeError("chunks must be an iterable of chunk mappings")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    claim_list = list(claims)
    if any(not isinstance(claim, str) for claim in claim_list):
        raise TypeError("each claim must be a string")

    chunk_list = list(chunks)
    for chunk in chunk_list:
        if not isinstance(chunk, Mapping):
            raise TypeError("each chunk must be a mapping")
        if not isinstance(chunk.get("text"), str):
            raise ValueError("each chunk must include text as a string")
        if not isinstance(chunk.get("source"), str):
            raise ValueError("each chunk must include source as a string")
        chunk_index = chunk.get("chunk_index")
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
            raise ValueError("each chunk must include chunk_index as an integer")

    tokenized_chunks = [_tokenize(chunk["text"]) for chunk in chunk_list]
    document_frequency: dict[str, int] = {}
    for tokens in tokenized_chunks:
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    document_count = len(chunk_list)
    results: list[ClaimEvidence] = []
    for claim in claim_list:
        claim_tokens = _tokenize(claim)
        evidence: list[RetrievedPassage] = []
        if claim_tokens and document_count:
            token_weights = {
                token: math.log(
                    (document_count + 1) / (document_frequency.get(token, 0) + 1)
                )
                + 1
                for token in claim_tokens
            }
            total_weight = sum(token_weights.values())

            for chunk, passage_tokens in zip(chunk_list, tokenized_chunks):
                matched_weight = sum(
                    weight
                    for token, weight in token_weights.items()
                    if token in passage_tokens
                )
                relevance_score = matched_weight / total_weight
                if relevance_score > 0:
                    evidence.append(
                        {
                            "text": chunk["text"],
                            "source": chunk["source"],
                            "chunk_index": chunk["chunk_index"],
                            "relevance_score": relevance_score,
                        }
                    )

        evidence.sort(
            key=lambda passage: (
                -passage["relevance_score"],
                passage["source"],
                passage["chunk_index"],
            )
        )
        results.append({"claim": claim, "evidence": evidence[:top_k]})

    return results
