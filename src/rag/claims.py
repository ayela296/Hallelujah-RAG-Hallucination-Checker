from collections.abc import Callable, MutableMapping, Sequence
from typing import TypedDict
from uuid import uuid4

from src.agents.claim_extractor import ClaimExtraction, extract_claims


class ClaimRecord(TypedDict):
    claim_id: str
    claim_text: str
    source: str
    status: str


ClaimExtractor = Callable[[str], ClaimExtraction]


def extract_and_store_claims(
    answer_text: str,
    shared_state: MutableMapping[str, object],
    extractor: ClaimExtractor = extract_claims,
) -> list[ClaimRecord]:
    """Extract claims once per answer and store them for the retrieval stage."""
    if not isinstance(answer_text, str):
        raise TypeError("answer_text must be a string")
    if not answer_text.strip():
        raise ValueError("The submitted answer is empty.")

    claims = shared_state.get("claims")
    if (
        shared_state.get("claims_source_answer") == answer_text
        and isinstance(claims, list)
    ):
        return claims

    shared_state["claims"] = []
    shared_state.pop("claims_source_answer", None)
    extraction = extractor(answer_text)
    claim_texts = extraction.get("claims")
    if not isinstance(claim_texts, Sequence) or isinstance(
        claim_texts, (str, bytes)
    ):
        raise ValueError("Claim extraction returned an invalid claims list.")
    if any(not isinstance(claim, str) or not claim.strip() for claim in claim_texts):
        raise ValueError("Claim extraction returned an empty or invalid claim.")

    records: list[ClaimRecord] = [
        {
            "claim_id": f"claim_{uuid4().hex}",
            "claim_text": claim_text,
            "source": "sprint_1",
            "status": "pending",
        }
        for claim_text in claim_texts
    ]
    shared_state["claims"] = records
    shared_state["claims_source_answer"] = answer_text
    return records
