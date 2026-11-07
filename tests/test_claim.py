import json
import re
from types import SimpleNamespace

import pytest

from src.agents import claim_extractor
from src.rag.claims import extract_and_store_claims


def mock_gemini(monkeypatch, response_text):
    captured = {}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured["contents"] = contents
            return SimpleNamespace(text=response_text)

    class FakeClient:
        def __init__(self, *, api_key):
            captured["api_key"] = api_key
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(claim_extractor.genai, "Client", FakeClient)
    return captured


@pytest.mark.parametrize(
    ("answer", "expected_claims"),
    [
        (
            "Pakistan gained independence in 1947. Islamabad became the capital "
            "of Pakistan in 1967. Pakistan's national language is Urdu.",
            [
                "Pakistan gained independence in 1947.",
                "Islamabad became the capital of Pakistan in 1967.",
                "Urdu is the national language of Pakistan.",
            ],
        ),
        (
            "The company was founded in 2010 and is headquartered in London, "
            "while its main manufacturing facility is located in Manchester.",
            [
                "The company was founded in 2010.",
                "The company is headquartered in London.",
                "The company's main manufacturing facility is located in Manchester.",
            ],
        ),
        (
            "Python was released in 1991. It is the best programming language "
            "for beginners. Python supports object-oriented programming.",
            [
                "Python was released in 1991.",
                "Python supports object-oriented programming.",
            ],
        ),
    ],
    ids=["multi-fact", "compound-sentence", "opinion-excluded"],
)
def test_extracts_expected_claims_from_mocked_gemini(
    monkeypatch, answer, expected_claims
):
    captured = mock_gemini(monkeypatch, json.dumps({"claims": expected_claims}))

    result = claim_extractor.extract_claims(answer)

    assert answer in captured["contents"]
    assert captured["api_key"] == "test-key"
    assert result == {"claims": expected_claims}


@pytest.mark.parametrize(
    ("response_text", "error_message"),
    [
        ("not JSON", "invalid JSON"),
        ("", "empty"),
        ('{"facts":[]}', "invalid claims"),
        ('{"claims":"not a list"}', "invalid claims"),
        ('{"claims":[42]}', "not text"),
        ('["unexpected response shape"]', "invalid claims"),
    ],
    ids=[
        "invalid-json",
        "empty-response",
        "missing-claims-field",
        "claims-not-a-list",
        "claim-not-text",
        "unexpected-response-shape",
    ],
)
def test_malformed_gemini_response_is_rejected_without_storing_claims(
    monkeypatch, response_text, error_message
):
    mock_gemini(monkeypatch, response_text)
    shared_state = {
        "claims": [{"claim_id": "stale"}],
        "claims_source_answer": "Previous answer",
    }

    with pytest.raises(ValueError, match=error_message):
        extract_and_store_claims("New answer.", shared_state)

    assert shared_state["claims"] == []
    assert "claims_source_answer" not in shared_state


def test_empty_gemini_claim_list_stores_no_records(monkeypatch):
    mock_gemini(monkeypatch, '{"claims":[]}')
    shared_state = {}

    records = extract_and_store_claims(
        "An answer with no factual claims.", shared_state
    )

    assert records == []
    assert shared_state["claims"] == []
    assert shared_state["claims_source_answer"] == "An answer with no factual claims."


def test_mocked_gemini_claims_are_stored_as_unique_shared_records(monkeypatch):
    expected_claims = ["First factual claim.", "Second factual claim."]
    mock_gemini(monkeypatch, json.dumps({"claims": expected_claims}))
    shared_state = {}

    records = extract_and_store_claims("An answer with two facts.", shared_state)

    assert shared_state["claims"] == records
    assert [record["claim_text"] for record in records] == expected_claims
    assert len({record["claim_id"] for record in records}) == len(records)
    assert all(
        re.fullmatch(r"claim_[0-9a-f]{32}", record["claim_id"])
        for record in records
    )
    assert all(record["source"] == "sprint_1" for record in records)
    assert all(record["status"] == "pending" for record in records)
