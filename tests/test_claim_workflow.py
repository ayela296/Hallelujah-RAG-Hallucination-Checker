from types import SimpleNamespace

import pytest

from src.agents import claim_extractor
from src.rag.claims import extract_and_store_claims


def test_agent_sends_the_stored_answer_to_gemini(monkeypatch):
    answer = "Venus is the second planet from the Sun."
    captured = {}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            return SimpleNamespace(text='{"claims":["Venus is the second planet from the Sun."]}')

    class FakeClient:
        def __init__(self, *, api_key):
            captured["api_key"] = api_key
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(claim_extractor.genai, "Client", FakeClient)

    result = claim_extractor.extract_claims(answer)

    assert answer in captured["contents"]
    assert captured["api_key"] == "test-key"
    assert result["claims"] == ["Venus is the second planet from the Sun."]


def test_extraction_receives_answer_and_stores_each_claim():
    answer = "Venus is the second planet from the Sun. Venus has a dense atmosphere."
    calls = []

    def fake_extractor(answer_text):
        calls.append(answer_text)
        return {
            "claims": [
                "Venus is the second planet from the Sun.",
                "Venus has a dense atmosphere.",
            ]
        }

    shared_state = {}
    claims = extract_and_store_claims(answer, shared_state, fake_extractor)

    assert calls == [answer]
    assert shared_state["claims"] == claims
    assert [claim["claim_text"] for claim in claims] == [
        "Venus is the second planet from the Sun.",
        "Venus has a dense atmosphere.",
    ]
    assert len({claim["claim_id"] for claim in claims}) == 2
    assert all(claim["source"] == "sprint_1" for claim in claims)
    assert all(claim["status"] == "pending" for claim in claims)


def test_repeated_check_for_same_answer_does_not_extract_twice():
    answer = "The Earth orbits the Sun."
    calls = []

    def fake_extractor(answer_text):
        calls.append(answer_text)
        return {"claims": ["The Earth orbits the Sun."]}

    shared_state = {}
    first = extract_and_store_claims(answer, shared_state, fake_extractor)
    second = extract_and_store_claims(answer, shared_state, fake_extractor)

    assert first == second
    assert calls == [answer]


def test_new_answer_replaces_previous_claims():
    shared_state = {}
    extract_and_store_claims(
        "Old answer.",
        shared_state,
        lambda answer: {"claims": ["Old claim."]},
    )

    def extract_new(answer_text):
        assert answer_text == "New answer."
        return {"claims": ["New claim."]}

    claims = extract_and_store_claims("New answer.", shared_state, extract_new)

    assert [claim["claim_text"] for claim in claims] == ["New claim."]
    assert shared_state["claims"] == claims


def test_empty_answer_and_extraction_errors_are_reported():
    shared_state = {"claims": [{"claim_id": "old"}]}

    with pytest.raises(ValueError, match="empty"):
        extract_and_store_claims(" ", shared_state)

    def failed_extractor(answer_text):
        raise RuntimeError("Gemini unavailable")

    with pytest.raises(RuntimeError, match="Gemini unavailable"):
        extract_and_store_claims("New answer.", shared_state, failed_extractor)

    assert shared_state["claims"] == []
    assert "claims_source_answer" not in shared_state


def test_non_string_claims_are_rejected():
    with pytest.raises(ValueError, match="invalid claim"):
        extract_and_store_claims(
            "An answer.",
            {},
            lambda answer: {"claims": ["Valid claim.", 42]},
        )
