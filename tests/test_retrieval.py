from src.rag.retrieval import retrieve_evidence


# Checks that the top three passages include relevance scores and rank correctly.
def test_retrieves_top_three_relevant_passages_in_rank_order():
    claims = ["Solar panels convert sunlight into electricity"]
    chunks = [
        {
            "text": "Solar panels convert sunlight into electricity.",
            "source": "energy.txt",
            "chunk_index": 0,
        },
        {
            "text": "Solar panels convert sunlight into heat.",
            "source": "energy.txt",
            "chunk_index": 1,
        },
        {
            "text": "Sunlight is used by solar panels.",
            "source": "climate.txt",
            "chunk_index": 0,
        },
        {
            "text": "The ocean contains many kinds of fish.",
            "source": "ocean.txt",
            "chunk_index": 0,
        },
    ]

    result = retrieve_evidence(claims, chunks)
    evidence = result[0]["evidence"]

    assert len(evidence) == 3
    assert evidence[0]["text"] == chunks[0]["text"]
    assert evidence[0]["relevance_score"] > evidence[1]["relevance_score"]
    assert all(0 < passage["relevance_score"] <= 1 for passage in evidence)


# Checks that evidence retains source names and chunk indexes across documents.
def test_preserves_source_names_and_chunk_indexes_from_multiple_documents():
    chunks = [
        {
            "text": "Bees pollinate flowers and support crop production.",
            "source": "pollination.txt",
            "chunk_index": 2,
        },
        {
            "text": "Pollination by bees helps many crops grow.",
            "source": "agriculture.pdf",
            "chunk_index": 4,
        },
    ]

    evidence = retrieve_evidence(["Bees help pollinate crops"], chunks)[0]["evidence"]

    assert {passage["source"] for passage in evidence} == {
        "pollination.txt",
        "agriculture.pdf",
    }
    assert {passage["chunk_index"] for passage in evidence} == {2, 4}


# Checks that a claim unrelated to the supplied chunks returns no evidence.
def test_unrelated_claim_has_empty_evidence():
    chunks = [
        {
            "text": "Bees pollinate flowers and support crop production.",
            "source": "pollination.txt",
            "chunk_index": 0,
        }
    ]

    result = retrieve_evidence(["Jupiter has many moons"], chunks)

    assert result == [{"claim": "Jupiter has many moons", "evidence": []}]


# Checks that retrieval returns the claim with fully attributed evidence.
def test_returns_claim_plus_evidence_structure():
    claim = "Rain clouds produce precipitation"
    chunks = [
        {
            "text": "Rain clouds can produce precipitation.",
            "source": "weather.txt",
            "chunk_index": 1,
        }
    ]

    result = retrieve_evidence([claim], chunks)

    assert result[0]["claim"] == claim
    assert result[0]["evidence"] == [
        {
            "text": chunks[0]["text"],
            "source": "weather.txt",
            "chunk_index": 1,
            "relevance_score": 1.0,
        }
    ]


# Checks that callers can set a custom maximum number of retrieved passages.
def test_accepts_a_custom_result_limit():
    chunks = [
        {"text": "Cats are mammals.", "source": "a.txt", "chunk_index": 0},
        {"text": "Cats are common pets.", "source": "b.txt", "chunk_index": 0},
        {"text": "Cats often sleep.", "source": "c.txt", "chunk_index": 0},
    ]

    result = retrieve_evidence(["Cats are mammals and pets"], chunks, top_k=2)

    assert len(result[0]["evidence"]) == 2
