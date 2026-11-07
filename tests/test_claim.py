import json
from src.agents.claim_extractor import extract_claims

def test_claim_extraction():
    sample_answer = (
        "Python was created by Guido van Rossum in 1991, and it supports multiple programming paradigms. "
        "It is widely used for data science and web development. "
        "Python is an amazing language that everyone should learn."
    )
    
    result = extract_claims(sample_answer)
    print("Extracted Claims:")
    print(json.dumps(result, indent=2))
    
    claims = result.get("claims", [])
    assert len(claims) >= 3, "Failed: Did not extract expected number of claims."
    print("Test passed successfully!")

if __name__ == "__main__":
    test_claim_extraction()