import os
import json
from typing import TypedDict
from dotenv import load_dotenv
from google import genai
from google.genai import types

EXTRACTION_SYSTEM_INSTRUCTION = """
You are an expert factual analysis assistant. Your job is to extract atomic, factual claims from a provided text.

Follow these strict rules:
1. Decompose compound sentences: If a sentence contains multiple facts, split them into separate individual claims.
2. Resolve pronouns: Replace ambiguous pronouns (e.g., 'it', 'he', 'she', 'they') with the actual subject name based on context.
3. Filter non-factual content: Skip opinions, subjective evaluations, filler words, and conversational fluff.
4. Maintain factual integrity: Do not alter the meaning of the facts.
"""


class ClaimExtraction(TypedDict):
    claims: list[str]


def extract_claims(answer_text: str) -> ClaimExtraction:
    """Extracts atomic factual claims from an answer text using Gemini with structured JSON output."""
    if not isinstance(answer_text, str):
        raise TypeError("answer_text must be a string")
    if not answer_text.strip():
        raise ValueError("answer_text must not be empty")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing API Key! Please ensure GEMINI_API_KEY is set in your environment.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    Analyze the following answer and extract all individual factual claims according to the system instructions.

    Answer to analyze:
    "{answer_text}"
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "claims": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    }
                },
                "required": ["claims"]
            }
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty claim-extraction response.")
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise ValueError("Gemini returned invalid JSON for claim extraction.") from error

    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise ValueError("Gemini returned an invalid claims response.")
    if any(not isinstance(claim, str) for claim in data["claims"]):
        raise ValueError("Gemini returned a claim that is not text.")
    return data