import os
import json
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

def extract_claims(answer_text: str) -> dict:
    """Extracts atomic factual claims from an answer text using Gemini with structured JSON output."""
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

    try:
        # Parse the JSON response safely
        raw_text = response.text.strip()
        # Handle potential markdown code block wrappers if any
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        data = json.loads(raw_text.strip())
        
        # Ensure the 'claims' key exists and is a list, otherwise default to empty list
        if not isinstance(data, dict) or "claims" not in data or not isinstance(data["claims"], list):
            return {"claims": []}
            
        return data
    except (json.JSONDecodeError, AttributeError):
        # Handle bad output / parsing errors gracefully
        return {"claims": []}