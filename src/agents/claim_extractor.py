import json
import os
from google import genai
from google.genai import types

client = genai.Client()

EXTRACTION_SYSTEM_INSTRUCTION = """
You are a precise factual claim extraction agent for a RAG hallucination checker. 
Your task is to break down the provided text answer into short, atomic (single-fact) statements.

Follow these strict rules:
1. Decompose compound sentences: If a sentence contains multiple facts (e.g. joined by 'and' or clauses), split them into separate individual claims.
2. Resolve pronouns: Replace ambiguous pronouns (e.g., 'it', 'he', 'she', 'they') with the actual subject name based on context.
3. Filter non-factual content: Skip opinions, subjective evaluations, filler words, and conversational fluff (e.g., "This is a great city", "I think", "In conclusion").
4. Maintain factual integrity: Do not alter the meaning of the facts.
"""

def extract_claims(answer_text: str) -> dict:
    """Extracts atomic factual claims from an answer text using Gemini with structured JSON output."""
    prompt = f"""
    Analyze the following answer and extract all individual factual claims according to the system instructions.
    
    Answer to analyze:
    "{answer_text}"
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "claims": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "id": {"type": "INTEGER"},
                                "text": {"type": "STRING"}
                            },
                            "required": ["id", "text"]
                        }
                    }
                },
                "required": ["claims"]
            },
            temperature=0.0
        ),
    )
    
    return json.loads(response.text)
