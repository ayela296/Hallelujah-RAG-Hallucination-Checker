import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

url = "https://generativelanguage.googleapis.com/v1beta/interactions"

headers = {
    "x-goog-api-key": api_key,
    "Content-Type": "application/json"
}

payload = {
    "model": "gemini-3.8-flash",
    "input": "Say hello, how are you and confirm this API connection is working."
}

response = requests.post(url, headers=headers, json=payload)
print(response.status_code)
print(response.json())