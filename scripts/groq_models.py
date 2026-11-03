import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# 1 level up from scripts/ is the project root
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    models = response.json().get("data", [])
    print(f"Total Models Available: {len(models)}\n")
    for m in sorted(models, key=lambda x: x["id"]):
        print(f"• {m['id']}")
else:
    print(f"Error {response.status_code}: {response.text}")
