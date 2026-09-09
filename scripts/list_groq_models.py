"""Quick utility: list the chat models this Groq API key can actually access.

Useful because Groq's public model catalog isn't always what a given key/tier
can call - `GROQ_TEXT_MODEL`/`GROQ_VISION_MODEL` in .env should be picked from
this list, not assumed from documentation.

Usage: python scripts/list_groq_models.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from groq import Groq  # noqa: E402

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("GROQ_API_KEY is not set - copy .env.example to .env and fill it in first.")
    raise SystemExit(1)

client = Groq(api_key=api_key)
models = client.models.list()
print(f"{len(models.data)} model(s) available to this key:\n")
for m in sorted(models.data, key=lambda m: m.id):
    print(f"  {m.id}")

print(
    "\nNote: this list doesn't say which models accept image input. If you expect a vision model "
    "(e.g. meta-llama/llama-4-scout-17b-16e-instruct) and don't see it here, check console.groq.com "
    "for preview-model access on your account."
)
