import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHARTS_DIR = DATA_DIR / "charts"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Optional: comma-separated list of additional Groq API keys (e.g. from separate free-tier
# accounts). Calls round-robin across all configured keys and fail over to the next key when
# one is rate-limited or its model is unavailable - a simple way to multiply daily quota and
# concurrent throughput without any other infra. GROQ_API_KEY is always included first if set.
_extra_keys = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
GROQ_API_KEYS = ([GROQ_API_KEY] if GROQ_API_KEY else []) + [k for k in _extra_keys if k != GROQ_API_KEY]

GROQ_TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Pages are I/O-bound (waiting on Groq API calls), so a thread pool here is a big win -
# the GIL is released while a thread is blocked on network I/O. Tune down if you're hitting
# Groq rate limits, up if you have plenty of headroom.
INGESTION_WORKERS = int(os.environ.get("INGESTION_WORKERS", "6"))

DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", "data/fact_layer.db"))
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = BASE_DIR / DATABASE_PATH

CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", "data/chroma"))
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = BASE_DIR / CHROMA_DIR

for d in (UPLOADS_DIR, CHARTS_DIR, CHROMA_DIR, DATABASE_PATH.parent):
    d.mkdir(parents=True, exist_ok=True)
