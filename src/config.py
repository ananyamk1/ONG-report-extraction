"""
Configuration and environment settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
PDF_DIR = ROOT_DIR / os.getenv("PDF_DIR", "data/raw_pdfs")
DB_PATH = ROOT_DIR / os.getenv("DB_PATH", "data/db/drilling.db")
VECTORSTORE_DIR = ROOT_DIR / os.getenv("VECTORSTORE_DIR", "vectorstore")

PDF_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ── Dataset ───────────────────────────────────────────────────────────────────
# Utah FORGE Well 78B-32 Daily Drilling Reports (DOE/OSTI, CC-BY 4.0)
# https://gdr.openei.org/submissions/1330
DATASET_URL = os.getenv("DATASET_URL", "https://gdr.openei.org/submissions/1330")
