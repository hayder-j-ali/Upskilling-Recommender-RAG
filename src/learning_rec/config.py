"""Central configuration. All paths and model choices flow through here."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", PROJECT_ROOT / "vector_store"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output"))

EMPLOYEES_FILE = Path(os.getenv("EMPLOYEES_FILE", DATA_DIR / "employees.csv"))
CONTENT_FILE = Path(os.getenv("CONTENT_FILE", DATA_DIR / "learning_content.csv"))

# Defaults target the Gemini API (a GOOGLE_API_KEY, not an OpenAI key — see
# .env.example). Verified live against the ListModels endpoint, not just
# docs — Google's model ids and aliases churn faster than most providers'.
# `gemini-flash-latest` is an alias Google repoints at their current
# recommended flash model, trading a small amount of run-to-run eval
# reproducibility for not going stale the way a pinned version id would.
# Pin CHAT_MODEL to a dated version instead if you need exact repeatability.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-flash-latest")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_K = int(os.getenv("TOP_K", "10"))
NUM_RECOMMENDATIONS = int(os.getenv("NUM_RECOMMENDATIONS", "5"))
