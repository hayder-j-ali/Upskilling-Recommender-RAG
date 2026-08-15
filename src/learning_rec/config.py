"""Central configuration. All paths and model choices flow through here."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


def shadowed_dotenv_vars() -> list[str]:
    """Names set in .env that a pre-existing environment variable overrides.

    `load_dotenv()` deliberately does not clobber variables already present
    in the environment, which is the right default — CI and production set
    real env vars and should win. The failure mode is that it happens
    silently: a stale `export GOOGLE_API_KEY=...` left in one shell makes
    .env inert for every process launched from it, so edits to .env appear
    to do nothing and the app authenticates with a credential the user has
    already replaced. That cost hours to find once; surfacing it is cheap.

    Precedence is left unchanged — this only reports the divergence so
    callers can say so out loud.
    """
    if not DOTENV_PATH.exists():
        return []
    return [
        name
        for name, file_value in dotenv_values(DOTENV_PATH).items()
        if file_value is not None and os.environ.get(name, file_value) != file_value
    ]
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
