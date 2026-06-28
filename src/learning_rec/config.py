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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_K = int(os.getenv("TOP_K", "10"))
NUM_RECOMMENDATIONS = int(os.getenv("NUM_RECOMMENDATIONS", "5"))
