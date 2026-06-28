"""CLI: build the FAISS vector store from the learning-content catalogue.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from learning_rec.config import CONTENT_FILE, INDEX_DIR  # noqa: E402
from learning_rec.ingest import build_vector_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing index before rebuilding.",
    )
    parser.add_argument(
        "--content",
        type=Path,
        default=CONTENT_FILE,
        help=f"Path to content catalogue (default: {CONTENT_FILE}).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
        help=f"Where to write the FAISS index (default: {INDEX_DIR}).",
    )
    args = parser.parse_args()

    build_vector_store(
        content_path=args.content,
        index_dir=args.index_dir,
        reset=args.reset,
    )
    print(f"Vector store written to {args.index_dir}")


if __name__ == "__main__":
    main()
