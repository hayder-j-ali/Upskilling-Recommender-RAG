"""Provenance metadata for a built FAISS index.

The index stores vectors but not the identity of the model that produced
them, and nothing downstream re-derives it. Swapping `EMBEDDING_MODEL`
without rebuilding therefore compares query vectors from one model against
document vectors from another.

That failure is silent here, which is what makes it worth guarding. Every
Gemini embedding model this project can use emits 3072 dimensions, so the
shapes still line up and FAISS returns a confident, wrongly-ordered result
instead of raising. A dimension mismatch would at least crash; identical
dimensions degrade quietly, and the recommendations still look plausible.

So the model name is recorded at build time and checked at load time.
"""

from __future__ import annotations

import json
from pathlib import Path

META_FILENAME = "index_meta.json"


class IndexMetadataError(RuntimeError):
    """The index cannot be trusted for the currently configured model."""


def meta_path(index_dir: Path) -> Path:
    return Path(index_dir) / META_FILENAME


def write(index_dir: Path, *, embedding_model: str, n_documents: int) -> None:
    """Record which model built this index. Called after the index is saved."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path(index_dir).write_text(
        json.dumps(
            {"embedding_model": embedding_model, "n_documents": n_documents},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def read(index_dir: Path) -> dict | None:
    """Return the recorded metadata, or None if absent/unreadable."""
    path = meta_path(index_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def verify(index_dir: Path, embedding_model: str) -> None:
    """Raise unless the index was built with `embedding_model`.

    Missing metadata is treated as a failure rather than waved through: an
    index of unknown provenance is exactly the case this guard exists for,
    and rebuilding costs well under a thousand tokens on the demo dataset.
    """
    rebuild = "Rebuild it with:  python scripts/build_index.py --reset"
    data = read(index_dir)

    if data is None:
        raise IndexMetadataError(
            f"The index at {index_dir} has no {META_FILENAME}, so the model "
            f"that built it is unknown and its vectors cannot be trusted "
            f"against the configured EMBEDDING_MODEL ({embedding_model}). "
            f"{rebuild}"
        )

    built_with = data.get("embedding_model")
    if built_with != embedding_model:
        raise IndexMetadataError(
            f"The index at {index_dir} was built with {built_with!r} but "
            f"EMBEDDING_MODEL is {embedding_model!r}. Querying across models "
            f"returns confidently wrong results rather than an error, "
            f"because these models share an output dimensionality. {rebuild}"
        )
