"""Tests for the index provenance guard.

The bug guarded against is specifically a *silent* one. Every Gemini
embedding model this project can use emits 3072 dimensions, so querying an
index built by model A with vectors from model B does not raise — FAISS
compares the shapes happily and returns a confidently mis-ordered list.
The recommendations still look plausible, which is what makes it dangerous.
"""

from __future__ import annotations

import json

import pytest

from learning_rec import index_meta
from learning_rec.index_meta import IndexMetadataError


class TestWriteAndRead:
    def test_round_trip(self, tmp_path):
        index_meta.write(tmp_path, embedding_model="models/x", n_documents=37)
        data = index_meta.read(tmp_path)
        assert data == {"embedding_model": "models/x", "n_documents": 37}

    def test_creates_directory_if_missing(self, tmp_path):
        target = tmp_path / "not-yet-there"
        index_meta.write(target, embedding_model="models/x", n_documents=1)
        assert index_meta.meta_path(target).exists()

    def test_read_returns_none_when_absent(self, tmp_path):
        assert index_meta.read(tmp_path) is None

    def test_read_returns_none_on_corrupt_json(self, tmp_path):
        index_meta.meta_path(tmp_path).write_text("{not json", encoding="utf-8")
        assert index_meta.read(tmp_path) is None

    def test_read_returns_none_when_payload_is_not_an_object(self, tmp_path):
        index_meta.meta_path(tmp_path).write_text("[1, 2]", encoding="utf-8")
        assert index_meta.read(tmp_path) is None


class TestVerify:
    def test_passes_when_model_matches(self, tmp_path):
        index_meta.write(tmp_path, embedding_model="models/same", n_documents=1)
        index_meta.verify(tmp_path, "models/same")  # must not raise

    def test_raises_on_model_mismatch(self, tmp_path):
        index_meta.write(tmp_path, embedding_model="models/built-with", n_documents=1)
        with pytest.raises(IndexMetadataError) as exc:
            index_meta.verify(tmp_path, "models/configured-now")
        msg = str(exc.value)
        assert "models/built-with" in msg
        assert "models/configured-now" in msg
        assert "build_index.py --reset" in msg  # message states the remedy

    def test_raises_when_metadata_is_missing(self, tmp_path):
        """An index of unknown provenance is exactly the risky case, so it
        fails rather than being waved through. Rebuilding is cheap.
        """
        with pytest.raises(IndexMetadataError) as exc:
            index_meta.verify(tmp_path, "models/whatever")
        assert "build_index.py --reset" in str(exc.value)

    def test_raises_when_metadata_is_corrupt(self, tmp_path):
        index_meta.meta_path(tmp_path).write_text("{broken", encoding="utf-8")
        with pytest.raises(IndexMetadataError):
            index_meta.verify(tmp_path, "models/whatever")

    def test_mismatch_message_explains_why_it_is_not_caught_automatically(
        self, tmp_path
    ):
        """Without the dimensionality note the error looks like pedantry;
        with it, the reader understands nothing else would have caught this.
        """
        index_meta.write(tmp_path, embedding_model="models/a", n_documents=1)
        with pytest.raises(IndexMetadataError) as exc:
            index_meta.verify(tmp_path, "models/b")
        assert "dimensionality" in str(exc.value)


def test_written_file_is_valid_json_on_disk(tmp_path):
    index_meta.write(tmp_path, embedding_model="models/x", n_documents=5)
    raw = index_meta.meta_path(tmp_path).read_text(encoding="utf-8")
    assert json.loads(raw)["n_documents"] == 5
    assert raw.endswith("\n")  # keeps the file diff-friendly
