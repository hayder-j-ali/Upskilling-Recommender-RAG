"""Tests for the Streamlit app.

Import smoke test: catches import-time syntax errors, missing imports,
broken sys.path bootstrap, removed config attributes. The interesting
*business* logic (retrieve, rerank_with_llm) is covered by its own tests.

The error-handling test below drives the app for real via `AppTest` —
worth the extra weight because it's a regression test for an actual
production incident (see TestApiErrorHandling docstring).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

streamlit = pytest.importorskip("streamlit")  # skip if streamlit not installed

from google.genai.errors import APIError
from langchain_google_genai._common import GoogleGenerativeAIError
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import learning_rec.recommender as recommender_module

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_streamlit_app_imports():
    spec = importlib.util.spec_from_file_location("streamlit_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["streamlit_app"] = module
    spec.loader.exec_module(module)  # will raise if the module is broken


class TestApiErrorHandling:
    """Regression tests for two real incidents where a Gemini API error
    crashed the app into a raw traceback instead of a clean message.
    Reproduces each error shape via a mock — no live API call, so these run
    offline in CI — and asserts the app shows an actionable message instead
    of crashing.
    """

    def test_transient_503_shows_clean_message_not_a_crash(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")

        # Exact shape Gemini returned in the incident this test guards against.
        fake_503 = APIError(
            503,
            {
                "error": {
                    "code": 503,
                    "message": (
                        "This model is currently experiencing high demand. "
                        "Spikes in demand are usually temporary. Please try "
                        "again later."
                    ),
                    "status": "UNAVAILABLE",
                }
            },
        )

        def raise_503(*_args, **_kwargs):
            raise fake_503

        with patch.object(recommender_module, "rerank_with_llm", side_effect=raise_503):
            at = AppTest.from_file(str(APP_PATH))
            at.run()
            # bm25 needs no FAISS index (keeps this test offline); force
            # rerank on so the patched rerank_with_llm actually gets hit.
            at.sidebar.radio[0].set_value("bm25").run()
            at.sidebar.checkbox[0].set_value(True).run()
            at.button[0].click().run()

        assert not at.exception, "app crashed instead of handling the API error"
        error_text = " ".join(e.value for e in at.error)
        assert "503" in error_text
        assert "again" in error_text.lower()  # retry guidance is present

    def test_embeddings_auth_error_shows_clean_message_not_a_crash(self, monkeypatch):
        """Second incident: a transient 401 (ACCESS_TOKEN_TYPE_UNSUPPORTED)
        on the embeddings endpoint that did not reproduce on retry with the
        same key — a Google-side hiccup, not a real credential problem.

        `GoogleGenerativeAIEmbeddings` (used by `retrieve()`, hit regardless
        of which retriever is selected once dense/hybrid needs an embedding)
        catches the real `google.genai.errors.APIError` internally and
        re-raises `GoogleGenerativeAIError` instead — NOT a subclass of
        `APIError` — so it slipped past the `except APIError` handler added
        for the first incident and crashed the app anyway.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")

        # Matches the real wrapped-message shape: langchain_google_genai's
        # embeddings code does `f"Error embedding content ({e.status}): {e}"`.
        fake_401 = GoogleGenerativeAIError(
            "Error embedding content (UNAUTHENTICATED): 401 UNAUTHENTICATED. "
            "{'error': {'code': 401, 'message': 'Request had invalid "
            "authentication credentials. Expected OAuth 2 access token, "
            "login cookie or other valid authentication credential.', "
            "'status': 'UNAUTHENTICATED'}}"
        )

        def raise_401(*_args, **_kwargs):
            raise fake_401

        with patch.object(recommender_module, "retrieve", side_effect=raise_401):
            at = AppTest.from_file(str(APP_PATH))
            at.run()
            # Any retriever kind reaches retrieve(); bm25 keeps this offline
            # since we never actually build a real BM25 index either.
            at.sidebar.radio[0].set_value("bm25").run()
            at.button[0].click().run()

        assert not at.exception, "app crashed instead of handling the API error"
        error_text = " ".join(e.value for e in at.error)
        assert "401" in error_text
        # A wrong credential never fixes itself, so this must NOT tell the
        # user to wait and retry (the advice a 503 gets) — it has to point
        # at the credential instead.
        assert "this specific endpoint" in error_text
        assert "AIza" not in error_text  # never steer users to the legacy format
