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
    """Regression test for a real incident: Gemini returned a 503 ("high
    demand") mid-demo, which crashed the app into a raw traceback because
    `rerank_with_llm()`'s call site had no exception handling. Reproduces
    the exact error shape from that incident via a mock — no live API call,
    so this runs offline in CI — and asserts the app now shows a clean,
    actionable message instead of crashing.
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
