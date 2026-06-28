"""Smoke test: the Streamlit app module imports cleanly.

We deliberately don't try to drive the app with `st.testing.v1` here —
the interesting logic (retrieve, rerank_with_llm) is covered by its own
tests. This catches: import-time syntax errors, missing imports,
broken sys.path bootstrap, removed config attributes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")  # skip if streamlit not installed

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_streamlit_app_imports():
    spec = importlib.util.spec_from_file_location("streamlit_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["streamlit_app"] = module
    spec.loader.exec_module(module)  # will raise if the module is broken
