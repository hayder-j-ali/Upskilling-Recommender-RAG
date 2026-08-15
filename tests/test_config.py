"""Tests for config, focused on the .env-shadowing detector.

Regression coverage for the hardest bug in this project's history: a stale
`export GOOGLE_API_KEY=...` in one shell silently made .env inert for every
process launched from it. Edits to .env appeared to do nothing, and the app
authenticated with a credential the user believed they had replaced. The
resulting 401 was misdiagnosed three times — as a transient outage, as a
categorically-unsupported credential type, and as a wrong key format —
because every hypothesis was tested in a shell WITHOUT the export, where
the good .env key was picked up and everything passed.
"""

from __future__ import annotations

import importlib

import pytest

import learning_rec.config as config_module


@pytest.fixture
def config_with_dotenv(tmp_path, monkeypatch):
    """Reload config against a throwaway .env so tests never touch the real one."""

    def _build(file_contents: str):
        dotenv = tmp_path / ".env"
        dotenv.write_text(file_contents, encoding="utf-8")
        monkeypatch.setattr(config_module, "DOTENV_PATH", dotenv)
        return config_module

    return _build


class TestShadowedDotenvVars:
    def test_no_shadowing_when_var_absent_from_environment(
        self, config_with_dotenv, monkeypatch
    ):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        cfg = config_with_dotenv("GOOGLE_API_KEY=from-dotenv\n")
        assert cfg.shadowed_dotenv_vars() == []

    def test_no_shadowing_when_values_agree(self, config_with_dotenv, monkeypatch):
        """load_dotenv() populated it from the file — not a conflict."""
        monkeypatch.setenv("GOOGLE_API_KEY", "same-value")
        cfg = config_with_dotenv("GOOGLE_API_KEY=same-value\n")
        assert cfg.shadowed_dotenv_vars() == []

    def test_detects_shell_export_overriding_dotenv(
        self, config_with_dotenv, monkeypatch
    ):
        """The actual bug: shell export wins, .env is silently ignored."""
        monkeypatch.setenv("GOOGLE_API_KEY", "stale-exported-key")
        cfg = config_with_dotenv("GOOGLE_API_KEY=the-key-the-user-just-edited-in\n")
        assert cfg.shadowed_dotenv_vars() == ["GOOGLE_API_KEY"]

    def test_reports_every_shadowed_name(self, config_with_dotenv, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "stale")
        monkeypatch.setenv("CHAT_MODEL", "stale-model")
        cfg = config_with_dotenv(
            "GOOGLE_API_KEY=fresh\nCHAT_MODEL=fresh-model\nTOP_K=10\n"
        )
        assert sorted(cfg.shadowed_dotenv_vars()) == ["CHAT_MODEL", "GOOGLE_API_KEY"]

    def test_empty_when_dotenv_file_is_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config_module, "DOTENV_PATH", tmp_path / "does-not-exist")
        monkeypatch.setenv("GOOGLE_API_KEY", "anything")
        assert config_module.shadowed_dotenv_vars() == []


def test_config_module_still_imports_cleanly():
    """Guards the import-time load_dotenv() call from regressions."""
    importlib.reload(config_module)
    assert config_module.EMBEDDING_MODEL
    assert config_module.CHAT_MODEL
