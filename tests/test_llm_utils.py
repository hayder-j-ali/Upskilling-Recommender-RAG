"""Tests for extract_text — normalizing AIMessage.content across providers.

Regression coverage for a real bug: ChatGoogleGenerativeAI returns `content`
as a list of typed blocks (with provider metadata attached) even for a plain
single-turn text reply, unlike OpenAI's wrapper which always returns a flat
string. json.loads(response.content) silently breaks on Gemini without this.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from learning_rec.llm_utils import extract_text, strip_markdown_fences


class TestExtractText:
    def test_plain_string_content(self):
        """OpenAI-style: content is already a flat string."""
        msg = AIMessage(content='{"a": 1}')
        assert extract_text(msg) == '{"a": 1}'

    def test_gemini_style_single_text_block(self):
        """Gemini-style: content is a list of typed blocks."""
        msg = AIMessage(
            content=[{"type": "text", "text": '{"a": 1}', "extras": {"signature": "x"}}]
        )
        assert extract_text(msg) == '{"a": 1}'

    def test_multiple_text_blocks_are_concatenated(self):
        msg = AIMessage(
            content=[
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"},
            ]
        )
        assert extract_text(msg) == "hello world"

    def test_non_text_blocks_are_skipped(self):
        msg = AIMessage(
            content=[
                {"type": "thinking", "text": "internal reasoning"},
                {"type": "text", "text": "the answer"},
            ]
        )
        assert extract_text(msg) == "the answer"

    def test_plain_strings_inside_list_are_kept(self):
        msg = AIMessage(content=["hello ", "world"])
        assert extract_text(msg) == "hello world"

    def test_empty_list_returns_empty_string(self):
        msg = AIMessage(content=[])
        assert extract_text(msg) == ""


class TestStripMarkdownFences:
    """Regression coverage: Gemini wraps JSON in ```json fences often enough
    in practice (despite explicit "no markdown fences" system-prompt
    instructions) that both parsing call sites needed this, not just one.
    """

    def test_strips_json_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert strip_markdown_fences(raw) == '{"a": 1}'

    def test_strips_bare_fence(self):
        raw = '```\n{"a": 1}\n```'
        assert strip_markdown_fences(raw) == '{"a": 1}'

    def test_leaves_unfenced_text_untouched(self):
        raw = '{"a": 1}'
        assert strip_markdown_fences(raw) == '{"a": 1}'

    def test_strips_surrounding_whitespace_around_fence(self):
        raw = '   ```json\n{"a": 1}\n```   '
        assert strip_markdown_fences(raw) == '{"a": 1}'

    def test_handles_multiline_json_inside_fence(self):
        raw = '```json\n[\n  {"a": 1},\n  {"b": 2}\n]\n```'
        assert strip_markdown_fences(raw) == '[\n  {"a": 1},\n  {"b": 2}\n]'
