"""Helpers shared by every LLM call site (rerank, judge)."""

from __future__ import annotations

import re

from langchain_core.messages import BaseMessage

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def extract_text(response: BaseMessage) -> str:
    """Normalize `AIMessage.content` to plain text.

    OpenAI's chat wrapper always returns `content` as a flat string. Gemini's
    wrapper returns a list of typed content blocks (each carrying provider
    metadata like a grounding signature) even for a plain single-turn text
    reply. Concatenate the text blocks so callers don't need to special-case
    the provider.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def strip_markdown_fences(text: str) -> str:
    """Strip a ```json ... ``` (or bare ```) fence some models wrap JSON in.

    Both system prompts in this project tell the model to return raw JSON
    with "no markdown fences" — Gemini ignores that instruction often enough
    in practice that relying on prompt compliance alone isn't robust. A
    light regex is standard practice for this rather than fighting the
    model into 100% compliance, which no provider reliably guarantees.
    """
    match = _FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text
