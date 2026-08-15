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


# Substrings Google puts in the `reason`/`status` fields of API errors. Matched
# against the stringified exception because the two client layers disagree on
# shape: google.genai raises APIError with structured .code/.status, while
# langchain_google_genai's embeddings code catches that and re-raises a plain
# GoogleGenerativeAIError whose only payload is the formatted string.
_AUTH_MARKERS = (
    "ACCESS_TOKEN_TYPE_UNSUPPORTED",  # right value, wrong *kind* of credential
    "API_KEY_INVALID",  # malformed or revoked key
    "UNAUTHENTICATED",
    "PERMISSION_DENIED",
)
_RATE_LIMIT_MARKERS = ("RESOURCE_EXHAUSTED", "429")
_TRANSIENT_MARKERS = ("UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "503", "500")


def classify_api_error(exc: BaseException) -> str:
    """Bucket a Gemini API error into 'auth' | 'rate_limit' | 'transient' | 'unknown'.

    The distinction matters because the remedies are opposites: an auth
    failure will *never* resolve by retrying, so telling a user to "wait and
    try again" (correct for a 503) actively wastes their time when the real
    problem is the credential.

    Auth is checked first: a 401 carries no transient marker, but keeping the
    order explicit documents the precedence rather than leaving it to luck.
    """
    text = str(exc).upper()
    if any(marker in text for marker in _AUTH_MARKERS):
        return "auth"
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return "rate_limit"
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return "transient"
    return "unknown"


def api_error_guidance(exc: BaseException) -> str:
    """Actionable next step for a Gemini API error, matched to its cause."""
    kind = classify_api_error(exc)
    if kind == "auth":
        return (
            "This is a credential problem, so retrying will not help. The "
            "Gemini Developer API expects an API key from "
            "https://aistudio.google.com/apikey — these start with `AIza`. "
            "An OAuth access token or a key from a different Google product "
            "is rejected here, often only on the embeddings endpoint, which "
            "is why chat can appear to work while retrieval fails. Set a "
            "valid key as `GOOGLE_API_KEY`, or switch to **bm25** with "
            "**LLM re-rank** off to run fully offline."
        )
    if kind == "rate_limit":
        return (
            "You have hit the API rate limit or quota. Wait a minute before "
            "trying again, or switch to **bm25** with **LLM re-rank** off to "
            "run fully offline."
        )
    if kind == "transient":
        return (
            "This is usually transient. Wait a few seconds and click "
            "**Generate recommendations** again — or switch to **bm25** with "
            "**LLM re-rank** off for an offline demo that does not depend on "
            "Gemini's availability."
        )
    return (
        "Try again in a few seconds. If it persists, switch to **bm25** with "
        "**LLM re-rank** off to run fully offline, and check the terminal for "
        "the full traceback."
    )
