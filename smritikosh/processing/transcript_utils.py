"""
Transcript utilities — anti-contamination helpers for passive extraction.

Three key operations:

1. strip_sentinels(text)
   Remove <!-- smritikosh:context-start/end --> blocks injected by the context
   builder. Prevents re-extracting facts that we already know (circular reinforcement).

2. user_turns_only(turns)
   Drop assistant turns before passing to the extraction LLM. The assistant may
   paraphrase injected context — extracting from it would inflate confidence without
   adding new information.

3. build_delta_prompt(user_turns, existing_facts)
   Produce the extraction prompt that tells the LLM to only extract NEW or
   CONTRADICTING facts relative to what is already known.

Usage:
    from smritikosh.processing.transcript_utils import (
        strip_sentinels,
        user_turns_only,
        build_delta_prompt,
    )

    clean_turns = user_turns_only(session_turns)
    text = "\\n".join(t["content"] for t in clean_turns)
    text = strip_sentinels(text)
    prompt = build_delta_prompt(clean_turns, existing_facts)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sentinel block pattern injected by ContextBuilder
_SENTINEL_RE = re.compile(
    r"<!--\s*smritikosh:context-start\s*-->.*?<!--\s*smritikosh:context-end\s*-->",
    re.DOTALL | re.IGNORECASE,
)

# Whitespace normalisation after stripping sentinels
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


@dataclass
class PreparedTranscript:
    """Output of prepare_transcript() — ready for the extraction LLM."""
    user_turns: list[dict]       # filtered to role=user, sentinels stripped
    combined_text: str           # concatenated content, newline-separated
    turns_count: int             # how many user turns remain after filtering
    stripped_sentinels: bool     # True if at least one sentinel block was found


# ── Core utilities ────────────────────────────────────────────────────────────

def strip_sentinels(text: str) -> str:
    """
    Remove all <!-- smritikosh:context-start --> … <!-- smritikosh:context-end -->
    blocks from text, then normalise resulting whitespace.

    Safe to call on text with no sentinels — returns text unchanged.
    """
    cleaned, n = _SENTINEL_RE.subn("", text)
    if n > 0:
        cleaned = _MULTI_BLANK_RE.sub("\n\n", cleaned).strip()
    return cleaned


def user_turns_only(turns: list[dict]) -> list[dict]:
    """
    Return only the turns where role == 'user', with sentinel blocks stripped
    from each turn's content.

    Args:
        turns: List of {'role': str, 'content': str} dicts.

    Returns:
        Filtered list, content cleaned.
    """
    result = []
    for turn in turns:
        if turn.get("role") != "user":
            continue
        content = strip_sentinels(str(turn.get("content", "")))
        if content.strip():
            result.append({**turn, "content": content})
    return result


def build_delta_prompt(
    user_turns: list[dict],
    existing_facts: list | None = None,
    last_turn_index: int = 0,
) -> str:
    """
    Build an extraction prompt that instructs the LLM to only extract NEW or
    CONTRADICTING facts relative to the user's existing fact set.

    Args:
        user_turns:       Already-filtered user turns (output of user_turns_only).
        existing_facts:   Current FactRecord list for this user (may be empty).
        last_turn_index:  For streaming extraction — skip turns at or before this index.

    Returns:
        Prompt string ready to pass to LLMAdapter.extract_structured().
    """
    relevant_turns = user_turns[last_turn_index:]
    conversation_text = "\n".join(
        f"User: {t['content']}" for t in relevant_turns
    ).strip()

    if not existing_facts:
        known_section = "(no existing knowledge — extract all clear personal facts)"
    else:
        lines = ["You already know the following about this user:"]
        for f in existing_facts:
            lines.append(f"  - {f.category}/{f.key}: {f.value}  (confidence: {f.confidence:.2f})")
        lines.append(
            "\nExtract ONLY facts that are NEW or that CONTRADICT existing knowledge. "
            "Do not repeat what is already known."
        )
        known_section = "\n".join(lines)

    return (
        f"{known_section}\n\n"
        f"Conversation (user turns only):\n"
        f"{conversation_text}"
    )


def prepare_transcript(turns: list[dict], last_turn_index: int = 0) -> PreparedTranscript:
    """
    Full preprocessing pipeline: filter to user turns, strip sentinels, check for
    any sentinel presence, and produce combined text.

    Args:
        turns:            Raw conversation turns.
        last_turn_index:  Skip turns at or before this index (for streaming windows).

    Returns:
        PreparedTranscript ready for trigger detection + LLM extraction.
    """
    # Check for sentinels before filtering (they could appear in assistant turns too)
    raw_text = " ".join(str(t.get("content", "")) for t in turns)
    had_sentinels = bool(_SENTINEL_RE.search(raw_text))

    # Filter and clean
    clean_turns = user_turns_only(turns)
    windowed = clean_turns[last_turn_index:]
    combined = "\n".join(t["content"] for t in windowed)

    return PreparedTranscript(
        user_turns=windowed,
        combined_text=combined,
        turns_count=len(windowed),
        stripped_sentinels=had_sentinels,
    )
