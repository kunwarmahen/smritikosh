"""
Typed return values for the Smritikosh SDK.

These mirror the API response schemas but live in the SDK layer so callers
don't need to depend on the server-side `smritikosh.api.schemas` module.
"""

from dataclasses import dataclass, field


@dataclass
class EncodedEvent:
    """Returned by SmritikoshClient.encode() after storing a memory."""
    event_id: str
    user_id: str
    importance_score: float
    facts_extracted: int
    extraction_failed: bool


@dataclass
class RecentEvent:
    """One event item from SmritikoshClient.get_recent()."""
    event_id: str
    raw_text: str
    importance_score: float
    consolidated: bool
    created_at: str


@dataclass
class MemoryContext:
    """
    Returned by SmritikoshClient.build_context().

    context_text  — ready to drop into any LLM system prompt.
    messages      — OpenAI-style [{"role": "system", "content": "..."}].
    """
    user_id: str
    query: str
    context_text: str
    messages: list[dict]
    total_memories: int
    embedding_failed: bool

    def is_empty(self) -> bool:
        return self.total_memories == 0


@dataclass
class HealthStatus:
    status: str
    version: str
