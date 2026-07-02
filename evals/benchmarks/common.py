"""
Shared types and prompts for the public-benchmark harness.

Both benchmarks normalize to the same shape: a *BenchUser* is one Smritikosh
user with a chat history to ingest and questions to answer against it.

  LoCoMo       → 1 user per conversation (10 users, ~150 questions each)
  LongMemEval  → 1 user per instance (500 users, 1 question each — every
                 instance has its own haystack history)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchTurn:
    speaker: str  # LoCoMo: speaker name; LongMemEval: "user" | "assistant"
    content: str


@dataclass
class BenchSession:
    session_id: str
    date: str  # human-readable date string from the dataset
    turns: list[BenchTurn] = field(default_factory=list)


@dataclass
class BenchQuestion:
    question_id: str
    question: str
    gold_answer: str
    category: str  # locomo: single-hop/temporal/multi-hop/open-domain; lme: question_type
    is_abstention: bool = False  # gold behavior is to say "I don't know"
    question_date: str = ""


@dataclass
class BenchUser:
    user_id: str
    sessions: list[BenchSession] = field(default_factory=list)
    questions: list[BenchQuestion] = field(default_factory=list)

    @property
    def total_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)


# ── Prompts ───────────────────────────────────────────────────────────────────

ANSWER_SYSTEM = (
    "You answer questions about a user's past conversations using retrieved "
    "memories. Be precise and concise: respond with just the answer (a short "
    "phrase or sentence), no preamble. If the memories do not contain the "
    "information needed, respond exactly: I don't know."
)

ANSWER_TEMPLATE = (
    "Retrieved memories:\n{context}\n\n"
    "{date_line}"
    "Question: {question}\n"
    "Answer:"
)

JUDGE_PROMPT = (
    "You are grading a memory system's answer against the gold answer.\n"
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Generated answer: {answer}\n\n"
    "Does the generated answer convey the same information as the gold answer? "
    "Minor wording, formatting, or date-format differences are acceptable; "
    "missing or contradicting the key information is not. "
    "Reply with a single word: CORRECT or WRONG."
)

JUDGE_ABSTENTION_PROMPT = (
    "A memory system was asked a question whose answer is NOT present in the "
    "conversation history — the correct behavior is to abstain (say it doesn't "
    "know or that the information isn't available).\n"
    "Question: {question}\n"
    "Generated answer: {answer}\n\n"
    "Did the system correctly abstain instead of inventing an answer? "
    "Reply with a single word: CORRECT or WRONG."
)


def build_answer_prompt(context: str, question: str, question_date: str = "") -> str:
    date_line = f"Current date: {question_date}\n" if question_date else ""
    return ANSWER_TEMPLATE.format(context=context, date_line=date_line, question=question)


def format_turn_content(session: BenchSession, turn: BenchTurn) -> str:
    """One ingestible line: date-stamped, speaker-attributed."""
    return f"[{session.date}] {turn.speaker}: {turn.content}"
