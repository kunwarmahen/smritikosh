"""
Tests for the public-benchmark harness (evals/benchmarks/).

Loaders are tested on tiny synthetic dataset files; ingestion/QA/judging on a
fake SDK client and fake LLM. No network, no server, no tokens.
"""

import json

import pytest

from evals.benchmarks.adapter import (
    IngestState,
    QAResult,
    answer_question,
    chunk_session_events,
    ingest_user,
    judge_result,
    looks_like_abstention,
)
from evals.benchmarks.common import BenchQuestion, BenchSession, BenchTurn, BenchUser
from evals.benchmarks.datasets import load_locomo, load_longmemeval
from evals.benchmarks.runner import BenchConfig, BenchReport, apply_limits, run_benchmark

# ── Fixture data ──────────────────────────────────────────────────────────────

LOCOMO_SAMPLE = [
    {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted a dog!"},
                {
                    "speaker": "Bob",
                    "dia_id": "D1:2",
                    "text": "Look at this.",
                    "blip_caption": "a man holding a trophy",
                },
            ],
            "session_1_date_time": "1 pm on 5 May, 2023",
            "session_2": [
                {"speaker": "Alice", "dia_id": "D2:1", "text": "Vet visit went fine."}
            ],
            "session_2_date_time": "2 pm on 9 May, 2023",
        },
        "qa": [
            {"question": "Who adopted a dog?", "answer": "Alice", "category": 1},
            {"question": "When was the vet visit?", "answer": "9 May 2023", "category": 2},
            {
                "question": "What did Bob win?",
                "adversarial_answer": "a medal",
                "category": 5,
            },
        ],
    }
]

LME_SAMPLE = [
    {
        "question_id": "q1_abc",
        "question_type": "single-session-user",
        "question": "What did I adopt?",
        "answer": "a dog",
        "question_date": "2023/05/10 (Wed) 10:00",
        # repr-strings on purpose — the real oracle file ships them this way
        "haystack_dates": "['2023/05/05 (Fri) 13:00']",
        "haystack_session_ids": "['answer_1']",
        "haystack_sessions": [
            [
                {"role": "user", "content": "I adopted a dog today!", "has_answer": True},
                {"role": "assistant", "content": "Congratulations!"},
            ]
        ],
    },
    {
        "question_id": "q2_abs",
        "question_type": "single-session-user",
        "question": "What is my cat's name?",
        "answer": "no answer",
        "question_date": "2023/05/10 (Wed) 10:00",
        "haystack_dates": ["2023/05/05 (Fri) 13:00"],
        "haystack_session_ids": ["answer_2"],
        "haystack_sessions": [[{"role": "user", "content": "I adopted a dog today!"}]],
    },
]


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "locomo10.json").write_text(json.dumps(LOCOMO_SAMPLE))
    (tmp_path / "longmemeval_oracle.json").write_text(json.dumps(LME_SAMPLE))
    return tmp_path


# ── Fakes ─────────────────────────────────────────────────────────────────────


class FakeContext:
    def __init__(self, text="Alice adopted a dog on 5 May.", total=2):
        self.context_text = text
        self.total_memories = total


class FakeClient:
    """Stands in for SmritikoshClient (async context manager + 3 methods)."""

    def __init__(self):
        self.encoded: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def encode(self, *, user_id, content, **kwargs):
        self.encoded.append((user_id, content))

    async def delete_user_memory(self, *, user_id, **kwargs):
        self.deleted.append(user_id)

    async def build_context(self, *, user_id, query, **kwargs):
        return FakeContext()


class FakeLLM:
    """Answers questions with a canned reply; judges based on a canned verdict."""

    def __init__(self, answer="a dog", verdict="CORRECT"):
        self.answer = answer
        self.verdict = verdict
        self.prompts: list[str] = []
        self._chat_model = "fake/model"

    async def complete(self, *, messages, **kwargs):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "Gold answer" in prompt or "correctly abstain" in prompt:
            return self.verdict
        return self.answer


class FakeBenchConfig(BenchConfig):
    """Injects the fake client and keeps state in a temp dir."""

    def __init__(self, client, tmp_path, **kwargs):
        super().__init__(benchmark="locomo", api_key="sk-test", data_dir=tmp_path, **kwargs)
        self._client = client

    def make_client(self):
        return self._client


def bench_user(questions=None):
    return BenchUser(
        user_id="locomo-conv-1",
        sessions=[
            BenchSession(
                session_id="session_1",
                date="1 pm on 5 May, 2023",
                turns=[
                    BenchTurn("Alice", "I adopted a dog!"),
                    BenchTurn("Bob", "Nice!"),
                    BenchTurn("Alice", "His name is Rex."),
                ],
            )
        ],
        questions=questions
        or [
            BenchQuestion("q1", "Who adopted a dog?", "Alice", "single-hop"),
        ],
    )


# ── Loaders ───────────────────────────────────────────────────────────────────


def test_load_locomo(data_dir):
    users = load_locomo(data_dir)
    assert len(users) == 1
    user = users[0]
    assert user.user_id == "locomo-conv-1"
    assert [s.session_id for s in user.sessions] == ["session_1", "session_2"]
    assert user.sessions[0].date == "1 pm on 5 May, 2023"
    # BLIP caption folded into the turn text
    assert "[shares a photo: a man holding a trophy]" in user.sessions[0].turns[1].content
    # Adversarial (cat 5) excluded by default
    assert len(user.questions) == 2
    assert {q.category for q in user.questions} == {"single-hop", "temporal"}


def test_load_locomo_include_adversarial(data_dir):
    users = load_locomo(data_dir, include_adversarial=True)
    questions = users[0].questions
    assert len(questions) == 3
    adversarial = questions[-1]
    assert adversarial.category == "adversarial"
    assert adversarial.is_abstention
    assert adversarial.gold_answer == ""


def test_load_longmemeval(data_dir):
    users = load_longmemeval(data_dir, variant="oracle")
    assert len(users) == 2
    first = users[0]
    assert first.user_id == "lme-oracle-q1_abc"
    assert first.sessions[0].date == "2023/05/05 (Fri) 13:00"  # repr-string parsed
    assert first.sessions[0].session_id == "answer_1"
    assert first.questions[0].question_date == "2023/05/10 (Wed) 10:00"
    assert not first.questions[0].is_abstention
    # _abs suffix → abstention; plain-list fields also accepted
    assert users[1].questions[0].is_abstention


def test_load_longmemeval_rejects_unknown_variant(data_dir):
    with pytest.raises(ValueError, match="variant"):
        load_longmemeval(data_dir, variant="xl")


# ── Chunking and ingestion ────────────────────────────────────────────────────


def test_chunk_session_events_per_turn():
    events = chunk_session_events(bench_user(), chunk_turns=1)
    assert len(events) == 3
    assert events[0] == "[1 pm on 5 May, 2023] Alice: I adopted a dog!"


def test_chunk_session_events_grouped():
    events = chunk_session_events(bench_user(), chunk_turns=2)
    assert len(events) == 2
    assert events[0].count("\n") == 1  # two turns joined


async def test_ingest_user_resumable(tmp_path):
    client = FakeClient()
    state = IngestState(tmp_path / "state.json")
    user = bench_user()

    stored = await ingest_user(client, user, state)
    assert stored == 3
    assert client.deleted == [user.user_id]  # wiped before ingest
    assert len(client.encoded) == 3

    # Second call: already done → no-op
    stored_again = await ingest_user(client, user, state)
    assert stored_again == 0
    assert len(client.encoded) == 3

    # Fresh state object reads the same file (resume across processes)
    assert IngestState(tmp_path / "state.json").is_done(user.user_id)


def test_ingest_state_reset(tmp_path):
    state = IngestState(tmp_path / "state.json")
    state.mark_done("u1")
    state.reset()
    assert not state.is_done("u1")
    assert not (tmp_path / "state.json").exists()


# ── QA + judging ──────────────────────────────────────────────────────────────


async def test_answer_question_prompt_and_latency():
    client, llm = FakeClient(), FakeLLM(answer="Alice")
    q = BenchQuestion("q1", "Who adopted a dog?", "Alice", "single-hop",
                      question_date="2023/05/10")
    result = await answer_question(client, llm, "u1", q)
    assert result.answer == "Alice"
    assert result.memories_used == 2
    assert result.retrieval_ms >= 0 and result.answer_ms >= 0
    prompt = llm.prompts[0]
    assert "Alice adopted a dog on 5 May." in prompt  # retrieved context included
    assert "Current date: 2023/05/10" in prompt
    assert "Who adopted a dog?" in prompt


async def test_judge_correct_and_wrong():
    result = QAResult("q1", "single-hop", "Who?", "Alice", answer="Alice")
    assert await judge_result(FakeLLM(verdict="CORRECT"), result) is True
    assert await judge_result(FakeLLM(verdict="WRONG"), result) is False


async def test_judge_abstention_fast_path():
    result = QAResult("q1", "adversarial", "What?", "", answer="I don't know.",
                      is_abstention=True)
    llm = FakeLLM(verdict="WRONG")  # judge would say WRONG, but fast path wins
    assert await judge_result(llm, result) is True
    assert llm.prompts == []  # no judge call needed


def test_looks_like_abstention():
    assert looks_like_abstention("I don't know")
    assert looks_like_abstention("That is not mentioned in the memories.")
    assert not looks_like_abstention("Alice")


# ── Limits ────────────────────────────────────────────────────────────────────


def test_apply_limits_users_and_round_robin():
    users = [
        bench_user(questions=[
            BenchQuestion("a1", "?", "x", "single-hop"),
            BenchQuestion("a2", "?", "x", "temporal"),
        ]),
        BenchUser(user_id="u2", sessions=[], questions=[
            BenchQuestion("b1", "?", "x", "multi-hop"),
        ]),
    ]
    limited = apply_limits(users, max_questions=2)
    picked = [q.question_id for u in limited for q in u.questions]
    assert picked == ["a1", "b1"]  # one from each user, not two from the first
    assert len(apply_limits(users, max_users=1)) == 1


# ── End-to-end with fakes ─────────────────────────────────────────────────────


async def test_run_benchmark_end_to_end(tmp_path):
    client = FakeClient()
    config = FakeBenchConfig(client, tmp_path)
    users = [bench_user(questions=[
        BenchQuestion("q1", "Who adopted a dog?", "Alice", "single-hop"),
        BenchQuestion("q2", "When was the vet visit?", "9 May", "temporal"),
    ])]
    report = await run_benchmark(users, config, llm=FakeLLM(verdict="CORRECT"),
                                 progress=False)
    assert report.ingested_events == 3
    assert len(report.results) == 2
    assert report.accuracy == 1.0
    assert report.by_category()["single-hop"]["accuracy"] == 1.0
    data = report.to_json()
    assert data["questions"] == 2
    assert "retrieval_p50_ms" in data["latency"]


async def test_run_benchmark_records_errors(tmp_path):
    class BrokenClient(FakeClient):
        async def build_context(self, **kwargs):
            raise RuntimeError("server down")

    client = BrokenClient()
    config = FakeBenchConfig(client, tmp_path)
    report = await run_benchmark([bench_user()], config, llm=FakeLLM(), progress=False)
    assert report.results[0].error
    assert report.accuracy == 0.0
    assert report.to_json()["errors"] == 1


async def test_empty_message_exception_still_counts_as_error(tmp_path):
    """httpx timeouts stringify to '' — they must land in errors, not scored-wrong."""

    class TimeoutClient(FakeClient):
        async def build_context(self, **kwargs):
            raise TimeoutError()  # str() == ""

    config = FakeBenchConfig(TimeoutClient(), tmp_path)
    report = await run_benchmark([bench_user()], config, llm=FakeLLM(), progress=False)
    result = report.results[0]
    assert result.error == "TimeoutError"  # type preserved despite empty message
    assert report.scored == []             # excluded from the accuracy denominator


async def test_run_benchmark_skip_ingest_and_cleanup(tmp_path):
    client = FakeClient()
    config = FakeBenchConfig(client, tmp_path)
    user = bench_user()
    await run_benchmark([user], config, llm=FakeLLM(), skip_ingest=True,
                        cleanup=True, progress=False)
    assert client.encoded == []                 # ingestion skipped
    assert user.user_id in client.deleted       # cleanup ran
