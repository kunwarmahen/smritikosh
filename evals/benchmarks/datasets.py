"""
Dataset download and loading for LoCoMo and LongMemEval.

Files are cached under evals/benchmarks/data/ (gitignored). Loaders normalize
both datasets to lists of BenchUser (see common.py).

LoCoMo (snap-research/locomo, ACL 2024): 10 very-long two-speaker
conversations (~6K turns total) with ~2K QA pairs in 5 categories:

    1 single-hop · 2 temporal · 3 multi-hop · 4 open-domain · 5 adversarial

Category 5 questions have no real answer (`adversarial_answer` is the trap)
and are excluded by default, matching the published Mem0/Zep/Memobase setups.

LongMemEval (xiaowu0162, ICLR 2025): 500 instances, each with its own
haystack of timestamped user-assistant sessions. Variants: `oracle` (evidence
sessions only, ~15MB), `s` (~115K-token haystacks), `m` (~1.5M-token
haystacks). `question_id` ending in `_abs` marks abstention questions.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

import httpx

from evals.benchmarks.common import BenchQuestion, BenchSession, BenchTurn, BenchUser

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LONGMEMEVAL_URL_TEMPLATE = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/{filename}"
)
LONGMEMEVAL_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

LOCOMO_CATEGORIES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}

_SESSION_KEY_RE = re.compile(r"session_(\d+)$")


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s → %s", url, dest)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        tmp = dest.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)
        tmp.rename(dest)
    return dest


def _as_list(value: object) -> list:
    """LongMemEval list fields are sometimes python-repr strings — handle both."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise TypeError(f"expected list or repr-string, got {type(value)}")


# ── LoCoMo ────────────────────────────────────────────────────────────────────


def load_locomo(
    data_dir: Path = DATA_DIR,
    *,
    include_adversarial: bool = False,
) -> list[BenchUser]:
    path = _download(LOCOMO_URL, data_dir / "locomo10.json")
    samples = json.loads(path.read_text())

    users: list[BenchUser] = []
    for sample in samples:
        conv = sample["conversation"]
        sessions: list[BenchSession] = []
        session_numbers = sorted(
            int(m.group(1))
            for key in conv
            if (m := _SESSION_KEY_RE.fullmatch(key)) and isinstance(conv[key], list)
        )
        for n in session_numbers:
            turns = []
            for raw in conv[f"session_{n}"]:
                text = raw.get("text", "")
                # Image turns carry a BLIP caption — keep it; the image itself
                # is out of scope for a text memory benchmark.
                caption = raw.get("blip_caption")
                if caption:
                    text = f"{text} [shares a photo: {caption}]".strip()
                turns.append(BenchTurn(speaker=raw["speaker"], content=text))
            sessions.append(
                BenchSession(
                    session_id=f"session_{n}",
                    date=conv.get(f"session_{n}_date_time", ""),
                    turns=turns,
                )
            )

        questions = []
        for i, qa in enumerate(sample["qa"]):
            category_num = int(qa["category"])
            if category_num == 5 and not include_adversarial:
                continue
            is_adversarial = category_num == 5
            questions.append(
                BenchQuestion(
                    question_id=f"{sample['sample_id']}-q{i}",
                    question=qa["question"],
                    gold_answer=str(qa.get("answer", "")) if not is_adversarial else "",
                    category=LOCOMO_CATEGORIES[category_num],
                    is_abstention=is_adversarial,
                )
            )

        users.append(
            BenchUser(
                user_id=f"locomo-{sample['sample_id']}",
                sessions=sessions,
                questions=questions,
            )
        )
    return users


# ── LongMemEval ───────────────────────────────────────────────────────────────


def load_longmemeval(
    data_dir: Path = DATA_DIR,
    *,
    variant: str = "oracle",
) -> list[BenchUser]:
    if variant not in LONGMEMEVAL_FILES:
        raise ValueError(f"unknown LongMemEval variant {variant!r} (oracle|s|m)")
    filename = LONGMEMEVAL_FILES[variant]
    path = _download(LONGMEMEVAL_URL_TEMPLATE.format(filename=filename), data_dir / filename)
    instances = json.loads(path.read_text())

    users: list[BenchUser] = []
    for inst in instances:
        question_id = inst["question_id"]
        raw_sessions = inst["haystack_sessions"]
        dates = _as_list(inst["haystack_dates"])
        session_ids = _as_list(inst["haystack_session_ids"])

        sessions = [
            BenchSession(
                session_id=str(session_ids[i]) if i < len(session_ids) else f"s{i}",
                date=str(dates[i]) if i < len(dates) else "",
                turns=[
                    BenchTurn(speaker=t["role"], content=t["content"])
                    for t in raw_sessions[i]
                ],
            )
            for i in range(len(raw_sessions))
        ]

        users.append(
            BenchUser(
                user_id=f"lme-{variant}-{question_id}",
                sessions=sessions,
                questions=[
                    BenchQuestion(
                        question_id=question_id,
                        question=inst["question"],
                        gold_answer=str(inst["answer"]),
                        category=inst["question_type"],
                        is_abstention=question_id.endswith("_abs"),
                        question_date=inst.get("question_date", ""),
                    )
                ],
            )
        )
    return users
