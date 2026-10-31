"""
Memory views — what a client's prompt is given about a user, and what
keeps a memory from ageing. An admin's choice, made in the dashboard.

Three switches and one rule, per user, with a default for everyone:

    facts      the user's facts from the graph (who they are, where they
               live, what they prefer, and those relevant to the question)
    beliefs    patterns the belief miner found, handed over as guesses
    memories   the statements themselves -- only those no fact covers, so
               nothing is said twice
    keep_by    "used": only an answer that leaned on a memory, fact or
               belief keeps it from ageing (POST /memory/used);
               "shown": being put in a prompt counts too

INHERITED, FIELD BY FIELD. A user's row overrides only the fields it
sets; the rest come from the ``"*"`` row (everyone's), and what that
leaves unset from ``DEFAULT``. So an admin can turn beliefs off for one
person without pinning their other switches to today's defaults.

USED IS THE DEFAULT. Being shown is not mattering: twenty memories are
shown on every turn, and a rule that kept whatever was shown would keep
everything, useful or not. "shown" is there for an admin who would
rather lose nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import MemoryView

#: The row that holds everyone's default.
EVERYONE = "*"
KEEP_BY = ("used", "shown")
FIELDS = ("show_facts", "show_beliefs", "show_memories", "keep_by")


@dataclass
class View:
    show_facts: bool = True
    show_beliefs: bool = True
    show_memories: bool = True
    keep_by: str = "used"

    def as_dict(self) -> dict:
        return asdict(self)


#: What applies when no admin has said anything.
DEFAULT = View()


async def _row(pg: AsyncSession, user_id: str, app_id: str) -> MemoryView | None:
    found = await pg.execute(select(MemoryView).where(
        MemoryView.user_id == user_id, MemoryView.app_id == app_id))
    return found.scalar_one_or_none()


async def effective(pg: AsyncSession, user_id: str,
                    app_id: str = "default") -> tuple[View, dict[str, str]]:
    """The view that applies to ``user_id``, and where each field came
    from: ``user``, ``everyone`` or ``default``."""
    mine = await _row(pg, user_id, app_id) if user_id != EVERYONE else None
    theirs = await _row(pg, EVERYONE, app_id)
    values, source = {}, {}
    for name in FIELDS:
        for where, row in (("user", mine), ("everyone", theirs)):
            value = getattr(row, name, None) if row is not None else None
            if value is not None:
                values[name], source[name] = value, where
                break
        else:
            values[name], source[name] = getattr(DEFAULT, name), "default"
    return View(**values), source


async def overrides(pg: AsyncSession, user_id: str, app_id: str = "default") -> dict:
    """The fields this row sets itself (None where it inherits)."""
    row = await _row(pg, user_id, app_id)
    return {name: (getattr(row, name) if row is not None else None) for name in FIELDS}


async def set_view(pg: AsyncSession, user_id: str, app_id: str = "default",
                   **fields) -> None:
    """Set ``user_id``'s overrides (``"*"``: everyone's). A field given as
    None goes back to inheriting; a field not given is left alone."""
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown view fields: {', '.join(sorted(unknown))}")
    if fields.get("keep_by") not in (None, *KEEP_BY):
        raise ValueError(f"keep_by is one of {', '.join(KEEP_BY)}")
    row = await _row(pg, user_id, app_id)
    if row is None:
        row = MemoryView(user_id=user_id, app_id=app_id)
        pg.add(row)
    for name, value in fields.items():
        setattr(row, name, value)
    await pg.flush()
