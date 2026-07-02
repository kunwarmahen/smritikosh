"""
ConsentService — cross-app memory sharing with per-category grants (item S4).

Multi-app isolation is the default: facts learned in app A are invisible to
app B. This service implements the deliberate inverse: a *user* grants app B
read access to their facts from app A, restricted to fact categories,
revocable at any time, with an audit-trail entry for every cross-app read.

Grants are directional and per-user. Enforcement happens at read time:
`consented_facts()` is called by the context builder when app B asks for a
user's context, pulling in facts from consented source apps (category-
filtered, provenance-tagged) and emitting one audit event per source read.

Postgres stores the grants (memory_consents); the facts themselves stay in
Neo4j under their source app — nothing is copied, so a revocation takes
effect on the next read.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from neo4j import AsyncSession as NeoSession
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import FactCategory, MemoryConsent
from smritikosh.memory.semantic import FactRecord, SemanticMemory

logger = logging.getLogger(__name__)

_VALID_CATEGORIES = {c.value for c in FactCategory}


class ConsentError(ValueError):
    """Raised for invalid grant parameters (bad category, self-grant, …)."""


class ConsentService:
    """Grant / revoke / resolve cross-app memory consents."""

    def __init__(self, semantic: SemanticMemory | None = None, audit=None) -> None:
        self.semantic = semantic
        self.audit = audit  # AuditLogger | None

    # ── Grant management ──────────────────────────────────────────────────

    async def grant(
        self,
        pg: AsyncSession,
        *,
        user_id: str,
        source_app_id: str,
        target_app_id: str,
        categories: list[str] | None = None,
        created_by: str,
    ) -> MemoryConsent:
        """
        Create or reactivate a consent grant.

        Upsert semantics: one row per (user, source, target). Re-granting a
        revoked consent reactivates it; granting an active one replaces its
        category list.

        Raises ConsentError on self-grants or unknown categories.
        """
        if source_app_id == target_app_id:
            raise ConsentError("source_app_id and target_app_id must differ.")
        categories = categories or []
        unknown = [c for c in categories if c not in _VALID_CATEGORIES]
        if unknown:
            raise ConsentError(
                f"Unknown fact categories: {unknown}. "
                f"Valid: {sorted(_VALID_CATEGORIES)}"
            )

        consent = await self._get(pg, user_id, source_app_id, target_app_id)
        if consent is None:
            consent = MemoryConsent(
                user_id=user_id,
                source_app_id=source_app_id,
                target_app_id=target_app_id,
                categories=categories,
                granted_at=datetime.now(timezone.utc),
                created_by=created_by,
            )
            pg.add(consent)
        else:
            consent.categories = categories
            consent.revoked_at = None
            consent.granted_at = datetime.now(timezone.utc)
            consent.created_by = created_by
        await pg.flush()

        await self._audit(
            "consent.granted", user_id, target_app_id,
            {"source_app_id": source_app_id, "target_app_id": target_app_id,
             "categories": categories, "created_by": created_by},
        )
        return consent

    async def revoke(
        self,
        pg: AsyncSession,
        *,
        user_id: str,
        source_app_id: str,
        target_app_id: str,
        revoked_by: str = "",
    ) -> bool:
        """Revoke a grant (keeps the row for audit). Returns False if absent/already revoked."""
        consent = await self._get(pg, user_id, source_app_id, target_app_id)
        if consent is None or not consent.is_active:
            return False
        consent.revoked_at = datetime.now(timezone.utc)
        await pg.flush()

        await self._audit(
            "consent.revoked", user_id, target_app_id,
            {"source_app_id": source_app_id, "target_app_id": target_app_id,
             "revoked_by": revoked_by},
        )
        return True

    async def list_for_user(
        self,
        pg: AsyncSession,
        user_id: str,
        *,
        include_revoked: bool = False,
    ) -> list[MemoryConsent]:
        stmt = select(MemoryConsent).where(MemoryConsent.user_id == user_id)
        if not include_revoked:
            stmt = stmt.where(MemoryConsent.revoked_at.is_(None))
        result = await pg.execute(stmt.order_by(MemoryConsent.granted_at.desc()))
        return list(result.scalars().all())

    async def active_sources_for(
        self,
        pg: AsyncSession,
        *,
        user_id: str,
        target_app_id: str,
    ) -> list[MemoryConsent]:
        """Active grants that let `target_app_id` read this user's other apps."""
        result = await pg.execute(
            select(MemoryConsent)
            .where(MemoryConsent.user_id == user_id)
            .where(MemoryConsent.target_app_id == target_app_id)
            .where(MemoryConsent.revoked_at.is_(None))
        )
        return list(result.scalars().all())

    # ── Read-path enforcement ─────────────────────────────────────────────

    async def consented_facts(
        self,
        pg: AsyncSession,
        neo: NeoSession,
        *,
        user_id: str,
        target_app_id: str,
        min_confidence: float = 0.0,
    ) -> list[FactRecord]:
        """
        Fetch facts shared into `target_app_id` under active consents.

        For each active grant: read the user's profile from the source app,
        filter to the consented categories (empty list = all), tag each fact
        with its provenance (``source_meta["shared_from_app"]``), and emit a
        ``consent.cross_app_read`` audit event.

        Failures on one source never break the read — cross-app facts are an
        enrichment, not a dependency.
        """
        if self.semantic is None:
            return []
        grants = await self.active_sources_for(
            pg, user_id=user_id, target_app_id=target_app_id
        )
        shared: list[FactRecord] = []
        for grant in grants:
            try:
                profile = await self.semantic.get_user_profile(
                    neo, user_id, grant.source_app_id, min_confidence=min_confidence
                )
            except Exception:
                logger.exception(
                    "Consented profile read failed",
                    extra={"user_id": user_id, "source_app": grant.source_app_id},
                )
                continue
            facts = profile.facts if profile else []
            if grant.categories:
                allowed = set(grant.categories)
                facts = [f for f in facts if f.category in allowed]
            for fact in facts:
                fact.source_meta = {**(fact.source_meta or {}),
                                    "shared_from_app": grant.source_app_id}
            shared.extend(facts)

            await self._audit(
                "consent.cross_app_read", user_id, target_app_id,
                {"source_app_id": grant.source_app_id,
                 "target_app_id": target_app_id,
                 "categories": grant.categories,
                 "facts_returned": len(facts)},
            )
        return shared

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    async def _get(
        pg: AsyncSession, user_id: str, source_app_id: str, target_app_id: str
    ) -> MemoryConsent | None:
        result = await pg.execute(
            select(MemoryConsent)
            .where(MemoryConsent.user_id == user_id)
            .where(MemoryConsent.source_app_id == source_app_id)
            .where(MemoryConsent.target_app_id == target_app_id)
        )
        return result.scalar_one_or_none()

    async def _audit(self, event_type: str, user_id: str, app_id: str, payload: dict) -> None:
        """Emit an audit event; never let audit failures break the operation."""
        if self.audit is None:
            return
        try:
            from smritikosh.audit.logger import AuditEvent

            await self.audit.emit(AuditEvent(
                event_type=event_type, user_id=user_id, app_id=app_id, payload=payload
            ))
        except Exception:  # pragma: no cover — audit must never break consent ops
            logger.exception("Consent audit emit failed")
