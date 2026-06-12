"""add user_activity table; backfill from events.

Indexed user discovery for background jobs (item A5) — replaces the
SELECT DISTINCT full scan of `events` on every scheduler tick. One row per
(user_id, app_id): last_event_at is touched on every event store; the
last_*_at watermarks record per-job completion so work can be ordered by
staleness.

The backfill seeds a row for every (user_id, app_id) that already has
events, using MAX(created_at) as last_event_at, so discovery keeps finding
existing tenants after the cutover.

Revision ID: 0024
Revises: 0023
Create Date: 2026-06-11
"""

import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_activity",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("last_event_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_consolidated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_pruned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_clustered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_belief_mined_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_synthesized_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("user_id", "app_id", name="uq_user_activity"),
    )
    op.create_index("ix_user_activity_last_event", "user_activity", ["last_event_at"])

    # Backfill one row per existing tenant. gen_random_uuid() needs the
    # pgcrypto extension on PG <13; generate ids client-side to be safe.
    conn = op.get_bind()
    rows = conn.execute(
        sa.text(
            "SELECT user_id, app_id, MAX(created_at) AS last_event_at "
            "FROM events GROUP BY user_id, app_id"
        )
    ).fetchall()
    if rows:
        conn.execute(
            sa.text(
                "INSERT INTO user_activity (id, user_id, app_id, last_event_at) "
                "VALUES (:id, :user_id, :app_id, :last_event_at)"
            ),
            [
                {
                    "id": str(uuid.uuid4()),
                    "user_id": r.user_id,
                    "app_id": r.app_id,
                    "last_event_at": r.last_event_at,
                }
                for r in rows
            ],
        )


def downgrade() -> None:
    op.drop_index("ix_user_activity_last_event", table_name="user_activity")
    op.drop_table("user_activity")
