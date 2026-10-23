"""add events.last_used_at (a memory in use does not age).

A client that recalls a memory and then answers with it says so
(POST /memory/used). Hybrid search's recency, the recent-events order
and the pruner's age all count from the later of created_at and
last_used_at, so a statement leaned on every week ranks like a new one
instead of sinking with the year it was written.

Revision ID: 0031
Revises: 0030
Create Date: 2026-10-23
"""

from alembic import op
import sqlalchemy as sa

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("events", "last_used_at")
