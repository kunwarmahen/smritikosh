"""add reconsolidation fields to events

Adds two columns to the events table to support Memory Reconsolidation:
  - reconsolidation_count  tracks how many times an event has been refined
  - last_reconsolidated_at enforces the per-event cooldown window

Revision ID: 0007
Revises: 0006
Create Date: 2026-03-16
"""

from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column(
            "reconsolidation_count",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "events",
        sa.Column(
            "last_reconsolidated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("events", "last_reconsolidated_at")
    op.drop_column("events", "reconsolidation_count")
