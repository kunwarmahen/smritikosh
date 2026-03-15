"""add recall_count to events

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-15
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column(
            "recall_count",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
    )
    op.create_index("ix_events_recall_count", "events", ["recall_count"])


def downgrade() -> None:
    op.drop_index("ix_events_recall_count", table_name="events")
    op.drop_column("events", "recall_count")
