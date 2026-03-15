"""add cluster_id and cluster_label to events

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-15
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column("cluster_id", sa.Integer, nullable=True),
    )
    op.add_column(
        "events",
        sa.Column("cluster_label", sa.Text, nullable=True),
    )
    op.create_index("ix_events_cluster_id", "events", ["cluster_id"])


def downgrade() -> None:
    op.drop_index("ix_events_cluster_id", table_name="events")
    op.drop_column("events", "cluster_label")
    op.drop_column("events", "cluster_id")
