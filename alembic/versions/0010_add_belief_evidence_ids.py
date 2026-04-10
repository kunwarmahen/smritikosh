"""add evidence_event_ids to user_beliefs

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-10

Adds evidence_event_ids (JSONB, default []) to user_beliefs so that each
inferred belief tracks the event IDs that contributed to it. This enables
provenance — users and admins can see which conversations produced a belief
and retract incorrect ones.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_beliefs",
        sa.Column(
            "evidence_event_ids",
            JSONB,
            nullable=False,
            server_default="[]",
        ),
    )


def downgrade() -> None:
    op.drop_column("user_beliefs", "evidence_event_ids")
