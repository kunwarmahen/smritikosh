"""Add fact_contradictions table for QC contradiction tracking

Revision ID: 0016
Revises: 0015
Create Date: 2026-04-23

Stores conflicts where an extracted fact has a different value from what is
already known for the same (user, app, category, key). The user resolves these
through the review dashboard; the system auto-resolves when the confidence delta
is large enough to overwrite without ambiguity.
"""

import sqlalchemy as sa
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fact_contradictions",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("key", sa.String(255), nullable=False),
        sa.Column("existing_value", sa.Text(), nullable=False),
        sa.Column("existing_confidence", sa.Float(), nullable=False),
        sa.Column("candidate_value", sa.Text(), nullable=False),
        sa.Column("candidate_source", sa.String(32), nullable=False,
                  server_default="api_explicit"),
        sa.Column("candidate_confidence", sa.Float(), nullable=False),
        sa.Column("resolved", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("resolution", sa.String(32), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fact_contradictions_user_app", "fact_contradictions",
                    ["user_id", "app_id"])
    op.create_index("ix_fact_contradictions_resolved", "fact_contradictions",
                    ["resolved"])


def downgrade() -> None:
    op.drop_index("ix_fact_contradictions_resolved", table_name="fact_contradictions")
    op.drop_index("ix_fact_contradictions_user_app", table_name="fact_contradictions")
    op.drop_table("fact_contradictions")
