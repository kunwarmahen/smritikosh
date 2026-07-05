"""add user_beliefs.status + retracted_at for belief retraction.

Item E2: DELETE /beliefs/{id} sets status=rejected (row is kept) so the
belief miner can see the rejection and never resurrect the statement.

Revision ID: 0027
Revises: 0026
Create Date: 2026-07-05
"""

from alembic import op
import sqlalchemy as sa

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_beliefs",
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
    )
    op.add_column(
        "user_beliefs",
        sa.Column("retracted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_user_beliefs_status", "user_beliefs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_user_beliefs_status", table_name="user_beliefs")
    op.drop_column("user_beliefs", "retracted_at")
    op.drop_column("user_beliefs", "status")
