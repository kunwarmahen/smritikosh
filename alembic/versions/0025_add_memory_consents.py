"""add memory_consents table.

Cross-app memory consent layer (item S4): a user grants one app read access
to facts learned in another, per fact category, revocable. Revocations keep
the row (revoked_at) so grant history stays auditable.

Revision ID: 0025
Revises: 0024
Create Date: 2026-07-01
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, UUID

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_consents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("source_app_id", sa.String(255), nullable=False),
        sa.Column("target_app_id", sa.String(255), nullable=False),
        sa.Column(
            "categories",
            ARRAY(sa.String(64)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.UniqueConstraint(
            "user_id", "source_app_id", "target_app_id", name="uq_memory_consent"
        ),
    )
    op.create_index(
        "ix_memory_consents_user_target",
        "memory_consents",
        ["user_id", "target_app_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_consents_user_target", table_name="memory_consents")
    op.drop_table("memory_consents")
