"""add scopes column to api_keys.

Revision ID: 0020
Revises: 0019_add_user_connectors
Create Date: 2026-04-27
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column(
            "scopes",
            ARRAY(sa.String(64)),
            nullable=False,
            server_default=sa.text("ARRAY['read','write']::varchar[]"),
        ),
    )


def downgrade() -> None:
    op.drop_column("api_keys", "scopes")
