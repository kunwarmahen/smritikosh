"""add raw_file / filename / context_note columns to media_ingests.

Persists the raw uploaded bytes and the request metadata a queued
media-processing task needs (item A3 — durable task queue), so the task
survives an API/worker restart. raw_file is cleared once processing finishes.

Revision ID: 0021
Revises: 0020
Create Date: 2026-05-20
"""

from alembic import op
import sqlalchemy as sa

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("media_ingests", sa.Column("raw_file", sa.LargeBinary(), nullable=True))
    op.add_column("media_ingests", sa.Column("filename", sa.String(512), nullable=True))
    op.add_column("media_ingests", sa.Column("context_note", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("media_ingests", "context_note")
    op.drop_column("media_ingests", "filename")
    op.drop_column("media_ingests", "raw_file")
