"""add events.consolidation_anchor for consolidated-event search semantics.

Item E1: consolidation re-embeds the distilled summary onto ONE event per
batch (the anchor). Hybrid search now down-weights consolidated NON-anchor
events (superseded noisy sources) so the clean anchor surfaces first.

Backfill note: pre-existing anchors cannot be identified retroactively
(the representative was never marked), so all legacy consolidated events
start as non-anchors and receive the down-weight. This is the conservative
choice — legacy raw-text sources are exactly what the penalty is for — but
it means legacy summary-carrying representatives are penalised too until
their batch is next touched. Acceptable for a P2; documented in
memory/episodic.py.

Revision ID: 0026
Revises: 0025
Create Date: 2026-07-05
"""

from alembic import op
import sqlalchemy as sa

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column(
            "consolidation_anchor",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("events", "consolidation_anchor")
