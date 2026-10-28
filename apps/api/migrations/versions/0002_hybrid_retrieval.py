"""Hybrid retrieval and vector cleanup.

Adds ``index_versions.sparse_config`` (NULL = dense-only; existing index versions keep
their fingerprints and collections) and the terminal ``deleted`` document-index status.

Frozen migration: does not import application models.
Revision ID: 0002_hybrid_retrieval
Revises: 0001_ingestion
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_hybrid_retrieval"
down_revision: str | None = "0001_ingestion"
branch_labels: str | None = None
depends_on: str | None = None

_OLD = "status IN ('building','ready','failed','deleting')"
_NEW = "status IN ('building','ready','failed','deleting','deleted')"


def upgrade() -> None:
    op.add_column(
        "index_versions",
        sa.Column(
            "sparse_config",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
    )
    op.drop_constraint(op.f("ck_document_indexes_status"), "document_indexes", type_="check")
    op.create_check_constraint(op.f("ck_document_indexes_status"), "document_indexes", _NEW)


def downgrade() -> None:
    # Rows already marked 'deleted' cannot be represented by the old constraint.
    op.execute("UPDATE document_indexes SET status = 'deleting' WHERE status = 'deleted'")
    op.drop_constraint(op.f("ck_document_indexes_status"), "document_indexes", type_="check")
    op.create_check_constraint(op.f("ck_document_indexes_status"), "document_indexes", _OLD)
    op.drop_column("index_versions", "sparse_config")
