"""Publication authority metadata and answer traces.

Frozen migration: does not import application models.
Revision ID: 0003_grounded_answers
Revises: 0002_hybrid_retrieval
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_grounded_answers"
down_revision: str | None = "0002_hybrid_retrieval"
branch_labels: str | None = None
depends_on: str | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "publication_status", sa.String(length=20), nullable=False, server_default="unknown"
        ),
    )
    op.add_column("documents", sa.Column("published_on", sa.Date(), nullable=True))
    op.add_column("documents", sa.Column("superseded_by_document_id", sa.Uuid(), nullable=True))
    op.add_column("documents", sa.Column("publication_note", sa.String(length=500), nullable=True))
    op.add_column(
        "documents",
        sa.Column("publication_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_documents_publication_status"),
        "documents",
        "publication_status IN ('current','superseded','withdrawn','unknown')",
    )
    op.create_check_constraint(
        op.f("ck_documents_not_self_superseded"), "documents", "superseded_by_document_id <> id"
    )
    op.create_foreign_key(
        "fk_documents_superseded_by",
        "documents",
        "documents",
        ["superseded_by_document_id", "org_id"],
        ["id", "org_id"],
    )
    op.create_table(
        "answer_runs",
        sa.Column("org_id", sa.Uuid(), nullable=False),
        sa.Column("index_version_id", sa.Uuid(), nullable=False),
        sa.Column("kb_ids", _JSON, nullable=False),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("question_sha256", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=10), nullable=False),
        sa.Column("reranked", sa.Boolean(), nullable=False),
        sa.Column("grounding", sa.String(length=30), nullable=False),
        sa.Column("abstain_reason", sa.String(length=500), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("trace", _JSON, nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["org_id"], ["organizations.id"], name=op.f("fk_answer_runs_org_id_organizations")
        ),
        sa.ForeignKeyConstraint(
            ["index_version_id"],
            ["index_versions.id"],
            name=op.f("fk_answer_runs_index_version_id_index_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_answer_runs")),
    )
    op.create_index("ix_answer_runs_org_created", "answer_runs", ["org_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_answer_runs_org_created", table_name="answer_runs")
    op.drop_table("answer_runs")
    op.drop_constraint("fk_documents_superseded_by", "documents", type_="foreignkey")
    op.drop_constraint(op.f("ck_documents_not_self_superseded"), "documents", type_="check")
    op.drop_constraint(op.f("ck_documents_publication_status"), "documents", type_="check")
    for column in (
        "publication_updated_at",
        "publication_note",
        "superseded_by_document_id",
        "published_on",
        "publication_status",
    ):
        op.drop_column("documents", column)
