"""Add the agricultural client registry.

Revision ID: 0002_clients
Revises: 0001_initial
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_clients"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("clients"):
        op.create_table(
            "clients",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("name", sa.String(length=120), nullable=False),
            sa.Column("description", sa.String(length=500), nullable=False),
            sa.Column("status", sa.String(length=24), nullable=False),
            sa.Column("partition_id", sa.Integer(), nullable=True),
            sa.Column("num_local_samples", sa.Integer(), nullable=True),
            sa.Column("last_seen_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
    _create_index_if_missing("clients", "ix_clients_name", ["name"])
    _create_index_if_missing("clients", "ix_clients_status", ["status"])


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("clients"):
        op.drop_table("clients")


def _create_index_if_missing(
    table_name: str,
    index_name: str,
    columns: list[str],
) -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {item["name"] for item in inspector.get_indexes(table_name)}
    if index_name not in existing:
        op.create_index(index_name, table_name, columns, unique=False)
