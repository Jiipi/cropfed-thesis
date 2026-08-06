"""Create the original experiment and metric tables.

Revision ID: 0001_initial
Revises: None
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("experiments"):
        op.create_table(
            "experiments",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("name", sa.String(length=120), nullable=False),
            sa.Column("status", sa.String(length=24), nullable=False),
            sa.Column("execution_mode", sa.String(length=32), nullable=False),
            sa.Column("algorithm", sa.String(length=16), nullable=False),
            sa.Column("partition_kind", sa.String(length=16), nullable=False),
            sa.Column("num_clients", sa.Integer(), nullable=False),
            sa.Column("num_rounds", sa.Integer(), nullable=False),
            sa.Column("local_epochs", sa.Integer(), nullable=False),
            sa.Column("learning_rate", sa.Float(), nullable=False),
            sa.Column("batch_size", sa.Integer(), nullable=False),
            sa.Column("dirichlet_alpha", sa.Float(), nullable=False),
            sa.Column("proximal_mu", sa.Float(), nullable=False),
            sa.Column("seed", sa.Integer(), nullable=False),
            sa.Column("result_json", sa.String(), nullable=True),
            sa.Column("error_message", sa.String(length=2000), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
    _create_index_if_missing("experiments", "ix_experiments_name", ["name"])
    _create_index_if_missing("experiments", "ix_experiments_status", ["status"])

    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("experiment_rounds"):
        op.create_table(
            "experiment_rounds",
            sa.Column("experiment_id", sa.String(length=36), nullable=False),
            sa.Column("round_number", sa.Integer(), nullable=False),
            sa.Column("metrics_json", sa.String(), nullable=False),
            sa.Column("train_loss", sa.Float(), nullable=True),
            sa.Column("evaluation_loss", sa.Float(), nullable=True),
            sa.Column("accuracy", sa.Float(), nullable=True),
            sa.Column("macro_f1", sa.Float(), nullable=True),
            sa.Column("harmful_missed_as_healthy_rate", sa.Float(), nullable=True),
            sa.Column("elapsed_seconds", sa.Float(), nullable=True),
            sa.Column("bytes_up", sa.Integer(), nullable=True),
            sa.Column("bytes_down", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
            sa.PrimaryKeyConstraint("experiment_id", "round_number"),
        )

    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("client_round_metrics"):
        op.create_table(
            "client_round_metrics",
            sa.Column("experiment_id", sa.String(length=36), nullable=False),
            sa.Column("round_number", sa.Integer(), nullable=False),
            sa.Column("client_id", sa.Integer(), nullable=False),
            sa.Column("phase", sa.String(length=16), nullable=False),
            sa.Column("node_id", sa.String(length=32), nullable=False),
            sa.Column("num_examples", sa.Integer(), nullable=False),
            sa.Column("metrics_json", sa.String(), nullable=False),
            sa.Column("payload_download_bytes", sa.Integer(), nullable=False),
            sa.Column("payload_upload_bytes", sa.Integer(), nullable=False),
            sa.Column("model_download_bytes", sa.Integer(), nullable=False),
            sa.Column("model_upload_bytes", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
            sa.PrimaryKeyConstraint(
                "experiment_id", "round_number", "client_id", "phase"
            ),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    for table_name in (
        "client_round_metrics",
        "experiment_rounds",
        "experiments",
    ):
        if inspector.has_table(table_name):
            op.drop_table(table_name)
            inspector = sa.inspect(op.get_bind())


def _create_index_if_missing(
    table_name: str,
    index_name: str,
    columns: list[str],
) -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {item["name"] for item in inspector.get_indexes(table_name)}
    if index_name not in existing:
        op.create_index(index_name, table_name, columns, unique=False)
