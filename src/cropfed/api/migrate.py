"""Versioned database migration entry points for CLI, Docker, and tests."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from alembic import command
from alembic.config import Config

from cropfed.api.settings import settings

PROJECT_ROOT = Path(os.getenv("CROPFED_PROJECT_ROOT", Path.cwd())).resolve()


def alembic_config(database_url: str | None = None) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option(
        "sqlalchemy.url",
        (database_url or settings.database_url).replace("%", "%%"),
    )
    return config


def upgrade_database(
    *,
    database_url: str | None = None,
    revision: str = "head",
) -> None:
    command.upgrade(alembic_config(database_url), revision)


def downgrade_database(
    *,
    database_url: str | None = None,
    revision: str = "-1",
) -> None:
    command.downgrade(alembic_config(database_url), revision)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cropfed-migrate")
    parser.add_argument("action", choices=["upgrade", "downgrade"])
    parser.add_argument("--revision")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "upgrade":
        upgrade_database(revision=args.revision or "head")
    else:
        downgrade_database(revision=args.revision or "-1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
