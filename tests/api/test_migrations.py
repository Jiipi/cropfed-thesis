import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlmodel import SQLModel

import cropfed.api.models  # noqa: F401
from cropfed.api.migrate import downgrade_database, upgrade_database


class DatabaseMigrationTests(unittest.TestCase):
    def test_upgrade_downgrade_and_reupgrade_clients_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "migration.sqlite"
            database_url = f"sqlite:///{database_path.as_posix()}"

            upgrade_database(database_url=database_url)
            engine = create_engine(database_url)
            self.assertIn("clients", inspect(engine).get_table_names())
            self.assertIn("alembic_version", inspect(engine).get_table_names())

            downgrade_database(
                database_url=database_url,
                revision="0001_initial",
            )
            self.assertNotIn("clients", inspect(engine).get_table_names())
            self.assertIn("experiments", inspect(engine).get_table_names())

            upgrade_database(database_url=database_url)
            self.assertIn("clients", inspect(engine).get_table_names())
            engine.dispose()

    def test_upgrade_adopts_schema_created_by_legacy_create_all(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "legacy.sqlite"
            database_url = f"sqlite:///{database_path.as_posix()}"
            engine = create_engine(database_url)
            SQLModel.metadata.create_all(engine)

            upgrade_database(database_url=database_url)

            tables = inspect(engine).get_table_names()
            self.assertIn("alembic_version", tables)
            self.assertIn("clients", tables)
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
