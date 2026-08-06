"""Print the Alembic sqlalchemy URL the migrate helper resolves to."""

from cropfed.api.migrate import alembic_config

cfg = alembic_config()
print(cfg.get_main_option("sqlalchemy.url"))
