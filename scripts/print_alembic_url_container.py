"""Print the Alembic sqlalchemy URL using the same env wiring as Docker."""

import os

os.environ.setdefault(
    "CROPFED_DATABASE_URL",
    "postgresql+psycopg://cropfed:rhSNh4-j2oDWMuK965Eey9X__yscpNPrd50Azg60lK8@db:5432/cropfed",
)

from cropfed.api.migrate import alembic_config

cfg = alembic_config()
print(cfg.get_main_option("sqlalchemy.url"))
