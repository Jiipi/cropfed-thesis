"""Probe run_worker_once inside the worker container."""

import logging
import os
import sys
import traceback

os.environ.setdefault(
    "CROPFED_DATABASE_URL",
    "postgresql+psycopg://cropfed:rhSNh4-j2oDWMuK965Eey9X__yscpNPrd50Azg60lK8@db:5432/cropfed",
)
os.environ["CROPFED_FLOWER_WORKER_ENABLED"] = "true"
os.environ["CROPFED_PROJECT_ROOT"] = "/app"
os.environ.setdefault("PYTHONPATH", "/app/src")

sys.path.insert(0, "/app/src")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# noqa below: settings depend on environment being set above.
from cropfed.api.settings import settings  # noqa: E402
from cropfed.api.worker import run_worker_once  # noqa: E402

print(f"database_url={settings.database_url}")
print(f"flower_data_root={settings.flower_data_root}")
print(f"flower_project_dir={settings.flower_project_dir}")
print(f"flower_worker_enabled={settings.flower_worker_enabled}")
try:
    claimed = run_worker_once()
    print(f"claim result: claimed={claimed}")
except Exception as exc:  # noqa: BLE001
    print(f"run_worker_once failed: {exc!r}")
    traceback.print_exc()
