"""Probe worker imports inside the API container."""

import importlib
import traceback

try:
    importlib.import_module("cropfed.api.db")
    print("cropfed.api.db OK")
except Exception as exc:  # noqa: BLE001
    print("db FAIL", repr(exc))

try:
    importlib.import_module("cropfed.api.worker")
    print("cropfed.api.worker OK")
except Exception as exc:  # noqa: BLE001
    print("worker FAIL", repr(exc))
    traceback.print_exc()

try:
    worker_module = importlib.import_module("cropfed.api.worker")
    _ = worker_module.run_worker_once
    print("run_worker_once OK")
except Exception as exc:  # noqa: BLE001
    print("run_worker_once FAIL", repr(exc))
    traceback.print_exc()
