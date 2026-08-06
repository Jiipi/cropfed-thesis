"""Quick probe of audit images section in container."""

import json
from pathlib import Path

audit = Path(
    "/app/artifacts/flower-api/21e79fec-a2da-4f1d-b3ed-feefa3e02c91/pre_run_data_audit.json"
)
data = json.loads(audit.read_text(encoding="utf-8"))
imgs = data.get("images") or {}
print("verified_images:", imgs.get("verified_images"))
print("invalid_images:", imgs.get("invalid_images"))
print("unique_paths_checked:", imgs.get("unique_paths_checked"))
print("format_counts:", imgs.get("format_counts"))
