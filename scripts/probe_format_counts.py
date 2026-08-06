"""Quick probe of audit format counts in container."""

import json
from pathlib import Path

audit = Path(
    "/app/artifacts/flower-api/21e79fec-a2da-4f1d-b3ed-feefa3e02c91/pre_run_data_audit.json"
)
data = json.loads(audit.read_text(encoding="utf-8"))
imgs = data.get("images") or {}
print("format_counts:", imgs.get("format_counts"))
print("invalid count:", len(imgs.get("invalid_images") or []))
print("verified count:", imgs.get("verified_images"))
print("unique_paths checked:", imgs.get("unique_paths_checked"))
