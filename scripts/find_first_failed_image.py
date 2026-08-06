"""Probe one specific manifest path to find the FileNotFoundError cause."""

import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
data = json.loads(audit_path.read_text(encoding="utf-8"))

errors = data.get("errors") or []
print(f"total errors: {len(errors)}")
for error in errors:
    image_ids = error.get("image_ids") or []
    reason = error.get("reason")
    code = error.get("code")
    if code != "invalid_image" or reason != "FileNotFoundError":
        continue
    if not image_ids:
        continue
    target = image_ids[0]
    print(f"--- searching image_id={target} ---")
    for manifest_name, manifest in data.get("manifests", {}).items():
        if not isinstance(manifest, dict):
            continue
        for key, value in manifest.items():
            if key == "num_records" or key == "sha256" or key == "split_counts":
                continue
            if isinstance(value, list) and any(target in str(v) for v in value):
                for v in value:
                    if target in str(v):
                        print(f"{manifest_name}.{key}: {v}")
                        break
                break