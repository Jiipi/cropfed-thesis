"""Show the structure of an audit JSON."""
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
data = json.loads(target.read_text(encoding="utf-8"))

print("top-level keys:", list(data.keys()))
manifests = data.get("manifests") or {}
print(f"manifests ({len(manifests)}):")
for name, value in manifests.items():
    if isinstance(value, dict):
        print(f"  {name}: keys={list(value.keys())[:8]}")
        first_key = next((k for k in value.keys() if isinstance(value[k], list)), None)
        if first_key:
            sample = value[first_key][:2]
            print(f"    sample[{first_key}]: {sample}")