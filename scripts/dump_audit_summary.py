"""Summarize audit errors."""

import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
data = json.loads(target.read_text(encoding="utf-8"))

print("status:", data.get("status"))
print("errors:")
for error in data.get("errors") or []:
    code = error.get("code")
    reason = error.get("reason")
    image_ids = error.get("image_ids") or []
    print(f"  {code} reason={reason} image_ids={len(image_ids)}")
print(f"taxonomy: status={data.get('taxonomy', {}).get('status')}")
print(f"client_assignment: status={data.get('client_assignment', {}).get('status')}")
print(f"global_split_overlap: status={data.get('global_split_overlap', {}).get('status')}")
print(f"duplicates: status={data.get('duplicates', {}).get('status')}")
print(f"images: status={data.get('images', {}).get('status')}")
print(f"privacy: status={data.get('privacy', {}).get('status')}")