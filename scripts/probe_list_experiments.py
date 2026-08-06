"""List experiments with admin auth token."""

import json
import os
import urllib.request

TOKEN = os.environ["CROPFED_API_ADMIN_TOKEN"]

req = urllib.request.Request("http://api:8000/api/v1/experiments")
req.add_header("Authorization", f"Bearer {TOKEN}")
data = json.loads(urllib.request.urlopen(req).read().decode())
print(f"count: {len(data)}")
for item in data[:5]:
    print(json.dumps(item, ensure_ascii=False)[:300])
