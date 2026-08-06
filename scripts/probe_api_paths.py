"""Dump OpenAPI paths from the API container."""

import json
import urllib.request

req = urllib.request.Request("http://api:8000/openapi.json")
data = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())
for path in sorted(data.get("paths", {}).keys()):
    methods = ",".join(sorted(data["paths"][path].keys())).upper()
    print(f"{methods:>10}  {path}")
