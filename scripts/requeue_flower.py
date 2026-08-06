"""Queue another experiment via the running API."""

import json
import os
import sys
import urllib.request

API = os.environ.get("API_URL", "http://localhost:8000")
TOKEN = os.environ.get(
    "ADMIN_TOKEN",
    "E553Hi5x5iTeUb8wBxytnXoKSJIKybFMNDcFGjU9xCKQMyKCvnV1DfOGJ-Nz9JCd",
)

if len(sys.argv) < 2:
    raise SystemExit("usage: requeue_flower.py <config.json>")
with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)

payload = json.dumps(config).encode()
request = urllib.request.Request(
    f"{API}/api/v1/experiments",
    data=payload,
    method="POST",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    },
)
with urllib.request.urlopen(request) as response:
    created = json.loads(response.read())
print("created", created["id"])

start_request = urllib.request.Request(
    f"{API}/api/v1/experiments/{created['id']}/start",
    method="POST",
    headers={"Authorization": f"Bearer {TOKEN}"},
)
with urllib.request.urlopen(start_request) as response:
    started = json.loads(response.read())
print("started", started["status"])
print(json.dumps({"id": created["id"]}))
