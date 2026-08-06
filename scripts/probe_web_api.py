"""End-to-end probe against the CropFed API through the web TLS proxy."""

from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def get_admin_token() -> str:
    env_path = Path(".env")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("CROPFED_API_ADMIN_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("CROPFED_API_ADMIN_TOKEN not found in .env")


def request_json(token: str, path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = f"https://localhost:8443/api/v1{path}"
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return json.loads(resp.read())


def probe(token: str) -> dict:
    project = request_json(token, "/project")
    classes = request_json(token, "/classes")
    profiles = request_json(token, "/data-profiles")
    experiments = request_json(token, "/experiments")
    return {
        "project": project,
        "classes": classes,
        "data_profiles": profiles,
        "experiments": experiments,
    }


def main() -> int:
    token = get_admin_token()
    try:
        result = probe(token)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("PROJECT:", json.dumps(result["project"], indent=2, ensure_ascii=False))
    print("\nCLASS COUNT:", result["classes"]["count"])
    print("DATA PROFILES:", json.dumps(result["data_profiles"], indent=2, ensure_ascii=False)[:500])
    print("\nEXPERIMENT COUNT:", len(result["experiments"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())