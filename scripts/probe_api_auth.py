"""Verify the API auth and project endpoint from inside the container."""

import os
import urllib.request

ADMIN_TOKEN = os.environ.get("CROPFED_ADMIN_TOKEN") or os.environ.get("ADMIN_TOKEN") or ""


def fetch(url: str, token: str | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:  # noqa: BLE001
        return exc.code, exc.read().decode()


# Project endpoint is public
status, body = fetch("http://api:8000/api/v1/project")
print(f"/api/v1/project -> {status} body={body[:200]}")

# Auth/me requires token
status, body = fetch("http://api:8000/api/v1/auth/me")
print(f"/api/v1/auth/me (no token) -> {status}")

# Listing experiments requires token
status, body = fetch("http://api:8000/api/v1/experiments", token=ADMIN_TOKEN)
print(f"/api/v1/experiments -> {status} body={body[:200]}")
