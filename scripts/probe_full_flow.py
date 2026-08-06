"""End-to-end probe: create experiment -> start smoke -> poll until completed.

Verifies the web UI control plane and the FastAPI synthetic executor work
through the public web proxy at https://localhost:8443.
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.error
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


def make_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def request_json(
    token: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    ctx: ssl.SSLContext | None = None,
) -> tuple[int, dict]:
    if ctx is None:
        ctx = make_ssl_context()
    url = f"https://localhost:8443/api/v1{path}"
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as error:
        return error.code, {"detail": error.read().decode()[:300]}


def main() -> int:
    token = get_admin_token()
    ctx = make_ssl_context()

    payload = {
        "name": "Probe smoke end-to-end",
        "execution_mode": "synthetic-smoke",
        "algorithm": "fedavg",
        "partition_kind": "iid",
        "num_clients": 4,
        "num_rounds": 3,
        "local_epochs": 1,
        "learning_rate": 0.05,
        "batch_size": 32,
        "dirichlet_alpha": 0.5,
        "proximal_mu": 0.01,
        "seed": 2026,
    }
    print("[1/4] Create experiment...")
    status, body = request_json(token, "/experiments", method="POST", payload=payload)
    print(f"  HTTP {status} id={body.get('id')} status={body.get('status')}")
    if status != 201:
        print("  FAILED", body)
        return 1
    exp_id = body["id"]

    print("[2/4] Start experiment (queued -> running)...")
    status, body = request_json(token, f"/experiments/{exp_id}/start", method="POST")
    print(f"  HTTP {status} status={body.get('status')}")
    if status not in (200, 202):
        print("  FAILED", body)
        return 1

    print("[3/4] Poll status until completed...")
    for attempt in range(60):
        status, body = request_json(token, f"/experiments/{exp_id}")
        if status != 200:
            print(f"  HTTP {status} {body}")
            return 1
        if body.get("status") in ("completed", "failed"):
            break
        time.sleep(2)
    print(f"  final status={body.get('status')} error={body.get('error_message')}")
    if body.get("status") != "completed":
        return 1

    print("[4/4] Inspect rounds + history...")
    status, rounds = request_json(token, f"/experiments/{exp_id}/rounds")
    items = rounds.get("items", [])
    print(f"  rounds: storage={rounds.get('storage')} count={len(items)}")
    if items:
        last = items[-1]
        print(f"  round={last.get('round')} macro_f1={last.get('macro_f1')}")

    result = body.get("result") or {}
    history = result.get("history", [])
    print(f"  history rounds: {len(history)}")
    if history:
        last = history[-1]
        print(f"  history last: round={last.get('round')} macro_f1={last.get('macro_f1')}")
    print("\nALL GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())