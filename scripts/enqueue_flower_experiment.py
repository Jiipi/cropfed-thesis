"""Create + start a fresh Flower experiment against the running API."""

import json
import os
import sys
import urllib.request

API = os.environ.get("API_URL", "http://localhost:8000")
TOKEN = os.environ.get(
    "ADMIN_TOKEN",
    "E553Hi5x5iTeUb8wBxytnXoKSJIKybFMNDcFGjU9xCKQMyKCvnV1DfOGJ-Nz9JCd",
)


def post(path: str, body: dict[str, object], *, auth: bool = True) -> dict[str, object]:
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(f"{API}{path}", data=payload, method="POST", headers=headers)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def main() -> int:
    config = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
        "name": "docker-fedavg-d0.5-pg-v3",
        "execution_mode": "flower",
        "algorithm": "fedavg",
        "partition_kind": "dirichlet",
        "num_clients": 4,
        "num_rounds": 1,
        "local_epochs": 1,
        "learning_rate": 0.001,
        "batch_size": 128,
        "dirichlet_alpha": 0.5,
        "proximal_mu": 0.01,
        "seed": 2026,
    }
    created = post("/api/v1/experiments", config)
    print(f"created {created['id']} status={created['status']}")
    started = post(f"/api/v1/experiments/{created['id']}/start", {})
    print(f"started status={started['status']}")
    print(json.dumps({"id": created["id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
