"""Hit API health + ready endpoints from inside the container."""

import urllib.request


def fetch(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:  # noqa: BLE001
        return exc.code, exc.read().decode()


for path in ("/health", "/health/ready"):
    url = f"http://api:8000{path}"
    status, body = fetch(url)
    print(f"{path:>14} -> {status} body={body!r}")
