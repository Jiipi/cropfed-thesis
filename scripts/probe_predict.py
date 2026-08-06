"""Predict endpoint smoke test against the local API inside a container.

Usage (run inside container):
    docker exec cropfed-thesis-api-1 python /tmp/probe_predict.py <path/to/image.jpg>
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def build_multipart(image_path: Path) -> tuple[bytes, str]:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    header = (
        "--" + boundary + "\r\n"
        'Content-Disposition: form-data; name="image"; filename="' + image_path.name + '"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    footer = ("\r\n--" + boundary + "--\r\n").encode()
    body = header + image_path.read_bytes() + footer
    return body, boundary


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_predict.py <image>", file=sys.stderr)
        return 2
    token = os.environ["CROPFED_API_ADMIN_TOKEN"]
    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 1

    body, boundary = build_multipart(image_path)
    request = urllib.request.Request(
        "http://localhost:8000/api/v1/predict",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        print("HTTP", error.code, error.read().decode())
        return 1
    payload = json.loads(response.read())
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())