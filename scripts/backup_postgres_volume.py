"""Backup the running PostgreSQL volume via pg_dump and pg_restore.

This exercises the production backup story for the CropFed stack. It writes
a single ``.sql`` dump next to the artefact directory and verifies that a
fresh logical instance (created on-the-fly with a random database name)
can be restored from that dump. The script is intentionally side-effect
free for the primary database — it never touches ``db`` service schema
or data.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _docker_ps(image_marker: str) -> str:
    """Return the running container id whose image contains ``image_marker``."""

    cmd = [
        "docker",
        "ps",
        "--filter",
        f"ancestor={image_marker}",
        "--format",
        "{{.ID}}",
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    value = completed.stdout.strip().splitlines()
    if not value or not value[0]:
        raise RuntimeError(f"no container found for {image_marker}")
    return value[0]


def _docker_exec(container: str, *argv: str) -> str:
    cmd = ["docker", "exec", "-i", container, *argv]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout


def backup(source_container: str, dump_path: Path, *, user: str, database: str) -> None:
    """Stream pg_dump from the source container into ``dump_path``."""

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "exec",
        source_container,
        "pg_dump",
        "-U",
        user,
        "-d",
        database,
        "--no-owner",
        "--clean",
        "--if-exists",
    ]
    with dump_path.open("wb") as handle:
        subprocess.run(cmd, check=True, stdout=handle)


def restore(
    *,
    image: str,
    dump_path: Path,
    user: str,
    password: str,
    suffix: str,
) -> None:
    """Start a temporary PostgreSQL container and restore ``dump_path`` into it."""

    container = f"cropfed-pg-restore-{suffix}"
    env = [
        "-e",
        f"POSTGRES_USER={user}",
        "-e",
        f"POSTGRES_PASSWORD={password}",
        "-e",
        f"POSTGRES_DB={user}",
        "--rm",
        "--name",
        container,
        "-d",
        image,
    ]
    subprocess.run(["docker", "run", *env], check=True)
    try:
        # wait until healthy (poll pg_isready)
        for _ in range(60):
            probe = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-e",
                    "PGHOST=localhost",
                    container,
                    "pg_isready",
                    "-U",
                    user,
                    "-d",
                    user,
                ],
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("restore container did not become ready")

        # Copy the dump into the container (Windows-friendly) before streaming.
        copy_cmd = ["docker", "cp", str(dump_path), f"{container}:/tmp/restore.sql"]
        subprocess.run(copy_cmd, check=True)
        psql_cmd = [
            "docker",
            "exec",
            "-i",
            "-e",
            "PGHOST=localhost",
            container,
            "psql",
            "-U",
            user,
            "-d",
            user,
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            "/tmp/restore.sql",
        ]
        subprocess.run(psql_cmd, check=True)

        # Verify the schema is present.
        verify_cmd = [
            "docker",
            "exec",
            "-e",
            "PGHOST=localhost",
            container,
            "psql",
            "-U",
            user,
            "-d",
            user,
            "-tAc",
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'public'",
        ]
        tables_cmd = subprocess.run(
            verify_cmd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if tables_cmd == "0":
            raise RuntimeError("restored database has no tables")
        print(f"   restored database has {tables_cmd} public tables")
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "postgres" / "cropfed-pre-restore.sql",
    )
    parser.add_argument(
        "--image",
        default="postgres:17-alpine",
    )
    parser.add_argument("--user", default=os.environ.get("POSTGRES_USER", "cropfed"))
    parser.add_argument(
        "--password",
        default=os.environ.get("POSTGRES_PASSWORD", ""),
    )
    parser.add_argument("--database", default=os.environ.get("POSTGRES_DB", "cropfed"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.password:
        raise SystemExit("POSTGRES_PASSWORD environment variable required")

    source = _docker_ps("postgres:17-alpine")
    print(f"== Backing up PostgreSQL container {source} -> {args.dump} ==")
    backup(source, args.dump, user=args.user, database=args.database)
    if not args.dump.is_file() or args.dump.stat().st_size == 0:
        raise SystemExit(f"dump was not written: {args.dump}")
    sha = hashlib.sha256(args.dump.read_bytes()).hexdigest()
    print(f"   dump bytes={args.dump.stat().st_size} sha256={sha[:16]}…")

    print("== Restoring dump into a fresh PostgreSQL container ==")
    suffix = secrets.token_hex(4)
    restore(
        image=args.image,
        dump_path=args.dump,
        user=args.user,
        password=args.password,
        suffix=suffix,
    )
    print("Backup and restore verified.")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
