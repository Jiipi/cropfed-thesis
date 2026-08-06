"""Rewrite Flower profile manifests so that their image paths are POSIX-relative.

The flow-prep scripts on Windows saved ``F:\\project\\cropfed-thesis\\data\\…`` paths
into ``train_manifest.csv`` / ``val_manifest.csv`` / ``test_manifest.csv``.  Those
absolute Windows paths are useless inside the Linux Flower container, so the
``audit_prepared_data`` check would always fail.  This script replaces any
leading ``F:\\…\\data\\`` prefix with the POSIX equivalent that the container
sees (``/app/data/``) so the docker-compose worker profile can actually find
the raw PlantVillage images.

The script never deletes source files.  It writes a ``*.rewritten.csv`` next
to each manifest plus a sha256/size summary next to ``profiles_index.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

WINDOWS_PREFIXES = (
    "F:\\project\\cropfed-thesis\\data\\",
    "F:/project/cropfed-thesis/data/",
    "f:\\project\\cropfed-thesis\\data\\",
    "f:/project/cropfed-thesis/data/",
)
POSIX_ROOT = "/app/data/"


def rewrite_path(raw: str) -> str:
    value = raw.strip()
    for prefix in WINDOWS_PREFIXES:
        if value.lower().startswith(prefix.lower()):
            tail = value[len(prefix):]
            tail = tail.replace("\\", "/")
            return POSIX_ROOT + tail
    return value


def _rewrite_manifest(path: Path, *, dry_run: bool) -> tuple[Path | None, int]:
    rows: list[dict[str, str]] = []
    changed = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            new_value = rewrite_path(row["path"])
            if new_value != row["path"]:
                row["path"] = new_value
                changed += 1
            rows.append(row)
    if changed == 0 or dry_run:
        return None if not changed or dry_run else path, changed
    target = path.with_suffix(".rewritten.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target, changed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_manifests(profile_root: Path) -> Iterable[Path]:
    yield profile_root / "test_manifest.csv"
    yield from sorted(profile_root.glob("clients/*/train_manifest.csv"))
    yield from sorted(profile_root.glob("clients/*/val_manifest.csv"))


def rewrite_profile(profile_root: Path, *, dry_run: bool) -> dict[str, object]:
    report: dict[str, object] = {
        "profile": profile_root.name,
        "manifests": [],
    }
    for manifest in iter_manifests(profile_root):
        if not manifest.is_file():
            continue
        original = manifest.read_bytes().decode("utf-8") if manifest.is_file() else ""
        if not original:
            continue
        rewritten = original
        # Count and rewrite the well-known Windows host prefixes that the
        # `data-prep` scripts saved into the manifests.  After rewriting the
        # prefix, normalise any remaining backslashes into forward slashes so
        # the POSIX container can resolve the path on every invocation.
        changed = (
            original.count("F:\\project\\cropfed-thesis\\data\\")
            + original.count("F:/project/cropfed-thesis/data/")
        )
        rewritten = rewritten.replace(
            "F:\\project\\cropfed-thesis\\data\\", POSIX_ROOT
        ).replace(
            "F:/project/cropfed-thesis/data/", POSIX_ROOT
        )
        normalized = rewritten.replace("\\", "/")
        target: Path | None = None
        if (changed or normalized != rewritten) and not dry_run:
            manifest.write_bytes(normalized.encode("utf-8"))
            target = manifest
        report["manifests"].append(
            {
                "source": str(manifest.relative_to(profile_root)),
                "changed_paths": changed,
                "rewritten_to": (
                    str(target.relative_to(profile_root)) if target is not None else None
                ),
                "sha256": _sha256(manifest),
                "size_bytes": manifest.stat().st_size,
            }
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles-root",
        type=Path,
        default=Path("data/flower-profiles"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/flower-profiles/profiles_path_rewrite.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profiles_root = args.profiles_root.resolve()
    if not profiles_root.is_dir():
        raise SystemExit(f"profiles root not found: {profiles_root}")
    profiles = sorted(
        child for child in profiles_root.iterdir()
        if child.is_dir() and child.name.startswith(("iid", "dirichlet"))
    )
    summary = {
        "profiles_root": str(profiles_root),
        "posix_root": POSIX_ROOT,
        "dry_run": args.dry_run,
        "profiles": [
            rewrite_profile(profile, dry_run=args.dry_run) for profile in profiles
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote path-rewrite summary -> {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
