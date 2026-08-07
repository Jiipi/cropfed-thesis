"""Migrate prepared profile manifests from absolute paths to portable ones.

The six 38-class profiles on disk cost hours of partitioning and were written
with absolute ``F:\\project\\cropfed-thesis\\data\\raw\\...`` paths, which name
the laptop that produced them.  Rewriting those paths relative to the dataset
root makes the same artifacts describe the same experiment on the GPU machine.

Two properties make this safe to do in place:

* ``image_id`` is ``sha1(relative_path)[:16]``, so the *stored* id proves whether
  a row's rewritten path is the one the preparation step actually used.  Every
  row is checked before anything is written; a single mismatch aborts the whole
  migration rather than leaving a half-rewritten profile set.
* Only the CSV manifests hold paths.  ``partition_summary.json``,
  ``data_audit.json`` and ``profile.json`` carry counts and hashes (D-025), so
  the class distributions the dashboard reads are untouched.

Rewriting a manifest changes its SHA-256, and those hashes appear in both
``profile.json`` and ``profiles_index.json``.  This script updates them, so
``--verify`` afterwards is what confirms the recorded checksums match the bytes
on disk.  It does not re-run the image audit: ``data_audit.json`` keeps its
original manifest hashes and is rewritten only by ``audit-data``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

MANIFEST_NAMES = (
    "train_manifest.csv",
    "test_manifest.csv",
    "pooled_train_manifest.csv",
    "validation_manifest.csv",
)
#: Maps a manifest file name to the ``*_sha256`` key recording it.
CHECKSUM_KEYS = {
    "train_manifest.csv": "train_manifest_sha256",
    "test_manifest.csv": "test_manifest_sha256",
    "pooled_train_manifest.csv": "pooled_train_manifest_sha256",
    "validation_manifest.csv": "validation_manifest_sha256",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_root(stored: str, root_prefixes: tuple[str, ...]) -> str | None:
    """Strip whichever spelling of the dataset root this row happens to use.

    Compared case-insensitively on the raw string rather than via ``resolve()``:
    the six profiles hold ~250k rows in total, and touching the filesystem once
    per row turns a seconds-long migration into a multi-minute one.
    """

    normalized = stored.strip().replace("\\", "/")
    lowered = normalized.lower()
    for prefix in root_prefixes:
        if lowered.startswith(prefix):
            return normalized[len(prefix):].lstrip("/")
    return None


def _root_prefixes(dataset_root: Path) -> tuple[str, ...]:
    posix = dataset_root.as_posix().rstrip("/")
    return tuple(
        sorted(
            {f"{posix.lower()}/", f"{str(dataset_root).replace(chr(92), '/').lower()}/"}
        )
    )


def migrate_manifest(
    path: Path,
    *,
    dataset_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Rewrite one manifest, refusing to write if any row fails its id check."""

    prefixes = _root_prefixes(dataset_root)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    rewritten = 0
    already_relative = 0
    problems: list[dict[str, str]] = []
    for row in rows:
        stored = row["path"]
        if not (stored[1:3] in (":/", ":\\") or stored.startswith("/")):
            already_relative += 1
            continue
        relative = _relative_to_root(stored, prefixes)
        if relative is None:
            problems.append({"image_id": row["image_id"], "reason": "outside_root"})
            continue
        expected_id = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]
        if expected_id != row["image_id"]:
            # The stored id was derived from the relative path at preparation
            # time. A mismatch means this root is not the one that produced the
            # profile, so rewriting would silently repoint the manifest.
            problems.append({"image_id": row["image_id"], "reason": "image_id_mismatch"})
            continue
        row["path"] = relative
        rewritten += 1

    report: dict[str, Any] = {
        "manifest": path.name,
        "rows": len(rows),
        "rewritten": rewritten,
        "already_relative": already_relative,
        "problems": problems[:10],
        "problem_count": len(problems),
    }
    if problems:
        return report
    if rewritten and not dry_run:
        # Write beside the original and replace atomically, so an interrupted run
        # cannot leave a truncated manifest where a valid one used to be.
        temporary = path.with_suffix(".csv.migrating")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    if not dry_run:
        report["sha256"] = _sha256_file(path)
    return report


def iter_profile_manifests(profile_root: Path):
    for name in MANIFEST_NAMES:
        candidate = profile_root / name
        if candidate.is_file():
            yield candidate
    yield from sorted(profile_root.glob("clients/*/train_manifest.csv"))
    yield from sorted(profile_root.glob("clients/*/val_manifest.csv"))


def migrate_profile_set(
    profiles_root: Path,
    *,
    dataset_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    index_path = profiles_root / "profiles_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"no profile set at {profiles_root}")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    profile_reports: list[dict[str, Any]] = []
    for row in index["profiles"]:
        profile_root = profiles_root / row["name"]
        manifests = [
            migrate_manifest(path, dataset_root=dataset_root, dry_run=dry_run)
            for path in iter_profile_manifests(profile_root)
        ]
        profile_reports.append({"profile": row["name"], "manifests": manifests})

    problems = sum(
        entry["problem_count"]
        for report in profile_reports
        for entry in report["manifests"]
    )
    if problems:
        raise SystemExit(
            f"refusing to migrate: {problems} rows failed the image_id check; "
            f"--dataset-root {dataset_root} is probably not the root that "
            "produced this profile set"
        )

    if not dry_run:
        _update_recorded_checksums(profiles_root, index, index_path)

    return {
        "profiles_root": str(profiles_root),
        "dataset_root": dataset_root.as_posix(),
        "dry_run": dry_run,
        "total_rewritten": sum(
            entry["rewritten"]
            for report in profile_reports
            for entry in report["manifests"]
        ),
        "profiles": profile_reports,
    }


def _update_recorded_checksums(
    profiles_root: Path,
    index: dict[str, Any],
    index_path: Path,
) -> None:
    """Bring profile.json and profiles_index.json back in step with the bytes."""

    for row in index["profiles"]:
        profile_root = profiles_root / row["name"]
        digests = {
            key: _sha256_file(profile_root / name)
            for name, key in CHECKSUM_KEYS.items()
            if (profile_root / name).is_file()
        }
        row.update(digests)

        profile_path = profile_root / "profile.json"
        if profile_path.is_file():
            metadata = json.loads(profile_path.read_text(encoding="utf-8"))
            metadata.update(digests)
            profile_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            row["profile_sha256"] = _sha256_file(profile_path)

    train_hashes = {row["train_manifest_sha256"] for row in index["profiles"]}
    test_hashes = {row["test_manifest_sha256"] for row in index["profiles"]}
    index["shared_split_invariants"] = {
        "same_train_manifest": len(train_hashes) == 1,
        "same_global_test_manifest": len(test_hashes) == 1,
        "train_manifest_sha256": next(iter(sorted(train_hashes))),
        "test_manifest_sha256": next(iter(sorted(test_hashes))),
    }
    if not index["shared_split_invariants"]["same_global_test_manifest"]:
        raise SystemExit(
            "migration broke D-024: the profiles no longer share one global test "
            "set; restore from backup and investigate before using these artifacts"
        )
    index["manifest_paths"] = "relative_to_dataset_root"
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def verify_profile_set(profiles_root: Path, *, dataset_root: Path) -> dict[str, Any]:
    """Confirm paths are relative, checksums match, and every image opens."""

    index = json.loads(
        (profiles_root / "profiles_index.json").read_text(encoding="utf-8")
    )
    findings: list[str] = []
    checked_rows = 0
    missing_files = 0
    for row in index["profiles"]:
        profile_root = profiles_root / row["name"]
        for name, key in CHECKSUM_KEYS.items():
            manifest = profile_root / name
            if not manifest.is_file():
                continue
            if _sha256_file(manifest) != row.get(key):
                findings.append(f"{row['name']}/{name}: recorded sha256 does not match")
        for manifest in iter_profile_manifests(profile_root):
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                for entry in csv.DictReader(handle):
                    stored = entry["path"]
                    checked_rows += 1
                    if stored[1:3] in (":/", ":\\") or stored.startswith("/"):
                        findings.append(
                            f"{row['name']}/{manifest.name}: still absolute"
                        )
                        break
                    if not (dataset_root / stored).is_file():
                        missing_files += 1
                        if missing_files < 5:
                            findings.append(
                                f"{row['name']}/{manifest.name}: unresolvable row"
                            )
    return {
        "profiles_root": str(profiles_root),
        "dataset_root": dataset_root.as_posix(),
        "rows_checked": checked_rows,
        "missing_files": missing_files,
        "status": "passed" if not findings else "failed",
        "findings": findings[:20],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="rewrite prepared manifests to be portable across machines"
    )
    parser.add_argument(
        "--profiles-root",
        type=Path,
        default=Path("data/flower-profiles-full"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="root the rewritten paths are relative to, e.g. "
        "data/raw/PlantVillage-Dataset/raw/color",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="where to write the JSON report; defaults to artifacts/, not the "
        "profile set, because the report names local paths and the profile set "
        "is what gets handed to another machine (D-025)",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        help="copy profiles_index.json and every profile.json here before writing",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check an already-migrated set instead of migrating",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profiles_root = args.profiles_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    if args.verify:
        report = verify_profile_set(profiles_root, dataset_root=dataset_root)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["status"] == "passed" else 2

    if args.backup and not args.dry_run:
        # Only the small JSON files are copied: the manifests are replaced
        # atomically, and duplicating 175 MB of CSV to guard a reversible
        # string rewrite is not a trade worth making.
        args.backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(profiles_root / "profiles_index.json", args.backup)
        for child in sorted(profiles_root.iterdir()):
            if (child / "profile.json").is_file():
                shutil.copy2(
                    child / "profile.json", args.backup / f"{child.name}.profile.json"
                )

    report = migrate_profile_set(
        profiles_root, dataset_root=dataset_root, dry_run=args.dry_run
    )
    summary_path = args.summary or (
        Path("artifacts") / "manifest_migration.json"
    )
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report["summary"] = str(summary_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
