"""Create deterministic local images/manifests for Flower integration smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from cropfed.constants import TOMATO_CLASSES
from cropfed.data.audit import audit_prepared_data, write_audit_report
from cropfed.data.manifest import ImageRecord, write_manifest


def create_fixture(output_root: Path, *, num_clients: int = 4) -> dict[str, object]:
    """Create a tiny, complete 10-class fixture without external data."""

    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"fixture output must be new or empty: {output_root}"
        )
    if num_clients != 4:
        raise ValueError("the canonical Flower smoke fixture requires four clients")

    image_root = output_root / "images"
    processed_root = output_root / "processed"
    client_root = output_root / "clients"
    train_records: list[ImageRecord] = []

    for client_id in range(num_clients):
        local_records: list[ImageRecord] = []
        for local_index in range(5):
            label_id = (client_id * 5 + local_index) % len(TOMATO_CLASSES)
            relative = Path("images") / f"client-{client_id}-{local_index}.png"
            image_path = output_root / relative
            _write_image(
                image_path,
                label_id=label_id,
                variant=(client_id * 10) + local_index,
            )
            record = _record(
                output_root=output_root,
                relative_path=relative,
                label_id=label_id,
                split="train",
            )
            train_records.append(record)
            local_records.append(record)

        local_train = [
            _replace_split(record, "local_train")
            for record in local_records[:-1]
        ]
        local_validation = [_replace_split(local_records[-1], "local_val")]
        client_dir = client_root / f"client_{client_id}"
        write_manifest(local_train, client_dir / "train_manifest.csv")
        write_manifest(local_validation, client_dir / "val_manifest.csv")

    test_records: list[ImageRecord] = []
    for label_id in range(len(TOMATO_CLASSES)):
        relative = Path("images") / f"global-test-{label_id}.png"
        _write_image(image_root.parent / relative, label_id=label_id, variant=100 + label_id)
        test_records.append(
            _record(
                output_root=output_root,
                relative_path=relative,
                label_id=label_id,
                split="test",
            )
        )

    train_manifest = processed_root / "train_manifest.csv"
    test_manifest = processed_root / "test_manifest.csv"
    write_manifest(train_records, train_manifest)
    write_manifest(test_records, test_manifest)
    report = audit_prepared_data(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        client_data_root=client_root,
        num_clients=num_clients,
    )
    audit_path = processed_root / "data_audit.json"
    write_audit_report(report, audit_path)
    if report["status"] != "passed":
        raise RuntimeError(f"generated fixture failed its own audit: {audit_path}")

    summary: dict[str, object] = {
        "fixture_kind": "synthetic_images_for_integration_only",
        "output_root": output_root.as_posix(),
        "client_data_root": client_root.as_posix(),
        "train_manifest": train_manifest.as_posix(),
        "test_manifest": test_manifest.as_posix(),
        "audit_report": audit_path.as_posix(),
        "num_clients": num_clients,
        "num_train": len(train_records),
        "num_test": len(test_records),
    }
    (output_root / "fixture.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _record(
    *,
    output_root: Path,
    relative_path: Path,
    label_id: int,
    split: str,
) -> ImageRecord:
    normalized = relative_path.as_posix()
    return ImageRecord(
        image_id=hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16],
        path=str((output_root / relative_path).resolve()),
        label_id=label_id,
        label_name=TOMATO_CLASSES[label_id],
        split=split,
    )


def _replace_split(record: ImageRecord, split: str) -> ImageRecord:
    return ImageRecord(
        image_id=record.image_id,
        path=record.path,
        label_id=record.label_id,
        label_name=record.label_name,
        split=split,
    )


def _write_image(path: Path, *, label_id: int, variant: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    background = (
        (37 * label_id + 11 * variant) % 256,
        (83 * label_id + 7 * variant) % 256,
        (19 * label_id + 13 * variant) % 256,
    )
    image = Image.new("RGB", (72, 72), background)
    draw = ImageDraw.Draw(image)
    inset = 5 + (variant % 12)
    draw.ellipse(
        (inset, inset, 71 - inset, 71 - inset),
        outline=((label_id * 23) % 256, 255 - background[1], background[2]),
        width=3,
    )
    draw.text((6, 56), f"{label_id}:{variant}", fill=(255, 255, 255))
    image.save(path, format="PNG")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = create_fixture(args.output_root, num_clients=args.clients)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
