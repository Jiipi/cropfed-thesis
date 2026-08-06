"""Run centralized and local-only image baselines on a synthetic fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from create_flower_smoke_fixture import create_fixture  # noqa: E402

from cropfed.constants import TOMATO_CLASSES  # noqa: E402
from cropfed.data.audit import audit_prepared_data, write_audit_report  # noqa: E402
from cropfed.experiments.centralized import run_centralized  # noqa: E402
from cropfed.experiments.local_only import run_local_only  # noqa: E402


def run_smoke(output_root: Path) -> dict[str, object]:
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"baseline smoke output must be new: {output_root}")
    fixture_root = output_root / "fixture"
    create_fixture(fixture_root)
    processed = fixture_root / "processed"
    audit = audit_prepared_data(
        train_manifest=processed / "train_manifest.csv",
        test_manifest=processed / "test_manifest.csv",
        client_data_root=fixture_root / "clients",
        num_clients=4,
        class_names=TOMATO_CLASSES,
    )
    write_audit_report(audit, output_root / "data_audit.json")
    if audit["status"] != "passed":
        raise RuntimeError("baseline fixture audit failed")

    centralized = run_centralized(
        train_manifest=processed / "pooled_train_manifest.csv",
        validation_manifest=processed / "validation_manifest.csv",
        test_manifest=processed / "test_manifest.csv",
        model_name="mobilenet_v2",
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        pretrained=False,
        seed=2026,
        output_dir=output_root / "centralized",
        research_result_valid=False,
    )
    local_only = run_local_only(
        client_data_root=fixture_root / "clients",
        test_manifest=processed / "test_manifest.csv",
        num_clients=4,
        model_name="mobilenet_v2",
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        pretrained=False,
        seed=2026,
        output_dir=output_root / "local-only",
        partition_kind="dirichlet",
        dirichlet_alpha=0.5,
        research_result_valid=False,
    )
    summary: dict[str, object] = {
        "status": "passed",
        "result_kind": "synthetic_images_baseline_integration_only",
        "research_result_valid": False,
        "data_audit_status": audit["status"],
        "centralized": {
            "num_train": centralized["num_train"],
            "num_test": centralized["num_test"],
            "checkpoint_bytes": centralized["checkpoint_bytes"],
        },
        "local_only": {
            "num_clients": local_only["num_clients"],
            "client_checkpoints": len(local_only["clients"]),
            "num_train_total": sum(
                int(client["num_train"]) for client in local_only["clients"]
            ),
        },
    }
    summary_path = output_root / "baseline_smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {**summary, "summary": str(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output_root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
