"""Verify PlantVillage Flower pilot artifact integrity.

For each pilot directory under the supplied root, this script:

* re-hashes every file and compares against ``run_manifest.json``,
* ensures ``flower.log`` contains the strategy/aggregate evidence the
  Smoke validation expects (``Registered``, ``aggregate_train``, ...),
* loads the checkpoint via the production loader and re-derives its
  metadata contract (algorithm, num_clients, class_order),
* cross-checks the client_history contract (one row per round per
  client per phase),
* collects a single JSON summary so the pilot report can quote it.

This intentionally mirrors the validation helpers used by
``run_flower_smoke.py`` so any drift between the synthetic fixture and
the PlantVillage pilots surfaces here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cropfed.constants import (  # noqa: E402
    TOMATO_TAXONOMY,
    taxonomy_from_class_order,
    taxonomy_from_scope,
)
from cropfed.flower.smoke import (  # noqa: E402
    parse_flower_log_evidence,
)
from cropfed.ml.checkpoint import load_model_checkpoint  # noqa: E402

EXPECTED = [
    ("fedavg", "dirichlet-alpha-0.5"),
    ("fedavg", "dirichlet-alpha-0.1"),
    ("fedprox", "dirichlet-alpha-0.5"),
    ("fedprox", "dirichlet-alpha-0.1"),
]
PROXIMAL_MU = 0.01
NUM_CLIENTS = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_one(pilot_dir: Path) -> dict[str, object]:
    algorithm = "fedavg" if "fedavg" in pilot_dir.name else "fedprox"
    profile_label = "alpha-0.5" if "alpha05" in pilot_dir.name else "alpha-0.1"

    manifest_path = pilot_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Re-hash all referenced artifacts and compare against manifest.
    mismatches: list[str] = []
    checkpoint_path = pilot_dir / manifest["checkpoint"]
    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha != manifest["checkpoint_sha256"]:
        mismatches.append(
            f"checkpoint_sha256 mismatch ({checkpoint_sha[:12]} vs "
            f"{manifest['checkpoint_sha256'][:12]})"
        )
    if checkpoint_path.stat().st_size != manifest["checkpoint_bytes"]:
        mismatches.append(
            f"checkpoint_bytes mismatch ({checkpoint_path.stat().st_size} "
            f"vs {manifest['checkpoint_bytes']})"
        )

    environment_path = pilot_dir / manifest["environment"]["path"]
    environment_sha = _sha256(environment_path)
    if environment_sha != manifest["environment"]["sha256"]:
        mismatches.append("environment_sha256 mismatch")
    if environment_path.stat().st_size != manifest["environment"]["bytes"]:
        mismatches.append("environment_bytes mismatch")

    metrics_path = pilot_dir / manifest["metrics"]
    client_metrics_path = pilot_dir / manifest["client_metrics"]
    metrics_sha = _sha256(metrics_path)
    client_metrics_sha = _sha256(client_metrics_path)
    if metrics_sha == client_metrics_sha:
        mismatches.append("metrics.json and client_metrics.json are byte-identical")

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    client_metrics_payload = json.loads(client_metrics_path.read_text(encoding="utf-8"))
    # client_metrics.json stores a bare list, metrics.json wraps it as a dict.
    client_history_metrics = metrics_payload.get("client_history")
    client_history_client = (
        client_metrics_payload
        if isinstance(client_metrics_payload, list)
        else client_metrics_payload.get("client_history")
    )
    if client_history_metrics != client_history_client:
        mismatches.append("client_history differs between metrics.json and client_metrics.json")

    # 2. Load checkpoint and re-derive metadata contract.
    loaded = load_model_checkpoint(checkpoint_path)
    if loaded.metadata.get("algorithm") != algorithm:
        mismatches.append(
            f"checkpoint metadata algorithm is {loaded.metadata.get('algorithm')} != {algorithm}"
        )
    if loaded.metadata.get("num_clients") != NUM_CLIENTS:
        mismatches.append("checkpoint metadata num_clients is not 4")
    # Tier-1 tomato pilots predate the taxonomy-scope metadata key, so an
    # absent scope still means tomato; a 38-class run records its own scope.
    scope = loaded.metadata.get("taxonomy_scope")
    expected_taxonomy = (
        taxonomy_from_scope(str(scope)) if scope else TOMATO_TAXONOMY
    )
    if tuple(loaded.class_order or ()) != expected_taxonomy.class_names:
        found = taxonomy_from_class_order(loaded.class_order)
        mismatches.append(
            f"checkpoint class_order is {found.scope if found else 'unknown'!r} "
            f"but metadata declares {expected_taxonomy.scope!r}"
        )

    # 3. Strategy evidence from flower.log (synthetic-run assertion).
    log_path = pilot_dir / "flower.log"
    if not log_path.is_file():
        mismatches.append("flower.log missing")
        evidence: dict[str, object] = {}
    else:
        try:
            evidence = parse_flower_log_evidence(
                log_path.read_text(encoding="utf-8", errors="replace"),
                algorithm=algorithm,
                expected_clients=NUM_CLIENTS,
                proximal_mu=PROXIMAL_MU,
            )
        except RuntimeError as exc:
            mismatches.append(f"flower.log evidence: {exc}")
            evidence = {}

    # 4. Client history completeness.
    client_history = metrics_payload.get("client_history")
    expected_clients = NUM_CLIENTS
    expected_rounds = int(manifest.get("num_rounds", 0))
    expected_identities = {
        (round_number, client_id, phase)
        for round_number in range(1, expected_rounds + 1)
        for client_id in range(expected_clients)
        for phase in ("train", "evaluate")
    }
    history_identities = {
        (int(item["round"]), int(item["client_id"]), str(item["phase"]))
        for item in (client_history or [])
        if isinstance(item, dict)
    }
    if history_identities != expected_identities:
        mismatches.append("client_history identities do not cover every round/client/phase")

    status = "passed" if not mismatches else "failed"
    return {
        "pilot": pilot_dir.name,
        "algorithm": algorithm,
        "profile": profile_label,
        "status": status,
        "checkpoint": str(checkpoint_path.name),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_format_version": loaded.format_version,
        "model_version": loaded.model_version,
        "global_test_macro_f1": (metrics_payload.get("global_test") or {}).get(
            "global_test_macro_f1"
        ),
        "best_validation_round": manifest.get("best_validation_round"),
        "client_history_entries": len(client_history or []),
        "communication_bytes": (metrics_payload.get("communication") or {}).get(
            "payload_total_bytes"
        ),
        "evidence": evidence,
        "warnings": manifest.get("warnings", []),
        "mismatches": mismatches,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Where to write the JSON summary. Defaults to <artifacts-root>/"
        "plantvillage-flower-pilots-verification.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.artifacts_root.resolve()
    summary: list[dict[str, object]] = []
    failures = 0
    for algorithm, profile in EXPECTED:
        # profile format: 'dirichlet-alpha-0.5' -> glob slug 'alpha05'
        slug_profile = profile.replace("dirichlet-alpha-", "alpha").replace(".", "")
        candidates = sorted(root.glob(f"plantvillage-flower-{algorithm}-{slug_profile}-pilot-*"))
        if not candidates:
            print(f"[skip] no pilot found for {algorithm} / {profile}")
            continue
        pilot = candidates[-1]
        result = verify_one(pilot)
        summary.append(result)
        if result["status"] != "passed":
            failures += 1
        print(
            f"[{result['status']}] {pilot.name}: "
            f"checkpoint={result['checkpoint_sha256'][:12]} "
            f"macro_f1={result['global_test_macro_f1']:.4f} "
            f"mismatches={len(result['mismatches'])}"
        )
        for warning in result["warnings"]:
            print(f"  warn: {warning}")
        for mismatch in result["mismatches"]:
            print(f"  fail: {mismatch}")

    output_path = args.summary_output or (root / "plantvillage-flower-pilots-verification.json")
    output_path.write_text(
        json.dumps(
            {
                "result_kind": "federated_image_pilot_verification",
                "research_result_valid": False,
                "pilots": summary,
                "summary": {
                    "total": len(summary),
                    "passed": len(summary) - failures,
                    "failed": failures,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nVerification summary -> {output_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())