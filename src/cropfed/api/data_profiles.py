"""Read privacy-safe client class-distribution summaries for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cropfed.api.settings import Settings
from cropfed.constants import taxonomy_from_scope
from cropfed.data.profiles import FULL_PROFILE_SPECS, MVP_PROFILE_SPECS


def data_profile_payload(application_settings: Settings) -> dict[str, Any]:
    """Return only aggregate counts from server-owned partition summaries."""

    data_root = _data_root(application_settings)
    taxonomy = taxonomy_from_scope(application_settings.taxonomy_scope)
    profile_specs = (
        FULL_PROFILE_SPECS
        if taxonomy.scope == "plantvillage-full"
        else MVP_PROFILE_SPECS
    )
    items = [
        _read_profile(
            data_root / spec.name / "clients" / "partition_summary.json",
            spec,
            num_classes=len(taxonomy.class_names),
        )
        for spec in profile_specs
    ]
    return {
        "items": items,
        "taxonomy_scope": taxonomy.scope,
        "class_order": list(taxonomy.class_names),
        "privacy": {
            "contains_image_bytes": False,
            "contains_local_image_paths": False,
        },
    }


def _read_profile(path: Path, spec: Any, *, num_classes: int) -> dict[str, Any]:
    expected_skew_type = (
        "quantity"
        if spec.quantity_skew
        else "feature"
        if spec.partition_kind == "feature_skew"
        else "label"
        if spec.partition_kind == "dirichlet"
        else "none"
    )
    base = {
        "name": spec.name,
        "partition_kind": spec.partition_kind,
        "skew_type": expected_skew_type,
        "dirichlet_alpha": spec.dirichlet_alpha,
        "quantity_skew": spec.quantity_skew,
        "feature_skew_strength": (
            spec.feature_skew_strength
            if spec.partition_kind == "feature_skew"
            else None
        ),
    }
    if not path.is_file():
        return {**base, "available": False, "status": "missing", "clients": []}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        clients = _validated_clients(document, num_classes=num_classes)
        if document.get("partition_kind") != spec.partition_kind:
            raise ValueError("partition kind mismatch")
        actual_alpha = document.get("dirichlet_alpha")
        if spec.dirichlet_alpha is not None and float(actual_alpha) != spec.dirichlet_alpha:
            raise ValueError("Dirichlet alpha mismatch")
        if bool(document.get("quantity_skew", False)) != spec.quantity_skew:
            raise ValueError("quantity skew mismatch")
        actual_skew_type = document.get("skew_type", expected_skew_type)
        if actual_skew_type != expected_skew_type:
            raise ValueError("skew type mismatch")
        if spec.partition_kind == "feature_skew" and float(
            document.get("feature_skew_strength")
        ) != spec.feature_skew_strength:
            raise ValueError("feature skew strength mismatch")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {**base, "available": False, "status": "invalid", "clients": []}
    return {
        **base,
        "available": True,
        "status": "ready",
        "num_clients": len(clients),
        "num_samples": sum(row["num_samples"] for row in clients),
        "clients": clients,
    }


def _validated_clients(
    document: object,
    *,
    num_classes: int,
) -> list[dict[str, Any]]:
    if not isinstance(document, dict) or document.get("num_clients") != 4:
        raise ValueError("MVP profile must contain four clients")
    if document.get("partition_kind") not in {"iid", "dirichlet", "feature_skew"}:
        raise ValueError("invalid partition kind")
    source_clients = document.get("clients")
    if not isinstance(source_clients, list) or len(source_clients) != 4:
        raise ValueError("invalid client list")

    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source in source_clients:
        if not isinstance(source, dict):
            raise ValueError("invalid client row")
        client_id = int(source["client_id"])
        counts = [int(value) for value in source["class_counts"]]
        if client_id in seen or client_id not in range(4):
            raise ValueError("invalid client id")
        if len(counts) != num_classes or any(value < 0 for value in counts):
            raise ValueError("invalid class counts")
        num_samples = int(source["num_samples"])
        num_train = int(source["num_train"])
        num_validation = int(source["num_validation"])
        if (
            num_samples != sum(counts)
            or num_samples != num_train + num_validation
            or min(num_samples, num_train, num_validation) < 1
        ):
            raise ValueError("inconsistent client totals")
        seen.add(client_id)
        result.append(
            {
                "client_id": client_id,
                "num_samples": num_samples,
                "num_train": num_train,
                "num_validation": num_validation,
                "class_counts": counts,
                "class_proportions": [value / num_samples for value in counts],
            }
        )
    return sorted(result, key=lambda row: row["client_id"])


def _data_root(application_settings: Settings) -> Path:
    configured = application_settings.flower_data_root.expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (application_settings.flower_project_dir.expanduser().resolve() / configured).resolve()
