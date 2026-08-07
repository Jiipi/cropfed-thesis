"""Tests for newly added API endpoints (clients, predict, checkpoints, compare, export)."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from cropfed.api.main import create_app
from cropfed.api.models import (
    ClientRoundMetricRecord,
    ExperimentRecord,
    ExperimentRoundRecord,
)
from cropfed.api.settings import Settings
from cropfed.constants import taxonomy_from_scope


def make_test_client(
    *,
    settings: Settings | None = None,
    prediction_executor=None,
    return_engine: bool = False,
):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    test_settings = settings or Settings(
        database_url="sqlite://",
        cors_origins=("http://localhost:5173",),
        api_auth_enabled=False,
    )
    app = create_app(
        database_engine=engine,
        application_settings=test_settings,
        prediction_executor=prediction_executor,
    )
    client = TestClient(app)
    return (client, engine) if return_engine else client


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (20, 80, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def fake_prediction(**_):
    return {
        "image_name": "temporary.png",
        "crop": "Tomato",
        "predicted_class_id": 0,
        "predicted_label": "Healthy",
        "predicted_group": "healthy",
        "confidence": 0.9,
        "model": "mobilenet_v2",
        "model_version": "test",
        "checkpoint_format_version": 1,
        "predictions": [
            {
                "class_id": 0,
                "label": "Healthy",
                "group": "healthy",
                "confidence": 0.9,
            }
        ],
        "inference_ms": 1.5,
        "warning": "test only",
        "image_uploaded": False,
    }


def test_client_crud():
    client = make_test_client()

    # List initially empty
    resp = client.get("/api/v1/clients")
    assert resp.status_code == 200
    assert resp.json() == []

    # Create a client
    resp = client.post(
        "/api/v1/clients",
        json={"name": "Nông trại Củ Chi", "description": "Trang trại mẫu", "partition_id": 0},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Nông trại Củ Chi"
    assert data["status"] == "registered"
    client_id = data["id"]

    # Update status
    resp = client.patch(
        f"/api/v1/clients/{client_id}/status",
        json={"status": "connected"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "connected"


def test_list_checkpoints():
    client = make_test_client()
    resp = client.get("/api/v1/checkpoints")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_experiments_compare_validation():
    client = make_test_client()
    # Fails if fewer than 2 IDs
    resp = client.get("/api/v1/experiments/compare?ids=id1")
    assert resp.status_code == 422


def test_export_csv_validation():
    client = make_test_client()
    resp = client.get("/api/v1/experiments/export-csv")
    assert resp.status_code == 422


def test_predict_requires_pinned_checkpoint():
    client = make_test_client(prediction_executor=fake_prediction)

    response = client.post(
        "/api/v1/predict",
        files={"image": ("leaf.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 503


def test_predict_happy_path_uses_verified_checkpoint_and_original_filename():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_root = Path(directory)
        checkpoint = checkpoint_root / "deployed.pt"
        checkpoint.write_bytes(b"pinned-test-checkpoint")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        client = make_test_client(
            settings=Settings(
                database_url="sqlite://",
                cors_origins=("http://localhost:5173",),
                checkpoint_dir=checkpoint_root,
                deployed_checkpoint=Path("deployed.pt"),
                deployed_checkpoint_sha256=digest,
            ),
            prediction_executor=fake_prediction,
        )

        response = client.post(
            "/api/v1/predict",
            files={"image": ("tomato-leaf.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["image_name"] == "tomato-leaf.png"
    assert response.json()["checkpoint_sha256"] == digest
    assert response.json()["image_uploaded"] is True


def test_predict_rejects_invalid_magic_mime_and_oversized_upload():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_root = Path(directory)
        checkpoint = checkpoint_root / "deployed.pt"
        checkpoint.write_bytes(b"pinned-test-checkpoint")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        client = make_test_client(
            settings=Settings(
                database_url="sqlite://",
                cors_origins=(),
                checkpoint_dir=checkpoint_root,
                deployed_checkpoint=Path("deployed.pt"),
                deployed_checkpoint_sha256=digest,
            ),
            prediction_executor=fake_prediction,
        )

        bad_magic = client.post(
            "/api/v1/predict",
            files={"image": ("leaf.png", b"not-a-png", "image/png")},
        )
        bad_mime = client.post(
            "/api/v1/predict",
            files={"image": ("leaf.png", png_bytes(), "image/jpeg")},
        )
        oversized = client.post(
            "/api/v1/predict",
            files={"image": ("leaf.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")},
        )

    assert bad_magic.status_code == 422
    assert bad_mime.status_code == 422
    assert oversized.status_code == 413


def test_checkpoint_list_marks_only_verified_deployment_without_absolute_path():
    import torch

    from cropfed.ml.checkpoint import save_model_checkpoint

    # The checkpoint class order is part of the deployment contract: the endpoint
    # only marks a checkpoint eligible when the order matches the configured
    # scope, so build the fixture from that scope instead of a fixed taxonomy.
    settings = Settings(database_url="sqlite://", cors_origins=())
    class_order = taxonomy_from_scope(settings.taxonomy_scope).class_names

    with tempfile.TemporaryDirectory() as directory:
        checkpoint_root = Path(directory)
        checkpoint = checkpoint_root / "models" / "deployed.pt"
        info = save_model_checkpoint(
            checkpoint,
            torch.nn.Linear(2, len(class_order)),
            model_name="tiny",
            metadata={"experiment_type": "test"},
            class_order=class_order,
        )
        client = make_test_client(
            settings=replace(
                settings,
                checkpoint_dir=checkpoint_root,
                deployed_checkpoint=Path("models/deployed.pt"),
                deployed_checkpoint_sha256=str(info["sha256"]),
            )
        )

        response = client.get("/api/v1/checkpoints")

    assert response.status_code == 200
    item = response.json()[0]
    assert item["path"] == "models/deployed.pt"
    assert item["deployed"] is True
    assert item["eligible_for_deployment"] is True
    assert item["sha256"] == info["sha256"]
    assert str(checkpoint_root) not in response.text


def test_compare_and_csv_reject_missing_ids_instead_of_silently_skipping():
    client = make_test_client()
    first = client.post("/api/v1/experiments", json={"name": "First run"}).json()
    second = client.post("/api/v1/experiments", json={"name": "Second run"}).json()

    happy = client.get(
        f"/api/v1/experiments/compare?ids={first['id']}&ids={second['id']}"
    )
    missing_compare = client.get(
        f"/api/v1/experiments/compare?ids={first['id']}&ids=missing"
    )
    missing_csv = client.get(
        f"/api/v1/experiments/export-csv?ids={first['id']}&ids=missing"
    )

    assert happy.status_code == 200
    assert len(happy.json()["items"]) == 2
    assert missing_compare.status_code == 404
    assert missing_csv.status_code == 404


def test_csv_export_neutralizes_spreadsheet_formulas():
    client = make_test_client()
    experiment = client.post(
        "/api/v1/experiments",
        json={"name": "=HYPERLINK(\"https://invalid\")"},
    ).json()

    response = client.get(
        f"/api/v1/experiments/export-csv?ids={experiment['id']}"
    )

    assert response.status_code == 200
    assert response.content.startswith(b"\xef\xbb\xbf")
    assert "'=HYPERLINK" in response.content.decode("utf-8-sig")


def test_cors_preflight_allows_patch_and_delete():
    client = make_test_client()

    for method in ("PATCH", "DELETE"):
        response = client.options(
            "/api/v1/clients/client-id/status",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": method,
            },
        )
        assert response.status_code == 200
        assert method in response.headers["access-control-allow-methods"]


def seed_flower_result(
    engine,
    experiment_id: str,
    *,
    client_f1s: list[float],
    accuracy: float = 0.80,
    macro_f1: float = 0.78,
    client_examples: list[int] | None = None,
) -> None:
    """Attach a one-round Flower result with per-client evaluate rows."""

    sizes = client_examples or [100] * len(client_f1s)
    with Session(engine) as session:
        record = session.get(ExperimentRecord, experiment_id)
        assert record is not None
        record.status = "completed"
        record.result_json = json.dumps(
            {
                "algorithm": record.algorithm,
                "history": [],
                "global_test": {
                    "global_test_accuracy": accuracy,
                    "global_test_macro_f1": macro_f1,
                },
                "selection": {"best_round": 1},
            }
        )
        session.add(record)
        session.add(
            ExperimentRoundRecord(
                experiment_id=experiment_id,
                round_number=1,
                metrics_json="{}",
                accuracy=accuracy,
                macro_f1=macro_f1,
            )
        )
        for client_id, score in enumerate(client_f1s):
            session.add(
                ClientRoundMetricRecord(
                    experiment_id=experiment_id,
                    round_number=1,
                    client_id=client_id,
                    phase="evaluate",
                    node_id=f"node-{client_id}",
                    num_examples=sizes[client_id],
                    metrics_json=json.dumps({"eval_macro_f1": score}),
                    payload_download_bytes=0,
                    payload_upload_bytes=0,
                    model_download_bytes=0,
                    model_upload_bytes=0,
                )
            )
        session.commit()


def write_centralized_baseline(
    directory: Path,
    *,
    seed: int = 2026,
    accuracy: float = 0.90,
    macro_f1: float = 0.88,
    research_result_valid: bool = True,
) -> Path:
    path = directory / "centralized_result.json"
    path.write_text(
        json.dumps(
            {
                "experiment_type": "centralized",
                "model": "mobilenet_v3_small",
                "seed": seed,
                "research_result_valid": research_result_valid,
                "metrics": {"accuracy": accuracy, "macro_f1": macro_f1},
            }
        ),
        encoding="utf-8",
    )
    return path


def compare_two(client):
    first = client.post("/api/v1/experiments", json={"name": "Run A"}).json()
    second = client.post("/api/v1/experiments", json={"name": "Run B"}).json()
    return first, second


def test_compare_reports_client_spread_not_only_the_floor():
    client, engine = make_test_client(return_engine=True)
    experiment = client.post("/api/v1/experiments", json={"name": "Run A"}).json()
    other = client.post("/api/v1/experiments", json={"name": "Run B"}).json()
    seed_flower_result(engine, experiment["id"], client_f1s=[0.4, 0.6, 0.8, 1.0])

    response = client.get(
        f"/api/v1/experiments/compare?ids={experiment['id']}&ids={other['id']}"
    )

    assert response.status_code == 200
    item = next(i for i in response.json()["items"] if i["id"] == experiment["id"])
    fairness = item["fairness"]
    assert fairness["worst"] == 0.4
    assert fairness["best"] == 1.0
    assert fairness["mean"] == 0.7
    assert fairness["spread"] == pytest.approx(0.6)
    assert fairness["std"] == pytest.approx(0.223606797749979)
    assert item["worst_client_f1"] == 0.4


def test_compare_fairness_exposes_a_small_client_left_behind():
    client, engine = make_test_client(return_engine=True)
    experiment = client.post("/api/v1/experiments", json={"name": "Run A"}).json()
    other = client.post("/api/v1/experiments", json={"name": "Run B"}).json()
    seed_flower_result(
        engine,
        experiment["id"],
        client_f1s=[0.10, 0.90, 0.90, 0.90],
        client_examples=[10, 1_000, 1_000, 1_000],
    )

    response = client.get(
        f"/api/v1/experiments/compare?ids={experiment['id']}&ids={other['id']}"
    )

    fairness = next(
        i for i in response.json()["items"] if i["id"] == experiment["id"]
    )["fairness"]
    # The weighted mean still looks healthy; the unweighted one does not. That
    # difference is the signal §7 asks for.
    assert fairness["weighted_mean"] > 0.88
    assert fairness["mean"] == pytest.approx(0.70)
    assert fairness["size_advantage"] > 0.18
    assert fairness["smallest_client_examples"] == 10


def test_compare_fairness_is_null_for_a_single_client():
    client, engine = make_test_client(return_engine=True)
    experiment = client.post("/api/v1/experiments", json={"name": "Run A"}).json()
    other = client.post("/api/v1/experiments", json={"name": "Run B"}).json()
    seed_flower_result(engine, experiment["id"], client_f1s=[0.7])

    response = client.get(
        f"/api/v1/experiments/compare?ids={experiment['id']}&ids={other['id']}"
    )

    item = next(i for i in response.json()["items"] if i["id"] == experiment["id"])
    assert item["fairness"] is None
    assert item["worst_client_f1"] == 0.7


def test_compare_gap_is_positive_when_federation_trails_centralized():
    with tempfile.TemporaryDirectory() as temporary:
        baseline = write_centralized_baseline(
            Path(temporary), seed=2026, accuracy=0.90, macro_f1=0.88
        )
        client, engine = make_test_client(
            settings=Settings(
                database_url="sqlite://",
                cors_origins=("http://localhost:5173",),
                centralized_baseline_result=baseline,
            ),
            return_engine=True,
        )
        first, second = compare_two(client)
        seed_flower_result(
            engine,
            first["id"],
            client_f1s=[0.7, 0.8],
            accuracy=0.85,
            macro_f1=0.83,
        )

        response = client.get(
            f"/api/v1/experiments/compare?ids={first['id']}&ids={second['id']}"
        )

        payload = response.json()
        item = next(i for i in payload["items"] if i["id"] == first["id"])
        assert item["gap_vs_centralized"]["accuracy"] == pytest.approx(0.05)
        assert item["gap_vs_centralized"]["macro_f1"] == pytest.approx(0.05)
        assert payload["centralized_baseline"]["macro_f1"] == 0.88


def test_compare_gap_is_null_when_baseline_seed_differs():
    with tempfile.TemporaryDirectory() as temporary:
        baseline = write_centralized_baseline(Path(temporary), seed=9999)
        client, engine = make_test_client(
            settings=Settings(
                database_url="sqlite://",
                cors_origins=("http://localhost:5173",),
                centralized_baseline_result=baseline,
            ),
            return_engine=True,
        )
        first, second = compare_two(client)
        seed_flower_result(engine, first["id"], client_f1s=[0.7, 0.8])

        response = client.get(
            f"/api/v1/experiments/compare?ids={first['id']}&ids={second['id']}"
        )

        item = next(i for i in response.json()["items"] if i["id"] == first["id"])
        assert item["gap_vs_centralized"] is None


def test_compare_refuses_a_pilot_baseline():
    """D-028: a pilot artifact must not become the thesis's headline comparison."""

    with tempfile.TemporaryDirectory() as temporary:
        baseline = write_centralized_baseline(
            Path(temporary), research_result_valid=False
        )
        client, engine = make_test_client(
            settings=Settings(
                database_url="sqlite://",
                cors_origins=("http://localhost:5173",),
                centralized_baseline_result=baseline,
            ),
            return_engine=True,
        )
        first, second = compare_two(client)
        seed_flower_result(engine, first["id"], client_f1s=[0.7, 0.8])

        response = client.get(
            f"/api/v1/experiments/compare?ids={first['id']}&ids={second['id']}"
        )

        payload = response.json()
        assert payload["centralized_baseline"] is None
        assert payload["items"][0]["gap_vs_centralized"] is None


def test_compare_refuses_a_baseline_trained_on_a_different_backbone():
    """An architecture difference must not be reported as a federation cost."""

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "centralized_result.json"
        path.write_text(
            json.dumps(
                {
                    "experiment_type": "centralized",
                    "model": "mobilenet_v2",
                    "seed": 2026,
                    "research_result_valid": True,
                    "metrics": {"accuracy": 0.90, "macro_f1": 0.88},
                }
            ),
            encoding="utf-8",
        )
        client, engine = make_test_client(
            settings=Settings(
                database_url="sqlite://",
                cors_origins=("http://localhost:5173",),
                flower_model_name="mobilenet_v3_small",
                centralized_baseline_result=path,
            ),
            return_engine=True,
        )
        first, second = compare_two(client)
        seed_flower_result(engine, first["id"], client_f1s=[0.7, 0.8])

        response = client.get(
            f"/api/v1/experiments/compare?ids={first['id']}&ids={second['id']}"
        )

        payload = response.json()
        assert payload["centralized_baseline"] is None
        assert payload["items"][0]["gap_vs_centralized"] is None


def test_compare_survives_a_missing_or_malformed_baseline_file():
    """The gap is one extra column; it must not take the comparison view down."""

    with tempfile.TemporaryDirectory() as temporary:
        broken = Path(temporary) / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        for configured in (broken, Path(temporary) / "absent.json"):
            client, engine = make_test_client(
                settings=Settings(
                    database_url="sqlite://",
                    cors_origins=("http://localhost:5173",),
                    centralized_baseline_result=configured,
                ),
                return_engine=True,
            )
            first, second = compare_two(client)
            seed_flower_result(engine, first["id"], client_f1s=[0.7, 0.8])

            response = client.get(
                f"/api/v1/experiments/compare?ids={first['id']}&ids={second['id']}"
            )

            assert response.status_code == 200
            assert response.json()["centralized_baseline"] is None
