"""Tests for newly added API endpoints (clients, predict, checkpoints, compare, export)."""

from __future__ import annotations

import hashlib
import io
import tempfile
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import SQLModel, create_engine
from sqlmodel.pool import StaticPool

from cropfed.api.main import create_app
from cropfed.api.settings import Settings
from cropfed.constants import taxonomy_from_scope


def make_test_client(
    *,
    settings: Settings | None = None,
    prediction_executor=None,
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
    return TestClient(app)


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
