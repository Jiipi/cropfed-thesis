import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from cropfed.api.main import create_app
from cropfed.api.settings import Settings
from cropfed.constants import OFFICIAL_TITLE, TOMATO_CLASSES, taxonomy_from_scope
from cropfed.data.profiles import FULL_PROFILE_SPECS, MVP_PROFILE_SPECS


def make_test_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = make_test_engine()
        self.settings = Settings(
            database_url="sqlite://",
            cors_origins=(),
            flower_project_dir=Path("__missing_test_project__"),
        )
        self.client_context = TestClient(
            create_app(
                database_engine=self.engine,
                application_settings=self.settings,
            )
        )
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.engine.dispose()

    def test_health_project_and_taxonomy_contract(self) -> None:
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(
            self.client.get("/health/ready").json(),
            {"status": "ok", "database": "ok"},
        )

        project = self.client.get("/api/v1/project")
        classes = self.client.get("/api/v1/classes")

        self.assertEqual(project.status_code, 200)
        self.assertEqual(project.json()["official_title"], OFFICIAL_TITLE)
        self.assertEqual(classes.status_code, 200)
        # Assert against the configured scope rather than a literal count, so
        # this contract test follows the taxonomy instead of pinning it.
        taxonomy = taxonomy_from_scope(self.settings.taxonomy_scope)
        self.assertEqual(classes.json()["count"], len(taxonomy.class_names))
        self.assertEqual(
            [item["name"] for item in classes.json()["items"]],
            list(taxonomy.class_names),
        )
        self.assertEqual(
            [item["group"] for item in classes.json()["items"]],
            list(taxonomy.class_groups),
        )
        profiles = self.client.get("/api/v1/data-profiles")
        self.assertEqual(profiles.status_code, 200)
        self.assertEqual(len(profiles.json()["items"]), len(FULL_PROFILE_SPECS))
        self.assertFalse(any(item["available"] for item in profiles.json()["items"]))
        self.assertFalse(profiles.json()["privacy"]["contains_image_bytes"])

    def test_default_scope_is_the_38_class_main_study_taxonomy(self) -> None:
        """The dataclass default and load_settings() must not disagree.

        They previously did (``tomato`` vs ``plantvillage-full``), so tests
        silently exercised a different taxonomy than production.
        """

        self.assertEqual(self.settings.taxonomy_scope, "plantvillage-full")
        self.assertEqual(self.client.get("/api/v1/classes").json()["count"], 38)

    def test_tomato_scope_still_serves_the_pilot_taxonomy(self) -> None:
        """Tier-1 pilot artefacts must stay readable under the legacy scope."""

        engine = make_test_engine()
        settings = replace(self.settings, taxonomy_scope="tomato")
        with TestClient(
            create_app(database_engine=engine, application_settings=settings)
        ) as client:
            classes = client.get("/api/v1/classes").json()
            self.assertEqual(classes["count"], 10)
            self.assertEqual(
                [item["name"] for item in classes["items"]], list(TOMATO_CLASSES)
            )
            profiles = client.get("/api/v1/data-profiles").json()
            self.assertEqual(len(profiles["items"]), len(MVP_PROFILE_SPECS))
        engine.dispose()

    def test_bearer_auth_enforces_viewer_and_admin_roles(self) -> None:
        admin_token = "admin-token-" + ("a" * 40)
        viewer_token = "viewer-token-" + ("v" * 40)
        auth_settings = replace(
            self.settings,
            api_auth_enabled=True,
            api_admin_token=admin_token,
            api_viewer_token=viewer_token,
        )
        auth_context = TestClient(
            create_app(
                database_engine=self.engine,
                application_settings=auth_settings,
            )
        )
        with auth_context as auth_client:
            missing = auth_client.get("/api/v1/project")
            invalid = auth_client.get(
                "/api/v1/project",
                headers={"Authorization": "Bearer invalid"},
            )
            viewer_headers = {"Authorization": f"Bearer {viewer_token}"}
            admin_headers = {"Authorization": f"Bearer {admin_token}"}
            viewer_read = auth_client.get("/api/v1/project", headers=viewer_headers)
            viewer_me = auth_client.get("/api/v1/auth/me", headers=viewer_headers)
            viewer_write = auth_client.post(
                "/api/v1/experiments",
                headers=viewer_headers,
                json={"name": "Viewer must not create"},
            )
            admin_write = auth_client.post(
                "/api/v1/experiments",
                headers=admin_headers,
                json={"name": "Admin may create"},
            )

            self.assertEqual(auth_client.get("/health/ready").status_code, 200)
            self.assertEqual(missing.status_code, 401)
            self.assertEqual(missing.headers["www-authenticate"], "Bearer")
            self.assertEqual(invalid.status_code, 401)
            self.assertEqual(viewer_read.status_code, 200)
            self.assertEqual(
                viewer_me.json(),
                {"role": "viewer", "authentication_enabled": True},
            )
            self.assertEqual(viewer_write.status_code, 403)
            self.assertEqual(admin_write.status_code, 201)

    def test_auth_settings_reject_weak_or_shared_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 32"):
            replace(
                self.settings,
                api_auth_enabled=True,
                api_admin_token="too-short",
                api_viewer_token="v" * 40,
            )
        with self.assertRaisesRegex(ValueError, "must be different"):
            replace(
                self.settings,
                api_auth_enabled=True,
                api_admin_token="s" * 40,
                api_viewer_token="s" * 40,
            )

    def test_data_profile_endpoint_returns_counts_without_paths(self) -> None:
        # Derive the fixture width from the configured scope; a literal class
        # count here silently stops exercising the endpoint once the taxonomy
        # changes, because a width mismatch is reported as `available: False`.
        num_classes = len(taxonomy_from_scope(self.settings.taxonomy_scope).class_names)
        num_samples = 2 * num_classes
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            summary = {
                "partition_kind": "iid",
                "dirichlet_alpha": None,
                "num_clients": 4,
                "seed": 2026,
                "clients": [
                    {
                        "client_id": client_id,
                        "num_samples": num_samples,
                        "num_train": num_samples - 4,
                        "num_validation": 4,
                        "class_counts": [2] * num_classes,
                        "class_proportions": [2 / num_samples] * num_classes,
                    }
                    for client_id in range(4)
                ],
            }
            destination = (
                project_root
                / "profiles"
                / "iid"
                / "clients"
                / "partition_summary.json"
            )
            destination.parent.mkdir(parents=True)
            destination.write_text(json.dumps(summary), encoding="utf-8")
            profile_client_context = TestClient(
                create_app(
                    database_engine=self.engine,
                    application_settings=replace(
                        self.settings,
                        flower_project_dir=project_root,
                        flower_data_root=Path("profiles"),
                    ),
                )
            )
            with profile_client_context as profile_client:
                response = profile_client.get("/api/v1/data-profiles")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["items"][0]["available"])
        self.assertEqual(payload["items"][0]["num_samples"], 4 * num_samples)
        self.assertEqual(
            payload["items"][0]["clients"][0]["class_counts"], [2] * num_classes
        )
        self.assertNotIn(str(project_root), response.text)

    def test_synthetic_experiment_lifecycle_and_rounds(self) -> None:
        created = self.client.post(
            "/api/v1/experiments",
            json={
                "name": "API lifecycle smoke",
                "algorithm": "fedprox",
                "num_clients": 4,
                "num_rounds": 1,
                "local_epochs": 1,
                "batch_size": 16,
            },
        )
        self.assertEqual(created.status_code, 201)
        experiment_id = created.json()["id"]
        self.assertEqual(created.json()["execution_mode"], "synthetic-smoke")
        self.assertEqual(created.json()["status"], "draft")

        started = self.client.post(f"/api/v1/experiments/{experiment_id}/start")
        completed = self.client.get(f"/api/v1/experiments/{experiment_id}")
        rounds = self.client.get(f"/api/v1/experiments/{experiment_id}/rounds")
        client_metrics = self.client.get(
            f"/api/v1/experiments/{experiment_id}/clients"
        )

        self.assertEqual(started.status_code, 202)
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.json()["status"], "completed")
        self.assertEqual(
            completed.json()["result"]["result_kind"], "synthetic_smoke_only"
        )
        self.assertEqual(rounds.json()["result_kind"], "synthetic_smoke_only")
        self.assertEqual(rounds.json()["storage"], "database")
        self.assertEqual(len(rounds.json()["items"]), 1)
        self.assertEqual(len(rounds.json()["summaries"]), 1)
        self.assertEqual(rounds.json()["summaries"][0]["round"], 1)
        self.assertIsNotNone(rounds.json()["summaries"][0]["macro_f1"])
        self.assertEqual(client_metrics.status_code, 200)
        self.assertEqual(client_metrics.json()["items"], [])
        self.assertEqual(
            self.client.post(
                f"/api/v1/experiments/{experiment_id}/start"
            ).status_code,
            409,
        )

    def test_create_list_get_and_not_found(self) -> None:
        first = self.client.post(
            "/api/v1/experiments",
            json={"name": "First experiment"},
        )
        second = self.client.post(
            "/api/v1/experiments",
            json={"name": "Second experiment", "algorithm": "fedprox"},
        )

        listed = self.client.get("/api/v1/experiments")

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()), 2)
        self.assertEqual(
            self.client.get(
                f"/api/v1/experiments/{first.json()['id']}"
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.get("/api/v1/experiments/missing").status_code,
            404,
        )
        self.assertEqual(
            self.client.get("/api/v1/experiments/missing/clients").status_code,
            404,
        )

    def test_flower_payload_is_whitelisted(self) -> None:
        wrong_clients = self.client.post(
            "/api/v1/experiments",
            json={
                "name": "Wrong Flower clients",
                "execution_mode": "flower",
                "num_clients": 3,
            },
        )
        wrong_alpha = self.client.post(
            "/api/v1/experiments",
            json={
                "name": "Wrong Flower alpha",
                "execution_mode": "flower",
                "num_clients": 4,
                "partition_kind": "dirichlet",
                "dirichlet_alpha": 0.2,
            },
        )
        arbitrary_mode = self.client.post(
            "/api/v1/experiments",
            json={"name": "Arbitrary command", "execution_mode": "shell"},
        )
        extra_command = self.client.post(
            "/api/v1/experiments",
            json={
                "name": "Ignored command must fail",
                "execution_mode": "flower",
                "command": "anything",
            },
        )

        self.assertEqual(wrong_clients.status_code, 422)
        self.assertEqual(wrong_alpha.status_code, 422)
        self.assertEqual(arbitrary_mode.status_code, 422)
        self.assertEqual(extra_command.status_code, 422)

    def test_flower_start_requires_enabled_worker(self) -> None:
        created = self.client.post(
            "/api/v1/experiments",
            json={
                "name": "Disabled Flower worker",
                "execution_mode": "flower",
                "num_clients": 4,
                "partition_kind": "iid",
                "num_rounds": 1,
                "local_epochs": 1,
                "learning_rate": 0.001,
            },
        )
        experiment_id = created.json()["id"]

        response = self.client.post(f"/api/v1/experiments/{experiment_id}/start")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            self.client.get(
                f"/api/v1/experiments/{experiment_id}"
            ).json()["status"],
            "draft",
        )

    def test_enabled_flower_job_stays_queued_for_external_worker(self) -> None:
        worker_client_context = TestClient(
            create_app(
                database_engine=self.engine,
                application_settings=replace(
                    self.settings, flower_worker_enabled=True
                ),
            )
        )
        with worker_client_context as worker_client:
            created = worker_client.post(
                "/api/v1/experiments",
                json={
                    "name": "Queued Flower job",
                    "execution_mode": "flower",
                    "num_clients": 4,
                    "partition_kind": "iid",
                    "num_rounds": 1,
                    "local_epochs": 1,
                    "learning_rate": 0.001,
                },
            )
            experiment_id = created.json()["id"]

            response = worker_client.post(
                f"/api/v1/experiments/{experiment_id}/start"
            )
            stored = worker_client.get(
                f"/api/v1/experiments/{experiment_id}"
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(stored.json()["status"], "queued")
        self.assertIsNone(stored.json()["result"])


if __name__ == "__main__":
    unittest.main()
