import json
import unittest
from dataclasses import replace
from pathlib import Path

from sqlmodel import Session, SQLModel, select
from test_api import make_test_engine

from cropfed.api.models import (
    ClientRoundMetricRecord,
    ExperimentRecord,
    ExperimentRoundRecord,
)
from cropfed.api.settings import Settings
from cropfed.api.worker import (
    FlowerRunSpec,
    build_flower_command,
    data_profile_name,
    run_worker_once,
)


def queued_record(identifier: str = "worker-job") -> ExperimentRecord:
    return ExperimentRecord(
        id=identifier,
        name="Worker integration",
        status="queued",
        execution_mode="flower",
        algorithm="fedprox",
        partition_kind="dirichlet",
        num_clients=4,
        num_rounds=1,
        local_epochs=1,
        learning_rate=0.001,
        batch_size=4,
        dirichlet_alpha=0.5,
        proximal_mu=0.01,
        seed=2026,
    )


class ApiWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = make_test_engine()
        SQLModel.metadata.create_all(self.engine)
        self.settings = Settings(
            database_url="sqlite://",
            cors_origins=(),
            flower_worker_enabled=True,
            flower_pretrained=False,
        )

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_data_profile_mapping_is_closed(self) -> None:
        self.assertEqual(data_profile_name("iid", 99), "iid")
        self.assertEqual(
            data_profile_name("dirichlet", 0.5), "dirichlet-alpha-0.5"
        )
        self.assertEqual(
            data_profile_name("dirichlet", 0.1), "dirichlet-alpha-0.1"
        )
        with self.assertRaises(ValueError):
            data_profile_name("dirichlet", 0.2)

    def test_command_is_argv_with_server_owned_paths(self) -> None:
        spec = FlowerRunSpec(
            experiment_id="safe-id",
            algorithm="fedprox",
            partition_kind="dirichlet",
            num_clients=4,
            num_rounds=3,
            local_epochs=2,
            learning_rate=0.001,
            batch_size=16,
            dirichlet_alpha=0.5,
            proximal_mu=0.01,
            seed=2026,
        )

        command = build_flower_command(
            spec,
            self.settings,
            project_root=Path("project").resolve(),
            client_data_root=Path("profiles/clients").resolve(),
            test_manifest=Path("profiles/test_manifest.csv").resolve(),
            output_dir=Path("artifacts/safe-id").resolve(),
            flower_executable=Path("flwr-test"),
        )

        self.assertEqual(command[:2], ["flwr-test", "run"])
        self.assertIn("local", command)
        self.assertIn("--federation-config", command)
        run_config = command[command.index("--run-config") + 1]
        self.assertIn("algorithm='fedprox'", run_config)
        self.assertIn("num-clients=4", run_config)
        self.assertIn("pretrained=false", run_config)
        self.assertIn("global-test-manifest=", run_config)
        self.assertNotIn("central-test-manifest=", run_config)
        self.assertNotIn("shell", command)

    def test_worker_completes_claimed_job_with_injected_executor(self) -> None:
        with Session(self.engine) as session:
            session.add(queued_record())
            session.commit()

        def fake_executor(
            spec: FlowerRunSpec, _: Settings
        ) -> dict[str, object]:
            self.assertEqual(spec.experiment_id, "worker-job")
            return {
                "result_kind": "flower_image_training_run",
                "research_result_valid": False,
                "history": [
                    {
                        "round": 1,
                        "train": {"train_loss": 0.8},
                        "central_evaluate": {
                            "central_loss": 0.7,
                            "central_accuracy": 0.6,
                            "central_macro_f1": 0.5,
                        },
                    }
                ],
                "client_history": [
                    {
                        "round": 1,
                        "phase": phase,
                        "client_id": 0,
                        "node_id": 100,
                        "num_examples": 2,
                        "metrics": {"train_loss": 0.8}
                        if phase == "train"
                        else {"eval_macro_f1": 0.5},
                        "payload_download_bytes": 100,
                        "payload_upload_bytes": 80,
                        "model_download_bytes": 90,
                        "model_upload_bytes": 70 if phase == "train" else 0,
                    }
                    for phase in ("train", "evaluate")
                ],
            }

        claimed = run_worker_once(
            database_engine=self.engine,
            application_settings=self.settings,
            executor=fake_executor,
        )

        with Session(self.engine) as session:
            stored = session.get(ExperimentRecord, "worker-job")
            assert stored is not None
            result = json.loads(stored.result_json or "{}")
            self.assertEqual(stored.status, "completed")
            self.assertIsNone(stored.error_message)
            self.assertFalse(result["research_result_valid"])
            round_record = session.get(
                ExperimentRoundRecord,
                ("worker-job", 1),
            )
            assert round_record is not None
            self.assertEqual(round_record.macro_f1, 0.5)
            client_rows = session.exec(
                select(ClientRoundMetricRecord).where(
                    ClientRoundMetricRecord.experiment_id == "worker-job"
                )
            ).all()
            self.assertEqual(len(client_rows), 2)
            self.assertEqual(
                {row.phase for row in client_rows},
                {"train", "evaluate"},
            )
        self.assertTrue(claimed)
        self.assertFalse(
            run_worker_once(
                database_engine=self.engine,
                application_settings=self.settings,
                executor=fake_executor,
            )
        )

    def test_worker_persists_executor_failure(self) -> None:
        with Session(self.engine) as session:
            session.add(queued_record("failed-job"))
            session.commit()

        def failing_executor(_: FlowerRunSpec, __: Settings) -> dict[str, object]:
            raise RuntimeError("controlled failure")

        run_worker_once(
            database_engine=self.engine,
            application_settings=self.settings,
            executor=failing_executor,
        )

        with Session(self.engine) as session:
            stored = session.get(ExperimentRecord, "failed-job")
            assert stored is not None
            self.assertEqual(stored.status, "failed")
            self.assertIn("controlled failure", stored.error_message or "")

    def test_disabled_worker_refuses_to_claim(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            run_worker_once(
                database_engine=self.engine,
                application_settings=replace(
                    self.settings, flower_worker_enabled=False
                ),
                executor=lambda *_: {},
            )


if __name__ == "__main__":
    unittest.main()
