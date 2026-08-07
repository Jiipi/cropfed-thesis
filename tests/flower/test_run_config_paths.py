"""``dataset-root`` and ``num-workers`` must survive the trip through run_config.

Flower's ``run_config.get(name, default)`` returns the default for any key that
is not declared in ``[tool.flwr.app.config]``, which has already cost this
project one silent misconfiguration.  These tests pin both halves: the keys are
declared in the packaged config, and the app refuses to invent a dataset root.
"""

import importlib.util
import tomllib
import unittest
from pathlib import Path

FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RunConfigDeclarationTests(unittest.TestCase):
    """Undeclared keys are the failure mode, so assert on the packaged file."""

    def test_dataset_root_and_num_workers_are_declared(self) -> None:
        pyproject = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        config = pyproject["tool"]["flwr"]["app"]["config"]

        self.assertIn("dataset-root", config)
        self.assertIn("num-workers", config)
        self.assertEqual(config["num-workers"], 0)


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower runtime is not installed")
class DatasetRootFromRunConfigTests(unittest.TestCase):
    def test_missing_dataset_root_raises_instead_of_defaulting(self) -> None:
        from cropfed.flower.client_app import _dataset_root

        with self.assertRaisesRegex(KeyError, "dataset-root"):
            _dataset_root(self._context({}))

    def test_blank_dataset_root_is_treated_as_missing(self) -> None:
        """An empty TOML string must not resolve to the launch directory."""

        from cropfed.flower.client_app import _dataset_root

        with self.assertRaisesRegex(KeyError, "dataset-root"):
            _dataset_root(self._context({"dataset-root": "   "}))

    def test_dataset_root_is_resolved_to_an_absolute_path(self) -> None:
        from cropfed.flower.client_app import _dataset_root

        resolved = _dataset_root(self._context({"dataset-root": "data/raw"}))

        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved.name, "raw")

    def test_server_and_client_resolve_the_root_identically(self) -> None:
        """A split brain here would evaluate the server on different files."""

        from cropfed.flower.client_app import _dataset_root as client_root
        from cropfed.flower.server_app import _dataset_root as server_root

        context = self._context({"dataset-root": "data/raw"})

        self.assertEqual(client_root(context), server_root(context))

    def test_num_workers_defaults_to_zero_and_reads_the_override(self) -> None:
        from cropfed.flower.client_app import _num_workers

        self.assertEqual(_num_workers(self._context({})), 0)
        self.assertEqual(_num_workers(self._context({"num-workers": 8})), 8)

    @staticmethod
    def _context(run_config: dict):
        from flwr.common import Context

        return Context(
            run_id=1,
            node_id=1,
            node_config={"partition-id": 0},
            state=None,
            run_config=run_config,
        )


if __name__ == "__main__":
    unittest.main()
