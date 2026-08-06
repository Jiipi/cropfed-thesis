"""Cover the dashboard's profile reader without needing the FastAPI app.

``api/data_profiles.py`` is the D-025 boundary: the dashboard receives class
counts and proportions from server-owned ``partition_summary.json`` files and
never image bytes or local paths.  The module imports without ``fastapi``, so
these checks run in the unit suite instead of the API suite.
"""

import json
import tempfile
import unittest
from pathlib import Path

from cropfed.api.data_profiles import _read_profile
from cropfed.data.profiles import DataProfileSpec


def _summary(**overrides) -> dict:
    clients = [
        {
            "client_id": client_id,
            "num_samples": 10,
            "num_train": 8,
            "num_validation": 2,
            "class_counts": [6, 4],
            "class_proportions": [0.6, 0.4],
        }
        for client_id in range(4)
    ]
    document = {
        "num_clients": 4,
        "partition_kind": "iid",
        "skew_type": "none",
        "dirichlet_alpha": None,
        "quantity_skew": False,
        "feature_skew_strength": None,
        "clients": clients,
    }
    document.update(overrides)
    return document


class ApiDataProfileTests(unittest.TestCase):
    def _read(self, spec: DataProfileSpec, document: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partition_summary.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return _read_profile(path, spec, num_classes=2)

    def test_quantity_skew_profile_is_reported_with_its_skew_type(self) -> None:
        spec = DataProfileSpec("quantity-skew", "iid", None, quantity_skew=True)
        payload = self._read(
            spec,
            _summary(skew_type="quantity", quantity_skew=True),
        )

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["skew_type"], "quantity")
        self.assertTrue(payload["quantity_skew"])
        self.assertEqual(payload["num_samples"], 40)

    def test_feature_skew_profile_reports_its_strength(self) -> None:
        spec = DataProfileSpec("feature-skew", "feature_skew", None)
        payload = self._read(
            spec,
            _summary(
                partition_kind="feature_skew",
                skew_type="feature",
                feature_skew_strength=0.5,
            ),
        )

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["skew_type"], "feature")
        self.assertEqual(payload["feature_skew_strength"], 0.5)

    def test_a_summary_that_contradicts_its_spec_is_rejected(self) -> None:
        """The whole point of A3: a mislabelled artifact must not be trusted.

        A ``quantity-skew`` directory holding a plain IID summary is exactly the
        failure this section was opened to catch, so it must read as invalid
        rather than render as a legitimate profile.
        """

        spec = DataProfileSpec("quantity-skew", "iid", None, quantity_skew=True)
        payload = self._read(spec, _summary())

        self.assertEqual(payload["status"], "invalid")
        self.assertFalse(payload["available"])
        self.assertEqual(payload["clients"], [])

    def test_a_missing_summary_is_reported_rather_than_raising(self) -> None:
        spec = DataProfileSpec("feature-skew", "feature_skew", None)
        with tempfile.TemporaryDirectory() as directory:
            payload = _read_profile(
                Path(directory) / "absent.json", spec, num_classes=2
            )

        self.assertEqual(payload["status"], "missing")
        self.assertFalse(payload["available"])

    def test_payload_carries_no_image_paths_or_bytes(self) -> None:
        """D-025: the dashboard receives counts only."""

        spec = DataProfileSpec("iid", "iid", None)
        payload = self._read(spec, _summary())

        serialised = json.dumps(payload)
        for forbidden in ("path", "image_id", "/", "\\\\"):
            self.assertNotIn(forbidden, serialised)


if __name__ == "__main__":
    unittest.main()
