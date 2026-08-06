import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ComposeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compose = yaml.safe_load(
            (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
        )

    def test_local_stack_exposes_only_the_tls_web_entrypoint(self) -> None:
        services = self.compose["services"]

        self.assertNotIn("ports", services["db"])
        self.assertNotIn("ports", services["api"])
        self.assertEqual(services["web"]["ports"], ["8080:80", "8443:443"])
        self.assertIn(
            "./secrets/web-tls:/etc/nginx/tls:ro",
            services["web"]["volumes"],
        )

    def test_web_healthcheck_uses_the_ipv4_loopback(self) -> None:
        command = self.compose["services"]["web"]["healthcheck"]["test"][1]

        self.assertIn("http://127.0.0.1/healthz", command)
        self.assertNotIn("http://localhost/healthz", command)

    def test_api_requires_server_provisioned_bearer_credentials(self) -> None:
        environment = self.compose["services"]["api"]["environment"]

        self.assertEqual(
            environment["CROPFED_API_AUTH_ENABLED"],
            "${CROPFED_API_AUTH_ENABLED:-true}",
        )
        self.assertIn(":?", environment["CROPFED_API_ADMIN_TOKEN"])
        self.assertIn(":?", environment["CROPFED_API_VIEWER_TOKEN"])


if __name__ == "__main__":
    unittest.main()
