import tempfile
import unittest
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from scripts.generate_local_secrets import generate_local_credentials


class LocalCredentialTests(unittest.TestCase):
    def test_generates_tls_bearer_and_flower_node_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_path = root / ".env"
            web = root / "web"
            flower = root / "flower"
            result = generate_local_credentials(
                env_path=env_path,
                web_tls_dir=web,
                flower_dir=flower,
            )

            values = dict(
                line.split("=", 1)
                for line in env_path.read_text(encoding="utf-8").splitlines()
                if line
            )
            certificate = x509.load_pem_x509_certificate(
                (web / "server.crt").read_bytes()
            )
            subject_alternative_name = certificate.extensions.get_extension_for_class(
                x509.SubjectAlternativeName
            ).value
            public_key = serialization.load_ssh_public_key(
                (flower / "supernode_0.pub").read_bytes()
            )

            self.assertEqual(values["CROPFED_API_AUTH_ENABLED"], "true")
            self.assertGreaterEqual(len(values["CROPFED_API_ADMIN_TOKEN"]), 32)
            self.assertNotEqual(
                values["CROPFED_API_ADMIN_TOKEN"],
                values["CROPFED_API_VIEWER_TOKEN"],
            )
            self.assertIn(
                "localhost",
                subject_alternative_name.get_values_for_type(x509.DNSName),
            )
            self.assertIsInstance(public_key, ec.EllipticCurvePublicKey)
            self.assertEqual(result["supernodes"], "4")
            self.assertTrue((flower / "superexec.secret").is_file())

    def test_refuses_to_overwrite_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_path = root / ".env"
            env_path.write_text("keep=true\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                generate_local_credentials(
                    env_path=env_path,
                    web_tls_dir=root / "web",
                    flower_dir=root / "flower",
                )

            self.assertEqual(env_path.read_text(encoding="utf-8"), "keep=true\n")


if __name__ == "__main__":
    unittest.main()
