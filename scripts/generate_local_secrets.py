"""Generate local-only credentials for Compose and Flower secure deployment.

The script refuses to overwrite existing credentials.  Generated private keys,
tokens, and passwords live under ignored paths and must never be committed.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def generate_local_credentials(
    *,
    env_path: Path,
    web_tls_dir: Path,
    flower_dir: Path,
    num_supernodes: int = 4,
) -> dict[str, str]:
    """Create a development CA, service certificates, and bearer credentials."""

    targets = [env_path, web_tls_dir, flower_dir]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing credential paths: " + ", ".join(existing)
        )
    if num_supernodes < 2:
        raise ValueError("num_supernodes must be at least 2")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    web_tls_dir.mkdir(parents=True)
    flower_dir.mkdir(parents=True)

    database_password = secrets.token_urlsafe(32)
    admin_token = secrets.token_urlsafe(48)
    viewer_token = secrets.token_urlsafe(48)
    env_path.write_text(
        "\n".join(
            [
                "POSTGRES_DB=cropfed",
                "POSTGRES_USER=cropfed",
                f"POSTGRES_PASSWORD={database_password}",
                "CROPFED_API_AUTH_ENABLED=true",
                f"CROPFED_API_ADMIN_TOKEN={admin_token}",
                f"CROPFED_API_VIEWER_TOKEN={viewer_token}",
                "CROPFED_CORS_ORIGINS=https://localhost:8443",
                "CROPFED_FLOWER_WORKER_ENABLED=false",
                "CROPFED_FLOWER_NUM_CPUS=4",
                "CROPFED_FLOWER_PRETRAINED=true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _restrict(env_path)

    web_fingerprint = _create_ca_and_server_certificate(
        destination=web_tls_dir,
        common_name="CropFed local web CA",
        server_common_name="localhost",
        dns_names=["localhost", "web"],
        ip_addresses=["127.0.0.1"],
    )
    flower_fingerprint = _create_ca_and_server_certificate(
        destination=flower_dir,
        common_name="CropFed local Flower CA",
        server_common_name="superlink",
        dns_names=["superlink", "localhost"],
        ip_addresses=["127.0.0.1"],
    )

    for client_id in range(num_supernodes):
        private_key = ec.generate_private_key(ec.SECP384R1())
        private_path = flower_dir / f"supernode_{client_id}.key"
        public_path = flower_dir / f"supernode_{client_id}.pub"
        private_path.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        public_path.write_bytes(
            private_key.public_key().public_bytes(
                encoding=serialization.Encoding.OpenSSH,
                format=serialization.PublicFormat.OpenSSH,
            )
            + b"\n"
        )
        _restrict(private_path)

    superexec_secret = flower_dir / "superexec.secret"
    superexec_secret.write_bytes(secrets.token_bytes(48))
    _restrict(superexec_secret)
    return {
        "env": str(env_path),
        "web_ca_sha256": web_fingerprint,
        "flower_ca_sha256": flower_fingerprint,
        "supernodes": str(num_supernodes),
    }


def _create_ca_and_server_certificate(
    *,
    destination: Path,
    common_name: str,
    server_common_name: str,
    dns_names: list[str],
    ip_addresses: list[str],
) -> str:
    now = datetime.now(UTC)
    ca_key = rsa.generate_private_key(public_exponent=65_537, key_size=3072)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    ca_certificate = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    server_key = rsa.generate_private_key(public_exponent=65_537, key_size=3072)
    server_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, server_common_name)]
    )
    alternative_names: list[x509.GeneralName] = [
        x509.DNSName(name) for name in dns_names
    ]
    alternative_names.extend(
        x509.IPAddress(ipaddress.ip_address(address)) for address in ip_addresses
    )
    server_certificate = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_certificate.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(alternative_names),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    _write_private_key(destination / "ca.key", ca_key)
    _write_certificate(destination / "ca.crt", ca_certificate)
    _write_private_key(destination / "server.key", server_key)
    _write_certificate(destination / "server.crt", server_certificate)
    return ca_certificate.fingerprint(hashes.SHA256()).hex()


def _write_private_key(path: Path, key: rsa.RSAPrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    _restrict(path)


def _write_certificate(path: Path, certificate: x509.Certificate) -> None:
    path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


def _restrict(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows ACLs are managed separately; ignored paths remain the primary guard.
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument(
        "--web-tls-dir", type=Path, default=Path("secrets/web-tls")
    )
    parser.add_argument(
        "--flower-dir", type=Path, default=Path("secrets/flower")
    )
    args = parser.parse_args()
    result = generate_local_credentials(
        env_path=args.env,
        web_tls_dir=args.web_tls_dir,
        flower_dir=args.flower_dir,
    )
    print(
        "local credentials generated; secrets were not printed: "
        f"env={result['env']}, supernodes={result['supernodes']}"
    )
    print(f"web CA SHA256: {result['web_ca_sha256']}")
    print(f"Flower CA SHA256: {result['flower_ca_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
