"""Environment-backed API settings with no hidden defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    cors_origins: tuple[str, ...]
    api_auth_enabled: bool = False
    api_admin_token: str | None = None
    api_viewer_token: str | None = None
    flower_worker_enabled: bool = False
    flower_project_dir: Path = Path(".")
    flower_data_root: Path = Path("data/flower-profiles-full")
    #: Where the profiles' relative image paths are anchored. Server-owned, like
    #: every other data path: HTTP clients never supply it (D-019).
    flower_dataset_root: Path = Path("data/raw")
    flower_num_workers: int = 0
    taxonomy_scope: str = "plantvillage-full"
    flower_model_name: str = "mobilenet_v3_small"
    flower_num_gpus: float = 0.0
    flower_output_root: Path = Path("artifacts/flower-api")
    flower_home: Path = Path(".flwr")
    flower_superlink: str = "local"
    flower_num_cpus: int = 4
    flower_timeout_seconds: int = 86_400
    flower_poll_interval: float = 2.0
    flower_pretrained: bool = True
    checkpoint_dir: Path = Path("artifacts")
    deployed_checkpoint: Path | None = None
    deployed_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if (self.deployed_checkpoint is None) != (
            self.deployed_checkpoint_sha256 is None
        ):
            raise ValueError(
                "deployed checkpoint and CROPFED_DEPLOYED_CHECKPOINT_SHA256 "
                "must be configured together"
            )
        if self.deployed_checkpoint_sha256 is not None:
            digest = self.deployed_checkpoint_sha256.lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "CROPFED_DEPLOYED_CHECKPOINT_SHA256 must be 64 hexadecimal characters"
                )
        if self.taxonomy_scope not in {"tomato", "plantvillage-full"}:
            raise ValueError("taxonomy_scope must be tomato or plantvillage-full")
        if self.flower_model_name not in {
            "mobilenet_v2",
            "mobilenet_v3_small",
            "efficientnet_lite0",
        }:
            raise ValueError("unsupported Flower model")
        if self.flower_num_gpus < 0:
            raise ValueError("flower_num_gpus cannot be negative")
        if not self.api_auth_enabled:
            return
        for name, token in (
            ("CROPFED_API_ADMIN_TOKEN", self.api_admin_token),
            ("CROPFED_API_VIEWER_TOKEN", self.api_viewer_token),
        ):
            if token is None or len(token) < 32:
                raise ValueError(f"{name} must contain at least 32 characters")
        if self.api_admin_token == self.api_viewer_token:
            raise ValueError("admin and viewer API tokens must be different")


def _load_dotenv_if_present() -> None:
    env_file = Path(".env")
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings() -> Settings:
    _load_dotenv_if_present()
    origins = tuple(
        value.strip()
        for value in os.getenv(
            "CROPFED_CORS_ORIGINS",
            (
                "http://localhost:5173,http://localhost:8080,http://localhost:8000,"
                "http://127.0.0.1:5173,http://127.0.0.1:8080,http://127.0.0.1:8000"
            ),
        ).split(",")
        if value.strip()
    )
    return Settings(
        database_url=os.getenv(
            "CROPFED_DATABASE_URL", "sqlite:///./artifacts/cropfed.db"
        ),
        cors_origins=origins,
        api_auth_enabled=_environment_bool("CROPFED_API_AUTH_ENABLED", default=False),
        api_admin_token=os.getenv("CROPFED_API_ADMIN_TOKEN"),
        api_viewer_token=os.getenv("CROPFED_API_VIEWER_TOKEN"),
        flower_worker_enabled=_environment_bool(
            "CROPFED_FLOWER_WORKER_ENABLED", default=False
        ),
        flower_project_dir=Path(
            os.getenv("CROPFED_FLOWER_PROJECT_DIR", ".")
        ),
        flower_data_root=Path(
            os.getenv("CROPFED_FLOWER_DATA_ROOT", "data/flower-profiles-full")
        ),
        flower_dataset_root=Path(
            os.getenv("CROPFED_FLOWER_DATASET_ROOT", "data/raw")
        ),
        flower_num_workers=_environment_nonnegative_int(
            "CROPFED_FLOWER_NUM_WORKERS", default=0
        ),
        taxonomy_scope=os.getenv(
            "CROPFED_TAXONOMY_SCOPE", "plantvillage-full"
        ),
        flower_model_name=os.getenv(
            "CROPFED_FLOWER_MODEL_NAME", "mobilenet_v3_small"
        ),
        flower_num_gpus=_environment_nonnegative_float(
            "CROPFED_FLOWER_NUM_GPUS", default=1.0
        ),
        flower_output_root=Path(
            os.getenv("CROPFED_FLOWER_OUTPUT_ROOT", "artifacts/flower-api")
        ),
        flower_home=Path(os.getenv("CROPFED_FLOWER_HOME", ".flwr")),
        flower_superlink=os.getenv("CROPFED_FLOWER_SUPERLINK", "local"),
        flower_num_cpus=_environment_int("CROPFED_FLOWER_NUM_CPUS", default=4),
        flower_timeout_seconds=_environment_int(
            "CROPFED_FLOWER_TIMEOUT_SECONDS", default=86_400
        ),
        flower_poll_interval=_environment_float(
            "CROPFED_FLOWER_POLL_INTERVAL", default=2.0
        ),
        flower_pretrained=_environment_bool(
            "CROPFED_FLOWER_PRETRAINED", default=True
        ),
        checkpoint_dir=Path(
            os.getenv("CROPFED_CHECKPOINT_DIR", "artifacts")
        ),
        deployed_checkpoint=_optional_path("CROPFED_DEPLOYED_CHECKPOINT"),
        deployed_checkpoint_sha256=(
            os.getenv("CROPFED_DEPLOYED_CHECKPOINT_SHA256") or None
        ),
    )


def _environment_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _environment_int(name: str, *, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _environment_nonnegative_int(name: str, *, default: int) -> int:
    """Zero is a meaningful value here, unlike for counts of CPUs or seconds.

    ``num_workers=0`` means "decode in the training process", which is the right
    default on Windows and inside a single-CPU Ray actor.
    """

    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _environment_float(name: str, *, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _environment_nonnegative_float(name: str, *, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


settings = load_settings()
