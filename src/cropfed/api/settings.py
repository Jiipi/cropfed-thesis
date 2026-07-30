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
    flower_data_root: Path = Path("data/flower-profiles")
    flower_output_root: Path = Path("artifacts/flower-api")
    flower_home: Path = Path(".flwr")
    flower_superlink: str = "local"
    flower_num_cpus: int = 4
    flower_timeout_seconds: int = 86_400
    flower_poll_interval: float = 2.0
    flower_pretrained: bool = True

    def __post_init__(self) -> None:
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


def load_settings() -> Settings:
    origins = tuple(
        value.strip()
        for value in os.getenv(
            "CROPFED_CORS_ORIGINS",
            (
                "http://localhost:5173,http://localhost:8080,"
                "http://127.0.0.1:5173,http://127.0.0.1:8080"
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
            os.getenv("CROPFED_FLOWER_DATA_ROOT", "data/flower-profiles")
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


def _environment_float(name: str, *, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


settings = load_settings()
