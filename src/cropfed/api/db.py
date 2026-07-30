"""Database engine and request-scoped sessions."""

from __future__ import annotations

from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine

from cropfed.api.settings import settings

connect_args = (
    {"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {}
)
engine = create_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=not settings.database_url.startswith("sqlite"),
    connect_args=connect_args,
)


def init_database() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session

