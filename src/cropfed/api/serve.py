"""Migrate the production database before starting the API server."""

from __future__ import annotations


def main() -> None:
    import uvicorn

    from cropfed.api.migrate import upgrade_database

    upgrade_database()
    uvicorn.run(
        "cropfed.api.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
