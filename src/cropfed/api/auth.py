"""Small, server-provisioned bearer-token authorization layer.

Tokens are read from the runtime environment.  They are never persisted in the
database or bundled into the frontend image.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hmac import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from cropfed.api.settings import Settings

BEARER_SCHEME = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class Principal:
    role: str
    authentication_enabled: bool


AuthDependency = Callable[..., Principal]


def build_auth_dependencies(
    settings: Settings,
) -> tuple[AuthDependency, AuthDependency, AuthDependency]:
    """Build request dependencies bound to one immutable settings object."""

    def authenticate(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(BEARER_SCHEME),
        ] = None,
    ) -> Principal:
        if not settings.api_auth_enabled:
            return Principal(role="admin", authentication_enabled=False)
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise _unauthorized()

        supplied = credentials.credentials
        if settings.api_admin_token and compare_digest(
            supplied, settings.api_admin_token
        ):
            return Principal(role="admin", authentication_enabled=True)
        if settings.api_viewer_token and compare_digest(
            supplied, settings.api_viewer_token
        ):
            return Principal(role="viewer", authentication_enabled=True)
        raise _unauthorized()

    def require_reader(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(BEARER_SCHEME),
        ] = None,
    ) -> Principal:
        return authenticate(credentials)

    def require_admin(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(BEARER_SCHEME),
        ] = None,
    ) -> Principal:
        principal = authenticate(credentials)
        if principal.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="admin role required",
            )
        return principal

    return authenticate, require_reader, require_admin


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="valid bearer token required",
        headers={"WWW-Authenticate": "Bearer"},
    )
