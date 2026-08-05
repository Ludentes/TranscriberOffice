"""Anonymous browser identity backed by an HttpOnly cookie."""
from __future__ import annotations

import re
import secrets
from urllib.parse import urlsplit

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import SessionConfig
from app.job_store import JobStore


_TOKEN_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def is_valid_session_token(value: str | None) -> bool:
    return bool(value and _TOKEN_PATTERN.fullmatch(value))


def set_session_cookie(
    response: Response,
    token: str,
    settings: SessionConfig,
) -> None:
    """Set the protected bearer cookie consistently across all session flows."""
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        max_age=settings.cookie_max_age_days * 86400,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


class SessionTokenImport(BaseModel):
    token: str


def _is_same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    host = request.headers.get("host")
    if not origin or not host:
        return False
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return (
        parsed.scheme.lower() == request.url.scheme.lower()
        and parsed.netloc.lower() == host.lower()
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
    )


def create_session_router(store: JobStore, settings: SessionConfig) -> APIRouter:
    """Create same-origin endpoints for deliberate token export and import."""
    router = APIRouter(prefix="/api/session", tags=["browser-session"])
    no_store = {"Cache-Control": "no-store"}

    @router.get("/token")
    def export_token(request: Request) -> JSONResponse:
        token = request.cookies.get(settings.cookie_name)
        owner_id = getattr(request.state, "owner_id", None)
        if (
            not is_valid_session_token(token)
            or store.find_owner_by_token(token) != owner_id
        ):
            return JSONResponse(
                {"detail": "Session token unavailable"},
                status_code=401,
                headers=no_store,
            )
        return JSONResponse({"token": token}, headers=no_store)

    @router.post("/import")
    def import_token(payload: SessionTokenImport, request: Request) -> JSONResponse:
        if not _is_same_origin(request):
            return JSONResponse(
                {"detail": "Session import is allowed only from this site"},
                status_code=403,
                headers=no_store,
            )
        token = payload.token.strip()
        if not is_valid_session_token(token) or store.find_owner_by_token(token) is None:
            return JSONResponse(
                {"detail": "Invalid access token"},
                status_code=400,
                headers=no_store,
            )
        response = JSONResponse({"success": True}, headers=no_store)
        set_session_cookie(response, token, settings)
        return response

    return router


class BrowserSessionMiddleware:
    """Resolve a private owner ID for every HTTP request."""

    def __init__(self, app: ASGIApp, store: JobStore, settings: SessionConfig):
        self.app = app
        self.store = store
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        token = request.cookies.get(self.settings.cookie_name)
        should_set_cookie = not is_valid_session_token(token)
        if should_set_cookie:
            token = secrets.token_hex(32)

        assert token is not None
        owner_id = self.store.resolve_owner(token)
        scope.setdefault("state", {})["owner_id"] = owner_id

        async def send_with_cookie(message: Message) -> None:
            if should_set_cookie and message["type"] == "http.response.start":
                cookie_response = Response()
                set_session_cookie(cookie_response, token, self.settings)
                cookie_headers = [
                    header
                    for header in cookie_response.raw_headers
                    if header[0].lower() == b"set-cookie"
                ]
                existing_headers = message.setdefault("headers", [])
                cookie_prefix = f"{self.settings.cookie_name}=".encode("latin-1")
                has_session_cookie = any(
                    name.lower() == b"set-cookie"
                    and value.lstrip().startswith(cookie_prefix)
                    for name, value in existing_headers
                )
                if not has_session_cookie:
                    existing_headers.extend(cookie_headers)
            await send(message)

        await self.app(scope, receive, send_with_cookie)
