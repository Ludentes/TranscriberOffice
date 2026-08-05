"""Anonymous browser identity backed by an HttpOnly cookie."""
from __future__ import annotations

import re
import secrets

from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import SessionConfig
from app.job_store import JobStore


_TOKEN_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def is_valid_session_token(value: str | None) -> bool:
    return bool(value and _TOKEN_PATTERN.fullmatch(value))


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
                cookie_response.set_cookie(
                    key=self.settings.cookie_name,
                    value=token,
                    max_age=self.settings.cookie_max_age_days * 86400,
                    httponly=True,
                    secure=self.settings.cookie_secure,
                    samesite="lax",
                    path="/",
                )
                cookie_headers = [
                    header
                    for header in cookie_response.raw_headers
                    if header[0].lower() == b"set-cookie"
                ]
                message.setdefault("headers", []).extend(cookie_headers)
            await send(message)

        await self.app(scope, receive, send_with_cookie)
