# Transferable Browser Access Token

## Goal

Allow a user to copy the anonymous browser access token from one browser and
paste it into another browser or computer. Both browsers then access the same
owner-scoped transcription history concurrently.

## User Experience

The persistent Gradio UI adds a collapsed section named "Access from another
device" with two deliberate actions:

- **Show token** reveals the current 64-character access token in a read-only,
  copyable field.
- **Use token** accepts a pasted token, switches the current browser to that
  existing history, and reloads the page.

A warning beside both controls states that the token is equivalent to a
password: anyone who has it can read, stop, delete, and download all
transcriptions belonging to that history.

Import replaces the current browser identity; it never merges or transfers
jobs between owners. The browser's previous history remains unchanged and can
only be reopened with its previous token. A shared token remains valid in all
browsers at the same time.

## Security Model

The token remains a random 256-bit bearer credential. SQLite continues to store
only its SHA-256 digest. The raw token is available to the server from the
request cookie but is never written to the database, application logs, URLs, or
redirect query parameters.

Displaying or importing a token is intentionally explicit. Normal page loads
do not expose it to JavaScript. The session cookie remains `HttpOnly`,
`SameSite=Lax`, scoped to `/`, long-lived, and `Secure` when HTTPS is enabled.

The transfer endpoints return `Cache-Control: no-store`. They do not include
the token in error messages. The UI labels the field as sensitive and does not
show a token until the user requests it.

## Components

### Owner lookup

`JobStore` gains a read-only `find_owner_by_token(token)` method. Unlike the
existing `resolve_owner`, it never creates an owner. Import uses this method so
a well-formed but unknown token is rejected rather than creating a new empty
history.

### Session router

A focused FastAPI router is created with the same `JobStore` and
`SessionConfig` used by middleware.

- `GET /api/session/token` returns the current cookie token only when it is
  structurally valid and resolves to the current request owner.
- `POST /api/session/import` accepts JSON `{ "token": "..." }`, validates the
  token format, requires an existing owner, and sets the configured session
  cookie to that token.

Successful import returns a small JSON success response. The browser reload is
performed by the UI only after this response succeeds.

### Same-origin protection

Import changes which documents subsequent uploads belong to. A cross-site
"login CSRF" could otherwise switch a victim into an attacker's history and
cause later uploads to become visible to the attacker.

The import endpoint therefore requires a browser request whose `Origin` matches
the request's effective scheme and host. Requests with a missing, invalid, or
foreign origin are rejected with `403`. Deployment behind a reverse proxy must
forward the original scheme and host correctly. The existing lack of permissive
CORS configuration remains unchanged.

Token export is a read endpoint protected by the browser same-origin policy and
has no permissive CORS headers. It still validates that the cookie maps to the
owner resolved by middleware.

### UI integration

The persistent Gradio UI uses a small same-origin JavaScript helper for the two
session endpoints:

1. Show token fetches `/api/session/token` and places the returned value into
   the transfer field.
2. Use token posts the pasted value to `/api/session/import` with
   `Content-Type: application/json`.
3. On success, the browser reloads and Gradio reconstructs history using the
   newly set HttpOnly cookie.
4. On failure, the UI shows a generic invalid-token or security message and
   does not change the current cookie.

No token is placed in a link, page location, local storage, or Gradio persistent
state.

## Error Handling

- Missing or malformed current cookie during export: `401`; middleware may set
  a fresh cookie on the response, but no token is disclosed by that failed
  request.
- Malformed or unknown imported token: `400` with a generic message.
- Missing or foreign `Origin`: `403`.
- Database error: `500` generic response; details remain in server logs without
  the raw token.
- Failed import leaves the current browser cookie and visible history intact.

## Testing

Automated tests cover:

- export returns the current raw token with `Cache-Control: no-store`;
- export never returns another owner's token;
- imported known token sets the configured protected cookie;
- a second TestClient sees the first browser's history after import;
- the first browser retains simultaneous access;
- import switches rather than merges histories;
- malformed and unknown tokens are rejected without creating owners;
- failed imports do not replace the existing cookie;
- missing and foreign origins are rejected;
- cookie flags match the existing session middleware;
- UI contains explicit reveal/import controls and security guidance;
- the existing owner-isolation and full test suites remain green.

## Documentation

README browser-history documentation will explain how to copy the access token,
use it on another device, preserve old tokens before switching, and treat every
token as a password. HTTPS remains strongly recommended outside localhost.

