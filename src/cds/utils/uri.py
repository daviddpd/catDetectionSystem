from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def redact_uri_password(source: str) -> str:
    """Redact URI passwords in userinfo while preserving the rest of the URI.

    Example:
        rtsp://user:secret@example.local/stream -> rtsp://user:***@example.local/stream
    """

    if not source or "://" not in source:
        return source

    try:
        parts = urlsplit(source)
    except Exception:
        return source

    if not parts.scheme or not parts.netloc:
        return source

    userinfo, sep, hostinfo = parts.netloc.rpartition("@")
    if not sep:
        return source

    user, has_password, _password = userinfo.partition(":")
    if not has_password:
        return source

    redacted_netloc = f"{user}:***@{hostinfo}"
    return urlunsplit(
        (
            parts.scheme,
            redacted_netloc,
            parts.path,
            parts.query,
            parts.fragment,
        )
    )
