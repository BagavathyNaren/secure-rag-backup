# db/url_utils.py  (already imported in app/server.py — extend with mask guard)
import re

_SECRET_RE = re.compile(r"(?<=://)[^:]+:[^@]+(?=@)")

_ALLOWED_SCHEMES = (
    "postgresql://",
    "postgresql+asyncpg://",
    "postgres://",
)


def redact_database_url(url: str) -> str:
    """
    Replace credentials in a DSN with ***.
    This is the ONLY function that should appear in log statements.

    Example
    -------
    postgresql://user:s3cr3t@host/db  →  postgresql://***@host/db
    """
    return _SECRET_RE.sub("***", url)


def validate_database_url(url: str) -> None:
    """
    Raise ValueError if the URL is empty or uses an unsupported scheme.
    Never logs or re-raises the raw URL in the exception message.
    """
    if not url:
        raise ValueError("DATABASE_URL is empty or not set.")

    if not any(url.startswith(scheme) for scheme in _ALLOWED_SCHEMES):
        raise ValueError(
            f"DATABASE_URL uses an unsupported scheme. "
            f"Expected one of: {', '.join(_ALLOWED_SCHEMES)}"
            # ↑ scheme list only — raw URL is intentionally excluded
        )