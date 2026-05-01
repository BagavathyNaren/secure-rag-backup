# db/url_utils.py
import re

_SECRET_RE = re.compile(r"(?<=://)[^:]+:[^@]+(?=@)")

_ALLOWED_SCHEMES = (
    "postgresql://",
    "postgresql+asyncpg://",
    "postgres://",
)


def redact_database_url(url: str) -> str:
    """
    Mask credentials AND query params in a DSN for safe logging.

    Example
    -------
    postgresql://user:s3cr3t@host/db?sslmode=require
    → postgresql://***@host/db

    Rationale
    ---------
    Query params like ?sslmode=require&channel_binding=require can leak
    internal infrastructure details; stripping them reduces the attack surface.
    """
    masked = _SECRET_RE.sub("***", url)
    
    # Strip query params (everything after ?)
    if "?" in masked:
        masked = masked.split("?")[0]
    
    return masked


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
        )