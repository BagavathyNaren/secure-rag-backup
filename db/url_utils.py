# db/url_utils.py

from urllib.parse import urlparse, urlunparse


def redact_database_url(url: str) -> str:
    """
    postgresql://user:pass@host:5432/db?sslmode=require
    -> postgresql://user:***@host:5432/db
    """
    if not url:
        return ""

    p = urlparse(url)
    netloc = p.netloc

    # redact password if present
    if "@" in netloc:
        creds, hostpart = netloc.split("@", 1)
        if ":" in creds:
            user = creds.split(":", 1)[0]
            netloc = f"{user}:***@{hostpart}"
        else:
            netloc = f"{creds}@{hostpart}"

    # strip query/fragment entirely
    return urlunparse((p.scheme, netloc, p.path, "", "", ""))


def validate_database_url(url: str) -> None:
    if not url or not url.strip():
        raise ValueError("DATABASE_URL is empty")

    p = urlparse(url)

    if p.scheme not in ("postgresql", "postgres", "postgresql+asyncpg"):
        raise ValueError(f"Unsupported DATABASE_URL scheme: {p.scheme}")

    if not p.hostname:
        raise ValueError("DATABASE_URL missing hostname")

    if not p.path or p.path == "/":
        raise ValueError("DATABASE_URL missing database name (path)")

    # Usually required in prod; if you use IAM/peer auth you can relax this later
    if p.username is None or not p.username.strip():
        raise ValueError("DATABASE_URL missing username")