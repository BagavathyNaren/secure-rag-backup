import os
from collections.abc import Mapping


REQUIRED_ENV_VARS = (
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "JWT_SECRET_KEY",
)

INSECURE_JWT_SECRET_VALUES = {
    "",
    "CHANGE_ME_IN_PRODUCTION_USE_LONG_RANDOM_STRING",
    "change-me",
    "changeme",
    "secret",
}


def _clean(value: object) -> str:
    return str(value or "").strip()


def get_csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def validate_required_env(env: Mapping[str, object] | None = None) -> None:
    env = env or os.environ
    missing = [name for name in REQUIRED_ENV_VARS if not _clean(env.get(name))]

    jwt_secret = _clean(env.get("JWT_SECRET_KEY"))
    insecure = []
    if jwt_secret in INSECURE_JWT_SECRET_VALUES or len(jwt_secret) < 32:
        insecure.append("JWT_SECRET_KEY")

    if missing or insecure:
        details = []
        if missing:
            details.append("missing required env vars: " + ", ".join(sorted(missing)))
        if insecure:
            details.append(
                "insecure env vars: "
                + ", ".join(sorted(insecure))
                + " (set a random value at least 32 characters long)"
            )
        raise RuntimeError("Invalid runtime configuration: " + "; ".join(details))

