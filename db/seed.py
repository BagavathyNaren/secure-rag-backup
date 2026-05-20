import asyncio
import logging
import os

from db.connection import AsyncSessionLocal
from db.user_repository import UserRepository
from models.database import UserRole


logger = logging.getLogger(__name__)


DEMO_USERS = [
    {
        "user_id": "usr_001",
        "username": "alice",
        "email": "alice@techcorp.com",
        "password_env": "SEED_ALICE_PASSWORD",
        "role": UserRole.employee,
    },
    {
        "user_id": "usr_002",
        "username": "bob",
        "email": "bob@techcorp.com",
        "password_env": "SEED_BOB_PASSWORD",
        "role": UserRole.manager,
    },
    {
        "user_id": "usr_003",
        "username": "carol",
        "email": "carol@techcorp.com",
        "password_env": "SEED_CAROL_PASSWORD",
        "role": UserRole.hr,
    },
    {
        "user_id": "usr_004",
        "username": "dave",
        "email": "dave@techcorp.com",
        "password_env": "SEED_DAVE_PASSWORD",
        "role": UserRole.finance,
    },
    {
        "user_id": "usr_005",
        "username": "eve",
        "email": "eve@techcorp.com",
        "password_env": "SEED_EVE_PASSWORD",
        "role": UserRole.executive,
    },
    {
        "user_id": "usr_006",
        "username": "frank",
        "email": "frank@techcorp.com",
        "password_env": "SEED_FRANK_PASSWORD",
        "role": UserRole.security,
    },
    {
        "user_id": "usr_007",
        "username": "grace",
        "email": "grace@techcorp.com",
        "password_env": "SEED_GRACE_PASSWORD",
        "role": UserRole.admin,
    },
]


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _password_for(user_data: dict) -> str | None:
    configured = os.getenv(user_data["password_env"], "").strip()
    if configured:
        return configured
    return None


async def seed_users():
    """
    Optionally seed demo users.

    Set SEED_DEMO_USERS=1 to seed. Provide per-user password secrets such as
    SEED_ALICE_PASSWORD. No default passwords are shipped with the app.
    """
    if not _truthy_env("SEED_DEMO_USERS"):
        logger.info("SEED_SKIP: SEED_DEMO_USERS is disabled")
        return

    async with AsyncSessionLocal() as db:
        repo = UserRepository(db)

        for user_data in DEMO_USERS:
            existing = await repo.get_by_username(user_data["username"])
            if existing:
                logger.info("SEED_SKIP: %s already exists", user_data["username"])
                continue

            password = _password_for(user_data)
            if not password:
                raise RuntimeError(
                    "Missing seed password for "
                    f"{user_data['username']}. Set {user_data['password_env']} "
                    "before enabling SEED_DEMO_USERS."
                )

            await repo.create_user(
                user_id=user_data["user_id"],
                username=user_data["username"],
                email=user_data["email"],
                password=password,
                role=user_data["role"],
            )
            logger.info(
                "SEED_INSERT: %s role=%s",
                user_data["username"],
                user_data["role"].value,
            )


if __name__ == "__main__":
    asyncio.run(seed_users())
