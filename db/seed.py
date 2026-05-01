# db/seed.py
# Recreates ALL your mock users in PostgreSQL

from db.connection import AsyncSessionLocal
from db.user_repository import UserRepository
from models.database import UserRole
import logging
import asyncio

logger = logging.getLogger(__name__)

# ── Same mock data, now production-grade ──────────────────────
SEED_USERS = [
    {
        "user_id":  "usr_001",
        "username": "alice",
        "email":    "alice@techcorp.com",
        "password": "employee_pass",      # Will be bcrypt hashed
        "role":     UserRole.employee
    },
    {
        "user_id":  "usr_002",
        "username": "bob",
        "email":    "bob@techcorp.com",
        "password": "manager_pass",
        "role":     UserRole.manager
    },
    {
        "user_id":  "usr_003",
        "username": "carol",
        "email":    "carol@techcorp.com",
        "password": "hr_pass",
        "role":     UserRole.hr
    },
    {
        "user_id":  "usr_004",
        "username": "dave",
        "email":    "dave@techcorp.com",
        "password": "finance_pass",
        "role":     UserRole.finance
    },
    {
        "user_id":  "usr_005",
        "username": "eve",
        "email":    "eve@techcorp.com",
        "password": "exec_pass",
        "role":     UserRole.executive
    },
]


async def seed_users():
    """
    Run once on startup.
    Skips users that already exist (idempotent).
    """
    async with AsyncSessionLocal() as db:
        repo = UserRepository(db)
        
        for user_data in SEED_USERS:
            # Check if already exists
            existing = await repo.get_by_username(user_data["username"])
            
            if existing:
                logger.info(
                    f"SEED_SKIP: {user_data['username']} already exists"
                )
                continue
            
            await repo.create_user(
                user_id  = user_data["user_id"],
                username = user_data["username"],
                email    = user_data["email"],
                password = user_data["password"],
                role     = user_data["role"]
            )
            logger.info(
                f"SEED_INSERT: {user_data['username']} "
                f"role={user_data['role'].value}"
            )


if __name__ == "__main__":
    asyncio.run(seed_users())