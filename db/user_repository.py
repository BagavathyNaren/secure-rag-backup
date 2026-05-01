# db/user_repository.py

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.sql import func
from models.database import User, UserRole
from passlib.context import CryptContext
from datetime import datetime, timezone
import logging

logger  = logging.getLogger(__name__)
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    # ─── Password Helpers ───────────────────────────────────────
    @staticmethod
    def hash_password(plain: str) -> str:
        return pwd_ctx.hash(plain)

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return pwd_ctx.verify(plain, hashed)

    # ─── Read ────────────────────────────────────────────────────
    async def get_by_username(self, username: str) -> User | None:
        result = await self.db.execute(
            select(User).where(
                User.username  == username,
                User.is_active == True
            )
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        result = await self.db.execute(
            select(User).where(
                User.email     == email,
                User.is_active == True
            )
        )
        return result.scalar_one_or_none()

    async def get_by_user_id(self, user_id: str) -> User | None:
        result = await self.db.execute(
            select(User).where(
                User.user_id   == user_id,
                User.is_active == True
            )
        )
        return result.scalar_one_or_none()

    # ─── Auth ────────────────────────────────────────────────────
    async def authenticate(
        self, 
        username: str, 
        password: str
    ) -> User | None:
        
        user = await self.get_by_username(username)
        
        if not user:
            logger.warning(f"Login attempt: unknown user '{username}'")
            return None
            
        if user.is_locked:
            logger.warning(f"Login attempt: locked account '{username}'")
            return None

        if not self.verify_password(password, user.hashed_password):
            # Increment failed attempts
            await self.db.execute(
                update(User)
                .where(User.id == user.id)
                .values(
                    failed_login_attempts=str(
                        int(user.failed_login_attempts or 0) + 1
                    )
                )
            )
            await self.db.commit()
            logger.warning(f"Failed login for '{username}'")
            return None

        # Success — reset failed attempts, update last login
        await self.db.execute(
            update(User)
            .where(User.id == user.id)
            .values(
                failed_login_attempts="0",
                last_login_at=func.now()
            )
        )
        await self.db.commit()
        return user

    # ─── Write ───────────────────────────────────────────────────
    async def create_user(
        self,
        user_id:  str,
        username: str,
        email:    str,
        password: str,
        role:     UserRole = UserRole.employee
    ) -> User:
        
        user = User(
            user_id         = user_id,
            username        = username,
            email           = email,
            hashed_password = self.hash_password(password),
            role            = role
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        logger.info(f"Created user: {username} role={role}")
        return user