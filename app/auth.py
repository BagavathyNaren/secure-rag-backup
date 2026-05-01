# app/auth.py

import os
from sqlalchemy import select, or_
from db.connection import AsyncSessionLocal
from models.database import User as UserModel
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ============================================================
# CONFIG
# ============================================================

# Set these in HF Spaces secrets:
# JWT_SECRET_KEY  → long random string (e.g. openssl rand -hex 32)
# JWT_ALGORITHM   → HS256 (default)
# JWT_EXPIRE_MINS → 60 (default)

SECRET_KEY  = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_USE_LONG_RANDOM_STRING")
ALGORITHM   = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINS = int(os.getenv("JWT_EXPIRE_MINS", "60"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ============================================================
# PASSWORD HELPERS (pure bcrypt — no passlib)
# ============================================================

def hash_password(plain: str) -> str:
    """Hash a plain text password using bcrypt."""
    return bcrypt.hashpw(
        plain[:72].encode("utf-8"),      # bcrypt hard limit = 72 bytes
        bcrypt.gensalt()
    ).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain text password against a bcrypt hash."""
    return bcrypt.checkpw(
        plain[:72].encode("utf-8"),      # same 72 byte limit
        hashed.encode("utf-8")
    )
def _user_to_dict(u: UserModel) -> dict:
    return {
        "user_id": u.user_id,
        "username": u.username,
        "email": u.email,
        "role": (u.role.value if hasattr(u.role, "value") else str(u.role)),
        "hashed_password": u.hashed_password,
        "active": bool(u.is_active),
        "is_locked": bool(getattr(u, "is_locked", False)),
    }


async def authenticate_user_pg(identifier: str, password: str) -> Optional[dict]:
    """
    Authenticate against PostgreSQL (Neon).
    Identifier may be username OR email OR user_id.
    Returns a user dict compatible with create_access_token().
    """
    async with AsyncSessionLocal() as session:
        stmt = select(UserModel).where(
            or_(
                UserModel.username == identifier,
                UserModel.email == identifier,
                UserModel.user_id == identifier,
            )
        )
        user_obj = (await session.execute(stmt)).scalar_one_or_none()

    if not user_obj:
        return None
    if not user_obj.is_active:
        return None
    if getattr(user_obj, "is_locked", False):
        return None
    if not verify_password(password, user_obj.hashed_password):
        return None

    return _user_to_dict(user_obj)

# ============================================================
# MOCK USER DATABASE
# Passwords are bcrypt hashed at module load time.
#
# In production → replace with real DB (PostgreSQL, etc.)
#
# To generate a hash manually in Python:
#   import bcrypt
#   print(bcrypt.hashpw(b"your_password", bcrypt.gensalt()).decode())
# ============================================================

def _make_user(user_id, username, email, role, password) -> dict:
    return {
        "user_id":         user_id,
        "username":        username,
        "email":           email,
        "role":            role,
        "hashed_password": hash_password(password),
        "active":          True,
    }

USERS_DB: Dict[str, dict] = {
    "alice": _make_user("usr_001", "alice", "alice@techcorp.com", "employee", "employee_pass"),
    "bob":   _make_user("usr_002", "bob",   "bob@techcorp.com",   "finance",  "finance_pass"),
    "carol": _make_user("usr_003", "carol", "carol@techcorp.com", "security", "security_pass"),
    "dave":  _make_user("usr_004", "dave",  "dave@techcorp.com",  "admin",    "admin_pass"),
}


# ============================================================
# USER AUTHENTICATION
# ============================================================

def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = USERS_DB.get(username)
    if not user:
        return None
    if not user["active"]:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


# ============================================================
# JWT HELPERS
# ============================================================

def create_access_token(user: dict) -> str:
    """
    Creates a signed JWT token containing:
    - sub      : username
    - user_id  : unique user ID
    - email    : user email
    - role     : role assigned by SERVER (cannot be faked by client)
    - exp      : expiry timestamp
    - iat      : issued at
    """
    expire = datetime.utcnow() + timedelta(minutes=EXPIRE_MINS)
    payload = {
        "sub":     user["username"],
        "user_id": user["user_id"],
        "email":   user["email"],
        "role":    user["role"],       # ← role set by SERVER, not client
        "exp":     expire,
        "iat":     datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decodes and validates JWT token.
    Raises HTTPException 401 if invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================
# FASTAPI DEPENDENCIES
# ============================================================

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    FastAPI dependency — validates JWT and returns user payload.
    Usage: current_user = Depends(get_current_user)
    """
    payload = decode_token(token)

    # Double-check user still exists and is active
    username = payload.get("sub")
    user = USERS_DB.get(username)
    if not user or not user["active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def require_role(allowed_roles: list):
    """
    Role guard dependency factory.
    Usage: current_user = Depends(require_role(["admin", "security"]))
    """
    def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access denied. "
                    f"Required: {allowed_roles}. "
                    f"Your role: {current_user['role']}"
                )
            )
        return current_user
    return role_checker