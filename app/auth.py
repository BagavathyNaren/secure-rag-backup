# app/auth.py

import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ============================================================
# CONFIG
# ============================================================

# Set these in HF Spaces secrets:
# JWT_SECRET_KEY  → long random string (e.g. openssl rand -hex 32)
# JWT_ALGORITHM   → HS256 (default)
# JWT_EXPIRE_MINS → 60 (default)

SECRET_KEY   = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_USE_LONG_RANDOM_STRING")
ALGORITHM    = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINS  = int(os.getenv("JWT_EXPIRE_MINS", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ============================================================
# MOCK USER DATABASE
# In production → replace with real DB (PostgreSQL, etc.)
# Passwords are bcrypt hashed.
#
# To generate a hashed password in Python:
#   from passlib.context import CryptContext
#   pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
#   print(pwd_context.hash("your_password_here"))
# ============================================================

USERS_DB: Dict[str, dict] = {
    "alice": {
        "user_id":       "usr_001",
        "username":      "alice",
        "email":         "alice@techcorp.com",
        "role":          "employee",
        "hashed_password": pwd_context.hash("employee_pass_123"),
        "active":        True,
    },
    "bob": {
        "user_id":       "usr_002",
        "username":      "bob",
        "email":         "bob@techcorp.com",
        "role":          "finance",
        "hashed_password": pwd_context.hash("finance_pass_456"),
        "active":        True,
    },
    "carol": {
        "user_id":       "usr_003",
        "username":      "carol",
        "email":         "carol@techcorp.com",
        "role":          "security",
        "hashed_password": pwd_context.hash("security_pass_789"),
        "active":        True,
    },
    "dave": {
        "user_id":       "usr_004",
        "username":      "dave",
        "email":         "dave@techcorp.com",
        "role":          "admin",
        "hashed_password": pwd_context.hash("admin_pass_000"),
        "active":        True,
    },
}


# ============================================================
# PASSWORD HELPERS
# ============================================================

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

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
    """
    expire = datetime.utcnow() + timedelta(minutes=EXPIRE_MINS)
    payload = {
        "sub":     user["username"],
        "user_id": user["user_id"],
        "email":   user["email"],
        "role":    user["role"],          # ← role set by SERVER
        "exp":     expire,
        "iat":     datetime.utcnow(),     # issued at
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    """
    Decodes and validates JWT token.
    Raises HTTPException if invalid or expired.
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
# FASTAPI DEPENDENCY — use in any endpoint
# ============================================================

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    FastAPI dependency.
    Usage: current_user = Depends(get_current_user)
    Returns the full decoded token payload.
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
    Role-based access dependency factory.
    Usage: Depends(require_role(["admin", "security"]))
    """
    def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {allowed_roles}. "
                       f"Your role: {current_user['role']}"
            )
        return current_user
    return role_checker