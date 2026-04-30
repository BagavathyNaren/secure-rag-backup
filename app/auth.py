# app/auth.py

import os
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ============================================================
# CONFIG
# ============================================================

SECRET_KEY  = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_USE_LONG_RANDOM_STRING")
ALGORITHM   = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINS = int(os.getenv("JWT_EXPIRE_MINS", "60"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ============================================================
# PASSWORD HELPERS
# ============================================================

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(
        plain[:72].encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(
        plain[:72].encode("utf-8"),
        hashed.encode("utf-8")
    )


# ============================================================
# USER DATABASE
# ✅ Passwords pre-hashed — ZERO bcrypt work at startup
#
# Passwords:
#   alice → "employee_pass"
#   bob   → "finance_pass"
#   carol → "security_pass"
#   dave  → "admin_pass"
# ============================================================

USERS_DB: Dict[str, dict] = {
    "alice": {
        "user_id":         "usr_001",
        "username":        "alice",
        "email":           "alice@techcorp.com",
        "role":            "employee",
        "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/HS.iQeO",
        "active":          True,
    },
    "bob": {
        "user_id":         "usr_002",
        "username":        "bob",
        "email":           "bob@techcorp.com",
        "role":            "finance",
        "hashed_password": "$2b$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi",
        "active":          True,
    },
    "carol": {
        "user_id":         "usr_003",
        "username":        "carol",
        "email":           "carol@techcorp.com",
        "role":            "security",
        "hashed_password": "$2b$12$yGMmN6kxrPlGl1QqQZ1oEO0gS0LB2ZZ8J6VcRzCdO5BqFKbVqCc.m",
        "active":          True,
    },
    "dave": {
        "user_id":         "usr_004",
        "username":        "dave",
        "email":           "dave@techcorp.com",
        "role":            "admin",
        "hashed_password": "$2b$12$GnFt1mEQH1c7rWP5Y8c.F.7vYhS4X3q7MZqCK0kL9KzMLzY3Q8jOu",
        "active":          True,
    },
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
    expire = datetime.utcnow() + timedelta(minutes=EXPIRE_MINS)
    payload = {
        "sub":     user["username"],
        "user_id": user["user_id"],
        "email":   user["email"],
        "role":    user["role"],
        "exp":     expire,
        "iat":     datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
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
    payload  = decode_token(token)
    username = payload.get("sub")
    user     = USERS_DB.get(username)
    if not user or not user["active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def require_role(allowed_roles: list):
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