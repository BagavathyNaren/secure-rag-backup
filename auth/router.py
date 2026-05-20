 # Updated login route
# auth/router.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from db.connection import get_db
from db.user_repository import UserRepository
from app.auth import create_access_token
import logging
import json

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/auth/login")
async def login(
    credentials: dict,
    db: AsyncSession = Depends(get_db)
):
    if "username" not in credentials or "password" not in credentials:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="username and password are required",
        )

    repo = UserRepository(db)
    
    user = await repo.authenticate(
        username = credentials["username"],
        password = credentials["password"]
    )
    
    if not user:
        logger.warning(json.dumps({
            "event": "LOGIN_FAILED",
            "username": credentials.get("username")
        }))
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid credentials"
        )
    
    token = create_access_token({
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "role": user.role.value,
    })
    
    logger.info(json.dumps({
        "trace_id": "auth",
        "event":    "LOGIN_SUCCESS",
        "user_id":  user.user_id,
        "username": user.username,
        "email":    user.email,
        "role":     user.role.value
    }))
    
    return {"access_token": token, "token_type": "bearer"}
