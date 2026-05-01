# models/database.py

from sqlalchemy import (
    Column, String, Boolean, DateTime, 
    Text, Enum as SAEnum, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum
import uuid

Base = declarative_base()


class UserRole(str, enum.Enum):
    employee   = "employee"
    manager    = "manager"
    hr         = "hr"
    finance    = "finance"
    executive  = "executive"
    admin      = "admin"    


class User(Base):
    __tablename__ = "users"

    # Primary Key
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Identity
    user_id    = Column(String(50),  unique=True, nullable=False, index=True)
    username   = Column(String(100), unique=True, nullable=False, index=True)
    email      = Column(String(255), unique=True, nullable=False, index=True)
    
    # Auth
    hashed_password = Column(Text, nullable=False)
    
    # Role & Access
    role       = Column(SAEnum(UserRole), nullable=False, default=UserRole.employee)
    is_active  = Column(Boolean, default=True, nullable=False)
    
    # Audit
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), onupdate=func.now())
    last_login_at   = Column(DateTime(timezone=True), nullable=True)
    
    # Security
    failed_login_attempts = Column(String(10), default="0")
    is_locked             = Column(Boolean, default=False)

    __table_args__ = (
        Index("ix_users_email_active", "email", "is_active"),
        Index("ix_users_role",         "role"),
    )

    def __repr__(self):
        return f"<User {self.username} role={self.role}>"