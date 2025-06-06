"""
Authentication Models
User models, token models, and authentication-related data structures.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field, validator


class UserRole(str, Enum):
    """User role enumeration."""

    ADMIN = "admin"
    POWER_USER = "power_user"
    STANDARD_USER = "standard_user"
    READ_ONLY = "read_only"


class Permission(str, Enum):
    """Permission enumeration."""

    READ = "read"
    WRITE = "write"
    UPLOAD = "upload"
    MANAGE_DOCS = "manage_docs"
    ADMIN = "admin"


# Role to permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.READ,
        Permission.WRITE,
        Permission.UPLOAD,
        Permission.MANAGE_DOCS,
        Permission.ADMIN,
    ],
    UserRole.POWER_USER: [
        Permission.READ,
        Permission.WRITE,
        Permission.UPLOAD,
        Permission.MANAGE_DOCS,
    ],
    UserRole.STANDARD_USER: [Permission.READ, Permission.WRITE, Permission.UPLOAD],
    UserRole.READ_ONLY: [Permission.READ],
}


class UserBase(BaseModel):
    """Base user model with common fields."""

    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: UserRole = UserRole.STANDARD_USER
    is_active: bool = True


class UserCreate(UserBase):
    """User creation model."""

    password: str = Field(..., min_length=8, max_length=128)

    @validator("password")
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")

        # Check for at least one uppercase, lowercase, digit, and special character
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v)

        if not all([has_upper, has_lower, has_digit, has_special]):
            raise ValueError(
                "Password must contain at least one uppercase letter, "
                "one lowercase letter, one digit, and one special character"
            )

        return v


class UserUpdate(BaseModel):
    """User update model."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """User model as stored in database."""

    id: UUID = Field(default_factory=uuid4)
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    is_locked: bool = False

    class Config:
        orm_mode = True


class User(UserBase):
    """User model for API responses."""

    id: UUID
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None

    @classmethod
    def from_db_user(cls, db_user: UserInDB) -> "User":
        """Create User from UserInDB."""
        return cls(
            id=db_user.id,
            email=db_user.email,
            first_name=db_user.first_name,
            last_name=db_user.last_name,
            role=db_user.role,
            is_active=db_user.is_active,
            permissions=ROLE_PERMISSIONS.get(db_user.role, []),
            created_at=db_user.created_at,
            last_login=db_user.last_login,
        )

    class Config:
        orm_mode = True


class LoginRequest(BaseModel):
    """Login request model."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User


class TokenRefreshRequest(BaseModel):
    """Token refresh request model."""

    refresh_token: str


class TokenData(BaseModel):
    """Token payload data."""

    user_id: str
    email: str
    role: UserRole
    permissions: List[Permission]
    token_type: str  # "access" or "refresh"
    exp: int
    iat: int
    jti: str  # JWT ID for token blacklisting


class PasswordChangeRequest(BaseModel):
    """Password change request model."""

    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)

    @validator("new_password")
    def validate_new_password(cls, v):
        """Validate new password strength."""
        return UserCreate.validate_password(v)


class PasswordResetRequest(BaseModel):
    """Password reset request model."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model."""

    token: str
    new_password: str = Field(..., min_length=8, max_length=128)

    @validator("new_password")
    def validate_new_password(cls, v):
        """Validate new password strength."""
        return UserCreate.validate_password(v)


class UserSettings(BaseModel):
    """User settings model."""

    default_search_limit: int = Field(default=10, ge=1, le=100)
    preferred_model: str = "gpt-4"
    theme: str = "light"
    notifications_enabled: bool = True
    language: str = "en"


class UserProfile(User):
    """Extended user profile with settings."""

    settings: UserSettings
