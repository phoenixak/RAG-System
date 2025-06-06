"""
Authentication API Endpoints
Login, logout, token refresh, and user management endpoints.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.jwt_utils import (
    JWTError,
    create_access_token,
    create_refresh_token,
    revoke_token,
    verify_access_token,
    verify_refresh_token,
)
from src.auth.models import (
    ROLE_PERMISSIONS,
    LoginRequest,
    TokenRefreshRequest,
    TokenResponse,
    User,
    UserRole,
)
from src.auth.security import verify_password
from src.core.config import get_settings
from src.core.logging import get_logger, log_security_event

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)
security = HTTPBearer()


# Temporary in-memory user store for Phase 1
# TODO: Replace with actual database in Phase 2
TEMP_USERS = {
    "admin@example.com": {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "email": "admin@example.com",
        "password_hash": "$2b$12$DDiv0KIkJECtopamHRsoMeP0m2fWXAxWpU.Bfu8ZvGwzFecY2zsGS",  # "admin123!"
        "first_name": "Admin",
        "last_name": "User",
        "role": UserRole.ADMIN,
        "is_active": True,
        "created_at": datetime.utcnow(),
    },
    "user@example.com": {
        "id": "550e8400-e29b-41d4-a716-446655440002",
        "email": "user@example.com",
        "password_hash": "$2b$12$vCkOSlxIK5lPah/s/anjF.YP3SQVDRXqSKK8leDgPlYzK0yjXcDP6",  # "password123!"
        "first_name": "Test",
        "last_name": "User",
        "role": UserRole.STANDARD_USER,
        "is_active": True,
        "created_at": datetime.utcnow(),
    },
}


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """Get current authenticated user from JWT token."""
    try:
        # Extract token from credentials
        token = credentials.credentials

        # Verify token
        token_data = verify_access_token(token)

        # Get user from temporary store
        # TODO: Replace with database query
        user_data = None
        for email, data in TEMP_USERS.items():
            if data["id"] == token_data.user_id:
                user_data = data
                break

        if not user_data:
            log_security_event(
                "user_not_found_for_valid_token",
                user_id=token_data.user_id,
                details={"email": token_data.email},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )

        if not user_data["is_active"]:
            log_security_event(
                "inactive_user_access_attempt",
                user_id=token_data.user_id,
                details={"email": token_data.email},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive user"
            )

        # Create User object
        user = User(
            id=user_data["id"],
            email=user_data["email"],
            first_name=user_data["first_name"],
            last_name=user_data["last_name"],
            role=user_data["role"],
            is_active=user_data["is_active"],
            permissions=ROLE_PERMISSIONS.get(user_data["role"], []),
            created_at=user_data["created_at"],
        )

        return user

    except JWTError as e:
        log_security_event("invalid_token_used", details={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login(login_data: LoginRequest):
    """
    User login endpoint.
    Validates credentials and returns JWT tokens.
    """
    try:
        # Find user in temporary store
        # TODO: Replace with database query
        user_data = TEMP_USERS.get(login_data.email.lower())

        if not user_data:
            log_security_event(
                "login_attempt_nonexistent_user", details={"email": login_data.email}
            )
            # Use generic error to prevent user enumeration
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        if not user_data["is_active"]:
            log_security_event(
                "login_attempt_inactive_user",
                user_id=user_data["id"],
                details={"email": login_data.email},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is inactive"
            )

        # Verify password
        if not verify_password(login_data.password, user_data["password_hash"]):
            log_security_event(
                "login_attempt_wrong_password",
                user_id=user_data["id"],
                details={"email": login_data.email},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Create tokens
        permissions = ROLE_PERMISSIONS.get(user_data["role"], [])

        access_token = create_access_token(
            user_id=user_data["id"],
            email=user_data["email"],
            role=user_data["role"],
            permissions=permissions,
        )

        refresh_token = create_refresh_token(
            user_id=user_data["id"], email=user_data["email"], role=user_data["role"]
        )

        # Update last login
        # TODO: Update in database
        user_data["last_login"] = datetime.utcnow()

        # Create user object
        user = User(
            id=user_data["id"],
            email=user_data["email"],
            first_name=user_data["first_name"],
            last_name=user_data["last_name"],
            role=user_data["role"],
            is_active=user_data["is_active"],
            permissions=permissions,
            created_at=user_data["created_at"],
            last_login=user_data["last_login"],
        )

        logger.info(
            "User logged in successfully",
            user_id=user_data["id"],
            email=user_data["email"],
        )

        log_security_event(
            "successful_login",
            user_id=user_data["id"],
            details={"email": login_data.email},
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.access_token_expire_minutes * 60,
            user=user,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login error", error=str(e), email=login_data.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_data: TokenRefreshRequest):
    """
    Token refresh endpoint.
    Exchanges a valid refresh token for new access and refresh tokens.
    """
    try:
        # Verify refresh token
        token_data = verify_refresh_token(refresh_data.refresh_token)

        # Get user from temporary store
        # TODO: Replace with database query
        user_data = None
        for email, data in TEMP_USERS.items():
            if data["id"] == token_data.user_id:
                user_data = data
                break

        if not user_data:
            log_security_event(
                "refresh_token_user_not_found", user_id=token_data.user_id
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )

        if not user_data["is_active"]:
            log_security_event(
                "refresh_token_inactive_user", user_id=token_data.user_id
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is inactive"
            )

        # Revoke old refresh token
        revoke_token(refresh_data.refresh_token)

        # Create new tokens
        permissions = ROLE_PERMISSIONS.get(user_data["role"], [])

        new_access_token = create_access_token(
            user_id=user_data["id"],
            email=user_data["email"],
            role=user_data["role"],
            permissions=permissions,
        )

        new_refresh_token = create_refresh_token(
            user_id=user_data["id"], email=user_data["email"], role=user_data["role"]
        )

        # Create user object
        user = User(
            id=user_data["id"],
            email=user_data["email"],
            first_name=user_data["first_name"],
            last_name=user_data["last_name"],
            role=user_data["role"],
            is_active=user_data["is_active"],
            permissions=permissions,
            created_at=user_data["created_at"],
            last_login=user_data.get("last_login"),
        )

        logger.info("Token refreshed successfully", user_id=user_data["id"])

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.access_token_expire_minutes * 60,
            user=user,
        )

    except JWTError as e:
        log_security_event("invalid_refresh_token", details={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed",
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    User logout endpoint.
    Revokes the current access token.
    """
    try:
        # Revoke current token
        token = credentials.credentials
        revoke_token(token)

        logger.info("User logged out successfully", user_id=current_user.id)

        log_security_event(
            "successful_logout",
            user_id=current_user.id,
            details={"email": current_user.email},
        )

        return {"message": "Logged out successfully"}

    except Exception as e:
        logger.error("Logout error", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Logout failed"
        )


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information.
    Returns the authenticated user's profile.
    """
    logger.debug("User info requested", user_id=current_user.id)
    return current_user


@router.get("/verify")
async def verify_token(current_user: User = Depends(get_current_user)):
    """
    Verify token validity.
    Returns basic verification status.
    """
    return {
        "valid": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
    }
