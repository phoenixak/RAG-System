"""
Security Utilities
Password hashing, verification, and security-related functions.
"""

import secrets
import string
from typing import Optional

from passlib.context import CryptContext

from src.core.logging import get_logger, log_security_event

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """Security management class for password operations."""

    def __init__(self):
        self.pwd_context = pwd_context

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        try:
            hashed = self.pwd_context.hash(password)
            logger.debug("Password hashed successfully")
            return hashed
        except Exception as e:
            logger.error("Failed to hash password", error=str(e))
            raise ValueError("Password hashing failed") from e

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            is_valid = self.pwd_context.verify(plain_password, hashed_password)
            if is_valid:
                logger.debug("Password verification successful")
            else:
                logger.warning("Password verification failed")
                log_security_event("password_verification_failed")
            return is_valid
        except Exception as e:
            logger.error("Password verification error", error=str(e))
            log_security_event("password_verification_error", details={"error": str(e)})
            return False

    def needs_rehash(self, hashed_password: str) -> bool:
        """Check if a password hash needs to be updated."""
        try:
            return self.pwd_context.needs_update(hashed_password)
        except Exception as e:
            logger.error("Error checking if hash needs update", error=str(e))
            return False

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        try:
            # Use URL-safe characters
            alphabet = string.ascii_letters + string.digits + "-_"
            token = "".join(secrets.choice(alphabet) for _ in range(length))
            logger.debug("Secure token generated", length=length)
            return token
        except Exception as e:
            logger.error("Failed to generate secure token", error=str(e))
            raise ValueError("Token generation failed") from e

    def generate_password_reset_token(self) -> str:
        """Generate a password reset token."""
        return self.generate_secure_token(64)

    def generate_api_key(self) -> str:
        """Generate an API key."""
        return self.generate_secure_token(48)

    def is_password_secure(self, password: str) -> tuple[bool, list[str]]:
        """
        Check if a password meets security requirements.
        Returns (is_secure, list_of_issues).
        """
        issues = []

        # Length check
        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")

        if len(password) > 128:
            issues.append("Password must be less than 128 characters long")

        # Character type checks
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

        if not has_upper:
            issues.append("Password must contain at least one uppercase letter")

        if not has_lower:
            issues.append("Password must contain at least one lowercase letter")

        if not has_digit:
            issues.append("Password must contain at least one digit")

        if not has_special:
            issues.append("Password must contain at least one special character")

        # Common password checks (basic)
        common_passwords = [
            "password",
            "123456",
            "password123",
            "admin",
            "qwerty",
            "letmein",
            "welcome",
            "monkey",
            "dragon",
            "sunshine",
        ]

        if password.lower() in common_passwords:
            issues.append("Password is too common")

        # Sequential character check
        if self._has_sequential_chars(password):
            issues.append("Password contains sequential characters")

        # Repeated character check
        if self._has_repeated_chars(password):
            issues.append("Password contains too many repeated characters")

        is_secure = len(issues) == 0
        return is_secure, issues

    def _has_sequential_chars(self, password: str, min_length: int = 3) -> bool:
        """Check for sequential characters in password."""
        sequences = [
            "abcdefghijklmnopqrstuvwxyz",
            "0123456789",
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm",
        ]

        password_lower = password.lower()

        for seq in sequences:
            for i in range(len(seq) - min_length + 1):
                substring = seq[i : i + min_length]
                if substring in password_lower or substring[::-1] in password_lower:
                    return True

        return False

    def _has_repeated_chars(self, password: str, max_repeated: int = 3) -> bool:
        """Check for repeated characters in password."""
        if len(password) < max_repeated:
            return False

        for i in range(len(password) - max_repeated + 1):
            char = password[i]
            if all(password[i + j] == char for j in range(max_repeated)):
                return True

        return False

    def sanitize_input(self, input_str: str, max_length: Optional[int] = None) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not isinstance(input_str, str):
            return ""

        # Remove null bytes and control characters
        sanitized = "".join(
            char for char in input_str if ord(char) >= 32 or char in "\t\n\r"
        )

        # Trim whitespace
        sanitized = sanitized.strip()

        # Apply length limit
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    def is_safe_filename(self, filename: str) -> bool:
        """Check if a filename is safe for storage."""
        if not filename:
            return False

        # Check for dangerous characters
        dangerous_chars = ["/", "\\", "..", "<", ">", ":", '"', "|", "?", "*"]

        for char in dangerous_chars:
            if char in filename:
                return False

        # Check for reserved names (Windows)
        reserved_names = [
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        ]

        name_without_ext = filename.split(".")[0].upper()
        if name_without_ext in reserved_names:
            return False

        # Check length
        if len(filename) > 255:
            return False

        return True


# Global security manager instance
security_manager = SecurityManager()


def hash_password(password: str) -> str:
    """Hash a password."""
    return security_manager.hash_password(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return security_manager.verify_password(plain_password, hashed_password)


def needs_password_rehash(hashed_password: str) -> bool:
    """Check if a password hash needs to be updated."""
    return security_manager.needs_rehash(hashed_password)


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return security_manager.generate_secure_token(length)


def generate_password_reset_token() -> str:
    """Generate a password reset token."""
    return security_manager.generate_password_reset_token()


def is_password_secure(password: str) -> tuple[bool, list[str]]:
    """Check if a password meets security requirements."""
    return security_manager.is_password_secure(password)


def sanitize_input(input_str: str, max_length: Optional[int] = None) -> str:
    """Sanitize user input."""
    return security_manager.sanitize_input(input_str, max_length)


def is_safe_filename(filename: str) -> bool:
    """Check if a filename is safe for storage."""
    return security_manager.is_safe_filename(filename)


# FastAPI authentication dependencies
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.models import TokenData

security = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> TokenData:
    """
    FastAPI dependency to get the current authenticated user from JWT token.
    """
    try:
        token = credentials.credentials

        # Import here to avoid circular imports
        from src.auth.jwt_utils import verify_access_token

        # Properly verify the JWT token
        token_data = verify_access_token(token)

        log_security_event(
            "user_authenticated",
            details={
                "user_id": token_data.user_id,
                "email": token_data.email,
                "role": token_data.role.value,
            },
        )

        return token_data

    except Exception as e:
        logger.error("Authentication error", error=str(e))
        log_security_event("authentication_error", details={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: Annotated[TokenData, Depends(get_current_user)],
) -> TokenData:
    """
    FastAPI dependency to get current active user (additional checks can be added here).
    """
    # Here you could add additional checks like:
    # - Check if user is active in database
    # - Check if user account is not disabled
    # - Check rate limiting, etc.

    return current_user
