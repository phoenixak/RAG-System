"""
Tests for authentication and security components.
Tests JWT token management, password hashing, and security utilities.
"""

import time
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.auth.models import (
    LoginRequest,
    Permission,
    ROLE_PERMISSIONS,
    TokenData,
    User,
    UserCreate,
    UserRole,
)


class TestUserRole:
    """Tests for UserRole enum and permissions."""

    def test_role_values(self):
        """Test UserRole enum values."""
        assert UserRole.ADMIN == "admin"
        assert UserRole.POWER_USER == "power_user"
        assert UserRole.STANDARD_USER == "standard_user"
        assert UserRole.READ_ONLY == "read_only"

    def test_admin_has_all_permissions(self):
        """Test admin role has all permissions."""
        admin_perms = ROLE_PERMISSIONS[UserRole.ADMIN]
        assert Permission.READ in admin_perms
        assert Permission.WRITE in admin_perms
        assert Permission.UPLOAD in admin_perms
        assert Permission.MANAGE_DOCS in admin_perms
        assert Permission.ADMIN in admin_perms

    def test_readonly_has_only_read(self):
        """Test read-only role has only read permission."""
        readonly_perms = ROLE_PERMISSIONS[UserRole.READ_ONLY]
        assert Permission.READ in readonly_perms
        assert len(readonly_perms) == 1

    def test_standard_user_permissions(self):
        """Test standard user has read, write, upload permissions."""
        perms = ROLE_PERMISSIONS[UserRole.STANDARD_USER]
        assert Permission.READ in perms
        assert Permission.WRITE in perms
        assert Permission.UPLOAD in perms
        assert Permission.ADMIN not in perms


class TestPasswordSecurity:
    """Tests for password hashing and verification."""

    def test_hash_password(self):
        """Test password hashing produces a hash."""
        from src.auth.security import hash_password

        hashed = hash_password("TestPassword123!")
        assert hashed != "TestPassword123!"
        assert len(hashed) > 0

    def test_verify_password_correct(self):
        """Test verifying correct password returns True."""
        from src.auth.security import hash_password, verify_password

        password = "SecurePassword456!"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password returns False."""
        from src.auth.security import hash_password, verify_password

        hashed = hash_password("CorrectPassword123!")
        assert verify_password("WrongPassword123!", hashed) is False

    def test_hash_is_unique_per_call(self):
        """Test that hashing same password produces different hashes (salted)."""
        from src.auth.security import hash_password

        hash1 = hash_password("SamePassword123!")
        hash2 = hash_password("SamePassword123!")
        assert hash1 != hash2  # bcrypt uses random salt

    def test_is_password_secure_valid(self):
        """Test secure password validation passes."""
        from src.auth.security import is_password_secure

        is_secure, issues = is_password_secure("StrongP@ssw0rd!")
        assert is_secure is True
        assert len(issues) == 0

    def test_is_password_secure_weak(self):
        """Test weak password validation fails."""
        from src.auth.security import is_password_secure

        is_secure, issues = is_password_secure("weak")
        assert is_secure is False
        assert len(issues) > 0


class TestSecurityUtilities:
    """Tests for security utility functions."""

    def test_sanitize_input(self):
        """Test input sanitization removes dangerous characters."""
        from src.auth.security import sanitize_input

        result = sanitize_input("<script>alert('xss')</script>")
        assert "<script>" not in result

    def test_sanitize_input_max_length(self):
        """Test input sanitization respects max length."""
        from src.auth.security import sanitize_input

        result = sanitize_input("a" * 1000, max_length=100)
        assert len(result) <= 100

    def test_is_safe_filename_valid(self):
        """Test safe filename validation."""
        from src.auth.security import is_safe_filename

        assert is_safe_filename("document.pdf") is True
        assert is_safe_filename("my_report_2024.docx") is True

    def test_is_safe_filename_invalid(self):
        """Test unsafe filename rejection."""
        from src.auth.security import is_safe_filename

        assert is_safe_filename("../../../etc/passwd") is False
        assert is_safe_filename("") is False

    def test_generate_secure_token(self):
        """Test secure token generation."""
        from src.auth.security import generate_secure_token

        token1 = generate_secure_token()
        token2 = generate_secure_token()
        assert token1 != token2
        assert len(token1) > 0


class TestJWTTokenManagement:
    """Tests for JWT token creation and verification."""

    def test_create_access_token(self):
        """Test access token creation."""
        from src.auth.jwt_utils import create_access_token

        token = create_access_token(
            user_id="test-user-id",
            email="test@example.com",
            role=UserRole.STANDARD_USER,
            permissions=[Permission.READ, Permission.WRITE],
        )
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        from src.auth.jwt_utils import create_refresh_token

        token = create_refresh_token(
            user_id="test-user-id",
            email="test@example.com",
            role=UserRole.STANDARD_USER,
        )
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_access_token(self):
        """Test access token verification returns TokenData."""
        from src.auth.jwt_utils import create_access_token, verify_access_token

        token = create_access_token(
            user_id="test-user-id",
            email="test@example.com",
            role=UserRole.ADMIN,
            permissions=[Permission.READ],
        )
        token_data = verify_access_token(token)
        assert isinstance(token_data, TokenData)
        assert token_data.user_id == "test-user-id"
        assert token_data.email == "test@example.com"

    def test_verify_refresh_token(self):
        """Test refresh token verification."""
        from src.auth.jwt_utils import create_refresh_token, verify_refresh_token

        token = create_refresh_token(
            user_id="test-user-id",
            email="test@example.com",
            role=UserRole.STANDARD_USER,
        )
        token_data = verify_refresh_token(token)
        assert token_data.user_id == "test-user-id"

    def test_revoke_token(self):
        """Test token revocation."""
        from src.auth.jwt_utils import (
            JWTError,
            create_access_token,
            revoke_token,
            verify_access_token,
        )

        token = create_access_token(
            user_id="test-user-id",
            email="test@example.com",
            role=UserRole.STANDARD_USER,
        )
        # Verify works before revocation
        verify_access_token(token)

        # Revoke
        revoke_token(token)

        # Should fail after revocation
        with pytest.raises(JWTError):
            verify_access_token(token)

    def test_verify_invalid_token_raises(self):
        """Test that invalid token raises JWTError."""
        from src.auth.jwt_utils import JWTError, verify_access_token

        with pytest.raises(JWTError):
            verify_access_token("invalid.token.string")


class TestAuthModels:
    """Tests for authentication Pydantic models."""

    def test_login_request_valid(self):
        """Test valid login request."""
        req = LoginRequest(email="user@example.com", password="password123")
        assert req.email == "user@example.com"

    def test_user_create_password_validation(self):
        """Test UserCreate password validation requires strong password."""
        with pytest.raises(ValueError):
            UserCreate(
                email="user@example.com",
                password="weak",  # Too short, no uppercase/special
            )

    def test_user_create_valid(self):
        """Test valid UserCreate."""
        user = UserCreate(
            email="user@example.com",
            password="StrongP@ss1234",
        )
        assert user.email == "user@example.com"

    def test_user_model(self):
        """Test User model creation."""
        user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            role=UserRole.STANDARD_USER,
            permissions=[Permission.READ],
            created_at=datetime.now(timezone.utc),
            is_active=True,
        )
        assert user.is_active is True
        assert user.role == UserRole.STANDARD_USER
