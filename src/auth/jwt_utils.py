"""
JWT Utilities
Token generation, validation, and management for authentication.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

import jwt
import redis as redis_sync

from src.auth.models import ROLE_PERMISSIONS, Permission, TokenData, UserRole
from src.core.config import get_settings
from src.core.logging import get_logger, log_security_event

settings = get_settings()
logger = get_logger(__name__)


class JWTError(Exception):
    """Custom JWT error exception."""

    pass


class TokenManager:
    """JWT token management class."""

    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
        self.refresh_token_expire_days = settings.refresh_token_expire_days

        # Redis-backed blacklist for revoked tokens with in-memory fallback
        self._redis_client = None
        self._use_fallback = False
        self._fallback_blacklist: set = set()
        self._redis_key_prefix = "rag:token_blacklist:"
        self._bl_logger = logging.getLogger(__name__)
        self._init_redis()

    def _init_redis(self) -> None:
        """Initialize sync Redis connection for token blacklist."""
        try:
            self._redis_client = redis_sync.from_url(
                settings.redis_url,
                decode_responses=True,
            )
            self._redis_client.ping()
        except Exception as e:
            self._bl_logger.warning(
                "Redis unavailable for token blacklist, falling back to in-memory: %s",
                e,
            )
            self._use_fallback = True

    def create_access_token(
        self,
        user_id: str,
        email: str,
        role: UserRole,
        permissions: Optional[list] = None,
    ) -> str:
        """Create an access token for the user."""

        if permissions is None:
            permissions = ROLE_PERMISSIONS.get(role, [])

        # FIX: Use consistent UTC time handling for JWT timestamps
        now_utc = datetime.now(timezone.utc)
        expire_utc = now_utc + timedelta(minutes=self.access_token_expire_minutes)

        # Use explicit UTC timestamps to avoid timezone issues
        now_timestamp = int(time.time())
        expire_timestamp = now_timestamp + (self.access_token_expire_minutes * 60)

        payload = {
            "user_id": user_id,
            "email": email,
            "role": role.value,
            "permissions": [p.value for p in permissions],
            "token_type": "access",
            "exp": expire_timestamp,
            "iat": now_timestamp,
            "jti": str(uuid4()),  # JWT ID for token blacklisting
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

            logger.info(
                "Access token created",
                user_id=user_id,
                email=email,
                expires_at=expire_utc.isoformat(),
            )

            return token

        except Exception as e:
            logger.error("Failed to create access token", error=str(e), user_id=user_id)
            raise JWTError("Failed to create access token") from e

    def create_refresh_token(self, user_id: str, email: str, role: UserRole) -> str:
        """Create a refresh token for the user."""

        # FIX: Use consistent UTC time handling for JWT timestamps
        now_utc = datetime.now(timezone.utc)
        expire_utc = now_utc + timedelta(days=self.refresh_token_expire_days)

        # Use explicit UTC timestamps to avoid timezone issues
        now_timestamp = int(time.time())
        expire_timestamp = now_timestamp + (
            self.refresh_token_expire_days * 24 * 60 * 60
        )

        payload = {
            "user_id": user_id,
            "email": email,
            "role": role.value,
            "token_type": "refresh",
            "exp": expire_timestamp,
            "iat": now_timestamp,
            "jti": str(uuid4()),
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

            logger.info(
                "Refresh token created",
                user_id=user_id,
                email=email,
                expires_at=expire_utc.isoformat(),
            )

            return token

        except Exception as e:
            logger.error(
                "Failed to create refresh token", error=str(e), user_id=user_id
            )
            raise JWTError("Failed to create refresh token") from e

    def decode_token(self, token: str) -> TokenData:
        """Decode and validate a JWT token."""

        try:
            # COMPREHENSIVE DEBUG: Full token validation analysis
            now_utc = datetime.now(timezone.utc)

            # First decode without verification to see token contents
            try:
                payload_preview = jwt.decode(
                    token, options={"verify_signature": False, "verify_exp": False}
                )
                exp_timestamp = payload_preview.get("exp", 0)
                iat_timestamp = payload_preview.get("iat", 0)
                exp_datetime_utc = datetime.fromtimestamp(
                    exp_timestamp, tz=timezone.utc
                )
                iat_datetime_utc = datetime.fromtimestamp(
                    iat_timestamp, tz=timezone.utc
                )

                logger.info(
                    f"Token issued at: {iat_datetime_utc.isoformat()} UTC (timestamp: {iat_timestamp})"
                )
                logger.info(
                    f"Token expires at: {exp_datetime_utc.isoformat()} UTC (timestamp: {exp_timestamp})"
                )
                logger.info(
                    f"Token age: {(now_utc - iat_datetime_utc).total_seconds()} seconds"
                )
                logger.info(
                    f"Time until expiry: {(exp_datetime_utc - now_utc).total_seconds()} seconds"
                )
                logger.info(
                    f"Is token expired by manual check? {now_utc > exp_datetime_utc}"
                )
            except Exception as preview_error:
                logger.error(f"Failed to preview token: {preview_error}")

            # Now try actual JWT validation
            try:
                payload = jwt.decode(
                    token, self.secret_key, algorithms=[self.algorithm]
                )
                logger.info("✅ JWT validation successful")
            except jwt.ExpiredSignatureError as exp_error:
                logger.error(f"❌ JWT ExpiredSignatureError: {exp_error}")
                raise
            except Exception as jwt_error:
                logger.error(f"❌ JWT validation error: {jwt_error}")
                raise

            # Debug: Check token expiration (FIX: Use UTC for both timestamps)
            exp_timestamp = payload.get("exp", 0)
            exp_datetime_utc = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)

            # Check if token is blacklisted
            jti = payload.get("jti")
            if jti and self._is_token_blacklisted(jti):
                log_security_event(
                    "token_blacklisted_used",
                    user_id=payload.get("user_id"),
                    details={"jti": jti},
                )
                raise JWTError("Token has been revoked")

            # Validate required fields
            required_fields = [
                "user_id",
                "email",
                "role",
                "token_type",
                "exp",
                "iat",
                "jti",
            ]
            for field in required_fields:
                if field not in payload:
                    raise JWTError(f"Missing required field: {field}")

            # Convert role and permissions
            role = UserRole(payload["role"])
            permissions = [Permission(p) for p in payload.get("permissions", [])]

            # Create TokenData instance
            token_data = TokenData(
                user_id=payload["user_id"],
                email=payload["email"],
                role=role,
                permissions=permissions,
                token_type=payload["token_type"],
                exp=payload["exp"],
                iat=payload["iat"],
                jti=payload["jti"],
            )

            return token_data

        except jwt.ExpiredSignatureError:
            logger.warning("Expired token used")
            raise JWTError("Token has expired")

        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token used", error=str(e))
            raise JWTError("Invalid token") from e

        except Exception as e:
            logger.error("Token decode error", error=str(e))
            raise JWTError("Token validation failed") from e

    def verify_access_token(self, token: str) -> TokenData:
        """Verify an access token and return token data."""

        token_data = self.decode_token(token)

        if token_data.token_type != "access":
            log_security_event(
                "wrong_token_type_used",
                user_id=token_data.user_id,
                details={"expected": "access", "actual": token_data.token_type},
            )
            raise JWTError("Invalid token type")

        return token_data

    def verify_refresh_token(self, token: str) -> TokenData:
        """Verify a refresh token and return token data."""

        token_data = self.decode_token(token)

        if token_data.token_type != "refresh":
            log_security_event(
                "wrong_token_type_used",
                user_id=token_data.user_id,
                details={"expected": "refresh", "actual": token_data.token_type},
            )
            raise JWTError("Invalid token type")

        return token_data

    def _is_token_blacklisted(self, jti: str) -> bool:
        """Check if a token JTI is blacklisted."""
        if self._use_fallback:
            return jti in self._fallback_blacklist
        try:
            return bool(
                self._redis_client.sismember(f"{self._redis_key_prefix}set", jti)
            )
        except Exception as e:
            self._bl_logger.warning(
                "Redis SISMEMBER failed for token blacklist, using fallback: %s", e
            )
            self._use_fallback = True
            return jti in self._fallback_blacklist

    def _add_to_blacklist(self, jti: str, ttl_seconds: int) -> None:
        """Add a token JTI to the blacklist with TTL."""
        if self._use_fallback:
            self._fallback_blacklist.add(jti)
            return
        try:
            # Store in a Redis SET for SISMEMBER lookups
            self._redis_client.sadd(f"{self._redis_key_prefix}set", jti)
            # Also store as individual key with TTL so it auto-expires
            self._redis_client.setex(f"{self._redis_key_prefix}{jti}", ttl_seconds, "1")
        except Exception as e:
            self._bl_logger.warning(
                "Redis SADD failed for token blacklist, using fallback: %s", e
            )
            self._use_fallback = True
            self._fallback_blacklist.add(jti)

    def revoke_token(self, token: str) -> None:
        """Revoke a token by adding it to the blacklist."""

        try:
            token_data = self.decode_token(token)

            # Calculate remaining TTL for the token
            remaining_seconds = max(0, token_data.exp - int(time.time()))
            # Use at least 1 hour TTL to cover clock skew
            ttl = max(remaining_seconds, 3600)

            self._add_to_blacklist(token_data.jti, ttl)

            log_security_event(
                "token_revoked",
                user_id=token_data.user_id,
                details={"jti": token_data.jti, "token_type": token_data.token_type},
            )

            logger.info(
                "Token revoked",
                user_id=token_data.user_id,
                jti=token_data.jti,
                token_type=token_data.token_type,
            )

        except JWTError:
            # Token is already invalid, but we still want to add it to blacklist
            # Extract jti without validation
            try:
                payload = jwt.decode(token, options={"verify_signature": False})
                jti = payload.get("jti")
                if jti:
                    exp = payload.get("exp", 0)
                    remaining = max(0, exp - int(time.time()))
                    ttl = max(remaining, 3600)
                    self._add_to_blacklist(jti, ttl)
            except Exception:
                pass  # Token is completely malformed

    def cleanup_blacklist(self) -> None:
        """Clean up expired tokens from blacklist.

        For Redis, expired individual keys auto-expire via TTL.
        This cleans up stale entries from the SET.
        """
        if self._use_fallback:
            return
        try:
            # Get all members of the blacklist set
            members = self._redis_client.smembers(f"{self._redis_key_prefix}set")
            for jti in members:
                # If the individual TTL key has expired, remove from SET
                if not self._redis_client.exists(f"{self._redis_key_prefix}{jti}"):
                    self._redis_client.srem(f"{self._redis_key_prefix}set", jti)
        except Exception as e:
            self._bl_logger.warning("Redis cleanup failed for token blacklist: %s", e)

    def get_token_info(self, token: str) -> Dict[str, Any]:
        """Get information about a token without full validation."""

        try:
            # Decode without verification to get payload
            payload = jwt.decode(token, options={"verify_signature": False})

            return {
                "user_id": payload.get("user_id"),
                "email": payload.get("email"),
                "role": payload.get("role"),
                "token_type": payload.get("token_type"),
                "issued_at": datetime.fromtimestamp(
                    payload.get("iat", 0), tz=timezone.utc
                ),
                "expires_at": datetime.fromtimestamp(
                    payload.get("exp", 0), tz=timezone.utc
                ),
                "jti": payload.get("jti"),
                "is_expired": datetime.now(timezone.utc)
                > datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
                "is_blacklisted": self._is_token_blacklisted(payload.get("jti", "")),
            }

        except Exception as e:
            logger.error("Failed to get token info", error=str(e))
            return {}


# Global token manager instance
token_manager = TokenManager()


def create_access_token(
    user_id: str, email: str, role: UserRole, permissions: Optional[list] = None
) -> str:
    """Create an access token."""
    return token_manager.create_access_token(user_id, email, role, permissions)


def create_refresh_token(user_id: str, email: str, role: UserRole) -> str:
    """Create a refresh token."""
    return token_manager.create_refresh_token(user_id, email, role)


def verify_access_token(token: str) -> TokenData:
    """Verify an access token."""
    return token_manager.verify_access_token(token)


def verify_refresh_token(token: str) -> TokenData:
    """Verify a refresh token."""
    return token_manager.verify_refresh_token(token)


def revoke_token(token: str) -> None:
    """Revoke a token."""
    token_manager.revoke_token(token)
