"""
Rate Limiting Configuration

Provides a shared slowapi Limiter instance for use across API endpoints.
Falls back gracefully if slowapi is not installed.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    from src.core.config import get_settings

    _settings = get_settings()

    def _get_rate_limit_key(request) -> str:
        """Extract user ID from JWT for rate limiting, falling back to IP."""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                import jwt as pyjwt

                token = auth_header.split(" ", 1)[1]
                payload = pyjwt.decode(
                    token,
                    options={"verify_signature": False, "verify_exp": False},
                )
                user_id = payload.get("user_id")
                if user_id:
                    return f"user:{user_id}"
            except Exception:
                pass
        return get_remote_address(request)

    limiter = Limiter(key_func=_get_rate_limit_key)

    RATE_LIMIT_ANONYMOUS = _settings.rate_limit_anonymous
    RATE_LIMIT_AUTHENTICATED = _settings.rate_limit_authenticated
    RATE_LIMIT_PREMIUM = _settings.rate_limit_premium

    SLOWAPI_AVAILABLE = True

except ImportError:
    logger.info("slowapi not installed; rate limiting is disabled")
    limiter = None
    RATE_LIMIT_ANONYMOUS = "10/minute"
    RATE_LIMIT_AUTHENTICATED = "100/minute"
    RATE_LIMIT_PREMIUM = "1000/minute"
    SLOWAPI_AVAILABLE = False
