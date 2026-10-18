"""JWT auth with four roles. Users come from the API_USERS environment variable.

API_USERS="alice:<bcrypt-hash>:admin,bob:<bcrypt-hash>:read_only"
Create a hash with:  python -m api.auth "my password"
"""

import os
import secrets
import sys
import time
import warnings
from dataclasses import dataclass
from enum import IntEnum

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer


class Role(IntEnum):
    """Each role can do everything the roles below it can."""

    read_only = 1  # search, ask, list documents
    standard_user = 2  # + upload documents
    power_user = 3  # + delete documents
    admin = 4  # + see users and index stats


@dataclass(frozen=True)
class User:
    username: str
    role: Role


SECRET = os.environ.get("SECRET_KEY") or ""
if not SECRET:
    SECRET = secrets.token_urlsafe(32)
    warnings.warn("SECRET_KEY is not set: using a random key, tokens will not survive a restart", stacklevel=1)
ALGORITHM = "HS256"
TOKEN_MINUTES = int(os.environ.get("ACCESS_TOKEN_MINUTES", "30"))

oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def load_users(spec: str | None = None) -> dict[str, tuple[str, Role]]:
    users = {}
    for entry in filter(None, (spec if spec is not None else os.environ.get("API_USERS", "")).split(",")):
        name, pw_hash, role = entry.strip().rsplit(":", 2) if entry.count(":") > 2 else entry.strip().split(":")
        users[name] = (pw_hash, Role[role])
    return users


USERS = load_users()


def authenticate(username: str, password: str) -> User | None:
    record = USERS.get(username)
    # always run bcrypt so response time does not reveal whether the user exists
    pw_hash = record[0] if record else "$2b$12$" + "x" * 53
    try:
        ok = bcrypt.checkpw(password.encode(), pw_hash.encode())
    except ValueError:
        ok = False
    return User(username, record[1]) if ok and record else None


def create_token(user: User) -> str:
    now = int(time.time())
    claims = {"sub": user.username, "role": user.role.name, "iat": now, "exp": now + TOKEN_MINUTES * 60}
    return jwt.encode(claims, SECRET, algorithm=ALGORITHM)


def current_user(token: str = Depends(oauth2)) -> User:
    try:
        claims = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        return User(claims["sub"], Role[claims["role"]])
    except (jwt.PyJWTError, KeyError):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
        ) from None


def require(role: Role):
    def check(user: User = Depends(current_user)) -> User:
        if user.role < role:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires the {role.name} role or higher")
        return user

    return check


if __name__ == "__main__":
    print(hash_password(sys.argv[1] if len(sys.argv) > 1 else input("password: ")))
