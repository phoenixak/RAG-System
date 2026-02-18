"""
Tests for launcher bootstrapping and startup preflight behavior.
"""

from pathlib import Path


def test_bootstrap_env_creates_env_with_generated_secrets(tmp_path, monkeypatch):
    """Missing .env should be created from .env.example with strong secrets."""
    import run as launcher

    env_example = tmp_path / ".env.example"
    env_example.write_text(
        (
            "SECRET_KEY=placeholder\n"
            "JWT_SECRET_KEY=placeholder\n"
            "SESSION_SECRET=placeholder\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(launcher, "project_root", tmp_path)
    monkeypatch.delenv("SECRET_KEY", raising=False)

    manager = launcher.ServiceManager()
    assert manager._bootstrap_env_file() is True
    assert manager.env_file.exists()

    env_values = manager._parse_env_file(manager.env_file)
    assert len(env_values["SECRET_KEY"]) >= 32
    assert len(env_values["JWT_SECRET_KEY"]) >= 32
    assert len(env_values["SESSION_SECRET"]) >= 32
    assert env_values["SECRET_KEY"] != "placeholder"


def test_validate_required_env_values_rejects_short_secret(tmp_path, monkeypatch):
    """Launcher preflight should reject short SECRET_KEY values."""
    import run as launcher

    env_file = tmp_path / ".env"
    env_file.write_text("SECRET_KEY=short\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("SECRET_KEY=placeholder\n", encoding="utf-8")

    monkeypatch.setattr(launcher, "project_root", tmp_path)
    monkeypatch.delenv("SECRET_KEY", raising=False)

    manager = launcher.ServiceManager()
    assert manager._validate_required_env_values() is False


def test_wait_for_service_uses_stdlib_http_client(monkeypatch):
    """Service polling should work through urllib and not require requests."""
    import run as launcher

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def getcode(self):
            return 200

    monkeypatch.setattr(
        launcher.urllib_request,
        "urlopen",
        lambda *args, **kwargs: _Response(),
    )

    manager = launcher.ServiceManager()
    assert manager.wait_for_service("http://localhost:8000/health", timeout=1) is True


def test_compose_frontend_uses_dedicated_streamlit_entrypoint():
    """Docker frontend service should point to frontend/app.py."""
    compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "frontend/app.py" in compose_text


def test_validate_python_runtime_rejects_python_312(monkeypatch):
    """Launcher should fail fast on unsupported Python 3.12+ local runtime."""
    import run as launcher

    monkeypatch.setattr(launcher.sys, "version_info", (3, 12, 0))
    manager = launcher.ServiceManager()
    assert manager._validate_python_runtime() is False
