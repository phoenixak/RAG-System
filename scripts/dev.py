#!/usr/bin/env python3
"""
Development Helper Script
Provides common development tasks for the Enterprise RAG System.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_command(cmd, check=True, cwd=None):
    """Run a shell command and return the result."""
    print(f"🔄 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, cwd=cwd)
    return result


def setup_env():
    """Set up development environment."""
    print("🏗️  Setting up development environment...")

    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        print("📋 Creating .env file from template...")
        run_command("cp .env.example .env")
        print("⚠️  Please edit .env file with your configuration!")

    # Install dependencies
    print("📦 Installing Python dependencies...")
    run_command("pip install -r requirements.txt")

    print("✅ Development environment setup complete!")
    print("📝 Next steps:")
    print("   1. Edit .env file with your configuration")
    print("   2. Run: python scripts/dev.py start")


def start_services():
    """Start development services."""
    print("🚀 Starting development services...")
    run_command("docker-compose up -d postgres redis chromadb")
    print("⏳ Waiting for services to be ready...")
    run_command("sleep 5")
    print("🔄 Starting API server...")
    run_command("uvicorn src.main:app --reload --host 0.0.0.0 --port 8000")


def stop_services():
    """Stop development services."""
    print("🛑 Stopping development services...")
    run_command("docker-compose down")
    print("✅ Services stopped")


def test():
    """Run tests."""
    print("🧪 Running tests...")
    run_command("python -m pytest tests/ -v")


def lint():
    """Run code linting."""
    print("🔍 Running code quality checks...")

    print("🖤 Formatting with black...")
    run_command("black src/", check=False)

    print("📚 Sorting imports with isort...")
    run_command("isort src/", check=False)

    print("🔍 Linting with flake8...")
    run_command("flake8 src/", check=False)

    print("🔒 Type checking with mypy...")
    run_command("mypy src/", check=False)

    print("✅ Code quality checks complete!")


def clean():
    """Clean up development artifacts."""
    print("🧹 Cleaning up...")

    # Remove Python cache
    run_command("find . -type d -name __pycache__ -delete", check=False)
    run_command("find . -type f -name '*.pyc' -delete", check=False)

    # Remove test artifacts
    run_command("rm -rf .pytest_cache", check=False)
    run_command("rm -rf htmlcov", check=False)
    run_command("rm -f .coverage", check=False)

    # Remove build artifacts
    run_command("rm -rf build dist *.egg-info", check=False)

    print("✅ Cleanup complete!")


def reset_db():
    """Reset databases."""
    print("🗃️  Resetting databases...")
    run_command("docker-compose down -v")
    run_command("docker-compose up -d postgres redis chromadb")
    print("⏳ Waiting for databases to be ready...")
    run_command("sleep 10")
    print("✅ Databases reset!")


def logs():
    """Show service logs."""
    print("📋 Showing service logs...")
    run_command("docker-compose logs -f")


def status():
    """Show service status."""
    print("📊 Service Status:")
    run_command("docker-compose ps")

    print("\n🔍 Health Checks:")
    health_checks = [
        (
            "API Health",
            "curl -s http://localhost:8000/api/v1/health | jq .status || echo 'API not available'",
        ),
        (
            "PostgreSQL",
            "docker-compose exec -T postgres pg_isready -U rag_user || echo 'PostgreSQL not available'",
        ),
        (
            "Redis",
            "docker-compose exec -T redis redis-cli ping || echo 'Redis not available'",
        ),
        (
            "ChromaDB",
            "curl -s http://localhost:8001/api/v1/heartbeat || echo 'ChromaDB not available'",
        ),
    ]

    for name, cmd in health_checks:
        print(f"\n{name}:")
        run_command(cmd, check=False)


def build():
    """Build Docker images."""
    print("🏗️  Building Docker images...")
    run_command("docker-compose build")
    print("✅ Build complete!")


def shell():
    """Open development shell."""
    print("🐚 Opening development shell...")
    try:
        # Try to use ipython if available
        import IPython

        IPython.start_ipython()
    except ImportError:
        # Fall back to regular Python shell
        import code

        code.interact(local=globals())


def docs():
    """Open API documentation."""
    print("📚 Opening API documentation...")
    print("🌐 API Docs: http://localhost:8000/docs")
    print("🌐 ReDoc: http://localhost:8000/redoc")

    # Try to open in browser
    try:
        import webbrowser

        webbrowser.open("http://localhost:8000/docs")
    except Exception:
        pass


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Enterprise RAG System Development Helper"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    subparsers.add_parser("setup", help="Set up development environment")

    # Service management
    subparsers.add_parser("start", help="Start development services")
    subparsers.add_parser("stop", help="Stop development services")
    subparsers.add_parser("restart", help="Restart development services")
    subparsers.add_parser("logs", help="Show service logs")
    subparsers.add_parser("status", help="Show service status")

    # Development tasks
    subparsers.add_parser("test", help="Run tests")
    subparsers.add_parser("lint", help="Run code linting")
    subparsers.add_parser("clean", help="Clean up artifacts")
    subparsers.add_parser("shell", help="Open development shell")

    # Database management
    subparsers.add_parser("reset-db", help="Reset databases")

    # Docker tasks
    subparsers.add_parser("build", help="Build Docker images")

    # Documentation
    subparsers.add_parser("docs", help="Open API documentation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Execute command
    commands = {
        "setup": setup_env,
        "start": start_services,
        "stop": stop_services,
        "restart": lambda: (stop_services(), start_services()),
        "test": test,
        "lint": lint,
        "clean": clean,
        "reset-db": reset_db,
        "logs": logs,
        "status": status,
        "build": build,
        "shell": shell,
        "docs": docs,
    }

    if args.command in commands:
        try:
            commands[args.command]()
        except KeyboardInterrupt:
            print("\n⏹️  Operation cancelled")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
