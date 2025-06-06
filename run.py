#!/usr/bin/env python3
"""
Unified Enterprise RAG System Launcher

This script provides a unified way to start the Enterprise RAG system components:
- Backend: FastAPI server with all microservices
- Frontend: Streamlit web interface
- All: Both backend and frontend services

Usage:
    python run.py                    # Start both services
    python run.py all                # Start both services
    python run.py backend            # Start only FastAPI backend
    python run.py frontend           # Start only Streamlit frontend
    python run.py --help             # Show help message

Environment:
    Set environment variables in .env file or use defaults
    Backend runs on: http://localhost:8000
    Frontend runs on: http://localhost:8501
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ============================================================================
# BACKEND CODE (FastAPI)
# ============================================================================


def create_fastapi_app():
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    # Import backend components
    try:
        from src.api.auth import router as auth_router
        from src.api.deduplication import router as deduplication_router
        from src.api.documents import router as documents_router
        from src.api.health import router as health_router
        from src.api.search import router as search_router
        from src.core.config import get_settings
        from src.core.logging import get_logger, log_request_response, setup_logging

        settings = get_settings()
        logger = get_logger(__name__)
    except ImportError as e:
        print(f"X Failed to import backend components: {e}")
        print(
            "Make sure all backend dependencies are installed and src/ directory exists"
        )
        return None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager for startup and shutdown events."""
        # Startup
        logger.info("Starting Enterprise RAG System", version=settings.app_version)

        # Initialize services
        try:
            from src.documents.embeddings import get_embedding_generator
            from src.documents.service import get_document_service
            from src.vector_store.chroma_client import get_chroma_client

            # Initialize ChromaDB connection
            chroma_client = get_chroma_client()
            await chroma_client.health_check()
            logger.info("ChromaDB connection established")

            # Initialize embedding generator
            embedding_generator = get_embedding_generator()
            model_info = embedding_generator.get_model_info()
            logger.info("Embedding model loaded", **model_info)

            # Initialize document processing service
            document_service = get_document_service()
            logger.info("Document processing service initialized")

            # Initialize search services
            from src.search.service import get_search_service

            search_service = get_search_service()
            logger.info("Search service initialized")

            # Initialize LLM service
            try:
                from src.llm.service import initialize_llm_service

                llm_service = await initialize_llm_service()
                logger.info("LLM service initialized")
            except Exception as e:
                logger.warning(
                    "LLM service initialization failed, responses will fall back to basic summaries",
                    error=str(e),
                )

            logger.info("All services initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize services", error=str(e), exc_info=True)
            raise

        yield

        # Shutdown
        logger.info("Shutting down Enterprise RAG System")

        try:
            from src.documents.embeddings import get_embedding_generator
            from src.documents.service import shutdown_document_service
            from src.search.service import close_search_service
            from src.vector_store.chroma_client import get_chroma_client

            await close_search_service()
            await shutdown_document_service()
            # Close embedding generator (if it has a close method)
            try:
                embedding_generator = get_embedding_generator()
                if hasattr(embedding_generator, "close"):
                    await embedding_generator.close()
            except:
                pass
            # Close chroma client
            try:
                chroma_client = get_chroma_client()
                if hasattr(chroma_client, "close"):
                    await chroma_client.close()
            except:
                pass

            logger.info("All services closed successfully")

        except Exception as e:
            logger.error("Error during service cleanup", error=str(e), exc_info=True)

    # Initialize logging
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Enterprise RAG System - Intelligent Document Search and Conversational AI",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        log_request_response(
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            response_time=process_time,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None,
        )

        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
        return response

    # Security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        if not settings.debug:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response

    # Include API routers
    app.include_router(health_router, prefix=settings.api_prefix, tags=["Health"])
    app.include_router(
        auth_router, prefix=f"{settings.api_prefix}/auth", tags=["Authentication"]
    )
    app.include_router(
        documents_router, prefix=f"{settings.api_prefix}/documents", tags=["Documents"]
    )
    app.include_router(
        search_router, prefix=f"{settings.api_prefix}/search", tags=["Search"]
    )
    app.include_router(
        deduplication_router,
        prefix=f"{settings.api_prefix}/deduplication",
        tags=["Deduplication"],
    )

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "environment": settings.environment,
            "docs_url": "/docs" if settings.debug else None,
        }

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        error_response = {
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": time.time(),
                "path": str(request.url.path),
            }
        }
        logger.error(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=exc.detail,
            path=str(request.url.path),
        )
        return JSONResponse(status_code=exc.status_code, content=error_response)

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        error_response = {
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "status_code": 500,
                "timestamp": time.time(),
                "path": str(request.url.path),
            }
        }
        logger.error(
            "Unexpected exception occurred",
            error=str(exc),
            path=str(request.url.path),
            exc_info=True,
        )
        return JSONResponse(status_code=500, content=error_response)

    return app, settings


def run_backend_server():
    """Run the FastAPI backend server."""
    import uvicorn

    app_info = create_fastapi_app()
    if not app_info:
        print("X Failed to create FastAPI app")
        return

    app, settings = app_info

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


# ============================================================================
# FRONTEND CODE (Streamlit)
# ============================================================================


def run_streamlit_app():
    """Run the Streamlit frontend application directly."""
    import streamlit as st

    # Import frontend components
    try:
        from frontend.components.auth import (
            check_auth_status,
            clear_session_state,
            init_auth_state,
            is_authenticated,
            show_login_page,
            show_logout_button,
        )
        from frontend.components.chat_ui import init_chat_state
        from frontend.config import SESSION_KEYS, STREAMLIT_CONFIG
    except ImportError as e:
        print(f"❌ Failed to import frontend components: {e}")
        print(
            "Make sure all frontend dependencies are installed and frontend/ directory exists"
        )
        return

    # Configure page
    st.set_page_config(
        page_title=STREAMLIT_CONFIG["page_title"],
        page_icon=STREAMLIT_CONFIG["page_icon"],
        layout=STREAMLIT_CONFIG["layout"],
        initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"],
    )

    # Initialize session state
    init_auth_state()
    init_chat_state()

    # Initialize current page if not set
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"

    # Check authentication status
    if is_authenticated():
        # Only validate token if we have user data, but not immediately after login
        # This prevents the race condition where token validation fails right after login
        user_data = st.session_state.get(SESSION_KEYS["user"])
        if user_data:
            # Show authenticated interface
            show_sidebar_navigation()
            show_main_content()
        else:
            # We're authenticated but missing user data, try to get it
            try:
                if not check_auth_status():
                    st.error("Session expired. Please log in again.")
                    st.rerun()
                else:
                    # Show authenticated interface after successful validation
                    show_sidebar_navigation()
                    show_main_content()
            except Exception:
                # If validation fails, clear session and show login
                clear_session_state()
                st.rerun()
    else:
        # Show login page
        show_login_page()


def show_sidebar_navigation():
    """Show navigation sidebar for authenticated users."""
    import streamlit as st

    st.sidebar.title("🤖 Enterprise RAG")

    # Show user info and logout
    from frontend.components.auth import show_logout_button

    show_logout_button()

    st.sidebar.markdown("---")

    # Navigation menu
    st.sidebar.markdown("### 📋 Navigation")

    # Main pages
    pages = {
        "💬 Chat": "chat",
        "📄 Documents": "documents",
        "🔍 Search": "search",
        "⚙️ Settings": "settings",
    }

    # Add admin page for admin users
    from frontend.components.auth import is_admin

    if is_admin():
        pages["👨‍💼 Admin"] = "admin"

    # Create navigation buttons
    for page_name, page_key in pages.items():
        if st.sidebar.button(
            page_name, key=f"nav_{page_key}", use_container_width=True
        ):
            st.session_state.current_page = page_key
            st.rerun()

    st.sidebar.markdown("---")

    # Quick stats
    show_quick_stats()


def show_quick_stats():
    """Show quick statistics in sidebar."""
    import streamlit as st

    st.sidebar.markdown("### 📊 Quick Stats")

    try:
        from frontend.components.api_client import get_api_client
        from frontend.config import SESSION_KEYS

        # Only try to get stats if we're fully authenticated and have user data
        if st.session_state.get(SESSION_KEYS["user"]):
            api_client = get_api_client()

            # Get document count
            docs_response = api_client.get_documents(limit=1)
            total_docs = docs_response.get("total", 0)

            # Show stats
            st.sidebar.metric("📄 Documents", total_docs)

        # Chat history count (this doesn't require API call)
        chat_history = st.session_state.get(SESSION_KEYS["chat_history"], [])
        user_messages = len([msg for msg in chat_history if msg.get("role") == "user"])
        st.sidebar.metric("💬 Messages", user_messages)

    except Exception:
        # Don't show warning, just show basic stats
        from frontend.config import SESSION_KEYS

        chat_history = st.session_state.get(SESSION_KEYS["chat_history"], [])
        user_messages = len([msg for msg in chat_history if msg.get("role") == "user"])
        st.sidebar.metric("💬 Messages", user_messages)


def show_main_content():
    """Show main content area based on current page."""
    import streamlit as st

    current_page = st.session_state.get("current_page", "chat")

    page_loaders = {
        "chat": ("frontend.pages.chat", "show_chat_interface"),
        "documents": ("frontend.pages.documents", "show_documents_interface"),
        "search": ("frontend.pages.search", "show_search_interface"),
        "settings": ("frontend.pages.settings", "show_settings_interface"),
        "admin": ("frontend.pages.admin", "show_admin_interface"),
    }

    if current_page in page_loaders:
        module_name, func_name = page_loaders[current_page]
        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
            func()
        except ImportError:
            st.error(
                f"{current_page.title()} page not available. Please check the installation."
            )
    else:
        # Default to chat page
        show_main_content()


# ============================================================================
# SERVICE MANAGER
# ============================================================================


class ServiceManager:
    """Manages the lifecycle of backend and frontend services."""

    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.shutdown_event = threading.Event()

    def start_backend(self) -> subprocess.Popen:
        """Start the FastAPI backend server."""
        print("🚀 Starting FastAPI Backend Server...")
        print("   URL: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        print("   Health Check: http://localhost:8000/api/v1/health")

        # Check if we're in development mode
        env = os.environ.copy()
        if not env.get("ENVIRONMENT"):
            env["ENVIRONMENT"] = "development"

        # Use multiprocessing to run the backend in a separate process
        try:
            # Create a wrapper script to run the backend
            backend_code = """
import sys
sys.path.insert(0, "{root}")
from run import run_backend_server
run_backend_server()
""".format(root=str(project_root).replace("\\", "\\\\"))

            cmd = [sys.executable, "-c", backend_code]

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Start background thread to monitor backend output
            threading.Thread(
                target=self._monitor_process_output,
                args=(process, "BACKEND"),
                daemon=True,
            ).start()

            return process

        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return None

    def start_frontend(self) -> subprocess.Popen:
        """Start the Streamlit frontend server."""
        print("🎨 Starting Streamlit Frontend...")
        print("   URL: http://localhost:8501")

        # Setup environment
        env = os.environ.copy()
        env["API_BASE_URL"] = "http://localhost:8000"
        env["STREAMLIT_SERVER_PORT"] = "8501"
        env["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

        # Create a temporary Python file with the Streamlit app code
        frontend_code = """
import sys
sys.path.insert(0, "{root}")
from run import run_streamlit_app
run_streamlit_app()
""".format(root=str(project_root).replace("\\", "\\\\"))

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--theme.primaryColor=#FF6B6B",
            "--theme.backgroundColor=#FFFFFF",
            "--theme.secondaryBackgroundColor=#F0F2F6",
            "-c",
            frontend_code,
        ]

        # Write the code to a temporary file
        temp_file = project_root / ".temp_streamlit_app.py"
        temp_file.write_text(frontend_code)

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(temp_file),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--theme.primaryColor=#FF6B6B",
            "--theme.backgroundColor=#FFFFFF",
            "--theme.secondaryBackgroundColor=#F0F2F6",
        ]

        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Start background thread to monitor frontend output
            threading.Thread(
                target=self._monitor_process_output,
                args=(process, "FRONTEND"),
                daemon=True,
            ).start()

            return process

        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return None

    def _monitor_process_output(self, process: subprocess.Popen, service_name: str):
        """Monitor and display process output with service prefixes."""
        try:
            for line in iter(process.stdout.readline, ""):
                if line.strip():
                    print(f"[{service_name}] {line.strip()}")
                if self.shutdown_event.is_set():
                    break
        except Exception as e:
            if not self.shutdown_event.is_set():
                print(f"[{service_name}] Error reading output: {e}")

    def wait_for_service(self, url: str, timeout: int = 30) -> bool:
        """Wait for a service to become available."""
        import requests

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        return False

    def start_services(self, backend: bool = True, frontend: bool = True):
        """Start the specified services."""
        print("=" * 60)
        print("🏢 Enterprise RAG System Launcher")
        print("=" * 60)

        # Validate environment
        if not self._validate_environment():
            sys.exit(1)

        try:
            # Start backend if requested
            if backend:
                backend_process = self.start_backend()
                if backend_process:
                    self.processes.append(backend_process)
                    print("✅ Backend started successfully")

                    # Wait for backend to be ready
                    print("⏳ Waiting for backend to be ready...")
                    if self.wait_for_service("http://localhost:8000/api/v1/health"):
                        print("✅ Backend is ready!")
                    else:
                        print("⚠️  Backend health check failed, but continuing...")
                else:
                    print("❌ Failed to start backend")
                    return

            # Start frontend if requested
            if frontend:
                # Small delay to ensure backend is fully ready
                if backend:
                    time.sleep(2)

                frontend_process = self.start_frontend()
                if frontend_process:
                    self.processes.append(frontend_process)
                    print("✅ Frontend started successfully")

                    # Wait for frontend to be ready
                    print("⏳ Waiting for frontend to be ready...")
                    if self.wait_for_service("http://localhost:8501"):
                        print("✅ Frontend is ready!")
                    else:
                        print("⚠️  Frontend not responding, but process started...")
                else:
                    print("❌ Failed to start frontend")
                    return

            # Print status summary
            print("\n" + "=" * 60)
            print("🎉 Services Started Successfully!")
            print("=" * 60)

            if backend:
                print("📊 Backend API: http://localhost:8000")
                print("📚 API Documentation: http://localhost:8000/docs")
                print("❤️  Health Check: http://localhost:8000/api/v1/health")

            if frontend:
                print("🎨 Frontend UI: http://localhost:8501")

            print("\n💡 Press Ctrl+C to stop all services")
            print("=" * 60)

            # Wait for shutdown signal
            self._wait_for_shutdown()

        except KeyboardInterrupt:
            print("\n🛑 Shutdown signal received...")
        except Exception as e:
            print(f"❌ Error during startup: {e}")
        finally:
            self._cleanup()

    def _validate_environment(self) -> bool:
        """Validate that the environment is set up correctly."""
        print("🔍 Validating environment...")

        # Check if required directories exist
        required_dirs = ["src", "frontend"]
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                print(f"❌ Required directory '{dir_name}' not found")
                return False

        # Check if pyproject.toml exists
        if not Path("pyproject.toml").exists():
            print("⚠️  pyproject.toml not found, some dependencies might be missing")

        print("✅ Environment validation passed")
        return True

    def _wait_for_shutdown(self):
        """Wait for shutdown signal or process termination."""
        try:
            while True:
                # Check if any processes have died
                for process in self.processes:
                    if process.poll() is not None:
                        print(f"⚠️  Process {process.pid} has terminated")
                        return

                time.sleep(1)
        except KeyboardInterrupt:
            pass

    def _cleanup(self):
        """Clean up all processes."""
        print("🧹 Cleaning up services...")
        self.shutdown_event.set()

        # Clean up temporary files
        temp_file = project_root / ".temp_streamlit_app.py"
        if temp_file.exists():
            temp_file.unlink()

        for process in self.processes:
            try:
                if process.poll() is None:  # Process is still running
                    print(f"🛑 Terminating process {process.pid}...")
                    process.terminate()

                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"⚡ Force killing process {process.pid}...")
                        process.kill()
                        process.wait()

            except Exception as e:
                print(f"⚠️  Error terminating process {process.pid}: {e}")

        print("✅ Cleanup completed")


def main():
    """Main entry point for the launcher."""
    parser = argparse.ArgumentParser(
        description="Enterprise RAG System Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Start both backend and frontend
  python run.py all                # Start both backend and frontend  
  python run.py backend            # Start only the FastAPI backend
  python run.py frontend           # Start only the Streamlit frontend

Services:
  Backend:  FastAPI server on http://localhost:8000
  Frontend: Streamlit UI on http://localhost:8501

Environment:
  Configure settings in .env file or environment variables
  Default environment is 'development' with auto-reload enabled
        """,
    )

    parser.add_argument(
        "service",
        nargs="?",
        choices=["backend", "frontend", "all"],
        default="all",
        help="Service to start (default: all)",
    )

    parser.add_argument(
        "--no-validate", action="store_true", help="Skip environment validation"
    )

    args = parser.parse_args()

    # Create service manager
    manager = ServiceManager()

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        manager._cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Determine which services to start
    start_backend = args.service in ["backend", "all"]
    start_frontend = args.service in ["frontend", "all"]

    # Start services
    manager.start_services(backend=start_backend, frontend=start_frontend)


if __name__ == "__main__":
    # Support direct function calls for subprocess
    if len(sys.argv) > 1 and sys.argv[1] == "__run_backend__":
        run_backend_server()
    elif len(sys.argv) > 1 and sys.argv[1] == "__run_frontend__":
        run_streamlit_app()
    else:
        main()
