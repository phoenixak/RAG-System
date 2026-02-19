"""
Authentication Components
Handles user authentication, session management, and protected routes.
"""

from typing import Any, Dict, Optional

import streamlit as st

from frontend.components.api_client import APIException, get_api_client
from frontend.config import SESSION_KEYS, SUCCESS_MESSAGES


def init_auth_state():
    """Initialize authentication state in session."""
    if SESSION_KEYS["authenticated"] not in st.session_state:
        st.session_state[SESSION_KEYS["authenticated"]] = False

    if SESSION_KEYS["user"] not in st.session_state:
        st.session_state[SESSION_KEYS["user"]] = None

    if SESSION_KEYS["token"] not in st.session_state:
        st.session_state[SESSION_KEYS["token"]] = None

    if SESSION_KEYS["refresh_token"] not in st.session_state:
        st.session_state[SESSION_KEYS["refresh_token"]] = None


def is_authenticated() -> bool:
    """Check if user is authenticated."""
    return st.session_state.get(SESSION_KEYS["authenticated"], False)


def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current authenticated user."""
    return st.session_state.get(SESSION_KEYS["user"])


def requires_auth(func):
    """Decorator to require authentication for a function."""

    def wrapper(*args, **kwargs):
        if not is_authenticated():
            st.warning("Please log in to access this feature.")
            show_login_form()
            return None
        return func(*args, **kwargs)

    return wrapper


def show_login_form():
    """Display login form."""
    st.subheader("🔐 Login")

    with st.form("login_form"):
        email = st.text_input("Email", placeholder="admin@example.com")
        password = st.text_input("Password", type="password", placeholder="admin123!")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if not email or not password:
                st.error("Please enter both email and password.")
                return

            try:
                with st.spinner("Logging in..."):
                    api_client = get_api_client()

                    # Attempt login
                    auth_response = api_client.login(email, password)

                    # Store authentication data
                    st.session_state[SESSION_KEYS["authenticated"]] = True
                    st.session_state[SESSION_KEYS["token"]] = auth_response[
                        "access_token"
                    ]
                    st.session_state[SESSION_KEYS["refresh_token"]] = auth_response.get(
                        "refresh_token"
                    )

                    # Use user info from login response instead of making another API call
                    # This prevents token expiration race condition
                    user_info = auth_response.get("user")
                    if user_info:
                        st.session_state[SESSION_KEYS["user"]] = user_info
                    else:
                        # Fallback to API call if user info not in response
                        user_info = api_client.get_current_user()
                        st.session_state[SESSION_KEYS["user"]] = user_info

                    st.success(SUCCESS_MESSAGES["login_success"])
                    st.rerun()

            except APIException as e:
                st.error(f"Login failed: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")


def show_logout_button():
    """Display logout button in sidebar."""
    if is_authenticated():
        user = get_current_user()
        username = user.get("email", "User") if user else "User"

        st.sidebar.write(f"👤 **{username}**")

        if user and user.get("role"):
            st.sidebar.write(f"🏷️ Role: {user['role'].title()}")

        if st.sidebar.button("🚪 Logout", key="logout_btn"):
            logout_user()


def logout_user():
    """Logout current user."""
    try:
        api_client = get_api_client()
        api_client.logout()
        st.success(SUCCESS_MESSAGES["logout_success"])

    except APIException as e:
        st.error(f"Logout failed: {str(e)}")

    except Exception as e:
        # Clear session even if API call fails
        st.error(f"Logout error: {str(e)}")

    finally:
        # Always clear local session
        clear_session_state()
        st.rerun()


def clear_session_state():
    """Clear all authentication-related session state."""
    auth_keys = [
        SESSION_KEYS["authenticated"],
        SESSION_KEYS["user"],
        SESSION_KEYS["token"],
        SESSION_KEYS["refresh_token"],
        SESSION_KEYS["conversation_id"],
        SESSION_KEYS["chat_history"],
        SESSION_KEYS["search_history"],
    ]

    for key in auth_keys:
        if key in st.session_state:
            del st.session_state[key]


def check_auth_status():
    """Check and validate current authentication status."""
    if not is_authenticated():
        return False

    # If we already have user data, don't make unnecessary API calls
    # This prevents race conditions after login
    if st.session_state.get(SESSION_KEYS["user"]):
        return True

    try:
        # Try to get current user info to validate token
        api_client = get_api_client()
        user_info = api_client.get_current_user()

        # Update user info in session
        st.session_state[SESSION_KEYS["user"]] = user_info
        return True

    except APIException as e:
        # Only clear session for specific authentication errors
        if (
            "401" in str(e)
            or "expired" in str(e).lower()
            or "invalid" in str(e).lower()
        ):
            clear_session_state()
            return False
        # For other API errors, assume still authenticated
        return True

    except Exception:
        # Connection error, assume still authenticated
        return True


def require_role(required_role: str):
    """Decorator to require specific role."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_authenticated():
                st.warning("Please log in to access this feature.")
                show_login_form()
                return None

            user = get_current_user()
            user_role = user.get("role", "user") if user else "user"

            if user_role != required_role and required_role != "user":
                st.error(f"Access denied. This feature requires {required_role} role.")
                return None

            return func(*args, **kwargs)

        return wrapper

    return decorator


def show_auth_status():
    """Show authentication status in sidebar."""
    if is_authenticated():
        user = get_current_user()
        if user:
            st.sidebar.success("✅ Authenticated")
            st.sidebar.write(f"**Email:** {user.get('email', 'N/A')}")
            st.sidebar.write(f"**Role:** {user.get('role', 'user').title()}")

            # Show role-specific capabilities
            if user.get("role") == "admin":
                st.sidebar.info("🔧 Admin capabilities enabled")
        else:
            st.sidebar.warning("⚠️ Authentication data missing")
    else:
        st.sidebar.error("❌ Not authenticated")


def get_user_role() -> str:
    """Get current user's role."""
    user = get_current_user()
    return user.get("role", "user") if user else "user"


def is_admin() -> bool:
    """Check if current user is admin."""
    return get_user_role() == "admin"


def show_login_page():
    """Display full login page with styled hero section."""
    # Hero section
    try:
        from frontend.components.styles import (
            render_login_hero,
            render_status_indicator,
        )

        render_login_hero()
    except Exception:
        st.title("Enterprise RAG System")

    # Two-column layout: status left, login form right
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # System status card
        st.markdown(
            "<div style='font-family:var(--font-mono,monospace); font-size:0.72rem; "
            "text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted,#475569); "
            "margin-bottom:0.6rem;'>System Status</div>",
            unsafe_allow_html=True,
        )
        try:
            api_client = get_api_client()
            health = api_client.health_check()
            online = health.get("status") == "healthy"
            version = health.get("version", "N/A")
            env = health.get("environment", "N/A")
            try:
                from frontend.components.styles import render_status_indicator

                render_status_indicator(online, "Backend API", f"v{version} / {env}")
            except Exception:
                if online:
                    st.success(f"Backend Online — v{version}")
                else:
                    st.warning("Backend Issues Detected")
        except Exception as e:
            try:
                from frontend.components.styles import render_status_indicator

                render_status_indicator(False, "Backend Offline", str(e)[:60])
            except Exception:
                st.error(f"Backend Offline: {str(e)}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature list
        st.markdown(
            """
            <div style="font-family:var(--font-mono,monospace); font-size:0.78rem;
                        color:var(--text-secondary,#94a3b8); line-height:2;">
                <div><span style="color:var(--accent-cyan,#00d4ff);">&#x25B6;</span>
                     &nbsp;Document Management (PDF, DOCX, TXT, CSV)</div>
                <div><span style="color:var(--accent-cyan,#00d4ff);">&#x25B6;</span>
                     &nbsp;Semantic &amp; Hybrid Vector Search</div>
                <div><span style="color:var(--accent-cyan,#00d4ff);">&#x25B6;</span>
                     &nbsp;Contextual Conversational AI</div>
                <div><span style="color:var(--accent-cyan,#00d4ff);">&#x25B6;</span>
                     &nbsp;LLM-Powered Response Generation</div>
                <div><span style="color:var(--accent-cyan,#00d4ff);">&#x25B6;</span>
                     &nbsp;Admin Dashboard &amp; Analytics</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            "<div style='font-family:var(--font-mono,monospace); font-size:0.72rem; "
            "text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted,#475569); "
            "margin-bottom:0.6rem;'>Sign In</div>",
            unsafe_allow_html=True,
        )
        show_login_form()

    st.markdown("---")

    # Demo credentials
    with st.expander("Demo Credentials"):
        st.markdown(
            """
            <div style="font-family:var(--font-mono,monospace); font-size:0.82rem;
                        color:var(--text-secondary,#94a3b8); line-height:1.8;">
                <strong style="color:var(--accent-cyan,#00d4ff);">Admin</strong><br>
                &nbsp;&nbsp;admin@example.com &nbsp;/&nbsp; admin123!<br><br>
                <strong style="color:var(--accent-cyan,#00d4ff);">User</strong><br>
                &nbsp;&nbsp;user@example.com &nbsp;/&nbsp; password123!
            </div>
            """,
            unsafe_allow_html=True,
        )
