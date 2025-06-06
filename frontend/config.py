"""
Frontend Configuration
Configuration settings for the Streamlit frontend application.
"""

import os
from typing import Any, Dict

# Backend API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"

# Full API URLs
API_URLS = {
    "auth": f"{API_BASE_URL}{API_PREFIX}/auth",
    "documents": f"{API_BASE_URL}{API_PREFIX}/documents",
    "search": f"{API_BASE_URL}{API_PREFIX}/search",
    "health": f"{API_BASE_URL}{API_PREFIX}/health",
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "Enterprise RAG System",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# File Upload Configuration
FILE_UPLOAD_CONFIG = {
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "allowed_types": ["pdf", "txt", "docx", "csv"],
    "upload_path": "uploads/",
}

# Search Configuration
SEARCH_CONFIG = {
    "default_search_type": "semantic",
    "max_results": 10,
    "similarity_threshold": 0.3,
    "conversation_timeout": 3600,  # 1 hour
}

# UI Configuration
UI_CONFIG = {
    "theme": {
        "primary_color": "#FF6B6B",
        "background_color": "#FFFFFF",
        "secondary_background_color": "#F0F2F6",
        "text_color": "#262730",
    },
    "sidebar_width": 300,
    "chat_height": 400,
    "results_per_page": 5,
}

# Session State Keys
SESSION_KEYS = {
    "authenticated": "authenticated",
    "user": "user",
    "token": "access_token",
    "refresh_token": "refresh_token",
    "conversation_id": "conversation_id",
    "chat_history": "chat_history",
    "search_history": "search_history",
    "current_page": "current_page",
}

# Error Messages
ERROR_MESSAGES = {
    "auth_failed": "Authentication failed. Please check your credentials.",
    "token_expired": "Session expired. Please log in again.",
    "upload_failed": "File upload failed. Please try again.",
    "search_failed": "Search failed. Please try again.",
    "connection_error": "Connection to backend failed. Please check if the server is running.",
    "file_too_large": f"File size exceeds the maximum limit of {FILE_UPLOAD_CONFIG['max_file_size'] // (1024 * 1024)}MB.",
    "invalid_file_type": f"Invalid file type. Allowed types: {', '.join(FILE_UPLOAD_CONFIG['allowed_types'])}",
}

# Success Messages
SUCCESS_MESSAGES = {
    "login_success": "Successfully logged in!",
    "logout_success": "Successfully logged out!",
    "upload_success": "File uploaded successfully!",
    "document_deleted": "Document deleted successfully!",
    "settings_saved": "Settings saved successfully!",
}


def get_api_url(endpoint: str) -> str:
    """Get full API URL for an endpoint."""
    return API_URLS.get(endpoint, f"{API_BASE_URL}{API_PREFIX}")


def get_config(section: str) -> Dict[str, Any]:
    """Get configuration section."""
    configs = {
        "streamlit": STREAMLIT_CONFIG,
        "file_upload": FILE_UPLOAD_CONFIG,
        "search": SEARCH_CONFIG,
        "ui": UI_CONFIG,
    }
    return configs.get(section, {})
