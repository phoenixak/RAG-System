"""
Utility Functions
Common utility functions for the Streamlit frontend.
"""

import base64
import hashlib
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import streamlit as st


def format_timestamp(timestamp: Union[int, float]) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "Unknown"


def format_time_ago(timestamp: Union[int, float]) -> str:
    """Format timestamp as time ago (e.g., '2 hours ago')."""
    try:
        dt = datetime.fromtimestamp(timestamp)
        now = datetime.now()
        diff = now - dt

        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"
    except (ValueError, TypeError):
        return "Unknown"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def highlight_search_terms(text: str, query: str) -> str:
    """Highlight search terms in text."""
    if not query:
        return text

    # Simple highlighting - in a real app you might want more sophisticated highlighting
    terms = query.lower().split()
    highlighted_text = text

    for term in terms:
        if term in text.lower():
            # Find all occurrences and replace with highlighted version
            import re

            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted_text = pattern.sub(f"**{term}**", highlighted_text)

    return highlighted_text


def show_loading_spinner(message: str = "Loading..."):
    """Show a loading spinner with message."""
    return st.spinner(message)


def show_success_message(message: str, duration: int = 3):
    """Show a success message that disappears after duration."""
    success_placeholder = st.empty()
    success_placeholder.success(message)
    time.sleep(duration)
    success_placeholder.empty()


def show_error_message(message: str, duration: int = 5):
    """Show an error message that disappears after duration."""
    error_placeholder = st.empty()
    error_placeholder.error(message)
    time.sleep(duration)
    error_placeholder.empty()


def create_download_link(
    data: Union[str, bytes], filename: str, mime_type: str = "application/octet-stream"
) -> str:
    """Create a download link for data."""
    if isinstance(data, str):
        data = data.encode("utf-8")

    b64_data = base64.b64encode(data).decode()
    return f'<a href="data:{mime_type};base64,{b64_data}" download="{filename}">Download {filename}</a>'


def validate_file_upload(
    uploaded_file, allowed_types: List[str], max_size: int
) -> tuple:
    """Validate uploaded file."""
    if uploaded_file is None:
        return False, "No file uploaded"

    # Check file type
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type not in allowed_types:
        return (
            False,
            f"File type '{file_type}' not allowed. Allowed types: {', '.join(allowed_types)}",
        )

    # Check file size
    if uploaded_file.size > max_size:
        max_size_mb = max_size / (1024 * 1024)
        return False, f"File size exceeds {max_size_mb:.1f}MB limit"

    return True, "File is valid"


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string."""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely serialize object to JSON."""
    try:
        return json.dumps(obj, default=str, indent=2)
    except (TypeError, ValueError):
        return default


def get_file_hash(file_content: bytes) -> str:
    """Generate hash for file content."""
    return hashlib.md5(file_content).hexdigest()


def paginate_results(results: List[Any], page: int, per_page: int) -> tuple:
    """Paginate a list of results."""
    total_items = len(results)
    total_pages = (total_items + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    page_results = results[start_idx:end_idx]

    return page_results, total_pages, total_items


def show_pagination_controls(
    current_page: int, total_pages: int, key_prefix: str = "page"
):
    """Show pagination controls."""
    if total_pages <= 1:
        return current_page

    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("⏮️ First", key=f"{key_prefix}_first", disabled=current_page <= 1):
            return 1

    with col2:
        if st.button("⬅️ Prev", key=f"{key_prefix}_prev", disabled=current_page <= 1):
            return current_page - 1

    with col3:
        st.write(f"Page {current_page} of {total_pages}")

    with col4:
        if st.button(
            "Next ➡️", key=f"{key_prefix}_next", disabled=current_page >= total_pages
        ):
            return current_page + 1

    with col5:
        if st.button(
            "Last ⏭️", key=f"{key_prefix}_last", disabled=current_page >= total_pages
        ):
            return total_pages

    return current_page


def create_status_badge(status: str) -> str:
    """Create a colored status badge."""
    status_colors = {
        "completed": "🟢",
        "processing": "🟡",
        "failed": "🔴",
        "pending": "⚪",
        "active": "🟢",
        "inactive": "⚪",
    }

    color = status_colors.get(status.lower(), "⚪")
    return f"{color} {status.title()}"


def format_search_score(score: float) -> str:
    """Format search relevance score."""
    if score >= 0.9:
        return f"🔥 {score:.1%}"  # Excellent match
    elif score >= 0.7:
        return f"⭐ {score:.1%}"  # Good match
    elif score >= 0.5:
        return f"👍 {score:.1%}"  # Fair match
    else:
        return f"👌 {score:.1%}"  # Weak match


def create_metric_card(
    title: str, value: Union[str, int, float], delta: Optional[str] = None
):
    """Create a metric display card."""
    st.metric(label=title, value=value, delta=delta)


def show_json_viewer(data: Dict[str, Any], title: str = "JSON Data"):
    """Show JSON data in an expandable viewer."""
    with st.expander(f"🔍 {title}"):
        st.json(data)


def create_progress_bar(current: int, total: int, label: str = "Progress") -> float:
    """Create a progress bar."""
    if total == 0:
        progress = 0.0
    else:
        progress = current / total

    st.progress(progress, text=f"{label}: {current}/{total} ({progress:.1%})")
    return progress


def format_document_type(filename: str) -> str:
    """Format document type with icon."""
    extension = filename.split(".")[-1].lower()

    type_icons = {
        "pdf": "📄",
        "docx": "📝",
        "doc": "📝",
        "txt": "📃",
        "csv": "📊",
        "xlsx": "📊",
        "xls": "📊",
    }

    icon = type_icons.get(extension, "📄")
    return f"{icon} {extension.upper()}"


def create_info_box(title: str, content: str, type: str = "info"):
    """Create an information box."""
    if type == "info":
        st.info(f"**{title}**\n\n{content}")
    elif type == "warning":
        st.warning(f"**{title}**\n\n{content}")
    elif type == "error":
        st.error(f"**{title}**\n\n{content}")
    elif type == "success":
        st.success(f"**{title}**\n\n{content}")


def debounce_input(key: str, delay: float = 0.5) -> bool:
    """Debounce input to avoid excessive API calls."""
    current_time = time.time()
    last_time_key = f"{key}_last_time"

    if last_time_key not in st.session_state:
        st.session_state[last_time_key] = 0

    if current_time - st.session_state[last_time_key] >= delay:
        st.session_state[last_time_key] = current_time
        return True

    return False


def get_theme_config():
    """Get theme configuration based on Streamlit settings."""
    # This is a placeholder - Streamlit doesn't expose theme info directly
    return {
        "primary_color": "#FF6B6B",
        "background_color": "#FFFFFF",
        "secondary_background_color": "#F0F2F6",
        "text_color": "#262730",
    }


def format_conversation_preview(messages: List[Dict]) -> str:
    """Create a preview of conversation for display."""
    if not messages:
        return "No messages"

    preview = ""
    for msg in messages[-3:]:  # Show last 3 messages
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            preview += f"👤 {truncate_text(content, 50)}\n"
        elif role == "assistant":
            preview += f"🤖 {truncate_text(content, 50)}\n"

    return preview


def normalize_document_response(doc_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize document response to ensure consistent field names."""
    normalized = {}

    # Handle ID field (backend uses 'id', frontend expects 'document_id')
    normalized["document_id"] = doc_data.get("id") or doc_data.get("document_id")

    # Handle filename
    normalized["filename"] = doc_data.get("filename", "Unknown")

    # Handle status field (backend uses 'processing_status', frontend expects 'status')
    status = doc_data.get("processing_status") or doc_data.get("status", "unknown")
    normalized["status"] = status

    # Handle file size (backend uses 'size', frontend expects 'file_size')
    file_size = doc_data.get("size") or doc_data.get("file_size", 0)
    normalized["file_size"] = file_size

    # Handle other fields with fallbacks
    normalized["chunk_count"] = doc_data.get("chunk_count", 0)
    normalized["upload_timestamp"] = doc_data.get("created_at") or doc_data.get(
        "upload_timestamp"
    )
    normalized["content_length"] = doc_data.get("text_length") or doc_data.get(
        "content_length", 0
    )
    normalized["page_count"] = doc_data.get("page_count")
    normalized["processing_error"] = doc_data.get("processing_error")
    normalized["content_preview"] = doc_data.get("content_preview")

    # Copy any remaining fields
    for key, value in doc_data.items():
        if key not in normalized:
            normalized[key] = value

    return normalized
