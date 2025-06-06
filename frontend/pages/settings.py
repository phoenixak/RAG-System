"""
Settings Page
User preferences and system configuration interface.
"""

import time

import streamlit as st

from frontend.components.api_client import get_api_client
from frontend.components.auth import get_current_user, requires_auth
from frontend.config import SEARCH_CONFIG, SESSION_KEYS, SUCCESS_MESSAGES, UI_CONFIG


@requires_auth
def show_settings_interface():
    """Main settings interface page."""
    st.title("⚙️ Settings")
    st.markdown("Configure your preferences and system settings.")

    # Settings tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Search", "🎨 Interface", "👤 Profile", "💾 Data"]
    )

    with tab1:
        show_search_settings()

    with tab2:
        show_interface_settings()

    with tab3:
        show_profile_settings()

    with tab4:
        show_data_settings()


# Alias for compatibility with main.py import
def show_settings_page():
    """Alias for show_settings_interface."""
    show_settings_interface()


def show_search_settings():
    """Show search configuration settings."""
    st.subheader("🔍 Search Configuration")
    st.markdown("Customize how search works across the application.")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Default Search Behavior**")

        # Default search type
        default_search_type = st.selectbox(
            "Default Search Type",
            options=["semantic", "hybrid", "contextual"],
            index=0,
            key="settings_default_search_type",
            help="Search type used by default in chat and search interfaces",
        )

        # Default max results
        default_max_results = st.slider(
            "Default Max Results",
            min_value=5,
            max_value=50,
            value=st.session_state.get(
                "settings_max_results", SEARCH_CONFIG["max_results"]
            ),
            key="settings_max_results",
            help="Default number of search results to retrieve",
        )

        # Default similarity threshold
        default_similarity_threshold = st.slider(
            "Default Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get(
                "settings_similarity_threshold", SEARCH_CONFIG["similarity_threshold"]
            ),
            step=0.05,
            key="settings_similarity_threshold",
            help="Minimum similarity score for search results",
        )

    with col2:
        st.write("**Hybrid Search Configuration**")

        # Semantic weight
        semantic_weight = st.slider(
            "Semantic Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("settings_semantic_weight", 0.7),
            step=0.1,
            key="settings_semantic_weight",
            help="Weight for semantic search in hybrid mode",
        )

        # Keyword weight
        keyword_weight = st.slider(
            "Keyword Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("settings_keyword_weight", 0.3),
            step=0.1,
            key="settings_keyword_weight",
            help="Weight for keyword search in hybrid mode",
        )

        # Weight validation
        total_weight = semantic_weight + keyword_weight
        if abs(total_weight - 1.0) > 0.01:
            st.warning(
                f"⚠️ Weights sum to {total_weight:.2f}. Consider adjusting to sum to 1.0 for optimal results."
            )

        # Conversation settings
        st.write("**Conversation Settings**")

        conversation_timeout = st.slider(
            "Conversation Timeout (minutes)",
            min_value=15,
            max_value=240,
            value=st.session_state.get("settings_conversation_timeout", 60),
            key="settings_conversation_timeout",
            help="How long to keep conversation context active",
        )

    # Advanced settings
    with st.expander("🔧 Advanced Search Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            enable_reranking = st.checkbox(
                "Enable Result Re-ranking",
                value=st.session_state.get("settings_enable_reranking", True),
                key="settings_enable_reranking",
                help="Use cross-encoder models to improve result relevance",
            )

            enable_caching = st.checkbox(
                "Enable Search Caching",
                value=st.session_state.get("settings_enable_caching", True),
                key="settings_enable_caching",
                help="Cache search results for faster repeated queries",
            )

        with col2:
            cache_ttl = st.slider(
                "Cache TTL (minutes)",
                min_value=5,
                max_value=120,
                value=st.session_state.get("settings_cache_ttl", 30),
                key="settings_cache_ttl",
                help="How long to cache search results",
                disabled=not enable_caching,
            )

            enable_query_expansion = st.checkbox(
                "Enable Query Expansion",
                value=st.session_state.get("settings_enable_query_expansion", False),
                key="settings_enable_query_expansion",
                help="Automatically expand queries with synonyms",
            )

    # Save search settings
    if st.button("💾 Save Search Settings", type="primary"):
        save_search_settings()


def show_interface_settings():
    """Show interface customization settings."""
    st.subheader("🎨 Interface Customization")
    st.markdown("Personalize your user interface experience.")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Display Preferences**")

        # Results per page
        results_per_page = st.slider(
            "Results Per Page",
            min_value=3,
            max_value=20,
            value=st.session_state.get(
                "settings_results_per_page", UI_CONFIG["results_per_page"]
            ),
            key="settings_results_per_page",
            help="Number of search results to show per page",
        )

        # Chat message limit
        chat_display_limit = st.slider(
            "Chat Messages to Display",
            min_value=10,
            max_value=100,
            value=st.session_state.get("settings_chat_display_limit", 50),
            key="settings_chat_display_limit",
            help="Maximum number of chat messages to display at once",
        )

        # Auto-scroll chat
        auto_scroll_chat = st.checkbox(
            "Auto-scroll Chat",
            value=st.session_state.get("settings_auto_scroll_chat", True),
            key="settings_auto_scroll_chat",
            help="Automatically scroll to latest message in chat",
        )

    with col2:
        st.write("**Content Display**")

        # Show relevance scores
        show_relevance_scores = st.checkbox(
            "Show Relevance Scores",
            value=st.session_state.get("settings_show_relevance_scores", True),
            key="settings_show_relevance_scores",
            help="Display search relevance scores in results",
        )

        # Show document metadata
        show_document_metadata = st.checkbox(
            "Show Document Metadata",
            value=st.session_state.get("settings_show_document_metadata", True),
            key="settings_show_document_metadata",
            help="Display document information in search results",
        )

        # Highlight search terms
        highlight_search_terms = st.checkbox(
            "Highlight Search Terms",
            value=st.session_state.get("settings_highlight_search_terms", True),
            key="settings_highlight_search_terms",
            help="Highlight matching terms in search results",
        )

        # Compact view
        compact_view = st.checkbox(
            "Compact View",
            value=st.session_state.get("settings_compact_view", False),
            key="settings_compact_view",
            help="Use a more compact layout to show more content",
        )

    # Notification settings
    with st.expander("🔔 Notification Settings", expanded=False):
        enable_success_notifications = st.checkbox(
            "Success Notifications",
            value=st.session_state.get("settings_success_notifications", True),
            key="settings_success_notifications",
            help="Show notifications for successful operations",
        )

        enable_error_notifications = st.checkbox(
            "Error Notifications",
            value=st.session_state.get("settings_error_notifications", True),
            key="settings_error_notifications",
            help="Show notifications for errors",
        )

        notification_duration = st.slider(
            "Notification Duration (seconds)",
            min_value=1,
            max_value=10,
            value=st.session_state.get("settings_notification_duration", 3),
            key="settings_notification_duration",
            help="How long to show notifications",
        )

    # Save interface settings
    if st.button("💾 Save Interface Settings", type="primary"):
        save_interface_settings()


def show_profile_settings():
    """Show user profile settings."""
    st.subheader("👤 User Profile")
    st.markdown("View and manage your account information.")

    # Get current user info
    user = get_current_user()
    if not user:
        st.error("Unable to load user information.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Account Information**")

        # Display user info (read-only)
        st.text_input("Email", value=user.get("email", ""), disabled=True)
        st.text_input("Role", value=user.get("role", "user").title(), disabled=True)

        # Account creation date
        if user.get("created_at"):
            from frontend.components.utils import format_timestamp

            created_date = format_timestamp(user["created_at"])
            st.text_input("Member Since", value=created_date, disabled=True)

    with col2:
        st.write("**Preferences**")

        # Display name
        display_name = st.text_input(
            "Display Name",
            value=st.session_state.get(
                "settings_display_name", user.get("email", "").split("@")[0]
            ),
            key="settings_display_name",
            help="Name to display in the interface",
        )

        # Email notifications (placeholder - would need backend support)
        email_notifications = st.checkbox(
            "Email Notifications",
            value=st.session_state.get("settings_email_notifications", False),
            key="settings_email_notifications",
            help="Receive email notifications for important updates",
            disabled=True,  # Disabled until backend supports it
        )

        if email_notifications:
            st.caption("⚠️ Email notifications not yet implemented")

    # Usage statistics
    show_usage_statistics()

    # Save profile settings
    if st.button("💾 Save Profile Settings", type="primary"):
        save_profile_settings()


def show_usage_statistics():
    """Show user usage statistics."""
    with st.expander("📊 Usage Statistics", expanded=False):
        try:
            # Get documents count
            api_client = get_api_client()
            docs_response = api_client.get_documents(limit=1)
            total_docs = docs_response.get("total", 0)

            # Get chat history count
            chat_history = st.session_state.get(SESSION_KEYS["chat_history"], [])
            total_messages = len(
                [msg for msg in chat_history if msg.get("role") == "user"]
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📄 Documents Uploaded", total_docs)

            with col2:
                st.metric("💬 Chat Messages", total_messages)

            with col3:
                current_session_length = len(chat_history)
                st.metric("📝 Current Session", current_session_length)

        except Exception:
            st.warning("Unable to load usage statistics")


def show_data_settings():
    """Show data management settings."""
    st.subheader("💾 Data Management")
    st.markdown("Manage your data, exports, and privacy settings.")

    # Export options
    st.write("**Export Your Data**")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "📄 Export Documents List", help="Export list of your uploaded documents"
        ):
            export_documents_list()

    with col2:
        if st.button("💬 Export Chat History", help="Export your conversation history"):
            export_chat_history()

    with col3:
        if st.button("⚙️ Export Settings", help="Export your current settings"):
            export_settings()

    st.markdown("---")

    # Clear data options
    st.write("**Clear Data**")
    st.warning("⚠️ These actions cannot be undone!")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Chat History", help="Clear all conversation history"):
            clear_chat_history()

    with col2:
        if st.button("🔄 Reset Settings", help="Reset all settings to defaults"):
            reset_settings()

    # Import settings
    st.markdown("---")
    st.write("**Import Settings**")

    uploaded_settings = st.file_uploader(
        "Upload Settings File",
        type=["json"],
        help="Import previously exported settings",
    )

    if uploaded_settings:
        if st.button("📥 Import Settings"):
            import_settings(uploaded_settings)


def save_search_settings():
    """Save search settings to session state."""
    settings_keys = [
        "settings_default_search_type",
        "settings_max_results",
        "settings_similarity_threshold",
        "settings_semantic_weight",
        "settings_keyword_weight",
        "settings_conversation_timeout",
        "settings_enable_reranking",
        "settings_enable_caching",
        "settings_cache_ttl",
        "settings_enable_query_expansion",
    ]

    # Save to session state (in a real app, this would save to backend)
    for key in settings_keys:
        if key in st.session_state:
            # Settings are already in session state
            pass

    st.success(SUCCESS_MESSAGES["settings_saved"])


def save_interface_settings():
    """Save interface settings to session state."""
    settings_keys = [
        "settings_results_per_page",
        "settings_chat_display_limit",
        "settings_auto_scroll_chat",
        "settings_show_relevance_scores",
        "settings_show_document_metadata",
        "settings_highlight_search_terms",
        "settings_compact_view",
        "settings_success_notifications",
        "settings_error_notifications",
        "settings_notification_duration",
    ]

    # Save to session state
    for key in settings_keys:
        if key in st.session_state:
            pass

    st.success(SUCCESS_MESSAGES["settings_saved"])


def save_profile_settings():
    """Save profile settings."""
    # In a real app, this would update the backend user profile
    st.success("Profile settings saved!")


def export_documents_list():
    """Export list of documents."""
    try:
        api_client = get_api_client()
        response = api_client.get_documents(limit=1000)  # Get all documents
        documents = response.get("documents", [])

        # Create CSV data
        import pandas as pd

        df_data = []
        for doc in documents:
            df_data.append(
                {
                    "Filename": doc.get("filename", ""),
                    "Upload_Date": doc.get("upload_timestamp", ""),
                    "File_Size": doc.get("file_size", ""),
                    "Status": doc.get("status", ""),
                    "Chunk_Count": doc.get("chunk_count", ""),
                }
            )

        df = pd.DataFrame(df_data)
        csv_data = df.to_csv(index=False)

        st.download_button(
            label="📥 Download Documents List",
            data=csv_data,
            file_name=f"documents_list_{int(time.time())}.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Failed to export documents list: {str(e)}")


def export_chat_history():
    """Export chat history."""
    from frontend.components.chat_ui import export_chat_history as export_chat

    export_chat()


def export_settings():
    """Export current settings."""
    import json
    import time

    # Collect all settings from session state
    settings_data = {}
    for key, value in st.session_state.items():
        if key.startswith("settings_"):
            settings_data[key] = value

    export_data = {
        "settings": settings_data,
        "export_timestamp": time.time(),
        "version": "1.0",
    }

    json_str = json.dumps(export_data, indent=2, default=str)

    st.download_button(
        label="📥 Download Settings",
        data=json_str,
        file_name=f"settings_{int(time.time())}.json",
        mime="application/json",
    )


def clear_chat_history():
    """Clear chat history with confirmation."""
    if "confirm_clear_chat" not in st.session_state:
        st.session_state.confirm_clear_chat = False

    if not st.session_state.confirm_clear_chat:
        if st.button("⚠️ Confirm Clear Chat History"):
            st.session_state.confirm_clear_chat = True
            st.rerun()
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Yes, Clear All"):
                st.session_state[SESSION_KEYS["chat_history"]] = []
                if SESSION_KEYS["conversation_id"] in st.session_state:
                    del st.session_state[SESSION_KEYS["conversation_id"]]
                st.session_state.confirm_clear_chat = False
                st.success("Chat history cleared!")
                st.rerun()

        with col2:
            if st.button("❌ Cancel"):
                st.session_state.confirm_clear_chat = False
                st.rerun()


def reset_settings():
    """Reset all settings to defaults."""
    if "confirm_reset_settings" not in st.session_state:
        st.session_state.confirm_reset_settings = False

    if not st.session_state.confirm_reset_settings:
        if st.button("⚠️ Confirm Reset All Settings"):
            st.session_state.confirm_reset_settings = True
            st.rerun()
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Yes, Reset All"):
                # Remove all settings from session state
                settings_keys = [
                    key
                    for key in st.session_state.keys()
                    if key.startswith("settings_")
                ]
                for key in settings_keys:
                    del st.session_state[key]

                st.session_state.confirm_reset_settings = False
                st.success("Settings reset to defaults!")
                st.rerun()

        with col2:
            if st.button("❌ Cancel"):
                st.session_state.confirm_reset_settings = False
                st.rerun()


def import_settings(uploaded_file):
    """Import settings from uploaded file."""
    try:
        import json

        # Read and parse the file
        settings_data = json.loads(uploaded_file.read())

        if "settings" not in settings_data:
            st.error("Invalid settings file format")
            return

        # Import settings
        imported_count = 0
        for key, value in settings_data["settings"].items():
            if key.startswith("settings_"):
                st.session_state[key] = value
                imported_count += 1

        st.success(f"Imported {imported_count} settings successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Failed to import settings: {str(e)}")


if __name__ == "__main__":
    show_settings_interface()
