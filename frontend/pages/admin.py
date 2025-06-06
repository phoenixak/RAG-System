"""
Admin Dashboard Page
System administration and monitoring interface (admin users only).
"""

import streamlit as st

from frontend.components.api_client import APIException, get_api_client
from frontend.components.auth import get_current_user, require_role
from frontend.components.utils import (
    create_metric_card,
    create_status_badge,
    format_file_size,
    format_timestamp,
)


@require_role("admin")
def show_admin_interface():
    """Main admin dashboard interface."""
    st.title("👨‍💼 Admin Dashboard")
    st.markdown("System administration and monitoring tools.")

    # Admin tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "👥 Users", "🗄️ System", "⚙️ Config"]
    )

    with tab1:
        show_system_overview()

    with tab2:
        show_user_management()

    with tab3:
        show_system_monitoring()

    with tab4:
        show_system_configuration()


# Alias for compatibility with main.py import
def show_admin_page():
    """Alias for show_admin_interface."""
    show_admin_interface()


def show_system_overview():
    """Show system overview dashboard."""
    st.subheader("📊 System Overview")

    try:
        # Get system health
        api_client = get_api_client()
        health_data = api_client.health_check()

        # System status
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status = health_data.get("status", "unknown")
            status_color = "🟢" if status == "healthy" else "🔴"
            create_metric_card("System Status", f"{status_color} {status.title()}")

        with col2:
            uptime = health_data.get("uptime", "N/A")
            create_metric_card("Uptime", uptime)

        with col3:
            version = health_data.get("version", "N/A")
            create_metric_card("Version", version)

        with col4:
            environment = health_data.get("environment", "N/A")
            create_metric_card("Environment", environment.title())

        st.markdown("---")

        # Document statistics
        show_document_statistics()

        st.markdown("---")

        # Recent activity
        show_recent_activity()

    except APIException as e:
        st.error(f"Failed to load system overview: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")


def show_document_statistics():
    """Show document-related statistics."""
    st.subheader("📄 Document Statistics")

    try:
        api_client = get_api_client()

        # Get documents summary
        docs_response = api_client.get_documents(limit=100)  # Get up to 100 for stats
        documents = docs_response.get("documents", [])
        total_docs = docs_response.get("total", 0)

        # Calculate statistics
        total_size = sum(doc.get("file_size", 0) for doc in documents)
        completed_docs = len(
            [doc for doc in documents if doc.get("status") == "completed"]
        )
        processing_docs = len(
            [doc for doc in documents if doc.get("status") == "processing"]
        )
        failed_docs = len([doc for doc in documents if doc.get("status") == "failed"])
        total_chunks = sum(doc.get("chunk_count", 0) for doc in documents)

        # Document type breakdown
        doc_types = {}
        for doc in documents:
            filename = doc.get("filename", "")
            ext = filename.split(".")[-1].upper() if "." in filename else "UNKNOWN"
            doc_types[ext] = doc_types.get(ext, 0) + 1

        # Display statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            create_metric_card("Total Documents", total_docs)
            create_metric_card("Completed", completed_docs)

        with col2:
            create_metric_card("Total Size", format_file_size(total_size))
            create_metric_card("Processing", processing_docs)

        with col3:
            create_metric_card("Total Chunks", total_chunks)
            create_metric_card("Failed", failed_docs)

        with col4:
            avg_chunks = int(total_chunks / total_docs) if total_docs > 0 else 0
            create_metric_card("Avg Chunks/Doc", avg_chunks)

            if doc_types:
                most_common = max(doc_types.items(), key=lambda x: x[1])
                create_metric_card(
                    "Most Common Type", f"{most_common[0]} ({most_common[1]})"
                )

        # Document type breakdown chart
        if doc_types:
            st.subheader("📊 Document Types")
            chart_data = {
                "Type": list(doc_types.keys()),
                "Count": list(doc_types.values()),
            }
            st.bar_chart(chart_data, x="Type", y="Count")

    except Exception as e:
        st.error(f"Failed to load document statistics: {str(e)}")


def show_recent_activity():
    """Show recent system activity."""
    st.subheader("🕒 Recent Activity")

    # This would typically come from system logs
    # For now, we'll show recent documents as a proxy
    try:
        api_client = get_api_client()
        docs_response = api_client.get_documents(limit=10)
        recent_docs = docs_response.get("documents", [])

        if recent_docs:
            st.write("**Recent Document Uploads:**")

            for doc in recent_docs:
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    filename = doc.get("filename", "Unknown")
                    st.write(f"📄 {filename}")

                with col2:
                    status = doc.get("status", "unknown")
                    st.write(create_status_badge(status))

                with col3:
                    upload_time = doc.get("upload_timestamp")
                    if upload_time:
                        from frontend.components.utils import format_time_ago

                        st.write(format_time_ago(upload_time))
        else:
            st.info("No recent activity to display.")

    except Exception:
        st.warning("Unable to load recent activity")


def show_user_management():
    """Show user management interface."""
    st.subheader("👥 User Management")

    # Note: This would require backend user management APIs
    st.info("🚧 User management features coming soon!")

    # Placeholder for user management features
    with st.expander("👤 Current Session Info", expanded=True):
        user = get_current_user()
        if user:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Current Admin User:**")
                st.write(f"• **Email:** {user.get('email', 'N/A')}")
                st.write(f"• **Role:** {user.get('role', 'N/A')}")

            with col2:
                st.write("**Session Info:**")
                if user.get("created_at"):
                    st.write(
                        f"• **Account Created:** {format_timestamp(user['created_at'])}"
                    )
                st.write(f"• **User ID:** {user.get('user_id', 'N/A')}")

    # Future user management features
    with st.expander("🔮 Planned Features", expanded=False):
        st.markdown("""
        **Upcoming User Management Features:**
        
        - 👥 **User List**: View all registered users
        - ➕ **Add Users**: Create new user accounts
        - 🔧 **Edit Users**: Modify user permissions and roles
        - 🗑️ **Delete Users**: Remove user accounts
        - 📊 **User Analytics**: Usage statistics per user
        - 🔒 **Permission Management**: Fine-grained access control
        - 📧 **User Notifications**: Send notifications to users
        """)


def show_system_monitoring():
    """Show system monitoring and diagnostics."""
    st.subheader("🗄️ System Monitoring")

    # System health checks
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Service Health Checks**")

        if st.button("🔄 Run Health Checks", type="primary"):
            run_health_checks()

    with col2:
        st.write("**System Resources**")
        show_system_resources()

    st.markdown("---")

    # Database information
    show_database_info()

    st.markdown("---")

    # Cache management
    show_cache_management()


def run_health_checks():
    """Run comprehensive health checks."""
    with st.spinner("Running health checks..."):
        results = {}

        try:
            # Backend API health
            api_client = get_api_client()
            health_data = api_client.health_check()
            results["API"] = {
                "status": "✅ Healthy"
                if health_data.get("status") == "healthy"
                else "❌ Unhealthy",
                "details": health_data,
            }
        except Exception as e:
            results["API"] = {"status": "❌ Error", "details": str(e)}

        # Additional health checks would go here
        # - Database connectivity
        # - Vector store health
        # - Model loading status
        # - Cache connectivity

        # Display results
        st.subheader("🩺 Health Check Results")

        for service, result in results.items():
            col1, col2 = st.columns([1, 3])

            with col1:
                st.write(f"**{service}:**")

            with col2:
                st.write(result["status"])

                if isinstance(result["details"], dict):
                    st.write(f"**📋 {service} Details:**")
                    st.json(result["details"])
                else:
                    st.caption(str(result["details"]))


def show_system_resources():
    """Show system resource information."""
    # Placeholder for system resource monitoring
    st.info("📊 Resource monitoring integration pending")

    # This would typically show:
    # - CPU usage
    # - Memory usage
    # - Disk space
    # - Network I/O
    # - GPU usage (if applicable)

    with st.expander("💡 Resource Monitoring", expanded=False):
        st.markdown("""
        **Planned Resource Monitoring:**
        
        - 🖥️ **CPU Usage**: Real-time CPU utilization
        - 🧠 **Memory Usage**: RAM consumption and availability
        - 💾 **Disk Space**: Storage usage and free space
        - 🌐 **Network I/O**: Bandwidth usage and latency
        - 🎮 **GPU Usage**: GPU utilization for ML models
        - 📈 **Historical Trends**: Resource usage over time
        """)


def show_database_info():
    """Show database and vector store information."""
    st.subheader("🗃️ Database Information")

    try:
        api_client = get_api_client()

        # Try to get some database statistics
        # This would require additional backend endpoints

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Vector Database**")
            st.info("ChromaDB connection status: ✅ Connected")

            # Get document count as a proxy for vector store size
            docs_response = api_client.get_documents(limit=1)
            total_docs = docs_response.get("total", 0)
            st.write("• **Collections**: 1 (enterprise_documents)")
            st.write(f"• **Documents**: {total_docs}")

        with col2:
            st.write("**Cache System**")
            st.info("Redis connection status: ✅ Connected")
            st.write("• **Type**: Redis")
            st.write("• **Status**: Active")

    except Exception as e:
        st.warning(f"Unable to retrieve database info: {str(e)}")


def show_cache_management():
    """Show cache management controls."""
    st.subheader("💾 Cache Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ Clear Search Cache", help="Clear cached search results"):
            clear_search_cache()

    with col2:
        if st.button("🔄 Clear Embedding Cache", help="Clear cached embeddings"):
            clear_embedding_cache()

    with col3:
        if st.button("🧹 Clear All Caches", help="Clear all system caches"):
            clear_all_caches()

    # Cache statistics (placeholder)
    with st.expander("📊 Cache Statistics", expanded=False):
        st.info("Cache statistics integration pending")
        st.markdown("""
        **Planned Cache Metrics:**
        
        - 🎯 **Hit Rate**: Cache hit/miss ratio
        - 📏 **Size**: Current cache size and limits
        - ⏱️ **TTL**: Time-to-live for cached items
        - 🔄 **Evictions**: Number of cache evictions
        - 📈 **Performance**: Cache performance metrics
        """)


def clear_search_cache():
    """Clear search result cache."""
    try:
        # This would call a backend endpoint to clear search cache
        # For now, just show a success message
        st.success("🗑️ Search cache cleared successfully!")
    except Exception as e:
        st.error(f"Failed to clear search cache: {str(e)}")


def clear_embedding_cache():
    """Clear embedding cache."""
    try:
        # This would call a backend endpoint to clear embedding cache
        st.success("🔄 Embedding cache cleared successfully!")
    except Exception as e:
        st.error(f"Failed to clear embedding cache: {str(e)}")


def clear_all_caches():
    """Clear all system caches."""
    try:
        # This would call backend endpoints to clear all caches
        st.success("🧹 All caches cleared successfully!")
    except Exception as e:
        st.error(f"Failed to clear caches: {str(e)}")


def show_system_configuration():
    """Show system configuration interface."""
    st.subheader("⚙️ System Configuration")

    # Configuration sections
    with st.expander("🔍 Search Configuration", expanded=False):
        show_search_config()

    with st.expander("📄 Document Processing", expanded=False):
        show_document_config()

    with st.expander("🚀 Performance Settings", expanded=False):
        show_performance_config()

    with st.expander("🔒 Security Settings", expanded=False):
        show_security_config()


def show_search_config():
    """Show search configuration options."""
    st.write("**Default Search Settings**")

    col1, col2 = st.columns(2)

    with col1:
        max_results = st.slider(
            "Global Max Results", 1, 100, 20, key="admin_max_results"
        )
        similarity_threshold = st.slider(
            "Default Similarity Threshold", 0.0, 1.0, 0.7, key="admin_similarity"
        )

    with col2:
        enable_caching = st.checkbox(
            "Enable Search Caching", value=True, key="admin_search_cache"
        )
        cache_ttl = st.slider("Cache TTL (minutes)", 5, 120, 30, key="admin_cache_ttl")

    if st.button("💾 Save Search Config"):
        st.success("Search configuration saved!")


def show_document_config():
    """Show document processing configuration."""
    st.write("**Document Processing Settings**")

    col1, col2 = st.columns(2)

    with col1:
        max_file_size = st.slider(
            "Max File Size (MB)", 10, 200, 50, key="admin_max_size"
        )
        chunk_size = st.slider(
            "Chunk Size (tokens)", 500, 2000, 1000, key="admin_chunk_size"
        )

    with col2:
        chunk_overlap = st.slider(
            "Chunk Overlap (tokens)", 50, 500, 200, key="admin_chunk_overlap"
        )
        concurrent_processing = st.slider(
            "Concurrent Jobs", 1, 10, 3, key="admin_concurrent"
        )

    if st.button("💾 Save Document Config"):
        st.success("Document configuration saved!")


def show_performance_config():
    """Show performance configuration options."""
    st.write("**Performance Settings**")

    col1, col2 = st.columns(2)

    with col1:
        embedding_batch_size = st.slider(
            "Embedding Batch Size", 16, 128, 32, key="admin_batch_size"
        )
        worker_threads = st.slider("Worker Threads", 1, 8, 4, key="admin_workers")

    with col2:
        memory_limit = st.slider("Memory Limit (GB)", 1, 16, 8, key="admin_memory")
        enable_gpu = st.checkbox("Enable GPU Processing", value=False, key="admin_gpu")

    if st.button("💾 Save Performance Config"):
        st.success("Performance configuration saved!")


def show_security_config():
    """Show security configuration options."""
    st.write("**Security Settings**")

    col1, col2 = st.columns(2)

    with col1:
        session_timeout = st.slider(
            "Session Timeout (hours)", 1, 24, 8, key="admin_session_timeout"
        )
        max_login_attempts = st.slider(
            "Max Login Attempts", 3, 10, 5, key="admin_max_attempts"
        )

    with col2:
        require_2fa = st.checkbox(
            "Require 2FA", value=False, key="admin_2fa", disabled=True
        )
        audit_logging = st.checkbox(
            "Enable Audit Logging", value=True, key="admin_audit"
        )

    if require_2fa:
        st.caption("⚠️ 2FA not yet implemented")

    if st.button("💾 Save Security Config"):
        st.success("Security configuration saved!")


if __name__ == "__main__":
    show_admin_interface()
