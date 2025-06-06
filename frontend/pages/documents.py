"""
Documents Management Page
Interface for uploading, viewing, and managing documents.
"""

import time
from typing import Any, Dict

import streamlit as st

from frontend.components.api_client import APIException, get_api_client
from frontend.components.auth import requires_auth
from frontend.components.utils import (
    create_status_badge,
    format_document_type,
    format_file_size,
    format_time_ago,
    format_timestamp,
    show_pagination_controls,
    truncate_text,
    validate_file_upload,
)
from frontend.config import FILE_UPLOAD_CONFIG


@requires_auth
def show_documents_interface():
    """Main documents management interface."""
    st.title("📄 Document Management")
    st.markdown("Upload, view, and manage your document collection.")

    # Document upload section
    show_upload_section()

    st.markdown("---")

    # Document list and management
    show_documents_list()


# Alias for compatibility with main.py import
def show_documents_page():
    """Alias for show_documents_interface."""
    show_documents_interface()


def show_upload_section():
    """Show file upload interface."""
    st.subheader("📤 Upload Documents")

    col1, col2 = st.columns([2, 1])

    with col1:
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=FILE_UPLOAD_CONFIG["allowed_types"],
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(FILE_UPLOAD_CONFIG['allowed_types'])}. Max size: {FILE_UPLOAD_CONFIG['max_file_size'] // (1024 * 1024)}MB per file.",
        )

        if uploaded_files:
            show_upload_preview(uploaded_files)

            if st.button("🚀 Upload Files", type="primary"):
                upload_files(uploaded_files)

    with col2:
        # Upload statistics and info
        show_upload_info()


def show_upload_preview(uploaded_files):
    """Show preview of files to be uploaded."""
    st.write("**Files to upload:**")

    total_size = 0
    valid_files = 0

    for file in uploaded_files:
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.write(f"📄 {file.name}")

        with col2:
            st.write(format_file_size(file.size))
            total_size += file.size

        with col3:
            # Validate file
            is_valid, message = validate_file_upload(
                file,
                FILE_UPLOAD_CONFIG["allowed_types"],
                FILE_UPLOAD_CONFIG["max_file_size"],
            )

            if is_valid:
                st.success("✅ Valid")
                valid_files += 1
            else:
                st.error(f"❌ {message}")

    # Summary
    st.write(
        f"**Summary:** {valid_files}/{len(uploaded_files)} valid files, Total size: {format_file_size(total_size)}"
    )


def upload_files(uploaded_files):
    """Upload files to the backend."""
    api_client = get_api_client()

    progress_bar = st.progress(0)
    status_text = st.empty()

    successful_uploads = 0
    failed_uploads = 0

    for i, file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Uploading {file.name}...")

            # Validate file
            is_valid, message = validate_file_upload(
                file,
                FILE_UPLOAD_CONFIG["allowed_types"],
                FILE_UPLOAD_CONFIG["max_file_size"],
            )

            if not is_valid:
                st.error(f"Skipped {file.name}: {message}")
                failed_uploads += 1
                continue

            # Upload file
            file_content = file.read()
            file.seek(0)  # Reset file pointer

            response = api_client.upload_document(file_content, file.name)

            st.success(f"✅ Uploaded: {file.name}")
            successful_uploads += 1

        except APIException as e:
            st.error(f"❌ Failed to upload {file.name}: {str(e)}")
            failed_uploads += 1

        except Exception as e:
            st.error(f"❌ Unexpected error uploading {file.name}: {str(e)}")
            failed_uploads += 1

    # Final status
    progress_bar.progress(1.0)
    status_text.text("Upload complete!")

    # Summary
    if successful_uploads > 0:
        st.success(f"🎉 Successfully uploaded {successful_uploads} file(s)")

    if failed_uploads > 0:
        st.error(f"❌ Failed to upload {failed_uploads} file(s)")

    # Refresh document list
    time.sleep(2)
    st.rerun()


def show_upload_info():
    """Show upload information and statistics."""
    st.subheader("ℹ️ Upload Info")

    # File format support
    with st.expander("📋 Supported Formats", expanded=False):
        formats = {
            "PDF": "Portable Document Format - Research papers, reports, manuals",
            "DOCX": "Microsoft Word documents - Articles, drafts, documentation",
            "TXT": "Plain text files - Notes, code, simple documents",
            "CSV": "Comma-separated values - Data tables, spreadsheets",
        }

        for format_name, description in formats.items():
            st.write(f"**{format_name}**: {description}")

    # Current limits
    st.write("**Upload Limits:**")
    st.write(
        f"• Max file size: {FILE_UPLOAD_CONFIG['max_file_size'] // (1024 * 1024)}MB"
    )
    st.write(f"• Supported types: {', '.join(FILE_UPLOAD_CONFIG['allowed_types'])}")

    # Processing info
    with st.expander("⚙️ Processing Info", expanded=False):
        st.write("""
        After upload, documents are automatically:
        1. **Text extracted** from the file format
        2. **Chunked** into searchable segments
        3. **Embedded** using AI models for semantic search
        4. **Indexed** for fast retrieval
        
        Processing time depends on document size and complexity.
        """)


def show_documents_list():
    """Show list of uploaded documents."""
    st.subheader("📚 Document Library")

    try:
        api_client = get_api_client()

        # Get current page
        current_page = st.session_state.get("docs_page", 1)
        per_page = 10

        # Fetch documents
        with st.spinner("Loading documents..."):
            skip = (current_page - 1) * per_page
            response = api_client.get_documents(skip=skip, limit=per_page)

        documents = response.get("documents", [])
        total_docs = response.get("total", 0)

        if not documents:
            show_empty_state()
            return

        # Documents header with stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Total Documents", total_docs)
        with col2:
            processing_count = len(
                [doc for doc in documents if doc.get("status") == "processing"]
            )
            st.metric("⚡ Processing", processing_count)
        with col3:
            completed_count = len(
                [doc for doc in documents if doc.get("status") == "completed"]
            )
            st.metric("✅ Ready", completed_count)

        # Search and filter controls
        show_document_controls()

        # Document list
        for doc in documents:
            show_document_card(doc)

        # Pagination
        total_pages = (total_docs + per_page - 1) // per_page
        if total_pages > 1:
            st.markdown("---")
            new_page = show_pagination_controls(current_page, total_pages, "docs")
            if new_page != current_page:
                st.session_state.docs_page = new_page
                st.rerun()

    except APIException as e:
        st.error(f"Failed to load documents: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")


def show_document_controls():
    """Show document search and filter controls."""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input(
            "🔍 Search documents",
            key="doc_search",
            placeholder="Search by filename, content, or metadata...",
        )

    with col2:
        status_filter = st.selectbox(
            "Status Filter",
            options=["All", "Completed", "Processing", "Failed"],
            key="status_filter",
        )

    with col3:
        sort_by = st.selectbox(
            "Sort By", options=["Upload Date", "Filename", "File Size"], key="sort_by"
        )


def show_document_card(document: Dict[str, Any]):
    """Show individual document card."""
    with st.container():
        st.markdown("---")

        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            # Document name and type
            filename = document.get("filename", "Unknown")
            doc_type = format_document_type(filename)
            st.write(f"**{doc_type} {filename}**")

            # Metadata
            upload_time = document.get("upload_timestamp")
            if upload_time:
                st.caption(f"Uploaded {format_time_ago(upload_time)}")

        with col2:
            # File size
            file_size = document.get("file_size", 0)
            st.write(f"📊 {format_file_size(file_size)}")

            # Chunk count
            chunk_count = document.get("chunk_count", 0)
            st.caption(f"{chunk_count} chunks")

        with col3:
            # Status
            status = document.get("status", "unknown")
            st.write(create_status_badge(status))

            # Processing progress if applicable
            if status == "processing":
                progress = document.get("processing_progress", 0)
                st.progress(progress / 100 if progress <= 100 else 0.5)

        with col4:
            # Action buttons
            doc_id = document.get("document_id")
            if doc_id:
                show_document_actions(doc_id, filename, status)


def show_document_actions(doc_id: str, filename: str, status: str):
    """Show action buttons for a document."""
    # View button
    if st.button("👁️ View", key=f"view_{doc_id}", help="View document details"):
        show_document_details(doc_id)

    # Delete button
    if st.button("🗑️ Delete", key=f"delete_{doc_id}", help="Delete document"):
        delete_document(doc_id, filename)

    # Show chunks button for completed documents
    if status == "completed":
        if st.button("🧩 Chunks", key=f"chunks_{doc_id}", help="View document chunks"):
            show_document_chunks(doc_id)


def show_document_details(doc_id: str):
    """Show detailed document information in a modal."""
    try:
        api_client = get_api_client()

        with st.spinner("Loading document details..."):
            document = api_client.get_document(doc_id)

        # Show in expander
        with st.expander(
            f"📄 Document Details: {document.get('filename', 'Unknown')}", expanded=True
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Basic Information:**")
                st.write(f"• **Filename:** {document.get('filename', 'N/A')}")
                st.write(
                    f"• **File Size:** {format_file_size(document.get('file_size', 0))}"
                )
                st.write(
                    f"• **Status:** {create_status_badge(document.get('status', 'unknown'))}"
                )
                st.write(
                    f"• **Upload Time:** {format_timestamp(document.get('upload_timestamp', 0))}"
                )

            with col2:
                st.write("**Processing Information:**")
                st.write(f"• **Chunks:** {document.get('chunk_count', 0)}")
                st.write(
                    f"• **Content Length:** {document.get('content_length', 0)} characters"
                )
                st.write(f"• **Page Count:** {document.get('page_count', 'N/A')}")

                if document.get("processing_error"):
                    st.error(f"**Error:** {document['processing_error']}")

            # Content preview
            if document.get("content_preview"):
                st.write("**Content Preview:**")
                preview = truncate_text(document["content_preview"], 500)
                st.text_area("Preview", preview, height=150, disabled=True)

    except APIException as e:
        st.error(f"Failed to load document details: {str(e)}")


def show_document_chunks(doc_id: str):
    """Show document chunks."""
    try:
        api_client = get_api_client()

        with st.spinner("Loading document chunks..."):
            response = api_client.get_document_chunks(doc_id)

        chunks = response.get("chunks", [])

        if not chunks:
            st.warning("No chunks found for this document.")
            return

        with st.expander(f"🧩 Document Chunks ({len(chunks)} total)", expanded=True):
            for i, chunk in enumerate(chunks, 1):
                st.write(f"**Chunk {i}**")

                # Chunk metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Page: {chunk.get('page_number', 'N/A')}")
                with col2:
                    st.caption(f"Tokens: {chunk.get('token_count', 'N/A')}")
                with col3:
                    st.caption(f"Length: {len(chunk.get('content', ''))} chars")

                # Chunk content
                content = chunk.get("content", "")
                st.text_area(
                    f"Content {i}", content, height=100, disabled=True, key=f"chunk_{i}"
                )

                if i < len(chunks):
                    st.divider()

    except APIException as e:
        st.error(f"Failed to load document chunks: {str(e)}")


def delete_document(doc_id: str, filename: str):
    """Delete a document with confirmation."""
    # Confirmation dialog
    confirm_key = f"confirm_delete_{doc_id}"

    if confirm_key not in st.session_state:
        st.session_state[confirm_key] = False

    if not st.session_state[confirm_key]:
        if st.button(f"⚠️ Confirm deletion of '{filename}'?", key=f"confirm_{doc_id}"):
            st.session_state[confirm_key] = True
            st.rerun()
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Yes, Delete", key=f"yes_{doc_id}"):
                try:
                    api_client = get_api_client()
                    api_client.delete_document(doc_id)
                    st.success(f"🗑️ Deleted '{filename}'")

                    # Reset confirmation and refresh
                    del st.session_state[confirm_key]
                    time.sleep(1)
                    st.rerun()

                except APIException as e:
                    st.error(f"Failed to delete document: {str(e)}")

        with col2:
            if st.button("❌ Cancel", key=f"cancel_{doc_id}"):
                del st.session_state[confirm_key]
                st.rerun()


def show_empty_state():
    """Show empty state when no documents are uploaded."""
    st.info("""
    📄 **No documents found**
    
    Get started by uploading your first document using the upload section above.
    
    **Supported formats:**
    • PDF files (research papers, reports, manuals)
    • Word documents (articles, documentation)
    • Text files (notes, code, simple documents)
    • CSV files (data tables, spreadsheets)
    """)


if __name__ == "__main__":
    show_documents_interface()
