"""
Chat UI Components
Interactive chat interface components for conversational search.
"""

import time
import uuid
from typing import Dict, List, Optional

import streamlit as st

from frontend.components.api_client import APIException, get_api_client
from frontend.config import SEARCH_CONFIG, SESSION_KEYS


def init_chat_state():
    """Initialize chat-related session state."""
    if SESSION_KEYS["conversation_id"] not in st.session_state:
        st.session_state[SESSION_KEYS["conversation_id"]] = None

    if SESSION_KEYS["chat_history"] not in st.session_state:
        st.session_state[SESSION_KEYS["chat_history"]] = []


def create_new_conversation():
    """Create a new conversation session."""
    try:
        api_client = get_api_client()
        conversation = api_client.create_conversation()

        # Handle different possible response structures
        if isinstance(conversation, dict):
            conversation_id = (
                conversation.get("conversation_id")
                or conversation.get("session_id")
                or conversation.get("id")
            )
        else:
            conversation_id = None

        if not conversation_id:
            raise APIException("Invalid conversation response structure")

        st.session_state[SESSION_KEYS["conversation_id"]] = conversation_id
        st.session_state[SESSION_KEYS["chat_history"]] = []

        return conversation_id

    except (APIException, KeyError, TypeError) as e:
        # Fall back to local conversation ID
        conversation_id = str(uuid.uuid4())
        st.session_state[SESSION_KEYS["conversation_id"]] = conversation_id
        st.session_state[SESSION_KEYS["chat_history"]] = []

        st.warning(
            f"Failed to create server conversation, using local session: {str(e)}"
        )
        return conversation_id


def get_conversation_id() -> str:
    """Get or create conversation ID."""
    if not st.session_state.get(SESSION_KEYS["conversation_id"]):
        return create_new_conversation()
    return st.session_state[SESSION_KEYS["conversation_id"]]


def add_message_to_chat(role: str, content: str, metadata: Optional[Dict] = None):
    """Add a message to chat history."""
    if SESSION_KEYS["chat_history"] not in st.session_state:
        st.session_state[SESSION_KEYS["chat_history"]] = []

    message = {
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "metadata": metadata or {},
    }

    st.session_state[SESSION_KEYS["chat_history"]].append(message)


def display_chat_history():
    """Display the chat conversation history."""
    chat_history = st.session_state.get(SESSION_KEYS["chat_history"], [])

    if not chat_history:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 3rem 1rem;
                color: var(--text-muted, #475569);
                font-family: var(--font-mono, monospace);
                font-size: 0.85rem;
                letter-spacing: 0.04em;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.8rem; opacity: 0.4;">&#x2B21;</div>
                No messages yet &mdash; ask a question to get started
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Display messages
    for message in chat_history:
        role = message["role"]
        content = message["content"]
        metadata = message.get("metadata", {})

        if role == "user":
            with st.chat_message("user"):
                st.write(content)

        elif role == "assistant":
            with st.chat_message("assistant"):
                st.write(content)

                # Show search results if available
                if "search_results" in metadata and metadata["search_results"]:
                    show_search_results_in_chat(metadata["search_results"])

                # Show search metadata
                if "search_metadata" in metadata and metadata["search_metadata"]:
                    show_search_metadata(metadata["search_metadata"])


def show_search_results_in_chat(search_results: List[Dict]):
    """Display search results within chat message."""
    if not search_results:
        return

    with st.expander(
        f"Retrieved Documents ({len(search_results)} sources)", expanded=False
    ):
        for i, result in enumerate(search_results, 1):
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    doc_name = result.get(
                        "document_filename",
                        result.get("document_name", "Unknown Document"),
                    )
                    st.markdown(f"**{i}. {doc_name}**")
                with col2:
                    score = result.get("score", 0)
                    st.caption(f"{score:.3f}")

                content = result.get("text", result.get("content", ""))
                preview = content[:200] + "..." if len(content) > 200 else content
                st.markdown(
                    f"<div style='font-size:0.84rem; color:var(--text-secondary,#94a3b8); "
                    f"line-height:1.5; margin-top:0.25rem;'>{preview}</div>",
                    unsafe_allow_html=True,
                )

                if result.get("page_number"):
                    st.caption(f"Page {result['page_number']}")

                if i < len(search_results):
                    st.divider()


def show_search_metadata(metadata: Dict):
    """Display search metadata information."""
    with st.expander("Search Details", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            search_type = metadata.get("search_type", "N/A")
            st.metric("Search Type", search_type.title())

        with col2:
            response_time = metadata.get("response_time", 0)
            st.metric("Response Time", f"{response_time:.2f}s")

        with col3:
            total_results = metadata.get("total_results", 0)
            st.metric("Results Found", total_results)


def chat_input_handler():
    """
    Handle chat input and process user queries.

    Adds messages to history then reruns so display_chat_history() renders
    everything — avoids double-rendering messages on the same frame.
    """
    user_input = st.chat_input("Ask a question about your documents...")

    if not user_input:
        return

    # Add user message to history first
    add_message_to_chat("user", user_input)

    # Show a quick spinner while processing (before rerun)
    with st.spinner("Searching documents..."):
        try:
            conversation_id = get_conversation_id()
            search_type = st.session_state.get("chat_search_type", "semantic")
            search_results, search_metadata = perform_search(
                user_input, search_type, conversation_id
            )
            response = generate_chat_response(user_input, search_results)

            add_message_to_chat(
                "assistant",
                response,
                {
                    "search_results": search_results,
                    "search_metadata": search_metadata,
                },
            )

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            add_message_to_chat("assistant", error_msg)

    # Rerun to render everything cleanly via display_chat_history()
    st.rerun()


def perform_search(query: str, search_type: str, conversation_id: str) -> tuple:
    """Perform search based on selected type."""
    api_client = get_api_client()
    start_time = time.time()

    try:
        limit = st.session_state.get("search_limit", SEARCH_CONFIG["max_results"])

        if search_type == "semantic":
            threshold = st.session_state.get(
                "similarity_threshold", SEARCH_CONFIG["similarity_threshold"]
            )
            result = api_client.semantic_search(
                query, limit=limit, similarity_threshold=threshold
            )

        elif search_type == "hybrid":
            semantic_weight = st.session_state.get("semantic_weight", 0.7)
            keyword_weight = st.session_state.get("keyword_weight", 0.3)
            result = api_client.hybrid_search(
                query,
                limit=limit,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )

        elif search_type == "contextual":
            result = api_client.contextual_search(query, conversation_id, limit=limit)

        else:
            result = api_client.semantic_search(query, limit=limit)

        response_time = time.time() - start_time
        search_results = result.get("results", [])
        search_metadata = {
            "search_type": search_type,
            "response_time": response_time,
            "total_results": len(search_results),
            "query": query,
        }

        return search_results, search_metadata

    except APIException as e:
        st.error(f"Search failed: {str(e)}")
        return [], {"search_type": search_type, "error": str(e)}


def generate_chat_response(
    query: str,
    search_results: List[Dict],
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    """Generate a chat response using LLM service or fallback to basic response."""
    try:
        from frontend.components.api_client import get_api_client

        api_client = get_api_client()
        llm_response = api_client.generate_llm_response(
            query=query,
            search_results=search_results,
            conversation_history=conversation_history or [],
        )
        return llm_response

    except Exception as e:
        st.warning(f"LLM service unavailable, using basic response: {str(e)}")
        return _generate_fallback_response(query, search_results)


def _generate_fallback_response(query: str, search_results: List[Dict]) -> str:
    """Generate a fallback response when LLM service is unavailable."""
    if not search_results:
        return """I couldn't find any relevant documents for your query. This could mean:

1. **No documents uploaded**: Please upload some documents first
2. **Query too specific**: Try using broader terms
3. **Different terminology**: Try rephrasing your question

You can upload documents using the Documents page and try searching again."""

    num_results = len(search_results)
    response = (
        f"I found {num_results} relevant document{'s' if num_results != 1 else ''} "
        "that match your query. Here's what I found:\n\n"
    )

    for i, result in enumerate(search_results[:3], 1):
        doc_name = result.get(
            "document_filename", result.get("document_name", "Unknown Document")
        )
        content = result.get("text", result.get("content", ""))
        score = result.get("score", 0)
        summary = content[:150] + "..." if len(content) > 150 else content

        response += f"**{i}. {doc_name}** (relevance: {score:.1%})\n"
        response += f"{summary}\n\n"

    if num_results > 3:
        response += f"*See retrieved documents below for {num_results - 3} additional source(s).*\n\n"

    response += "Expand **Retrieved Documents** below to see full source content."
    return response


def show_chat_controls():
    """
    Display chat control buttons in a horizontal row.
    Search type is NOT included here — it lives in the Settings expander
    to avoid duplicate widget key conflicts.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "Clear Chat",
            key="ctrl_clear_chat",
            help="Clear conversation history",
            use_container_width=True,
        ):
            clear_chat_history()

    with col2:
        if st.button(
            "New Session",
            key="ctrl_new_session",
            help="Start a new conversation session",
            use_container_width=True,
        ):
            create_new_conversation()
            st.rerun()

    with col3:
        if st.button(
            "Export Chat",
            key="ctrl_export_chat",
            help="Export conversation as JSON",
            use_container_width=True,
        ):
            export_chat_history()


def clear_chat_history():
    """Clear the chat history."""
    st.session_state[SESSION_KEYS["chat_history"]] = []

    conversation_id = st.session_state.get(SESSION_KEYS["conversation_id"])
    if conversation_id:
        try:
            api_client = get_api_client()
            api_client.clear_conversation(conversation_id)
        except Exception:
            pass

    st.rerun()


def export_chat_history():
    """Export chat history as JSON download button."""
    import json

    chat_history = st.session_state.get(SESSION_KEYS["chat_history"], [])

    if not chat_history:
        st.warning("No chat history to export.")
        return

    export_data = {
        "conversation_id": st.session_state.get(SESSION_KEYS["conversation_id"]),
        "export_timestamp": time.time(),
        "message_count": len(chat_history),
        "messages": chat_history,
    }

    json_str = json.dumps(export_data, indent=2, default=str)

    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name=f"chat_history_{int(time.time())}.json",
        mime="application/json",
        key="download_chat_json",
    )


def show_conversation_stats():
    """Display conversation statistics."""
    chat_history = st.session_state.get(SESSION_KEYS["chat_history"], [])
    conversation_id = st.session_state.get(SESSION_KEYS["conversation_id"])

    if chat_history:
        col1, col2, col3 = st.columns(3)

        with col1:
            user_messages = len([msg for msg in chat_history if msg["role"] == "user"])
            st.metric("User Messages", user_messages)

        with col2:
            assistant_messages = len(
                [msg for msg in chat_history if msg["role"] == "assistant"]
            )
            st.metric("AI Responses", assistant_messages)

        with col3:
            if conversation_id:
                st.metric("Session ID", conversation_id[:8] + "...")
