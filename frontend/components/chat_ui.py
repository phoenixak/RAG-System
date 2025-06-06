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
        st.info("💬 Start a conversation by asking a question about your documents!")
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
                if "search_results" in metadata:
                    show_search_results_in_chat(metadata["search_results"])

                # Show search metadata
                if "search_metadata" in metadata:
                    show_search_metadata(metadata["search_metadata"])


def show_search_results_in_chat(search_results: List[Dict]):
    """Display search results within chat message."""
    if not search_results:
        return

    with st.expander(
        f"📄 Retrieved Documents ({len(search_results)} results)", expanded=False
    ):
        for i, result in enumerate(search_results, 1):
            with st.container():
                # Document header
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Fix: Backend returns multiple possible field names for document
                    doc_name = result.get(
                        "document_filename",
                        result.get("document_name", "Unknown Document"),
                    )
                    st.write(f"**{i}. {doc_name}**")
                with col2:
                    score = result.get("score", 0)
                    st.write(f"⭐ {score:.3f}")

                # Content preview
                # Fix: Backend returns 'text' field, not 'content'
                content = result.get("text", result.get("content", ""))
                if len(content) > 200:
                    st.write(f"{content[:200]}...")
                else:
                    st.write(content)

                # Metadata
                if result.get("page_number"):
                    st.caption(f"Page {result['page_number']}")

                st.divider()


def show_search_metadata(metadata: Dict):
    """Display search metadata information."""
    with st.expander("🔍 Search Details", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            search_type = metadata.get("search_type", "N/A")
            st.metric("Search Type", search_type)

        with col2:
            response_time = metadata.get("response_time", 0)
            st.metric("Response Time", f"{response_time:.2f}s")

        with col3:
            total_results = metadata.get("total_results", 0)
            st.metric("Results Found", total_results)


def chat_input_handler():
    """Handle chat input and process user queries."""
    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Add user message to chat
        add_message_to_chat("user", user_input)

        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)

        # Process query and get response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            try:
                with st.spinner("Searching documents..."):
                    # Get conversation ID
                    conversation_id = get_conversation_id()

                    # Perform search based on selected type
                    search_type = st.session_state.get("chat_search_type", "semantic")
                    search_results, search_metadata = perform_search(
                        user_input, search_type, conversation_id
                    )

                    # Generate response based on search results
                    response = generate_chat_response(user_input, search_results)

                    # Display response
                    response_placeholder.write(response)

                    # Show search results
                    if search_results:
                        show_search_results_in_chat(search_results)
                        show_search_metadata(search_metadata)

                    # Add assistant message to chat
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
                response_placeholder.error(error_msg)
                add_message_to_chat("assistant", error_msg)


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
            # Default to semantic
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
        # Try to use LLM service for intelligent response generation
        from frontend.components.api_client import get_api_client

        api_client = get_api_client()

        # Call the LLM service through the API
        llm_response = api_client.generate_llm_response(
            query=query,
            search_results=search_results,
            conversation_history=conversation_history or [],
        )

        return llm_response

    except Exception as e:
        # Fallback to the original simple response generation
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

    # Create a response based on search results
    num_results = len(search_results)

    if num_results == 1:
        response = "I found 1 relevant document that matches your query. "
    else:
        response = f"I found {num_results} relevant documents that match your query. "

    # Add summary of top results
    top_results = search_results[:3]  # Show top 3
    response += "Here's what I found:\n\n"

    for i, result in enumerate(top_results, 1):
        # Fix: Backend returns multiple possible field names
        doc_name = result.get(
            "document_filename", result.get("document_name", "Unknown Document")
        )
        content = result.get("text", result.get("content", ""))
        score = result.get("score", 0)

        # Truncate content for summary
        summary = content[:150] + "..." if len(content) > 150 else content

        response += f"**{i}. {doc_name}** (relevance: {score:.1%})\n"
        response += f"{summary}\n\n"

    if num_results > 3:
        response += f"*See the expanded results below for {num_results - 3} additional documents.*\n\n"

    response += "💡 **Tip**: You can click on the 'Retrieved Documents' section below to see the full content and details."

    return response


def show_chat_controls():
    """Display chat control buttons."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
            clear_chat_history()

    with col2:
        if st.button("🔄 New Session", help="Start a new conversation"):
            create_new_conversation()
            st.rerun()

    with col3:
        if st.button("📥 Export Chat", help="Export conversation"):
            export_chat_history()

    with col4:
        # Search type selector
        search_type = st.selectbox(
            "Search Type",
            options=["semantic", "hybrid", "contextual"],
            key="chat_search_type",
            help="Choose the search method",
        )


def clear_chat_history():
    """Clear the chat history."""
    st.session_state[SESSION_KEYS["chat_history"]] = []

    # Clear conversation on server if available
    conversation_id = st.session_state.get(SESSION_KEYS["conversation_id"])
    if conversation_id:
        try:
            api_client = get_api_client()
            api_client.clear_conversation(conversation_id)
        except Exception:
            pass  # Ignore server errors for clearing

    st.success("Chat history cleared!")
    st.rerun()


def export_chat_history():
    """Export chat history as JSON."""
    chat_history = st.session_state.get(SESSION_KEYS["chat_history"], [])

    if not chat_history:
        st.warning("No chat history to export.")
        return

    import json

    # Prepare export data
    export_data = {
        "conversation_id": st.session_state.get(SESSION_KEYS["conversation_id"]),
        "export_timestamp": time.time(),
        "message_count": len(chat_history),
        "messages": chat_history,
    }

    # Convert to JSON
    json_str = json.dumps(export_data, indent=2, default=str)

    # Download button
    st.download_button(
        label="📥 Download Chat History",
        data=json_str,
        file_name=f"chat_history_{int(time.time())}.json",
        mime="application/json",
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
