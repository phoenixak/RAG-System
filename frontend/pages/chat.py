"""
Chat Page Interface
Interactive conversational search interface.
"""

import streamlit as st

from frontend.components.auth import requires_auth
from frontend.components.chat_ui import (
    chat_input_handler,
    display_chat_history,
    init_chat_state,
    show_chat_controls,
    show_conversation_stats,
)
from frontend.components.utils import create_info_box


@requires_auth
def show_chat_interface():
    """Main chat interface page."""
    st.title("💬 Conversational Search")
    st.markdown("Ask questions about your documents and get intelligent responses.")

    # Initialize chat state
    init_chat_state()

    # Chat controls and settings
    show_chat_header()

    # Main chat area
    chat_container = st.container()

    with chat_container:
        # Display conversation history
        display_chat_history()

        # Chat input handler
        chat_input_handler()

    # Show conversation stats and controls
    st.markdown("---")
    show_chat_footer()


def show_chat_header():
    """Show chat header with controls and settings."""
    col1, col2 = st.columns([3, 1])

    with col1:
        # Search settings in expandable section
        with st.expander("🔧 Search Settings", expanded=False):
            show_search_settings()

    with col2:
        # Chat controls
        show_chat_controls()


def show_search_settings():
    """Show search configuration options."""
    st.subheader("Search Configuration")

    # Search type selection
    search_type = st.selectbox(
        "Search Type",
        options=["semantic", "hybrid", "contextual"],
        index=0,
        key="search_type",
        help="""
        - **Semantic**: Vector-based similarity search
        - **Hybrid**: Combines semantic and keyword search  
        - **Contextual**: Uses conversation history for context
        """,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Number of results
        search_limit = st.slider(
            "Max Results",
            min_value=1,
            max_value=20,
            value=10,
            key="search_limit",
            help="Maximum number of document chunks to retrieve",
        )

        # Similarity threshold (for semantic search)
        if search_type in ["semantic", "contextual"]:
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="similarity_threshold",
                help="Minimum similarity score for results",
            )

    with col2:
        # Hybrid search weights
        if search_type == "hybrid":
            semantic_weight = st.slider(
                "Semantic Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.30,
                step=0.1,
                key="semantic_weight",
                help="Weight for semantic search results",
            )

            keyword_weight = st.slider(
                "Keyword Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                key="keyword_weight",
                help="Weight for keyword search results",
            )

            # Ensure weights sum to 1.0
            if semantic_weight + keyword_weight != 1.0:
                st.warning(
                    f"Weights sum to {semantic_weight + keyword_weight:.1f}. Consider adjusting to sum to 1.0."
                )


def show_chat_footer():
    """Show chat footer with stats and additional controls."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Conversation Stats")
        show_conversation_stats()

    with col2:
        st.subheader("💡 Tips")
        show_chat_tips()

    with col3:
        st.subheader("🔗 Quick Actions")
        show_quick_actions()


def show_chat_tips():
    """Show helpful tips for using the chat interface."""
    tips = [
        "Ask specific questions about your documents",
        "Use contextual search for follow-up questions",
        "Try different search types for better results",
        "Check the retrieved documents for source information",
        "Export your conversation for later reference",
    ]

    for tip in tips:
        st.write(f"• {tip}")


def show_quick_actions():
    """Show quick action buttons."""
    if st.button("📄 View Documents", use_container_width=True):
        st.session_state.current_page = "documents"
        st.rerun()

    if st.button("🔍 Advanced Search", use_container_width=True):
        st.session_state.current_page = "search"
        st.rerun()

    if st.button("⚙️ Settings", use_container_width=True):
        st.session_state.current_page = "settings"
        st.rerun()


def show_welcome_message():
    """Show welcome message for new users."""
    from frontend.config import SESSION_KEYS

    chat_history = st.session_state.get(SESSION_KEYS["chat_history"], [])

    if not chat_history:
        create_info_box(
            "Welcome to Conversational Search!",
            """
            Get started by asking questions about your documents:
            
            • **"What is this document about?"** - Get document summaries
            • **"Find information about [topic]"** - Search for specific topics  
            • **"Compare [A] and [B]"** - Find similarities and differences
            • **"Show me examples of [concept]"** - Find specific examples
            
            The AI will search your documents and provide relevant answers with source references.
            """,
            "info",
        )


def show_example_queries():
    """Show example queries users can try."""
    st.subheader("🎯 Example Queries")

    examples = [
        {
            "category": "Document Analysis",
            "queries": [
                "What are the main topics covered in my documents?",
                "Summarize the key findings from the research papers",
                "What are the conclusions in the uploaded reports?",
            ],
        },
        {
            "category": "Information Retrieval",
            "queries": [
                "Find all mentions of [specific term]",
                "What does the document say about [topic]?",
                "Show me examples of [concept] from the documents",
            ],
        },
        {
            "category": "Comparative Analysis",
            "queries": [
                "Compare the approaches mentioned in different documents",
                "What are the differences between [A] and [B]?",
                "Find conflicting information across documents",
            ],
        },
    ]

    for example in examples:
        with st.expander(f"📂 {example['category']}", expanded=False):
            for query in example["queries"]:
                if st.button(f"Try: {query}", key=f"example_{hash(query)}"):
                    # Add query to chat
                    from frontend.components.chat_ui import add_message_to_chat

                    add_message_to_chat("user", query)
                    st.rerun()


# Additional helper function for the chat interface
def show_search_type_explanation():
    """Show explanation of different search types."""
    with st.expander("❓ Search Types Explained", expanded=False):
        st.markdown("""
        ### 🎯 Semantic Search
        Uses AI embeddings to find documents based on meaning and context.
        - **Best for**: Conceptual questions, finding related topics
        - **Example**: "What are the benefits?" finds docs mentioning advantages, positives, etc.
        
        ### 🔄 Hybrid Search  
        Combines semantic search with keyword matching for comprehensive results.
        - **Best for**: Balanced search combining meaning and exact terms
        - **Example**: Gets both conceptual matches and exact keyword matches
        
        ### 💭 Contextual Search
        Uses conversation history to understand follow-up questions.
        - **Best for**: Follow-up questions, clarifications, deep-dive discussions
        - **Example**: "Tell me more about that" understands what "that" refers to
        """)


if __name__ == "__main__":
    show_chat_interface()
