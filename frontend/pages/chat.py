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
from frontend.config import SEARCH_CONFIG


@requires_auth
def show_chat_interface():
    """Main chat interface page."""
    # Page header
    st.markdown(
        "<div style='font-family:var(--font-mono,monospace); font-size:0.72rem; "
        "color:var(--text-muted,#475569); text-transform:uppercase; letter-spacing:0.1em; "
        "margin-bottom:0.2rem;'>Conversational Search</div>",
        unsafe_allow_html=True,
    )
    st.title("Chat with your Documents")
    st.markdown(
        "Ask questions in natural language — the system retrieves relevant document passages and generates a contextual answer."
    )

    # Initialize chat state
    init_chat_state()

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Settings + Controls row                                              #
    # ------------------------------------------------------------------ #
    with st.expander("Search Settings", expanded=False):
        show_search_settings()

    show_chat_controls()

    # ------------------------------------------------------------------ #
    # Chat area                                                           #
    # ------------------------------------------------------------------ #
    display_chat_history()

    # Chat input is always rendered at the bottom of the main area
    chat_input_handler()

    # ------------------------------------------------------------------ #
    # Footer stats                                                        #
    # ------------------------------------------------------------------ #
    st.markdown("---")
    show_chat_footer()


def show_search_settings():
    """Show search configuration options."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<div style='font-family:var(--font-mono,monospace); font-size:0.75rem; "
            "text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted,#475569); "
            "margin-bottom:0.5rem;'>Search Mode</div>",
            unsafe_allow_html=True,
        )
        # This is the ONE selectbox for search type — key: chat_search_type
        st.selectbox(
            "Search Type",
            options=["semantic", "hybrid", "contextual"],
            index=0,
            key="chat_search_type",
            help=(
                "**Semantic** — vector similarity search\n"
                "**Hybrid** — semantic + keyword combination\n"
                "**Contextual** — uses conversation history for follow-ups"
            ),
            label_visibility="collapsed",
        )

        st.slider(
            "Max Results",
            min_value=1,
            max_value=20,
            value=10,
            key="search_limit",
            help="Maximum number of document chunks to retrieve",
        )

    with col2:
        search_type = st.session_state.get("chat_search_type", "semantic")

        if search_type in ["semantic", "contextual"]:
            st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="similarity_threshold",
                help="Minimum similarity score — lower values return more (but less precise) results",
            )

        elif search_type == "hybrid":
            sem_w = st.slider(
                "Semantic Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="semantic_weight",
            )
            kw_w = st.slider(
                "Keyword Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                key="keyword_weight",
            )
            if abs(sem_w + kw_w - 1.0) > 0.05:
                st.warning(
                    f"Weights sum to {sem_w + kw_w:.1f} — ideally they should sum to 1.0"
                )


def show_chat_footer():
    """Show chat footer with stats, tips, and quick actions."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "<div style='font-family:var(--font-mono,monospace); font-size:0.75rem; "
            "text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted,#475569); "
            "margin-bottom:0.6rem;'>Session Stats</div>",
            unsafe_allow_html=True,
        )
        show_conversation_stats()

    with col2:
        st.markdown(
            "<div style='font-family:var(--font-mono,monospace); font-size:0.75rem; "
            "text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted,#475569); "
            "margin-bottom:0.6rem;'>Tips</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <ul style="font-size:0.83rem; color:var(--text-secondary,#94a3b8);
                       line-height:1.8; padding-left:1.1rem; margin:0;">
                <li>Ask specific questions for better results</li>
                <li>Use <em>Contextual</em> mode for follow-up questions</li>
                <li>Lower similarity threshold to broaden results</li>
                <li>Expand sources to verify retrieved passages</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            "<div style='font-family:var(--font-mono,monospace); font-size:0.75rem; "
            "text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted,#475569); "
            "margin-bottom:0.6rem;'>Quick Actions</div>",
            unsafe_allow_html=True,
        )

        if st.button("View Documents", use_container_width=True, key="qa_docs"):
            st.session_state.current_page = "documents"
            st.rerun()

        if st.button("Advanced Search", use_container_width=True, key="qa_search"):
            st.session_state.current_page = "search"
            st.rerun()

        if st.button("Settings", use_container_width=True, key="qa_settings"):
            st.session_state.current_page = "settings"
            st.rerun()


if __name__ == "__main__":
    show_chat_interface()
