"""
Search Interface Page
Advanced search functionality with filters and result visualization.
"""

import time
from typing import Any, Dict, List

import streamlit as st

from frontend.components.api_client import APIException, get_api_client
from frontend.components.auth import requires_auth
from frontend.components.utils import (
    create_metric_card,
    format_search_score,
    format_timestamp,
    highlight_search_terms,
    paginate_results,
    show_json_viewer,
    show_pagination_controls,
    truncate_text,
)
from frontend.config import SEARCH_CONFIG


@requires_auth
def show_search_interface():
    """Main search interface page."""
    st.title("🔍 Advanced Search")
    st.markdown("Explore your documents with powerful search capabilities.")

    # Search form
    search_query, search_params = show_search_form()

    # Perform search if query exists
    if search_query:
        show_search_results(search_query, search_params)
    else:
        show_search_help()


# Alias for compatibility with main.py import
def show_search_page():
    """Alias for show_search_interface."""
    show_search_interface()


def show_search_form():
    """Show the search form with advanced options."""
    st.subheader("🔎 Search Query")

    # Main search input
    search_query = st.text_input(
        "Enter your search query",
        placeholder="Search your documents...",
        key="main_search_query",
    )

    # Advanced options
    with st.expander("⚙️ Advanced Search Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Search Configuration**")

            search_type = st.selectbox(
                "Search Type",
                options=["semantic", "hybrid", "contextual"],
                index=0,
                key="adv_search_type",
                help="Type of search algorithm to use",
            )

            max_results = st.slider(
                "Maximum Results",
                min_value=1,
                max_value=50,
                value=SEARCH_CONFIG["max_results"],
                key="adv_max_results",
            )

            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=SEARCH_CONFIG["similarity_threshold"],
                step=0.05,
                key="adv_similarity_threshold",
                help="Minimum similarity score for results",
            )

        with col2:
            st.write("**Hybrid Search Weights**")

            semantic_weight = st.slider(
                "Semantic Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="adv_semantic_weight",
                disabled=search_type != "hybrid",
            )

            keyword_weight = st.slider(
                "Keyword Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                key="adv_keyword_weight",
                disabled=search_type != "hybrid",
            )

            # Document type filter
            st.write("**Filters**")
            doc_types = st.multiselect(
                "Document Types",
                options=["PDF", "DOCX", "TXT", "CSV"],
                default=[],
                key="doc_type_filter",
                help="Filter by document type",
            )

    # Prepare search parameters
    search_params = {
        "search_type": search_type,
        "max_results": max_results,
        "similarity_threshold": similarity_threshold,
        "semantic_weight": semantic_weight,
        "keyword_weight": keyword_weight,
        "doc_types": doc_types,
    }

    return search_query, search_params


def show_search_results(query: str, params: Dict[str, Any]):
    """Show search results."""
    st.markdown("---")
    st.subheader("📊 Search Results")

    try:
        # Perform search
        with st.spinner("Searching documents..."):
            results, search_metadata = perform_advanced_search(query, params)

        if not results:
            show_no_results(query)
            return

        # Search metadata
        show_search_metadata(search_metadata)

        # Results section
        show_results_list(results, query)

        # Export options
        show_export_options(results, query, search_metadata)

    except APIException as e:
        st.error(f"Search failed: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")


def perform_advanced_search(query: str, params: Dict[str, Any]) -> tuple:
    """Perform search with advanced parameters."""
    api_client = get_api_client()
    start_time = time.time()

    search_type = params["search_type"]
    max_results = params["max_results"]

    if search_type == "semantic":
        result = api_client.semantic_search(
            query=query,
            limit=max_results,
            similarity_threshold=params["similarity_threshold"],
        )
    elif search_type == "hybrid":
        result = api_client.hybrid_search(
            query=query,
            limit=max_results,
            semantic_weight=params["semantic_weight"],
            keyword_weight=params["keyword_weight"],
        )
    elif search_type == "contextual":
        # Use conversation ID if available
        from frontend.config import SESSION_KEYS

        conversation_id = st.session_state.get(SESSION_KEYS["conversation_id"])
        if not conversation_id:
            # Create new conversation for contextual search
            from frontend.components.chat_ui import create_new_conversation

            conversation_id = create_new_conversation()

        result = api_client.contextual_search(
            query=query, conversation_id=conversation_id, limit=max_results
        )
    else:
        raise ValueError(f"Unknown search type: {search_type}")

    response_time = time.time() - start_time

    search_results = result.get("results", [])

    # Apply document type filter if specified
    if params["doc_types"]:
        filtered_results = []
        for res in search_results:
            doc_name = res.get("document_filename", res.get("document_name", ""))
            doc_ext = doc_name.split(".")[-1].upper() if "." in doc_name else ""
            if doc_ext in params["doc_types"]:
                filtered_results.append(res)
        search_results = filtered_results

    search_metadata = {
        "search_type": search_type,
        "query": query,
        "response_time": response_time,
        "total_results": len(search_results),
        "parameters": params,
        "timestamp": time.time(),
    }

    return search_results, search_metadata


def show_search_metadata(metadata: Dict[str, Any]):
    """Display search metadata and statistics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        create_metric_card("Search Type", metadata["search_type"].title())

    with col2:
        create_metric_card("Results Found", metadata["total_results"])

    with col3:
        response_time = metadata["response_time"]
        create_metric_card("Response Time", f"{response_time:.2f}s")

    with col4:
        search_quality = (
            "Excellent"
            if response_time < 1.0
            else "Good"
            if response_time < 3.0
            else "Slow"
        )
        create_metric_card("Performance", search_quality)


def show_results_list(results: List[Dict[str, Any]], query: str):
    """Display the list of search results."""
    # Pagination
    current_page = st.session_state.get("search_page", 1)
    per_page = 5

    page_results, total_pages, total_items = paginate_results(
        results, current_page, per_page
    )

    if total_pages > 1:
        st.write(
            f"Showing results {(current_page - 1) * per_page + 1}-{min(current_page * per_page, total_items)} of {total_items}"
        )

    # Display results
    for i, result in enumerate(page_results, 1):
        show_search_result_card(result, query, (current_page - 1) * per_page + i)

    # Pagination controls
    if total_pages > 1:
        new_page = show_pagination_controls(current_page, total_pages, "search")
        if new_page != current_page:
            st.session_state.search_page = new_page
            st.rerun()


def show_search_result_card(result: Dict[str, Any], query: str, index: int):
    """Display a single search result card."""
    with st.container():
        st.markdown("---")

        # Header
        col1, col2 = st.columns([4, 1])

        with col1:
            doc_name = result.get(
                "document_filename", result.get("document_name", "Unknown Document")
            )
            st.write(f"**{index}. {doc_name}**")

            # Content preview with highlighting
            # Fix: Backend returns 'text' field, not 'content'
            content = result.get("text", result.get("content", ""))
            highlighted_content = highlight_search_terms(content, query)

            # Truncate content for display
            if len(highlighted_content) > 300:
                preview = highlighted_content[:300] + "..."
            else:
                preview = highlighted_content

            st.markdown(preview)

        with col2:
            # Relevance score
            score = result.get("score", 0)
            st.write(format_search_score(score))

        # Metadata row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            page_num = result.get("page_number")
            if page_num:
                st.caption(f"📄 Page {page_num}")
            else:
                st.caption("📄 No page info")

        with col2:
            chunk_index = result.get("chunk_index")
            if chunk_index is not None:
                st.caption(f"🧩 Chunk {chunk_index + 1}")

        with col3:
            content_length = len(content)
            st.caption(f"📏 {content_length} chars")

        with col4:
            # Action buttons
            if st.button("🔍 View Details", key=f"details_{index}"):
                st.session_state[f"show_details_{index}"] = not st.session_state.get(
                    f"show_details_{index}", False
                )
                st.rerun()

    # Show details outside the main container to avoid nesting expanders
    if st.session_state.get(f"show_details_{index}", False):
        show_result_details(result)


def show_result_details(result: Dict[str, Any]):
    """Show detailed information about a search result."""
    with st.expander(
        f"📄 Result Details: {result.get('document_filename', result.get('document_name', 'Unknown'))}",
        expanded=True,
    ):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Document Information:**")
            st.write(
                f"• **Document:** {result.get('document_filename', result.get('document_name', 'N/A'))}"
            )
            st.write(f"• **Score:** {format_search_score(result.get('score', 0))}")
            st.write(f"• **Page:** {result.get('page_number', 'N/A')}")
            st.write(f"• **Chunk:** {result.get('chunk_index', 'N/A')}")

        with col2:
            st.write("**Content Metrics:**")
            # Fix: Backend returns 'text' field, not 'content'
            content = result.get("text", result.get("content", ""))
            st.write(f"• **Length:** {len(content)} characters")
            st.write(f"• **Words:** {len(content.split())} words")
            st.write(f"• **Document ID:** {result.get('document_id', 'N/A')}")

        # Full content
        st.write("**Full Content:**")
        st.text_area(
            "Content",
            content,
            height=200,
            disabled=True,
            key=f"content_{hash(content)}",
        )

        # Show raw result data
        show_json_viewer(result, "Raw Result Data")


def show_no_results(query: str):
    """Show message when no results are found."""
    st.warning(f"No results found for: **{query}**")

    st.markdown("""
    **Suggestions to improve your search:**
    
    1. **Try different keywords** - Use synonyms or related terms
    2. **Broaden your search** - Use fewer or more general terms
    3. **Check spelling** - Ensure your query is spelled correctly
    4. **Adjust search type** - Try hybrid or semantic search
    5. **Lower similarity threshold** - Reduce the minimum similarity score
    6. **Upload more documents** - Expand your document collection
    """)


def show_export_options(results: List[Dict], query: str, metadata: Dict):
    """Show options to export search results."""
    st.markdown("---")
    st.subheader("📥 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Export as CSV"):
            export_results_csv(results, query, metadata)

    with col2:
        if st.button("📋 Export as JSON"):
            export_results_json(results, query, metadata)

    with col3:
        if st.button("📝 Export Summary"):
            export_results_summary(results, query, metadata)


def export_results_csv(results: List[Dict], query: str, metadata: Dict):
    """Export results as CSV."""
    import io

    import pandas as pd

    # Prepare data for CSV
    csv_data = []
    for i, result in enumerate(results, 1):
        csv_data.append(
            {
                "Rank": i,
                "Document": result.get(
                    "document_filename", result.get("document_name", "")
                ),
                "Score": result.get("score", 0),
                "Page": result.get("page_number", ""),
                # Fix: Backend returns 'text' field, not 'content'
                "Content": result.get("text", result.get("content", "")),
                "Content_Length": len(result.get("text", result.get("content", ""))),
            }
        )

    df = pd.DataFrame(csv_data)

    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # Download button
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"search_results_{int(time.time())}.csv",
        mime="text/csv",
    )


def export_results_json(results: List[Dict], query: str, metadata: Dict):
    """Export results as JSON."""
    import json

    export_data = {
        "search_metadata": metadata,
        "results": results,
        "export_timestamp": time.time(),
    }

    json_str = json.dumps(export_data, indent=2, default=str)

    st.download_button(
        label="📥 Download JSON",
        data=json_str,
        file_name=f"search_results_{int(time.time())}.json",
        mime="application/json",
    )


def export_results_summary(results: List[Dict], query: str, metadata: Dict):
    """Export results summary as text."""
    summary = f"""Search Results Summary
Query: {query}
Search Type: {metadata["search_type"]}
Results Found: {metadata["total_results"]}
Response Time: {metadata["response_time"]:.2f}s
Generated: {format_timestamp(metadata["timestamp"])}

Results:
"""

    for i, result in enumerate(results, 1):
        summary += f"\n{i}. {result.get('document_filename', result.get('document_name', 'Unknown'))} (Score: {result.get('score', 0):.3f})\n"
        content = result.get("content", "")
        summary += f"   {truncate_text(content, 150)}\n"

    st.download_button(
        label="📥 Download Summary",
        data=summary,
        file_name=f"search_summary_{int(time.time())}.txt",
        mime="text/plain",
    )


def show_search_help():
    """Show search help and examples."""
    st.markdown("---")
    st.subheader("💡 Search Help")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Search Types
        
        **Semantic Search**
        - Finds documents by meaning, not just keywords
        - Good for: Conceptual questions, finding related topics
        - Example: "machine learning benefits" finds docs about AI advantages
        
        **Hybrid Search**
        - Combines semantic + keyword search
        - Good for: Balanced results with exact matches
        - Best of both semantic understanding and keyword precision
        
        **Contextual Search**  
        - Uses conversation history for context
        - Good for: Follow-up questions, clarifications
        - Understands "that", "it", "them" references
        """)

    with col2:
        st.markdown("""
        ### 📝 Search Tips
        
        **Query Formulation**
        - Use natural language questions
        - Be specific but not overly narrow
        - Try multiple phrasings if needed
        
        **Advanced Options**
        - Adjust similarity threshold for more/fewer results
        - Use document type filters to narrow search
        - Experiment with hybrid search weights
        
        **Performance**
        - Semantic search: Slower but more intelligent
        - Hybrid search: Balanced speed and accuracy
        - More results = longer processing time
        """)

    # Example queries
    st.markdown("### 🔍 Example Queries")

    examples = [
        "What are the main conclusions?",
        "Find information about data processing",
        "Compare different methodologies",
        "Show examples of implementation",
        "What challenges are mentioned?",
        "Summarize the key findings",
    ]

    st.write("Try these example queries:")
    for example in examples:
        if st.button(f"🔍 {example}", key=f"help_example_{hash(example)}"):
            st.session_state.main_search_query = example
            st.rerun()


if __name__ == "__main__":
    show_search_interface()
