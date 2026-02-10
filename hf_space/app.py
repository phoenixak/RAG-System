"""
RAG System Demo - Streamlit Application
A self-contained RAG (Retrieval-Augmented Generation) demo using
google/flan-t5-small for text generation and sentence-transformers
for semantic search with ChromaDB as the vector store.
"""

import io
import os
import uuid
import logging
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* General layout */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Source document cards */
    .source-card {
        background: var(--background-secondary, #f8f9fa);
        border: 1px solid var(--border-color, #dee2e6);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.9em;
    }

    .source-card .source-header {
        font-weight: 600;
        color: var(--text-color, #1a1a2e);
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .source-card .relevance-badge {
        background: #4361ee;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 500;
    }

    .source-card .source-text {
        color: var(--text-color-secondary, #495057);
        font-size: 0.88em;
        line-height: 1.5;
        margin-top: 6px;
    }

    /* Document list in sidebar */
    .doc-item {
        background: var(--background-secondary, #f0f2f6);
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.85em;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px 0 10px 0;
        font-size: 0.8em;
        color: var(--text-color-secondary, #6c757d);
        border-top: 1px solid var(--border-color, #dee2e6);
        margin-top: 40px;
    }

    /* Chat area spacing */
    .stChatMessage {
        margin-bottom: 8px;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .source-card {
            background: #1e1e2e;
            border-color: #383850;
        }
        .source-card .source-header {
            color: #e0e0e0;
        }
        .source-card .source-text {
            color: #b0b0c0;
        }
        .doc-item {
            background: #1e1e2e;
        }
    }
</style>
"""


# ---------------------------------------------------------------------------
# RAG System
# ---------------------------------------------------------------------------
class RAGSystem:
    """Core RAG pipeline: document processing, retrieval, and generation."""

    def __init__(self) -> None:
        """Initialize the RAG system (models loaded separately via cache)."""
        self.chunk_size = 500
        self.chunk_overlap = 50

    # ------------------------------------------------------------------
    # Model loading (cached by Streamlit)
    # ------------------------------------------------------------------
    @staticmethod
    @st.cache_resource
    def load_models() -> tuple:
        """Load flan-t5-small and sentence-transformers.

        Returns:
            Tuple of (text-generation pipeline, SentenceTransformer model).
        """
        from transformers import pipeline as hf_pipeline
        from sentence_transformers import SentenceTransformer

        with st.spinner("Loading language model (flan-t5-small)..."):
            llm = hf_pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_new_tokens=200,
                do_sample=False,
            )

        with st.spinner("Loading embedding model (all-MiniLM-L6-v2)..."):
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        return llm, embedder

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------
    @staticmethod
    def setup_vector_store():
        """Create an in-memory ChromaDB client and collection.

        Returns:
            Tuple of (chromadb.Client, Collection).
        """
        import chromadb

        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"},
        )
        return client, collection

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------
    def extract_text(self, uploaded_file) -> dict:
        """Extract text content from an uploaded file.

        Args:
            uploaded_file: Streamlit UploadedFile object.

        Returns:
            Dict with keys 'filename', 'text', and 'type'.

        Raises:
            ValueError: If the file type is unsupported or extraction fails.
        """
        filename = uploaded_file.name
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext == ".txt":
            text = self._extract_txt(uploaded_file)
        elif file_ext == ".pdf":
            text = self._extract_pdf(uploaded_file)
        elif file_ext == ".docx":
            text = self._extract_docx(uploaded_file)
        elif file_ext == ".csv":
            text = self._extract_csv(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        if not text or not text.strip():
            raise ValueError(f"No text content could be extracted from {filename}")

        return {
            "filename": filename,
            "text": text.strip(),
            "type": file_ext.lstrip("."),
        }

    @staticmethod
    def _extract_txt(uploaded_file) -> str:
        raw = uploaded_file.read()
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                return raw.decode(encoding)
            except (UnicodeDecodeError, AttributeError):
                continue
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _extract_pdf(uploaded_file) -> str:
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise ValueError("PyPDF2 is required for PDF processing") from exc

        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        return "\n\n".join(pages)

    @staticmethod
    def _extract_docx(uploaded_file) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise ValueError("python-docx is required for DOCX processing") from exc

        doc = Document(io.BytesIO(uploaded_file.read()))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    @staticmethod
    def _extract_csv(uploaded_file) -> str:
        df = pd.read_csv(uploaded_file)
        rows = []
        for _, row in df.iterrows():
            parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            rows.append(". ".join(parts))
        return "\n\n".join(rows)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> list[str]:
        """Split text into overlapping chunks, breaking at sentence boundaries.

        Args:
            text: The full document text.
            chunk_size: Maximum characters per chunk.
            overlap: Number of overlapping characters between chunks.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            if current_length + sentence_len > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Keep tail sentences for overlap
                overlap_chunk: list[str] = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) > overlap:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_len += len(s)

                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(sentence)
            current_length += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Naive sentence splitter on '.', '!', '?'."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------
    def add_document(self, doc_data: dict, embedder, collection) -> int:
        """Chunk, embed, and add a document to the vector store.

        Args:
            doc_data: Dict with 'filename' and 'text'.
            embedder: SentenceTransformer instance.
            collection: ChromaDB collection.

        Returns:
            Number of chunks added.
        """
        chunks = self.chunk_text(doc_data["text"], self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0

        embeddings = embedder.encode(chunks, show_progress_bar=False)
        ids = [f"{doc_data['filename']}_{uuid.uuid4().hex[:8]}" for _ in chunks]
        metadatas = [
            {
                "source": doc_data["filename"],
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        embedder,
        collection,
        n_results: int = 5,
    ) -> list[dict]:
        """Perform semantic search over the vector store.

        Args:
            query: User query string.
            embedder: SentenceTransformer instance.
            collection: ChromaDB collection.
            n_results: Maximum number of results to return.

        Returns:
            List of dicts with 'text', 'source', 'similarity', and 'chunk_index'.
        """
        if collection.count() == 0:
            return []

        query_embedding = embedder.encode([query], show_progress_bar=False)
        actual_n = min(n_results, collection.count())
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )

        formatted: list[dict] = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance is in [0, 2]; convert to similarity
                similarity = max(0.0, 1.0 - dist)
                formatted.append(
                    {
                        "text": doc,
                        "source": meta.get("source", "Unknown"),
                        "similarity": round(similarity, 4),
                        "chunk_index": meta.get("chunk_index", 0),
                    }
                )

        formatted.sort(key=lambda x: x["similarity"], reverse=True)
        return formatted

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------
    def generate_response(
        self,
        query: str,
        context_docs: list[dict],
        llm_pipeline,
    ) -> str:
        """Generate an answer with flan-t5-small using retrieved context.

        Args:
            query: User question.
            context_docs: Retrieved documents from search().
            llm_pipeline: HuggingFace text2text-generation pipeline.

        Returns:
            Generated answer string.
        """
        if not context_docs:
            return (
                "I don't have any documents to reference. "
                "Please upload documents first, then ask your question."
            )

        # Build context from top results
        context_parts: list[str] = []
        for doc in context_docs[:3]:
            text = doc["text"][:400]
            context_parts.append(text)
        context = "\n\n".join(context_parts)

        prompt = (
            "Answer the following question based on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )

        try:
            result = llm_pipeline(prompt, max_new_tokens=200)
            answer = result[0]["generated_text"].strip()

            if not answer or len(answer) < 3:
                return self._fallback_response(query, context_docs)

            return answer

        except Exception as exc:
            logger.error("Generation error: %s", exc)
            return self._fallback_response(query, context_docs)

    def _fallback_response(self, query: str, context_docs: list[dict]) -> str:
        """Provide relevant excerpts when the LLM response is inadequate.

        Args:
            query: User question.
            context_docs: Retrieved documents.

        Returns:
            Formatted fallback response with source excerpts.
        """
        if not context_docs:
            return "No relevant information found in the uploaded documents."

        response_parts = [
            "Here are the most relevant excerpts from your documents:\n"
        ]
        for i, doc in enumerate(context_docs[:3], 1):
            excerpt = doc["text"][:300].strip()
            source = doc["source"]
            score = doc["similarity"]
            response_parts.append(
                f"**Source {i}** ({source}, relevance: {score:.0%}):\n"
                f"> {excerpt}...\n"
            )

        return "\n".join(response_parts)


# ---------------------------------------------------------------------------
# Streamlit UI Helpers
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize all required Streamlit session state variables."""
    defaults = {
        "documents": [],
        "chat_history": [],
        "embedder": None,
        "llm_pipeline": None,
        "chroma_client": None,
        "collection": None,
        "rag_system": None,
        "models_loaded": False,
        "retrieved_docs": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_models_into_state() -> None:
    """Load models and vector store into session state if not already loaded."""
    if st.session_state.models_loaded:
        return

    rag = RAGSystem()
    st.session_state.rag_system = rag

    try:
        llm, embedder = RAGSystem.load_models()
        st.session_state.llm_pipeline = llm
        st.session_state.embedder = embedder

        client, collection = rag.setup_vector_store()
        st.session_state.chroma_client = client
        st.session_state.collection = collection
        st.session_state.models_loaded = True
        logger.info("All models and vector store loaded successfully")

    except Exception as exc:
        st.error(f"Failed to load models: {exc}")
        logger.error("Model loading failed: %s", exc)


def render_sidebar() -> None:
    """Render the sidebar with document upload and management."""
    with st.sidebar:
        st.header("Document Management")
        st.markdown("---")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "docx", "csv"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, CSV",
        )

        if uploaded_files and st.button("Process Documents", type="primary"):
            process_uploaded_files(uploaded_files)

        st.markdown("---")

        # Document list
        st.subheader("Loaded Documents")
        if st.session_state.documents:
            for doc in st.session_state.documents:
                st.markdown(
                    f'<div class="doc-item">'
                    f'<strong>{doc["filename"]}</strong><br>'
                    f'<small>{doc["chunks"]} chunks | {doc["type"].upper()}</small>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            collection = st.session_state.collection
            if collection is not None:
                st.caption(f"Total chunks in vector store: {collection.count()}")

            if st.button("Clear All Documents", type="secondary"):
                clear_all_documents()
                st.rerun()
        else:
            st.info("No documents loaded yet. Upload files above to get started.")

        st.markdown("---")

        # Sample document loader
        sample_path = os.path.join(
            os.path.dirname(__file__), "sample_documents", "sample.txt"
        )
        if os.path.exists(sample_path):
            if st.button("Load Sample Document"):
                load_sample_document(sample_path)
                st.rerun()

        # Info
        st.markdown("---")
        st.markdown(
            "**Models:**\n"
            "- LLM: `google/flan-t5-small`\n"
            "- Embeddings: `all-MiniLM-L6-v2`\n"
            "- Vector Store: ChromaDB"
        )


def process_uploaded_files(uploaded_files) -> None:
    """Process a list of uploaded files and add to the vector store."""
    rag = st.session_state.rag_system
    embedder = st.session_state.embedder
    collection = st.session_state.collection

    if rag is None or embedder is None or collection is None:
        st.error("Models not loaded. Please wait for initialization.")
        return

    progress = st.sidebar.progress(0)
    total = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                doc_data = rag.extract_text(uploaded_file)
                chunk_count = rag.add_document(doc_data, embedder, collection)

                st.session_state.documents.append(
                    {
                        "filename": doc_data["filename"],
                        "type": doc_data["type"],
                        "chunks": chunk_count,
                        "text_length": len(doc_data["text"]),
                    }
                )
                st.sidebar.success(
                    f"Added {uploaded_file.name} ({chunk_count} chunks)"
                )

        except Exception as exc:
            st.sidebar.error(f"Error processing {uploaded_file.name}: {exc}")
            logger.error("File processing error: %s", exc)

        progress.progress((idx + 1) / total)

    progress.empty()


def load_sample_document(sample_path: str) -> None:
    """Load the bundled sample document into the vector store."""
    rag = st.session_state.rag_system
    embedder = st.session_state.embedder
    collection = st.session_state.collection

    if rag is None or embedder is None or collection is None:
        st.error("Models not loaded yet.")
        return

    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc_data = {
            "filename": "sample.txt",
            "text": text,
            "type": "txt",
        }

        chunk_count = rag.add_document(doc_data, embedder, collection)
        st.session_state.documents.append(
            {
                "filename": "sample.txt",
                "type": "txt",
                "chunks": chunk_count,
                "text_length": len(text),
            }
        )
        st.sidebar.success(f"Loaded sample document ({chunk_count} chunks)")

    except Exception as exc:
        st.sidebar.error(f"Error loading sample: {exc}")
        logger.error("Sample loading error: %s", exc)


def clear_all_documents() -> None:
    """Clear all documents and reset the vector store."""
    st.session_state.documents = []
    st.session_state.chat_history = []
    st.session_state.retrieved_docs = []

    rag = st.session_state.rag_system
    if rag is not None:
        try:
            client, collection = rag.setup_vector_store()
            st.session_state.chroma_client = client
            st.session_state.collection = collection
        except Exception as exc:
            logger.error("Error resetting vector store: %s", exc)

    st.sidebar.success("All documents cleared.")


def render_retrieved_docs(docs: list[dict]) -> None:
    """Render retrieved source documents in the right column."""
    if not docs:
        st.info("Ask a question to see relevant source documents here.")
        return

    for i, doc in enumerate(docs[:5], 1):
        similarity_pct = f"{doc['similarity']:.0%}"
        excerpt = doc["text"][:250].strip()
        if len(doc["text"]) > 250:
            excerpt += "..."

        st.markdown(
            f'<div class="source-card">'
            f'<div class="source-header">'
            f"<span>Source {i}: {doc['source']}</span>"
            f'<span class="relevance-badge">{similarity_pct}</span>'
            f"</div>"
            f'<div class="source-text">{excerpt}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_chat_area() -> None:
    """Render the main chat interface."""
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        handle_user_query(user_query)


def handle_user_query(query: str) -> None:
    """Process a user query: search, generate, and display results."""
    rag = st.session_state.rag_system
    embedder = st.session_state.embedder
    collection = st.session_state.collection
    llm_pipeline = st.session_state.llm_pipeline

    if rag is None or embedder is None or llm_pipeline is None:
        st.error("Models are not loaded yet. Please wait.")
        return

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.chat_history.append({"role": "user", "content": query})

    # Retrieve and generate
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            # Search
            if collection is not None and collection.count() > 0:
                context_docs = rag.search(query, embedder, collection, n_results=5)
            else:
                context_docs = []

            st.session_state.retrieved_docs = context_docs

            # Generate
            response = rag.generate_response(query, context_docs, llm_pipeline)

            # Display response
            st.markdown(response)

            # Show source attribution inline
            if context_docs:
                with st.expander("View Sources", expanded=False):
                    for i, doc in enumerate(context_docs[:3], 1):
                        score = doc["similarity"]
                        source = doc["source"]
                        st.caption(f"Source {i}: {source} (relevance: {score:.0%})")

    st.session_state.chat_history.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the Streamlit RAG demo application."""
    st.set_page_config(
        page_title="RAG System Demo",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize
    init_session_state()
    load_models_into_state()

    # Header
    st.title("RAG System Demo")
    st.caption(
        "Upload documents and ask questions -- powered by open-source models on "
        "Hugging Face."
    )

    # Sidebar
    render_sidebar()

    # Main content: two columns (chat | retrieved docs)
    col_chat, col_docs = st.columns([2, 1])

    with col_chat:
        st.subheader("Chat")
        render_chat_area()

    with col_docs:
        st.subheader("Retrieved Documents")
        render_retrieved_docs(st.session_state.retrieved_docs)

    # Footer
    st.markdown(
        '<div class="footer">'
        "Powered by "
        "<strong>google/flan-t5-small</strong> | "
        "<strong>sentence-transformers/all-MiniLM-L6-v2</strong> | "
        "<strong>ChromaDB</strong> | "
        "<strong>Streamlit</strong>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
