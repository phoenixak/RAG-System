# RAG System Demo -- Instructions

## Getting Started

1. **Wait for models to load.** The first launch downloads and caches `google/flan-t5-small` and `all-MiniLM-L6-v2`. This may take a minute on the initial run.

2. **Upload documents.** Use the sidebar file uploader to add one or more documents (PDF, TXT, DOCX, or CSV). Click **Process Documents** to ingest them into the vector store. Alternatively, click **Load Sample Document** to use the bundled sample about AI and machine learning.

3. **Ask questions.** Type a question in the chat input at the bottom of the main area. The system will:
   - Search for the most relevant chunks in your uploaded documents.
   - Pass the top results as context to the language model.
   - Return a generated answer with source attribution.

4. **Review sources.** The right column shows the retrieved document chunks ranked by relevance score. Click **View Sources** in the chat to see which chunks informed each answer.

## Tips

- Upload multiple documents and ask comparative questions.
- Shorter, focused questions tend to produce better answers.
- The relevance score (0-100%) indicates how closely a chunk matches your query.
- Use **Clear All Documents** in the sidebar to reset and start fresh.

## Limitations

- `flan-t5-small` is a compact model. For complex reasoning, a larger model would perform better.
- Very long documents are split into chunks; some context may be lost at chunk boundaries.
- The vector store is in-memory and resets when the app restarts.
