"""
RAG Prompt Templates
Prompt templates for generating contextual responses using retrieved documents.
"""

from typing import Dict, List

from .models import RAGContext


class RAGPromptTemplates:
    """Collection of prompt templates for RAG responses."""

    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for RAG responses."""
        return """You are an intelligent document assistant. Your role is to provide accurate, helpful responses based on the provided document context.

Instructions:
1. Use ONLY the information provided in the context documents to answer questions
2. If the context doesn't contain relevant information, clearly state this limitation
3. Cite specific documents when referencing information
4. Provide clear, concise, and well-structured responses
5. If asked about something not in the documents, explain what you cannot answer and suggest alternatives
6. Maintain a professional and helpful tone

Response Guidelines:
- Be factual and precise
- Use bullet points or numbered lists when appropriate
- Quote relevant passages when helpful
- Reference document names and page numbers when available
- If multiple documents contain relevant information, synthesize the information coherently"""

    @staticmethod
    def create_user_prompt(query: str, context_docs: List[RAGContext]) -> str:
        """Create the user prompt with query and context."""
        # Build context section
        context_section = "Context Documents:\n\n"

        for i, doc in enumerate(context_docs, 1):
            context_section += f"Document {i}: {doc.document_name}"
            if doc.page_number:
                context_section += f" (Page {doc.page_number})"
            context_section += f" [Relevance: {doc.score:.1%}]\n"
            context_section += f"Content: {doc.content}\n\n"

        # Build complete prompt
        prompt = f"{context_section}User Question: {query}\n\n"
        prompt += "Please provide a comprehensive answer based on the context documents above. "
        prompt += "Reference specific documents and page numbers when applicable."

        return prompt

    @staticmethod
    def create_conversation_prompt(
        query: str, context_docs: List[RAGContext], conversation_history: List[Dict]
    ) -> str:
        """Create a prompt including conversation history."""
        # Build conversation history
        history_section = ""
        if conversation_history:
            history_section = "Previous Conversation:\n"
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    history_section += f"User: {content}\n"
                elif role == "assistant":
                    history_section += f"Assistant: {content}\n"
            history_section += "\n"

        # Build context section
        context_section = "Current Context Documents:\n\n"
        for i, doc in enumerate(context_docs, 1):
            context_section += f"Document {i}: {doc.document_name}"
            if doc.page_number:
                context_section += f" (Page {doc.page_number})"
            context_section += f" [Relevance: {doc.score:.1%}]\n"
            context_section += f"Content: {doc.content}\n\n"

        # Build complete prompt
        prompt = history_section + context_section
        prompt += f"Current Question: {query}\n\n"
        prompt += "Please provide a response that takes into account both the conversation history "
        prompt += (
            "and the current context documents. Maintain conversation continuity while "
        )
        prompt += "incorporating relevant information from the documents."

        return prompt

    @staticmethod
    def create_no_context_prompt(query: str) -> str:
        """Create a prompt when no relevant documents are found."""
        return f"""User Question: {query}

I don't have access to relevant documents that can answer your question. This could mean:

1. No documents have been uploaded to the system yet
2. The uploaded documents don't contain information related to your query
3. Your question might need to be rephrased to match the document content

To get better results:
- Try using different keywords or phrases
- Check if documents related to your topic have been uploaded
- Consider breaking complex questions into simpler parts
- Upload relevant documents if they're not yet in the system

Is there a specific document or topic you'd like me to help you find, or would you like to rephrase your question?"""

    @staticmethod
    def create_summarization_prompt(content: str, max_length: int = 150) -> str:
        """Create a prompt for summarizing document content."""
        return f"""Please provide a concise summary of the following content in approximately {max_length} words:

{content}

Focus on the main points and key information that would be most relevant for answering user questions."""

    @staticmethod
    def create_source_citation_prompt() -> str:
        """Create instructions for proper source citation."""
        return """When referencing information from documents, use this format:
- For specific facts: "According to [Document Name]..."
- For quotes: "As stated in [Document Name], '[quote]'"
- For page references: "([Document Name], page X)"
- For multiple sources: "Multiple documents confirm that..." followed by source list"""
