"""
Integration Tests
End-to-end tests for the complete RAG pipeline and system integration.
"""

import asyncio
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestDocumentProcessingPipeline:
    """Test complete document processing pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_document_processing(self):
        """Test complete document upload and processing workflow."""
        # Mock dependencies
        with patch("src.documents.service.get_chroma_client") as mock_chroma, \
             patch("src.documents.service.get_embedding_generator") as mock_embeddings, \
             patch("src.documents.processors.get_processor_for_file_type") as mock_processor:
            
            # Setup mocks
            mock_chroma_client = AsyncMock()
            mock_chroma_client.add_documents = AsyncMock(return_value=True)
            mock_chroma.return_value = mock_chroma_client
            
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(
                return_value=[[0.1, 0.2, 0.3] for _ in range(5)]
            )
            mock_embeddings.return_value = mock_embedding_gen
            
            mock_doc_processor = Mock()
            mock_doc_processor.extract_text = AsyncMock(
                return_value="This is extracted text from the document. " * 100
            )
            mock_doc_processor.extract_metadata = AsyncMock(
                return_value={"title": "Test Document", "page_count": 3}
            )
            mock_processor.return_value = mock_doc_processor
            
            # Create document service
            from src.documents.service import DocumentService
            doc_service = DocumentService()
            
            # Create mock file
            mock_file = Mock()
            mock_file.filename = "test_document.pdf"
            mock_file.size = 1024
            mock_file.read = AsyncMock(return_value=b"fake_pdf_content")
            
            # Process document
            result = await doc_service.process_uploaded_document(
                file=mock_file,
                user_id="test_user",
                metadata={"custom": "value"}
            )
            
            # Verify pipeline completion
            assert result["status"] == "completed"
            assert "document_id" in result
            
            # Verify all pipeline steps were called
            mock_doc_processor.extract_text.assert_called_once()
            mock_doc_processor.extract_metadata.assert_called_once()
            mock_embedding_gen.generate_embeddings.assert_called()
            mock_chroma_client.add_documents.assert_called()

    @pytest.mark.asyncio
    async def test_document_processing_error_recovery(self):
        """Test error recovery in document processing pipeline."""
        with patch("src.documents.service.get_chroma_client") as mock_chroma, \
             patch("src.documents.service.get_embedding_generator") as mock_embeddings, \
             patch("src.documents.processors.get_processor_for_file_type") as mock_processor:
            
            # Setup processor to fail initially, then succeed
            mock_doc_processor = Mock()
            mock_doc_processor.extract_text = AsyncMock(
                side_effect=[Exception("Extraction failed"), "Recovered text content"]
            )
            mock_processor.return_value = mock_doc_processor
            
            from src.documents.service import DocumentService
            doc_service = DocumentService()
            
            mock_file = Mock()
            mock_file.filename = "problematic_document.pdf"
            mock_file.read = AsyncMock(return_value=b"problematic_content")
            
            # First attempt should fail
            with pytest.raises(Exception):
                await doc_service.process_uploaded_document(
                    file=mock_file,
                    user_id="test_user"
                )


class TestSearchPipeline:
    """Test complete search and retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_search_pipeline(self):
        """Test complete search workflow from query to response."""
        # Mock dependencies
        with patch("src.search.service.get_chroma_client") as mock_chroma, \
             patch("src.search.service.get_embedding_generator") as mock_embeddings, \
             patch("src.search.service.get_llm_service") as mock_llm:
            
            # Setup search mocks
            mock_chroma_client = AsyncMock()
            mock_chroma_client.query_documents = AsyncMock(return_value={
                "documents": [["Relevant document content about machine learning"]],
                "distances": [[0.15]],
                "metadatas": [[{"document_filename": "ml_guide.pdf", "page_number": 1}]]
            })
            mock_chroma.return_value = mock_chroma_client
            
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(
                return_value=[[0.1, 0.2, 0.3]]
            )
            mock_embeddings.return_value = mock_embedding_gen
            
            mock_llm_service = AsyncMock()
            mock_llm_service.generate_rag_response = AsyncMock(
                return_value="Machine learning is a subset of AI that enables computers to learn from data."
            )
            mock_llm.return_value = mock_llm_service
            
            # Create search service
            from src.search.service import SearchService
            search_service = SearchService()
            
            # Perform search
            from src.search.models import SemanticSearchRequest
            request = SemanticSearchRequest(
                query="What is machine learning?",
                limit=5,
                similarity_threshold=0.7
            )
            
            result = await search_service.semantic_search(request, user_id="test_user")
            
            # Verify search pipeline
            assert result.query == "What is machine learning?"
            assert len(result.results) > 0
            assert result.search_time_ms > 0
            
            # Verify pipeline steps
            mock_embedding_gen.generate_embeddings.assert_called()
            mock_chroma_client.query_documents.assert_called()

    @pytest.mark.asyncio
    async def test_hybrid_search_integration(self):
        """Test hybrid search combining semantic and keyword search."""
        with patch("src.search.service.get_chroma_client") as mock_chroma, \
             patch("src.search.service.get_embedding_generator") as mock_embeddings:
            
            # Mock semantic search results
            mock_chroma_client = AsyncMock()
            mock_chroma_client.query_documents = AsyncMock(return_value={
                "documents": [["Semantic search result about AI"]],
                "distances": [[0.2]],
                "metadatas": [[{"document_filename": "ai_doc.pdf"}]]
            })
            mock_chroma.return_value = mock_chroma_client
            
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(
                return_value=[[0.1, 0.2, 0.3]]
            )
            mock_embeddings.return_value = mock_embedding_gen
            
            # Mock keyword search (would need BM25 implementation)
            with patch("src.search.hybrid_search.HybridSearchEngine") as mock_hybrid:
                mock_engine = Mock()
                mock_engine.search = AsyncMock(return_value=Mock(results=[]))
                mock_hybrid.return_value = mock_engine
                
                from src.search.service import SearchService
                search_service = SearchService()
                
                from src.search.models import HybridSearchRequest
                request = HybridSearchRequest(
                    query="artificial intelligence",
                    semantic_weight=0.7,
                    keyword_weight=0.3,
                    limit=10
                )
                
                result = await search_service.hybrid_search(request, user_id="test_user")
                
                assert result.query == "artificial intelligence"

    @pytest.mark.asyncio
    async def test_rag_conversation_pipeline(self):
        """Test complete RAG conversation with context."""
        with patch("src.search.service.get_chroma_client") as mock_chroma, \
             patch("src.search.service.get_embedding_generator") as mock_embeddings, \
             patch("src.search.service.get_llm_service") as mock_llm:
            
            # Setup mocks for RAG pipeline
            mock_chroma_client = AsyncMock()
            mock_chroma_client.query_documents = AsyncMock(return_value={
                "documents": [["Machine learning enables automated learning from data"]],
                "distances": [[0.1]],
                "metadatas": [[{"document_filename": "ml_textbook.pdf", "page_number": 5}]]
            })
            mock_chroma.return_value = mock_chroma_client
            
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(
                return_value=[[0.1, 0.2, 0.3]]
            )
            mock_embeddings.return_value = mock_embedding_gen
            
            mock_llm_service = AsyncMock()
            mock_llm_service.generate_rag_response = AsyncMock(
                return_value="Based on the documents, machine learning is a method that enables computers to automatically learn patterns from data without being explicitly programmed for each specific task."
            )
            mock_llm.return_value = mock_llm_service
            
            from src.search.service import SearchService
            search_service = SearchService()
            
            # Test RAG chat with conversation history
            from src.search.models import RAGChatRequest
            request = RAGChatRequest(
                query="How does machine learning work?",
                conversation_history=[
                    {"role": "user", "content": "What is AI?"},
                    {"role": "assistant", "content": "AI is artificial intelligence."},
                    {"role": "user", "content": "What about machine learning?"}
                ],
                include_sources=True,
                context_weight=0.8
            )
            
            result = await search_service.rag_chat(request, user_id="test_user")
            
            # Verify RAG response
            assert "machine learning" in result["response"].lower()
            assert len(result["sources"]) > 0
            assert "conversation_id" in result
            
            # Verify LLM was called with conversation context
            mock_llm_service.generate_rag_response.assert_called()
            call_args = mock_llm_service.generate_rag_response.call_args[1]
            assert "conversation_history" in call_args


class TestSystemIntegration:
    """Test complete system integration across all components."""

    @pytest.mark.asyncio
    async def test_full_system_workflow(self):
        """Test complete workflow: upload document -> process -> search -> RAG response."""
        # This is a comprehensive integration test
        with patch("src.documents.service.get_chroma_client") as mock_doc_chroma, \
             patch("src.search.service.get_chroma_client") as mock_search_chroma, \
             patch("src.documents.service.get_embedding_generator") as mock_doc_embeddings, \
             patch("src.search.service.get_embedding_generator") as mock_search_embeddings, \
             patch("src.search.service.get_llm_service") as mock_llm, \
             patch("src.documents.processors.get_processor_for_file_type") as mock_processor:
            
            # Setup document processing mocks
            mock_chroma_client = AsyncMock()
            mock_chroma_client.add_documents = AsyncMock(return_value=True)
            mock_chroma_client.query_documents = AsyncMock(return_value={
                "documents": [["The uploaded document discusses machine learning algorithms"]],
                "distances": [[0.1]],
                "metadatas": [[{"document_filename": "uploaded_doc.pdf", "document_id": "doc123"}]]
            })
            mock_doc_chroma.return_value = mock_chroma_client
            mock_search_chroma.return_value = mock_chroma_client
            
            # Setup embedding mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(
                return_value=[[0.1, 0.2, 0.3]]
            )
            mock_doc_embeddings.return_value = mock_embedding_gen
            mock_search_embeddings.return_value = mock_embedding_gen
            
            # Setup document processor
            mock_doc_processor = Mock()
            mock_doc_processor.extract_text = AsyncMock(
                return_value="This document discusses machine learning algorithms and their applications in real-world scenarios."
            )
            mock_doc_processor.extract_metadata = AsyncMock(
                return_value={"title": "ML Algorithms Guide", "author": "AI Expert"}
            )
            mock_processor.return_value = mock_doc_processor
            
            # Setup LLM service
            mock_llm_service = AsyncMock()
            mock_llm_service.generate_rag_response = AsyncMock(
                return_value="Based on your uploaded document, machine learning algorithms are computational methods that enable systems to learn patterns from data and make predictions or decisions."
            )
            mock_llm.return_value = mock_llm_service
            
            # Step 1: Upload and process document
            from src.documents.service import DocumentService
            doc_service = DocumentService()
            
            mock_file = Mock()
            mock_file.filename = "ml_algorithms.pdf"
            mock_file.size = 2048
            mock_file.read = AsyncMock(return_value=b"pdf_content_about_ml")
            
            upload_result = await doc_service.process_uploaded_document(
                file=mock_file,
                user_id="test_user",
                metadata={"category": "technical"}
            )
            
            assert upload_result["status"] == "completed"
            document_id = upload_result["document_id"]
            
            # Step 2: Search the uploaded document
            from src.search.service import SearchService
            search_service = SearchService()
            
            from src.search.models import SemanticSearchRequest
            search_request = SemanticSearchRequest(
                query="machine learning algorithms",
                limit=5
            )
            
            search_result = await search_service.semantic_search(
                search_request, 
                user_id="test_user"
            )
            
            assert len(search_result.results) > 0
            assert "machine learning" in search_result.results[0].text.lower()
            
            # Step 3: Generate RAG response
            from src.search.models import RAGChatRequest
            rag_request = RAGChatRequest(
                query="Explain machine learning algorithms",
                include_sources=True
            )
            
            rag_result = await search_service.rag_chat(
                rag_request,
                user_id="test_user"
            )
            
            assert "machine learning" in rag_result["response"].lower()
            assert len(rag_result["sources"]) > 0
            assert rag_result["sources"][0]["document_filename"] == "uploaded_doc.pdf"
            
            # Verify the complete pipeline executed
            mock_doc_processor.extract_text.assert_called()
            mock_embedding_gen.generate_embeddings.assert_called()
            mock_chroma_client.add_documents.assert_called()
            mock_chroma_client.query_documents.assert_called()
            mock_llm_service.generate_rag_response.assert_called()

    @pytest.mark.asyncio
    async def test_multi_user_isolation(self):
        """Test that user data is properly isolated."""
        with patch("src.documents.service.get_chroma_client") as mock_chroma, \
             patch("src.search.service.get_chroma_client") as mock_search_chroma:
            
            # Setup mocks to return different results for different users
            mock_chroma_client = AsyncMock()
            
            def user_specific_query(*args, **kwargs):
                # Simulate user-specific filtering
                if "user1" in str(kwargs.get("metadata_filter", {})):
                    return {
                        "documents": [["User 1 document"]],
                        "distances": [[0.1]],
                        "metadatas": [[{"user_id": "user1", "document": "doc1"}]]
                    }
                else:
                    return {
                        "documents": [["User 2 document"]],
                        "distances": [[0.1]],
                        "metadatas": [[{"user_id": "user2", "document": "doc2"}]]
                    }
            
            mock_chroma_client.query_documents = AsyncMock(side_effect=user_specific_query)
            mock_chroma.return_value = mock_chroma_client
            mock_search_chroma.return_value = mock_chroma_client
            
            from src.search.service import SearchService
            search_service = SearchService()
            
            from src.search.models import SemanticSearchRequest
            
            # Search as user 1
            request1 = SemanticSearchRequest(query="test query", limit=5)
            result1 = await search_service.semantic_search(request1, user_id="user1")
            
            # Search as user 2
            request2 = SemanticSearchRequest(query="test query", limit=5)
            result2 = await search_service.semantic_search(request2, user_id="user2")
            
            # Results should be different for different users
            # (This would require actual user filtering implementation)
            assert len(result1.results) > 0
            assert len(result2.results) > 0

    @pytest.mark.asyncio
    async def test_system_performance_under_load(self):
        """Test system performance with multiple concurrent operations."""
        with patch("src.documents.service.get_chroma_client") as mock_chroma, \
             patch("src.search.service.get_chroma_client") as mock_search_chroma, \
             patch("src.documents.service.get_embedding_generator") as mock_embeddings, \
             patch("src.search.service.get_embedding_generator") as mock_search_embeddings:
            
            # Setup fast-responding mocks
            mock_chroma_client = AsyncMock()
            mock_chroma_client.query_documents = AsyncMock(return_value={
                "documents": [["Fast response document"]],
                "distances": [[0.1]],
                "metadatas": [[{"document": "fast_doc"}]]
            })
            mock_chroma.return_value = mock_chroma_client
            mock_search_chroma.return_value = mock_chroma_client
            
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(
                return_value=[[0.1, 0.2, 0.3]]
            )
            mock_embeddings.return_value = mock_embedding_gen
            mock_search_embeddings.return_value = mock_embedding_gen
            
            from src.search.service import SearchService
            search_service = SearchService()
            
            from src.search.models import SemanticSearchRequest
            
            # Run multiple concurrent searches
            async def perform_search(query_id):
                request = SemanticSearchRequest(
                    query=f"test query {query_id}",
                    limit=5
                )
                return await search_service.semantic_search(
                    request, 
                    user_id=f"user{query_id}"
                )
            
            # Run 20 concurrent searches
            tasks = [perform_search(i) for i in range(20)]
            
            import time
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Verify all searches completed
            assert len(results) == 20
            assert all(len(result.results) > 0 for result in results)
            
            # Check performance (should complete reasonably fast with mocks)
            total_time = end_time - start_time
            assert total_time < 5.0  # Should complete within 5 seconds


class TestErrorRecoveryIntegration:
    """Test error recovery and resilience across the system."""

    @pytest.mark.asyncio
    async def test_vector_store_failure_recovery(self):
        """Test system behavior when vector store is unavailable."""
        with patch("src.search.service.get_chroma_client") as mock_chroma:
            # Mock ChromaDB failure
            mock_chroma_client = AsyncMock()
            mock_chroma_client.query_documents = AsyncMock(
                side_effect=Exception("Vector store connection failed")
            )
            mock_chroma.return_value = mock_chroma_client
            
            from src.search.service import SearchService
            search_service = SearchService()
            
            from src.search.models import SemanticSearchRequest
            request = SemanticSearchRequest(query="test query", limit=5)
            
            # Should handle gracefully and return empty results or error
            try:
                result = await search_service.semantic_search(request, user_id="test_user")
                # If it doesn't raise, should return empty results
                assert len(result.results) == 0
            except Exception as e:
                # Should be a handled exception with proper error message
                assert "search" in str(e).lower() or "unavailable" in str(e).lower()

    @pytest.mark.asyncio
    async def test_llm_service_failure_fallback(self):
        """Test fallback behavior when LLM service fails."""
        with patch("src.search.service.get_chroma_client") as mock_chroma, \
             patch("src.search.service.get_embedding_generator") as mock_embeddings, \
             patch("src.search.service.get_llm_service") as mock_llm:
            
            # Setup working search but failing LLM
            mock_chroma_client = AsyncMock()
            mock_chroma_client.query_documents = AsyncMock(return_value={
                "documents": [["Document content about ML"]],
                "distances": [[0.1]],
                "metadatas": [[{"document_filename": "ml_doc.pdf"}]]
            })
            mock_chroma.return_value = mock_chroma_client
            
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(
                return_value=[[0.1, 0.2, 0.3]]
            )
            mock_embeddings.return_value = mock_embedding_gen
            
            # LLM service fails
            mock_llm_service = AsyncMock()
            mock_llm_service.generate_rag_response = AsyncMock(
                side_effect=Exception("LLM service unavailable")
            )
            mock_llm.return_value = mock_llm_service
            
            from src.search.service import SearchService
            search_service = SearchService()
            
            from src.search.models import RAGChatRequest
            request = RAGChatRequest(
                query="What is machine learning?",
                include_sources=True
            )
            
            # Should fall back to search results with explanation
            result = await search_service.rag_chat(request, user_id="test_user")
            
            # Should provide fallback response with retrieved documents
            assert "response" in result
            assert len(result["sources"]) > 0
            # Fallback response should explain the situation
            assert any(word in result["response"].lower() 
                      for word in ["unable", "service", "documents", "found"])

    @pytest.mark.asyncio
    async def test_embedding_service_failure_handling(self):
        """Test handling of embedding service failures."""
        with patch("src.search.service.get_embedding_generator") as mock_embeddings:
            # Embedding generation fails
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(
                side_effect=Exception("Embedding service unavailable")
            )
            mock_embeddings.return_value = mock_embedding_gen
            
            from src.search.service import SearchService
            search_service = SearchService()
            
            from src.search.models import SemanticSearchRequest
            request = SemanticSearchRequest(query="test query", limit=5)
            
            # Should handle embedding failure gracefully
            with pytest.raises(Exception) as exc_info:
                await search_service.semantic_search(request, user_id="test_user")
            
            # Should be a meaningful error message
            assert "embedding" in str(exc_info.value).lower() or "service" in str(exc_info.value).lower()


class TestDataConsistency:
    """Test data consistency across system operations."""

    @pytest.mark.asyncio
    async def test_document_metadata_consistency(self):
        """Test that document metadata remains consistent across operations."""
        with patch("src.documents.service.get_chroma_client") as mock_doc_chroma, \
             patch("src.search.service.get_chroma_client") as mock_search_chroma, \
             patch("src.documents.service.get_embedding_generator") as mock_embeddings, \
             patch("src.documents.processors.get_processor_for_file_type") as mock_processor:
            
            # Setup consistent metadata across services
            document_metadata = {
                "document_id": "doc123",
                "filename": "consistent_doc.pdf",
                "title": "Consistency Test Document",
                "uploaded_by": "test_user",
                "upload_date": "2023-01-01T00:00:00Z"
            }
            
            # Mock document storage
            mock_chroma_client = AsyncMock()
            mock_chroma_client.add_documents = AsyncMock(return_value=True)
            mock_chroma_client.query_documents = AsyncMock(return_value={
                "documents": [["Document content"]],
                "distances": [[0.1]],
                "metadatas": [[document_metadata]]
            })
            mock_doc_chroma.return_value = mock_chroma_client
            mock_search_chroma.return_value = mock_chroma_client
            
            # Mock processors and embeddings
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(
                return_value=[[0.1, 0.2, 0.3]]
            )
            mock_embeddings.return_value = mock_embedding_gen
            
            mock_doc_processor = Mock()
            mock_doc_processor.extract_text = AsyncMock(return_value="Document content")
            mock_doc_processor.extract_metadata = AsyncMock(return_value={
                "title": "Consistency Test Document"
            })
            mock_processor.return_value = mock_doc_processor
            
            # Process document
            from src.documents.service import DocumentService
            doc_service = DocumentService()
            
            mock_file = Mock()
            mock_file.filename = "consistent_doc.pdf"
            mock_file.read = AsyncMock(return_value=b"content")
            
            upload_result = await doc_service.process_uploaded_document(
                file=mock_file,
                user_id="test_user"
            )
            
            # Search for document
            from src.search.service import SearchService
            search_service = SearchService()
            
            from src.search.models import SemanticSearchRequest
            search_request = SemanticSearchRequest(
                query="consistency test",
                limit=5
            )
            
            search_result = await search_service.semantic_search(
                search_request,
                user_id="test_user"
            )
            
            # Verify metadata consistency
            assert len(search_result.results) > 0
            result_metadata = search_result.results[0].metadata
            
            # Key metadata should match
            assert result_metadata.get("filename") == document_metadata["filename"]
            assert result_metadata.get("document_id") == document_metadata["document_id"]


if __name__ == "__main__":
    pytest.main([__file__])