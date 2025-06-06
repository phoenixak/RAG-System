"""
Search API Endpoints
REST API endpoints for search and retrieval functionality.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer

from src.auth.security import get_current_user
from src.core.logging import get_logger
from src.search.models import (
    ContextualSearchRequest,
    HybridSearchRequest,
    QuerySuggestionsRequest,
    QuerySuggestionsResponse,
    SearchResponse,
    SearchResult,
    SemanticSearchRequest,
    SimilarDocumentsRequest,
)
from src.search.service import get_search_service

router = APIRouter()
security = HTTPBearer()
logger = get_logger(__name__)


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    current_user: dict = Depends(get_current_user),
) -> SearchResponse:
    """
    Perform semantic search using vector similarity.

    Semantic search finds documents based on meaning and context rather than exact keyword matches.
    It uses embeddings to understand the semantic similarity between the query and document content.
    """
    try:
        logger.info(
            "Semantic search request",
            query=request.query,
            user_id=current_user.user_id,
            limit=request.limit,
        )

        search_service = get_search_service()
        response = await search_service.semantic_search(
            request=request,
            user_id=current_user.user_id,
        )

        logger.info(
            "Semantic search completed",
            query=request.query,
            results_count=len(response.results),
            search_time_ms=response.search_time_ms,
        )

        return response

    except Exception as e:
        logger.error(
            "Semantic search failed",
            query=request.query,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}",
        )


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    current_user: dict = Depends(get_current_user),
) -> SearchResponse:
    """
    Perform hybrid search combining semantic and keyword search with optional re-ranking.

    Hybrid search combines the benefits of both semantic search (understanding meaning)
    and keyword search (exact term matching) to provide the most relevant results.
    Results can be re-ranked using cross-encoder models for improved relevance.
    """
    try:
        logger.info(
            "Hybrid search request",
            query=request.query,
            user_id=current_user.user_id,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
            enable_rerank=request.enable_rerank,
        )

        search_service = get_search_service()
        response = await search_service.hybrid_search(
            request=request,
            user_id=current_user.user_id,
        )

        logger.info(
            "Hybrid search completed",
            query=request.query,
            results_count=len(response.results),
            search_time_ms=response.search_time_ms,
            rerank_time_ms=response.rerank_time_ms,
        )

        return response

    except Exception as e:
        logger.error(
            "Hybrid search failed",
            query=request.query,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(e)}",
        )


@router.post("/contextual", response_model=SearchResponse)
async def contextual_search(
    request: ContextualSearchRequest,
    current_user: dict = Depends(get_current_user),
) -> SearchResponse:
    """
    Perform context-aware search using conversation history.

    Contextual search enhances queries with conversation history to provide more relevant results.
    It maintains session context and can understand follow-up questions and related queries.
    """
    try:
        logger.info(
            "Contextual search request",
            query=request.query,
            session_id=request.session_id,
            use_context=request.use_context,
            user_id=current_user.user_id,
        )

        search_service = get_search_service()
        response = await search_service.contextual_search(
            request=request,
            user_id=current_user.user_id,
        )

        logger.info(
            "Contextual search completed",
            query=request.query,
            session_id=request.session_id,
            results_count=len(response.results),
            search_time_ms=response.search_time_ms,
        )

        return response

    except Exception as e:
        logger.error(
            "Contextual search failed",
            query=request.query,
            session_id=request.session_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Contextual search failed: {str(e)}",
        )


@router.post("/similar-documents", response_model=List[SearchResult])
async def find_similar_documents(
    request: SimilarDocumentsRequest,
    current_user: dict = Depends(get_current_user),
) -> List[SearchResult]:
    """
    Find documents similar to a reference document.

    This endpoint finds documents that are semantically similar to a given reference document
    using vector similarity search. Useful for discovering related content.
    """
    try:
        logger.info(
            "Similar documents request",
            reference_document_id=request.document_id,
            user_id=current_user.user_id,
            limit=request.limit,
        )

        search_service = get_search_service()
        results = await search_service.find_similar_documents(
            request=request,
            user_id=current_user.user_id,
        )

        logger.info(
            "Similar documents search completed",
            reference_document_id=request.document_id,
            results_count=len(results),
        )

        return results

    except Exception as e:
        logger.error(
            "Similar documents search failed",
            reference_document_id=request.document_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar documents search failed: {str(e)}",
        )


@router.get("/suggestions", response_model=QuerySuggestionsResponse)
async def get_query_suggestions(
    q: str = Query(..., description="Partial query for suggestions"),
    limit: int = Query(default=5, ge=1, le=20, description="Number of suggestions"),
    current_user=Depends(get_current_user),
) -> QuerySuggestionsResponse:
    """
    Get query suggestions and autocomplete.

    Provides intelligent query suggestions and autocomplete functionality
    to help users formulate better search queries.
    """
    try:
        logger.info(
            "Query suggestions request",
            partial_query=q,
            user_id=current_user.user_id,
            limit=limit,
        )

        request = QuerySuggestionsRequest(
            partial_query=q,
            limit=limit,
        )

        search_service = get_search_service()
        response = await search_service.get_query_suggestions(
            request=request,
            user_id=current_user.user_id,
        )

        logger.info(
            "Query suggestions completed",
            partial_query=q,
            suggestions_count=len(response.suggestions),
        )

        return response

    except Exception as e:
        logger.error(
            "Query suggestions failed",
            partial_query=q,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query suggestions failed: {str(e)}",
        )


@router.get("/stats")
async def get_search_stats(
    current_user=Depends(get_current_user),
) -> dict:
    """
    Get search system statistics and performance metrics.

    Returns comprehensive statistics about search performance, cache hit rates,
    and system health. Requires appropriate permissions.
    """
    try:
        # Check if user has admin permissions for stats
        user_role = current_user.role.value
        if user_role not in ["admin", "moderator"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view search statistics",
            )

        logger.info(
            "Search stats request",
            user_id=current_user.user_id,
            user_role=user_role,
        )

        search_service = get_search_service()
        stats = search_service.get_search_stats()

        return {
            "status": "success",
            "stats": stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Search stats failed",
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get search statistics: {str(e)}",
        )


@router.post("/cache/clear")
async def clear_search_cache(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Clear the search result cache.

    Clears all cached search results. This can be useful after updating documents
    or when you want to ensure fresh results. Requires admin permissions.
    """
    try:
        # Check if user has admin permissions
        user_role = current_user.role.value
        if user_role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can clear the search cache",
            )

        logger.info(
            "Search cache clear request",
            user_id=current_user.user_id,
        )

        search_service = get_search_service()
        search_service.clear_cache()

        logger.info(
            "Search cache cleared successfully",
            user_id=current_user.user_id,
        )

        return {
            "status": "success",
            "message": "Search cache cleared successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Clear search cache failed",
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear search cache: {str(e)}",
        )


@router.post("/conversations")
async def create_conversation_session(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Create a new conversation session.

    Creates a new conversation session for contextual search and returns the session ID.
    """
    try:
        logger.info(
            "Create conversation session request",
            user_id=current_user.user_id,
        )

        search_service = get_search_service()
        conversation_manager = search_service.conversation_manager

        # Generate new session ID
        import uuid

        session_id = str(uuid.uuid4())

        # Create session
        session = conversation_manager.get_or_create_session(session_id)

        logger.info(
            "Conversation session created",
            session_id=session_id,
            user_id=current_user.user_id,
        )

        return {
            "status": "success",
            "session_id": session_id,
            "created_at": session.created_at.isoformat()
            if hasattr(session, "created_at")
            else None,
        }

    except Exception as e:
        logger.error(
            "Create conversation session failed",
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation session: {str(e)}",
        )


@router.get("/conversation/{session_id}")
async def get_conversation_summary(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Get conversation session summary.

    Returns information about a conversation session including query history,
    identified topics, and context summary.
    """
    try:
        logger.info(
            "Conversation summary request",
            session_id=session_id,
            user_id=current_user.user_id,
        )

        search_service = get_search_service()
        conversation_manager = search_service.conversation_manager
        summary = conversation_manager.get_session_summary(session_id)

        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation session {session_id} not found",
            )

        return {
            "status": "success",
            "session": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Get conversation summary failed",
            session_id=session_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation summary: {str(e)}",
        )


@router.delete("/conversation/{session_id}")
async def clear_conversation_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Clear a conversation session.

    Removes all conversation history and context for the specified session.
    """
    try:
        logger.info(
            "Clear conversation session request",
            session_id=session_id,
            user_id=current_user.user_id,
        )

        search_service = get_search_service()
        conversation_manager = search_service.conversation_manager
        success = conversation_manager.clear_session(session_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation session {session_id} not found",
            )

        logger.info(
            "Conversation session cleared",
            session_id=session_id,
            user_id=current_user.user_id,
        )

        return {
            "status": "success",
            "message": f"Conversation session {session_id} cleared successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Clear conversation session failed",
            session_id=session_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear conversation session: {str(e)}",
        )


@router.post("/generate_response")
async def generate_llm_response(
    request: dict,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Generate an LLM response based on search results.

    Takes a query, search results, and optional conversation history to generate
    an intelligent response using the configured LLM service.
    """
    try:
        logger.info(
            "LLM response generation request",
            user_id=current_user.user_id,
            query=request.get("query", "")[:100],  # Log first 100 chars
        )

        query = request.get("query", "")
        search_results = request.get("search_results", [])
        conversation_history = request.get("conversation_history", [])

        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Query is required"
            )

        # Import here to avoid circular dependencies
        from src.llm.service import get_llm_service

        llm_service = get_llm_service()
        if not llm_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service is not available",
            )

        response = await llm_service.generate_rag_response(
            query=query,
            context_docs=search_results,
            conversation_history=conversation_history,
        )

        logger.info(
            "LLM response generated successfully",
            user_id=current_user.user_id,
            response_length=len(response),
        )

        return {"response": response}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "LLM response generation failed",
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM response generation failed: {str(e)}",
        )
