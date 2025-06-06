"""
Conversation Context Management
Handles conversation sessions and context-aware search.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from src.core.config import get_settings
from src.core.logging import LoggerMixin
from src.search.models import (
    ConversationContext,
)

settings = get_settings()


class ConversationMemory:
    """In-memory conversation storage (will be replaced with Redis in production)."""

    def __init__(self, max_sessions: int = 1000, session_ttl: int = 7200):
        self.sessions: Dict[str, ConversationContext] = {}
        self.max_sessions = max_sessions
        self.session_ttl = session_ttl  # 2 hours default

    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation session by ID."""
        session = self.sessions.get(session_id)

        if session:
            # Check if session is expired
            age = (datetime.utcnow() - session.updated_at).total_seconds()
            if age > self.session_ttl:
                del self.sessions[session_id]
                return None

        return session

    def create_session(self, session_id: Optional[str] = None) -> ConversationContext:
        """Create a new conversation session."""
        if session_id is None:
            session_id = str(uuid4())

        session = ConversationContext(session_id=session_id)

        # Clean up old sessions if we're at capacity
        if len(self.sessions) >= self.max_sessions:
            self._cleanup_old_sessions()

        self.sessions[session_id] = session
        return session

    def update_session(self, session: ConversationContext) -> None:
        """Update an existing session."""
        session.updated_at = datetime.utcnow()
        self.sessions[session.session_id] = session

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def _cleanup_old_sessions(self) -> None:
        """Remove oldest 20% of sessions."""
        if not self.sessions:
            return

        # Sort by last updated time
        sorted_sessions = sorted(self.sessions.items(), key=lambda x: x[1].updated_at)

        # Remove oldest 20%
        remove_count = len(self.sessions) // 5
        for session_id, _ in sorted_sessions[:remove_count]:
            del self.sessions[session_id]

    def get_stats(self) -> Dict[str, any]:
        """Get memory statistics."""
        now = datetime.utcnow()

        active_sessions = 0
        for session in self.sessions.values():
            age = (now - session.updated_at).total_seconds()
            if age <= 3600:  # Active in last hour
                active_sessions += 1

        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "max_sessions": self.max_sessions,
            "session_ttl": self.session_ttl,
        }


class ContextAnalyzer(LoggerMixin):
    """Analyzes conversation context to improve search."""

    def __init__(self):
        # Common topic keywords for context analysis
        self.topic_keywords = {
            "technical": [
                "api",
                "code",
                "programming",
                "software",
                "system",
                "development",
            ],
            "business": [
                "revenue",
                "profit",
                "sales",
                "market",
                "strategy",
                "customer",
            ],
            "legal": [
                "contract",
                "agreement",
                "terms",
                "policy",
                "compliance",
                "regulation",
            ],
            "financial": [
                "budget",
                "cost",
                "expense",
                "investment",
                "finance",
                "accounting",
            ],
            "research": [
                "study",
                "analysis",
                "research",
                "data",
                "findings",
                "methodology",
            ],
        }

    def extract_topics(self, queries: List[str]) -> List[str]:
        """
        Extract topics from conversation queries.

        Args:
            queries: List of previous queries

        Returns:
            List of identified topics
        """
        topics = set()

        # Combine all queries
        all_text = " ".join(queries).lower()

        # Check for topic keywords
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                topics.add(topic)

        return list(topics)

    def generate_context_summary(self, queries: List[str]) -> str:
        """
        Generate a summary of conversation context.

        Args:
            queries: List of previous queries

        Returns:
            Context summary string
        """
        if not queries:
            return ""

        # Extract key terms from recent queries
        recent_queries = queries[-3:]  # Last 3 queries

        # Simple keyword extraction (could be enhanced with NLP)
        key_terms = set()
        for query in recent_queries:
            words = query.lower().split()
            # Filter meaningful words (length > 3, not common stop words)
            meaningful_words = [
                word
                for word in words
                if len(word) > 3
                and word not in {"this", "that", "what", "when", "where", "how"}
            ]
            key_terms.update(meaningful_words[:3])  # Top 3 per query

        if key_terms:
            return f"Recent discussion about: {', '.join(sorted(key_terms)[:5])}"

        return "General conversation"

    def calculate_query_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate similarity between two queries.

        Args:
            query1: First query
            query2: Second query

        Returns:
            Similarity score (0-1)
        """
        # Simple token overlap similarity
        tokens1 = set(query1.lower().split())
        tokens2 = set(query2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def identify_query_intent(self, query: str) -> str:
        """
        Identify the intent of a query.

        Args:
            query: Search query

        Returns:
            Query intent classification
        """
        query_lower = query.lower()

        # Question patterns
        if any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
            return "question"

        # Search patterns
        if any(word in query_lower for word in ["find", "search", "show", "list"]):
            return "search"

        # Definition patterns
        if any(word in query_lower for word in ["define", "definition", "meaning"]):
            return "definition"

        # Comparison patterns
        if any(
            word in query_lower for word in ["compare", "difference", "versus", "vs"]
        ):
            return "comparison"

        return "general"


class ConversationManager(LoggerMixin):
    """Manages conversation sessions and context-aware search."""

    def __init__(self):
        self.memory = ConversationMemory()
        self.analyzer = ContextAnalyzer()

    def get_or_create_session(
        self, session_id: Optional[str] = None
    ) -> ConversationContext:
        """
        Get existing session or create a new one.

        Args:
            session_id: Optional session ID

        Returns:
            Conversation context
        """
        if session_id:
            session = self.memory.get_session(session_id)
            if session:
                return session

        # Create new session
        session = self.memory.create_session(session_id)

        self.logger.info(
            "Created new conversation session",
            session_id=session.session_id,
        )

        return session

    def add_query_to_session(
        self,
        session_id: str,
        query: str,
        results_count: int = 0,
    ) -> ConversationContext:
        """
        Add a query to the conversation session.

        Args:
            session_id: Session ID
            query: Search query
            results_count: Number of results returned

        Returns:
            Updated conversation context
        """
        session = self.get_or_create_session(session_id)

        # Add query to history (limit to 10 queries)
        session.previous_queries.append(query)
        if len(session.previous_queries) > 10:
            session.previous_queries = session.previous_queries[-10:]

        # Update topics
        session.topics = self.analyzer.extract_topics(session.previous_queries)

        # Update context summary
        session.context_summary = self.analyzer.generate_context_summary(
            session.previous_queries
        )

        # Update session
        self.memory.update_session(session)

        self.logger.debug(
            "Added query to session",
            session_id=session_id,
            query=query,
            total_queries=len(session.previous_queries),
            topics=session.topics,
        )

        return session

    def enhance_query_with_context(
        self,
        query: str,
        context: ConversationContext,
        context_weight: float = 0.2,
    ) -> str:
        """
        Enhance a query with conversation context.

        Args:
            query: Original query
            context: Conversation context
            context_weight: Weight for context influence

        Returns:
            Enhanced query
        """
        if not context.previous_queries or context_weight <= 0:
            return query

        # Get recent relevant queries
        relevant_queries = []
        for prev_query in context.previous_queries[-3:]:  # Last 3 queries
            similarity = self.analyzer.calculate_query_similarity(query, prev_query)
            if similarity > 0.3:  # Threshold for relevance
                relevant_queries.append(prev_query)

        if not relevant_queries:
            return query

        # Extract context terms
        context_terms = set()
        for prev_query in relevant_queries:
            words = prev_query.lower().split()
            meaningful_words = [
                word for word in words if len(word) > 3 and word not in query.lower()
            ]
            context_terms.update(meaningful_words[:2])  # Top 2 per query

        # Enhance query with context
        if context_terms:
            context_part = " ".join(sorted(context_terms)[:3])  # Top 3 context terms
            enhanced_query = f"{query} {context_part}"

            self.logger.debug(
                "Enhanced query with context",
                original_query=query,
                enhanced_query=enhanced_query,
                context_terms=list(context_terms),
            )

            return enhanced_query

        return query

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, any]]:
        """
        Get a summary of the conversation session.

        Args:
            session_id: Session ID

        Returns:
            Session summary or None if not found
        """
        session = self.memory.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "query_count": len(session.previous_queries),
            "topics": session.topics,
            "context_summary": session.context_summary,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "recent_queries": session.previous_queries[-3:]
            if session.previous_queries
            else [],
        }

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a conversation session.

        Args:
            session_id: Session ID

        Returns:
            True if session was cleared, False if not found
        """
        success = self.memory.delete_session(session_id)

        if success:
            self.logger.info("Cleared conversation session", session_id=session_id)

        return success

    def get_conversation_stats(self) -> Dict[str, any]:
        """
        Get conversation management statistics.

        Returns:
            Dictionary of statistics
        """
        memory_stats = self.memory.get_stats()

        # Calculate additional stats
        total_queries = 0
        for session in self.memory.sessions.values():
            total_queries += len(session.previous_queries)

        return {
            **memory_stats,
            "total_queries": total_queries,
            "avg_queries_per_session": (
                total_queries / len(self.memory.sessions) if self.memory.sessions else 0
            ),
        }


# Global conversation manager instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager instance."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager


async def close_conversation_manager():
    """Close the global conversation manager."""
    global _conversation_manager
    if _conversation_manager:
        # Clear all sessions for cleanup
        _conversation_manager.memory.sessions.clear()
        _conversation_manager = None
