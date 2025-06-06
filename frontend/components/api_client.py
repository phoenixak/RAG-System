"""
API Client
Handles all communication with the FastAPI backend.
"""

from typing import Any, Dict

import requests
import streamlit as st

from frontend.config import SESSION_KEYS, get_api_url


class APIClient:
    """Client for communicating with the FastAPI backend."""

    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 30

    def _get_headers(self, include_auth: bool = True) -> Dict[str, str]:
        """Get request headers with optional authentication."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if include_auth and SESSION_KEYS["token"] in st.session_state:
            token = st.session_state[SESSION_KEYS["token"]]
            headers["Authorization"] = f"Bearer {token}"

        return headers

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and extract data."""
        try:
            if response.status_code == 401:
                # Token expired, clear session
                self._clear_auth_state()
                st.error("Session expired. Please log in again.")
                st.rerun()

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 422:
                # Validation error
                error_detail = response.json().get("detail", "Validation error")
                raise APIException(f"Validation error: {error_detail}")
            else:
                error_msg = response.json().get("error", {}).get("message", str(e))
                raise APIException(f"API error: {error_msg}")

        except ValueError:
            # Invalid JSON response
            raise APIException("Invalid response from server")

    def _clear_auth_state(self):
        """Clear authentication state from session."""
        auth_keys = [
            SESSION_KEYS["authenticated"],
            SESSION_KEYS["user"],
            SESSION_KEYS["token"],
            SESSION_KEYS["refresh_token"],
        ]
        for key in auth_keys:
            if key in st.session_state:
                del st.session_state[key]

    def health_check(self) -> Dict[str, Any]:
        """Check backend health status."""
        try:
            url = get_api_url("health")
            response = self.session.get(
                url, headers=self._get_headers(include_auth=False)
            )
            return self._handle_response(response)
        except Exception as e:
            raise APIException(f"Health check failed: {str(e)}")

    # Authentication Methods
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user and return tokens."""
        try:
            url = f"{get_api_url('auth')}/login"
            data = {"email": email, "password": password}

            response = self.session.post(
                url,
                json=data,  # JSON data to match backend expectations
                headers=self._get_headers(include_auth=False),
            )

            return self._handle_response(response)

        except Exception as e:
            raise APIException(f"Login failed: {str(e)}")

    def logout(self) -> bool:
        """Logout user and clear tokens."""
        try:
            url = f"{get_api_url('auth')}/logout"
            response = self.session.post(url, headers=self._get_headers())
            self._handle_response(response)
            self._clear_auth_state()
            return True

        except Exception as e:
            # Clear local state even if logout fails
            self._clear_auth_state()
            raise APIException(f"Logout failed: {str(e)}")

    def get_current_user(self) -> Dict[str, Any]:
        """Get current user information."""
        try:
            url = f"{get_api_url('auth')}/me"
            response = self.session.get(url, headers=self._get_headers())
            return self._handle_response(response)

        except Exception as e:
            raise APIException(f"Failed to get user info: {str(e)}")

    # Document Methods
    def upload_document(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Upload a document file."""
        try:
            url = f"{get_api_url('documents')}/upload"

            files = {"file": (filename, file_data)}
            headers = {}
            if SESSION_KEYS["token"] in st.session_state:
                headers["Authorization"] = (
                    f"Bearer {st.session_state[SESSION_KEYS['token']]}"
                )

            response = self.session.post(url, files=files, headers=headers)
            return self._handle_response(response)

        except Exception as e:
            raise APIException(f"File upload failed: {str(e)}")

    def get_documents(self, skip: int = 0, limit: int = 50) -> Dict[str, Any]:
        """Get list of documents."""
        try:
            url = (
                f"{get_api_url('documents')}"  # Remove trailing slash to avoid redirect
            )
            params = {
                "page": (skip // limit) + 1,
                "page_size": limit,
            }  # Use page-based pagination

            response = self.session.get(url, params=params, headers=self._get_headers())
            data = self._handle_response(response)

            # Normalize document responses to handle field name mismatches
            if "documents" in data:
                normalized_docs = []
                for doc in data["documents"]:
                    normalized = {}
                    # Handle ID field (backend uses 'id', frontend expects 'document_id')
                    normalized["document_id"] = doc.get("id") or doc.get("document_id")
                    normalized["filename"] = doc.get("filename", "Unknown")
                    # Handle status field (backend uses 'processing_status', frontend expects 'status')
                    normalized["status"] = doc.get("processing_status") or doc.get(
                        "status", "unknown"
                    )
                    # Handle file size (backend uses 'size', frontend expects 'file_size')
                    normalized["file_size"] = doc.get("size") or doc.get("file_size", 0)
                    # Copy other fields
                    for key, value in doc.items():
                        if key not in normalized:
                            normalized[key] = value
                    normalized_docs.append(normalized)
                data["documents"] = normalized_docs

            return data

        except Exception as e:
            raise APIException(f"Failed to get documents: {str(e)}")

    def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get single document details."""
        try:
            url = f"{get_api_url('documents')}/{document_id}"
            response = self.session.get(url, headers=self._get_headers())
            return self._handle_response(response)

        except Exception as e:
            raise APIException(f"Failed to get document: {str(e)}")

    def delete_document(self, document_id: str) -> bool:
        """Delete a document."""
        try:
            url = f"{get_api_url('documents')}/{document_id}"
            response = self.session.delete(url, headers=self._get_headers())
            self._handle_response(response)
            return True

        except Exception as e:
            raise APIException(f"Failed to delete document: {str(e)}")

    def get_document_chunks(
        self, document_id: str, skip: int = 0, limit: int = 20
    ) -> Dict[str, Any]:
        """Get document chunks."""
        try:
            url = f"{get_api_url('documents')}/{document_id}/chunks"
            params = {"skip": skip, "limit": limit}

            response = self.session.get(url, params=params, headers=self._get_headers())
            return self._handle_response(response)

        except Exception as e:
            raise APIException(f"Failed to get document chunks: {str(e)}")

    # Search Methods
    def semantic_search(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Perform semantic search."""
        try:
            url = f"{get_api_url('search')}/semantic"
            data = {
                "query": query,
                "limit": limit,
                "similarity_threshold": similarity_threshold,
            }

            response = self.session.post(url, json=data, headers=self._get_headers())
            return self._handle_response(response)

        except Exception as e:
            raise APIException(f"Semantic search failed: {str(e)}")

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """Perform hybrid search."""
        try:
            url = f"{get_api_url('search')}/hybrid"
            data = {
                "query": query,
                "limit": limit,
                "semantic_weight": semantic_weight,
                "keyword_weight": keyword_weight,
            }

            response = self.session.post(url, json=data, headers=self._get_headers())
            return self._handle_response(response)

        except Exception as e:
            raise APIException(f"Hybrid search failed: {str(e)}")

    def contextual_search(
        self, query: str, conversation_id: str, limit: int = 10
    ) -> Dict[str, Any]:
        """Perform contextual search with conversation history."""
        try:
            url = f"{get_api_url('search')}/contextual"
            data = {
                "query": query,
                "session_id": conversation_id,
                "limit": limit,
                "use_context": True,
            }

            response = self.session.post(url, json=data, headers=self._get_headers())
            return self._handle_response(response)

        except Exception as e:
            raise APIException(f"Contextual search failed: {str(e)}")

    def get_similar_documents(self, document_id: str, limit: int = 5) -> Dict[str, Any]:
        """Get similar documents."""
        try:
            url = f"{get_api_url('search')}/similar/{document_id}"
            params = {"limit": limit}

            response = self.session.get(url, params=params, headers=self._get_headers())
            return self._handle_response(response)

        except Exception as e:
            raise APIException(f"Failed to get similar documents: {str(e)}")

    # Conversation Methods
    def create_conversation(self) -> Dict[str, Any]:
        """Create a new conversation session."""
        try:
            url = f"{get_api_url('search')}/conversations"
            response = self.session.post(url, headers=self._get_headers())
            return self._handle_response(response)

        except Exception as e:
            raise APIException(f"Failed to create conversation: {str(e)}")

    def get_conversation_history(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation history."""
        try:
            url = f"{get_api_url('search')}/conversations/{conversation_id}/history"
            response = self.session.get(url, headers=self._get_headers())
            return self._handle_response(response)

        except Exception as e:
            raise APIException(f"Failed to get conversation history: {str(e)}")

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history."""
        try:
            url = f"{get_api_url('search')}/conversations/{conversation_id}/clear"
            response = self.session.delete(url, headers=self._get_headers())
            self._handle_response(response)
            return True

        except Exception as e:
            raise APIException(f"Failed to clear conversation: {str(e)}")

    def generate_llm_response(
        self, query: str, search_results: list, conversation_history: list = None
    ) -> str:
        """Generate LLM response based on search results."""
        try:
            url = f"{get_api_url('search')}/generate_response"
            data = {
                "query": query,
                "search_results": search_results,
                "conversation_history": conversation_history or [],
            }

            response = self.session.post(url, json=data, headers=self._get_headers())
            result = self._handle_response(response)
            return result.get("response", "Sorry, I couldn't generate a response.")

        except Exception as e:
            raise APIException(f"Failed to generate LLM response: {str(e)}")


class APIException(Exception):
    """Custom exception for API errors."""

    pass


# Global API client instance
@st.cache_resource
def get_api_client() -> APIClient:
    """Get cached API client instance."""
    return APIClient()
