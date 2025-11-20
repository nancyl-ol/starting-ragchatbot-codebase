"""
API endpoint tests for the RAG system FastAPI application.

These tests verify the correct behavior of the REST API endpoints:
- POST /api/query: Query processing endpoint
- GET /api/courses: Course statistics endpoint
- GET /: Root/health check endpoint
"""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.api
class TestRootEndpoint:
    """Tests for the root endpoint (/)"""

    def test_root_endpoint_returns_ok(self, test_client):
        """Test that root endpoint returns 200 OK with status message"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "message" in data

    def test_root_endpoint_structure(self, test_client):
        """Test that root endpoint returns expected JSON structure"""
        response = test_client.get("/")
        data = response.json()

        # Verify response structure
        assert isinstance(data, dict)
        assert isinstance(data.get("status"), str)
        assert isinstance(data.get("message"), str)


@pytest.mark.api
class TestQueryEndpoint:
    """Tests for the query endpoint (/api/query)"""

    def test_query_with_session_id(self, test_client, sample_query_request, mock_rag_system):
        """Test query endpoint with provided session ID"""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

        # Verify session ID matches request
        assert data["session_id"] == sample_query_request["session_id"]

        # Verify RAG system was called
        mock_rag_system.query.assert_called_once_with(
            sample_query_request["query"],
            sample_query_request["session_id"]
        )

    def test_query_without_session_id(self, test_client, sample_query_request_no_session, mock_rag_system):
        """Test query endpoint creates session when none provided"""
        response = test_client.post("/api/query", json=sample_query_request_no_session)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify session was created
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_returns_correct_data(self, test_client, sample_query_request, mock_rag_system):
        """Test that query endpoint returns expected answer and sources"""
        response = test_client.post("/api/query", json=sample_query_request)
        data = response.json()

        # Verify the mocked response is returned
        assert data["answer"] == "This is a test response about course materials."
        assert len(data["sources"]) == 2
        assert "Source 1: Test course material" in data["sources"]
        assert "Source 2: Another reference" in data["sources"]

    def test_query_missing_required_field(self, test_client):
        """Test query endpoint rejects request missing 'query' field"""
        invalid_request = {"session_id": "test-123"}
        response = test_client.post("/api/query", json=invalid_request)

        assert response.status_code == 422  # Unprocessable Entity

    def test_query_with_empty_query(self, test_client):
        """Test query endpoint handles empty query string"""
        empty_query = {"query": ""}
        response = test_client.post("/api/query", json=empty_query)

        # Should still be accepted (validation happens in RAG system)
        assert response.status_code == 200

    def test_query_with_invalid_json(self, test_client):
        """Test query endpoint handles invalid JSON"""
        response = test_client.post(
            "/api/query",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_endpoint_handles_rag_exception(self, test_client, sample_query_request, mock_rag_system):
        """Test query endpoint returns 500 when RAG system throws exception"""
        # Make the mock raise an exception
        mock_rag_system.query.side_effect = Exception("Database connection failed")

        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]

    def test_query_with_special_characters(self, test_client):
        """Test query endpoint handles special characters in query"""
        special_query = {
            "query": "What about C++, Python & Java?!?",
            "session_id": "test-123"
        }
        response = test_client.post("/api/query", json=special_query)

        assert response.status_code == 200

    def test_query_with_long_text(self, test_client):
        """Test query endpoint handles long query text"""
        long_query = {
            "query": "x" * 10000,  # Very long query
            "session_id": "test-123"
        }
        response = test_client.post("/api/query", json=long_query)

        assert response.status_code == 200

    def test_query_response_model_validation(self, test_client, sample_query_request):
        """Test that query response adheres to QueryResponse model"""
        response = test_client.post("/api/query", json=sample_query_request)
        data = response.json()

        # Verify all required fields are present
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data, f"Required field '{field}' missing from response"

        # Verify types according to QueryResponse model
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

        # Verify sources are either strings or objects with 'text' and optional 'url'
        for source in data["sources"]:
            assert isinstance(source, (str, dict))
            if isinstance(source, dict):
                assert "text" in source


@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for the courses endpoint (/api/courses)"""

    def test_get_courses_returns_statistics(self, test_client, mock_rag_system):
        """Test courses endpoint returns course statistics"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        # Verify RAG system method was called
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_courses_returns_correct_data(self, test_client, mock_rag_system):
        """Test courses endpoint returns expected course data"""
        response = test_client.get("/api/courses")
        data = response.json()

        # Verify the mocked data is returned
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Course 1" in data["course_titles"]
        assert "Course 2" in data["course_titles"]
        assert "Course 3" in data["course_titles"]

    def test_get_courses_handles_empty_catalog(self, test_client, mock_rag_system):
        """Test courses endpoint handles empty course catalog"""
        # Mock empty catalog
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")
        data = response.json()

        assert response.status_code == 200
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_handles_exception(self, test_client, mock_rag_system):
        """Test courses endpoint returns 500 when analytics throws exception"""
        # Make the mock raise an exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics error" in data["detail"]

    def test_get_courses_response_model_validation(self, test_client):
        """Test that courses response adheres to CourseStats model"""
        response = test_client.get("/api/courses")
        data = response.json()

        # Verify all required fields are present
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data, f"Required field '{field}' missing from response"

        # Verify types according to CourseStats model
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        # Verify all course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)

    def test_get_courses_count_matches_titles_length(self, test_client):
        """Test that total_courses count matches number of course titles"""
        response = test_client.get("/api/courses")
        data = response.json()

        assert data["total_courses"] == len(data["course_titles"])


@pytest.mark.api
class TestCORS:
    """Tests for CORS configuration"""

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in responses"""
        response = test_client.get("/")

        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers

    def test_options_request_allowed(self, test_client):
        """Test that OPTIONS requests are handled (for CORS preflight)"""
        response = test_client.options("/api/query")

        # Should allow OPTIONS requests
        assert response.status_code in [200, 204]


@pytest.mark.api
class TestErrorHandling:
    """Tests for general error handling"""

    def test_404_on_invalid_endpoint(self, test_client):
        """Test that invalid endpoints return 404"""
        response = test_client.get("/api/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self, test_client):
        """Test that wrong HTTP methods return 405"""
        # GET on POST endpoint
        response = test_client.get("/api/query")

        assert response.status_code == 405


@pytest.mark.api
class TestIntegration:
    """Integration tests across multiple endpoints"""

    def test_query_then_check_courses(self, test_client, sample_query_request):
        """Test querying then checking courses list"""
        # First make a query
        query_response = test_client.post("/api/query", json=sample_query_request)
        assert query_response.status_code == 200

        # Then check courses
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200

        # Both should succeed
        query_data = query_response.json()
        courses_data = courses_response.json()

        assert query_data["answer"]
        assert courses_data["total_courses"] > 0

    def test_multiple_queries_same_session(self, test_client):
        """Test multiple queries with the same session ID"""
        session_id = "persistent-session"

        # First query
        query1 = {"query": "What is machine learning?", "session_id": session_id}
        response1 = test_client.post("/api/query", json=query1)
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["session_id"] == session_id

        # Second query with same session
        query2 = {"query": "Tell me more", "session_id": session_id}
        response2 = test_client.post("/api/query", json=query2)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == session_id

    def test_multiple_queries_different_sessions(self, test_client):
        """Test multiple queries with different session IDs"""
        query1 = {"query": "Question 1", "session_id": "session-1"}
        query2 = {"query": "Question 2", "session_id": "session-2"}

        response1 = test_client.post("/api/query", json=query1)
        response2 = test_client.post("/api/query", json=query2)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Sessions should be different
        assert data1["session_id"] != data2["session_id"]
