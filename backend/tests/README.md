# API Testing for RAG System

This directory contains comprehensive API endpoint tests for the RAG system FastAPI application.

## Test Structure

- `conftest.py`: Shared pytest fixtures and configuration
  - Mock fixtures for RAG system components
  - Test app fixture (without static file mounting)
  - Sample request fixtures
  - Test client fixture

- `test_api.py`: API endpoint tests
  - Root endpoint tests (`/`)
  - Query endpoint tests (`/api/query`)
  - Courses endpoint tests (`/api/courses`)
  - CORS configuration tests
  - Error handling tests
  - Integration tests

## Running Tests

### Install dependencies (if not already installed)
```bash
cd /Users/nancyli/Documents/Learning/ClaudeCode/starting-ragchatbot-codebase
uv sync --dev
```

### Run all tests
```bash
cd backend
uv run pytest
```

### Run only API tests
```bash
cd backend
uv run pytest -m api
```

### Run specific test file
```bash
cd backend
uv run pytest tests/test_api.py
```

### Run specific test class
```bash
cd backend
uv run pytest tests/test_api.py::TestQueryEndpoint
```

### Run specific test
```bash
cd backend
uv run pytest tests/test_api.py::TestQueryEndpoint::test_query_with_session_id
```

### Run with verbose output
```bash
cd backend
uv run pytest -v
```

### Run with coverage report
```bash
cd backend
uv run pytest --cov=. --cov-report=html
# Open htmlcov/index.html to view coverage report
```

## Test Markers

Tests are organized with pytest markers:
- `@pytest.mark.api`: API endpoint tests
- `@pytest.mark.unit`: Unit tests (for other test files)
- `@pytest.mark.integration`: Integration tests

Use markers to run specific test categories:
```bash
uv run pytest -m api        # Run only API tests
uv run pytest -m unit       # Run only unit tests
```

## Test Coverage

The test suite covers:

### Root Endpoint (`/`)
- ✓ Health check returns 200 OK
- ✓ Response structure validation

### Query Endpoint (`/api/query`)
- ✓ Query with session ID
- ✓ Query without session ID (auto-creation)
- ✓ Correct data returned
- ✓ Missing required fields (422 error)
- ✓ Empty query handling
- ✓ Invalid JSON (422 error)
- ✓ RAG system exceptions (500 error)
- ✓ Special characters in query
- ✓ Long text handling
- ✓ Response model validation

### Courses Endpoint (`/api/courses`)
- ✓ Returns course statistics
- ✓ Correct data returned
- ✓ Empty catalog handling
- ✓ Analytics exceptions (500 error)
- ✓ Response model validation
- ✓ Count matches titles length

### CORS
- ✓ CORS headers present
- ✓ OPTIONS requests handled

### Error Handling
- ✓ 404 on invalid endpoints
- ✓ 405 on wrong HTTP methods

### Integration
- ✓ Query then check courses
- ✓ Multiple queries same session
- ✓ Multiple queries different sessions

## Key Design Decisions

### Test App Fixture
The `test_app` fixture creates a FastAPI app **without** mounting static files. This design choice:
- Avoids import errors from missing frontend directory in test environment
- Keeps tests focused on API behavior, not static file serving
- Defines API endpoints inline with mocked RAG system
- Maintains same endpoint signatures as production app

### Mock RAG System
The tests use a mocked RAG system (`mock_rag_system` fixture) to:
- Isolate API layer tests from business logic
- Avoid external dependencies (Anthropic API, ChromaDB)
- Enable fast test execution
- Control test scenarios (success, errors, edge cases)

## Adding New Tests

To add new API tests:

1. Add test methods to existing test classes or create new classes
2. Use descriptive test names: `test_<feature>_<scenario>`
3. Add `@pytest.mark.api` marker to new test classes
4. Use fixtures from `conftest.py` for common test data
5. Follow the AAA pattern: Arrange, Act, Assert

Example:
```python
@pytest.mark.api
class TestNewEndpoint:
    """Tests for new endpoint"""
    
    def test_endpoint_success_case(self, test_client):
        """Test successful request"""
        # Arrange
        request_data = {"key": "value"}
        
        # Act
        response = test_client.post("/api/new", json=request_data)
        
        # Assert
        assert response.status_code == 200
        assert "expected_field" in response.json()
```
