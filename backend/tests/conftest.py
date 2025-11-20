"""Shared test fixtures and configuration for pytest"""

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

# Add backend directory to path so we can import modules
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults


@pytest.fixture
def sample_lesson():
    """Create a sample lesson for testing"""
    return Lesson(
        lesson_number=1,
        title="Introduction to Testing",
        lesson_link="https://example.com/course/lesson-1",
    )


@pytest.fixture
def sample_course(sample_lesson):
    """Create a sample course for testing"""
    return Course(
        title="Testing Fundamentals",
        course_link="https://example.com/course",
        instructor="Dr. Test",
        lessons=[
            sample_lesson,
            Lesson(
                lesson_number=2,
                title="Advanced Testing Techniques",
                lesson_link="https://example.com/course/lesson-2",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunk():
    """Create a sample course chunk for testing"""
    return CourseChunk(
        content="This is a sample chunk of course content about testing fundamentals.",
        course_title="Testing Fundamentals",
        lesson_number=1,
        chunk_index=0,
    )


@pytest.fixture
def sample_search_results():
    """Create sample successful search results"""
    return SearchResults(
        documents=[
            "Content from lesson 1 about testing basics",
            "Content from lesson 2 about advanced testing",
        ],
        metadata=[
            {"course_title": "Testing Fundamentals", "lesson_number": 1},
            {"course_title": "Testing Fundamentals", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
        error=None,
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results"""
    return SearchResults(documents=[], metadata=[], distances=[], error=None)


@pytest.fixture
def error_search_results():
    """Create search results with error"""
    return SearchResults.empty("Course not found")


@pytest.fixture
def mock_vector_store(sample_search_results):
    """Create a mock VectorStore with default successful search"""
    mock_store = Mock()
    mock_store.search.return_value = sample_search_results
    mock_store.get_lesson_link.return_value = "https://example.com/course/lesson-1"
    mock_store.get_course_link.return_value = "https://example.com/course"
    mock_store._resolve_course_name.return_value = "Testing Fundamentals"
    mock_store.get_all_courses_metadata.return_value = [
        {
            "title": "Testing Fundamentals",
            "course_link": "https://example.com/course",
            "instructor": "Dr. Test",
            "lesson_count": 2,
            "lessons": [
                {
                    "lesson_number": 1,
                    "lesson_title": "Introduction to Testing",
                    "lesson_link": "https://example.com/course/lesson-1",
                },
                {
                    "lesson_number": 2,
                    "lesson_title": "Advanced Testing Techniques",
                    "lesson_link": "https://example.com/course/lesson-2",
                },
            ],
        }
    ]
    mock_store.get_existing_course_titles.return_value = ["Testing Fundamentals"]
    mock_store.get_course_count.return_value = 1
    mock_store.add_course_metadata.return_value = None
    mock_store.add_course_content.return_value = None
    mock_store.clear_all_data.return_value = None
    return mock_store


@pytest.fixture
def mock_anthropic_response_simple():
    """Create a mock Anthropic API response without tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_content = Mock()
    mock_content.text = "This is a simple response from Claude."
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_anthropic_response_with_tool():
    """Create a mock Anthropic API response with tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    # Create mock tool use content block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"query": "testing basics"}

    mock_response.content = [mock_tool_block]
    return mock_response


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response_simple):
    """Create a mock Anthropic client"""
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_anthropic_response_simple
    return mock_client


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager"""
    mock_manager = Mock()
    mock_manager.execute_tool.return_value = "Search results returned successfully"
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    mock_manager.get_last_sources.return_value = []
    mock_manager.reset_sources.return_value = None
    return mock_manager


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test_session_123"
    mock_manager.get_conversation_history.return_value = (
        "User: Previous question\nAssistant: Previous answer"
    )
    mock_manager.add_exchange.return_value = None
    mock_manager.clear_session.return_value = None
    return mock_manager


@pytest.fixture
def mock_document_processor(sample_course, sample_course_chunk):
    """Create a mock DocumentProcessor"""
    mock_processor = Mock()
    mock_processor.process_course_document.return_value = (
        sample_course,
        [sample_course_chunk],
    )
    return mock_processor


@pytest.fixture
def mock_ai_generator():
    """Create a mock AIGenerator"""
    mock_generator = Mock()
    mock_generator.generate_response.return_value = "Generated response from AI"
    return mock_generator


# Multi-round tool calling fixtures


@pytest.fixture
def mock_anthropic_response_with_second_tool():
    """Create a mock Anthropic API response with a different tool use (for round 2)"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    # Create mock tool use content block for outline tool
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "get_course_outline"
    mock_tool_block.id = "tool_456"
    mock_tool_block.input = {"course_name": "Testing Fundamentals"}

    mock_response.content = [mock_tool_block]
    return mock_response


@pytest.fixture
def mock_anthropic_final_response():
    """Create a mock final text response after tool rounds"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_content = Mock()
    mock_content.text = "Final answer synthesizing tool results"
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_two_round_responses(
    mock_anthropic_response_with_tool,
    mock_anthropic_response_with_second_tool,
    mock_anthropic_final_response,
):
    """Create a sequence of responses for two-round tool calling"""
    return [
        mock_anthropic_response_with_tool,  # Round 1: search_course_content
        mock_anthropic_response_with_second_tool,  # Round 2: get_course_outline
        mock_anthropic_final_response,  # Final: text response
    ]


@pytest.fixture
def mock_max_rounds_responses(
    mock_anthropic_response_with_tool, mock_anthropic_final_response
):
    """Create responses where Claude exhausts max rounds"""
    return [
        mock_anthropic_response_with_tool,  # Round 1: tool_use
        mock_anthropic_response_with_tool,  # Round 2: tool_use (exhausts limit)
        mock_anthropic_final_response,  # Final call without tools
    ]


@pytest.fixture
def mock_early_termination_responses(
    mock_anthropic_response_with_tool, mock_anthropic_final_response
):
    """Create responses where Claude stops using tools after one round"""
    return [
        mock_anthropic_response_with_tool,  # Round 1: tool_use
        mock_anthropic_final_response,  # Round 2: Claude decides to answer
    ]


# API Testing Fixtures


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    from config import Config

    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"

    return config


@pytest.fixture
def mock_rag_system(mock_config):
    """Mock RAG system for API testing"""
    mock = MagicMock()
    mock.config = mock_config
    mock.session_manager.create_session.return_value = "test-session-123"
    mock.query.return_value = (
        "This is a test response about course materials.",
        ["Source 1: Test course material", "Source 2: Another reference"],
    )
    mock.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course 1", "Course 2", "Course 3"],
    }

    return mock


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a test FastAPI app without static file mounting.
    This avoids issues with missing frontend directory in test environment.
    """
    from typing import List, Optional, Union

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    # Create test app
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Define Pydantic models (same as in app.py)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, SourceItem]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Define API endpoints (same as in app.py)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint for health check"""
        return {"status": "ok", "message": "RAG System API is running"}

    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app"""
    from fastapi.testclient import TestClient

    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request data for testing"""
    return {
        "query": "What is covered in the machine learning course?",
        "session_id": "test-session-123",
    }


@pytest.fixture
def sample_query_request_no_session():
    """Sample query request without session ID"""
    return {"query": "Tell me about deep learning basics"}
