"""Tests for rag_system module - RAGSystem integration and orchestration"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import os
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class TestRAGSystemInitialization:
    """Test RAGSystem initialization and component setup"""

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_initialization_creates_all_components(
        self, mock_doc_processor, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test that all components are initialized correctly"""
        # Create a mock config
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "./test_chroma"
        mock_config.EMBEDDING_MODEL = "test-model"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test_key"
        mock_config.ANTHROPIC_MODEL = "test_model"
        mock_config.MAX_HISTORY = 2

        system = RAGSystem(mock_config)

        # Verify all components were initialized with correct params
        mock_doc_processor.assert_called_once_with(800, 100)
        mock_vector_store.assert_called_once_with("./test_chroma", "test-model", 5)
        mock_ai_gen.assert_called_once_with("test_key", "test_model")
        mock_session_mgr.assert_called_once_with(2)

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_initialization_registers_tools(
        self, mock_doc_processor, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test that both search tools are registered"""
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "./test_chroma"
        mock_config.EMBEDDING_MODEL = "test-model"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test_key"
        mock_config.ANTHROPIC_MODEL = "test_model"
        mock_config.MAX_HISTORY = 2

        system = RAGSystem(mock_config)

        # Verify both tools are registered
        assert "search_course_content" in system.tool_manager.tools
        assert "get_course_outline" in system.tool_manager.tools


class TestRAGSystemAddCourseDocument:
    """Test adding individual course documents"""

    @pytest.fixture
    def rag_system(self, mock_document_processor, mock_vector_store, mock_ai_generator, mock_session_manager):
        """Create RAGSystem with mocked components"""
        with patch('rag_system.DocumentProcessor', return_value=mock_document_processor), \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator', return_value=mock_ai_generator), \
             patch('rag_system.SessionManager', return_value=mock_session_manager):

            mock_config = Mock()
            mock_config.CHUNK_SIZE = 800
            mock_config.CHUNK_OVERLAP = 100
            mock_config.CHROMA_PATH = "./test_chroma"
            mock_config.EMBEDDING_MODEL = "test-model"
            mock_config.MAX_RESULTS = 5
            mock_config.ANTHROPIC_API_KEY = "test_key"
            mock_config.ANTHROPIC_MODEL = "test_model"
            mock_config.MAX_HISTORY = 2

            return RAGSystem(mock_config)

    def test_add_course_document_success(self, rag_system, sample_course, sample_course_chunk):
        """Test successful addition of course document"""
        # Setup mock to return course and chunks
        chunks = [sample_course_chunk]
        rag_system.document_processor.process_course_document.return_value = (sample_course, chunks)

        course, chunk_count = rag_system.add_course_document("/path/to/course.txt")

        # Verify document was processed
        rag_system.document_processor.process_course_document.assert_called_once_with("/path/to/course.txt")

        # Verify metadata and content were added to vector store
        rag_system.vector_store.add_course_metadata.assert_called_once_with(sample_course)
        rag_system.vector_store.add_course_content.assert_called_once_with(chunks)

        # Verify return values
        assert course == sample_course
        assert chunk_count == 1

    def test_add_course_document_error_handling(self, rag_system, capsys):
        """Test error handling when document processing fails"""
        # Setup mock to raise exception
        rag_system.document_processor.process_course_document.side_effect = Exception("File not found")

        course, chunk_count = rag_system.add_course_document("/invalid/path.txt")

        # Verify error was handled gracefully
        assert course is None
        assert chunk_count == 0

        # Verify error message was printed
        captured = capsys.readouterr()
        assert "Error processing course document" in captured.out
        assert "File not found" in captured.out

    def test_add_course_document_with_multiple_chunks(self, rag_system, sample_course):
        """Test adding document with multiple chunks"""
        chunks = [
            CourseChunk(content="Content 1", course_title="Test Course", lesson_number=1, chunk_index=0),
            CourseChunk(content="Content 2", course_title="Test Course", lesson_number=1, chunk_index=1),
            CourseChunk(content="Content 3", course_title="Test Course", lesson_number=2, chunk_index=0)
        ]
        rag_system.document_processor.process_course_document.return_value = (sample_course, chunks)

        course, chunk_count = rag_system.add_course_document("/path/to/course.txt")

        assert chunk_count == 3
        rag_system.vector_store.add_course_content.assert_called_once_with(chunks)


class TestRAGSystemAddCourseFolder:
    """Test adding multiple course documents from a folder"""

    @pytest.fixture
    def rag_system_with_mocks(self, mock_document_processor, mock_vector_store, mock_ai_generator, mock_session_manager):
        """Create RAGSystem with all mocked components"""
        with patch('rag_system.DocumentProcessor', return_value=mock_document_processor), \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator', return_value=mock_ai_generator), \
             patch('rag_system.SessionManager', return_value=mock_session_manager):

            mock_config = Mock()
            mock_config.CHUNK_SIZE = 800
            mock_config.CHUNK_OVERLAP = 100
            mock_config.CHROMA_PATH = "./test_chroma"
            mock_config.EMBEDDING_MODEL = "test-model"
            mock_config.MAX_RESULTS = 5
            mock_config.ANTHROPIC_API_KEY = "test_key"
            mock_config.ANTHROPIC_MODEL = "test_model"
            mock_config.MAX_HISTORY = 2

            return RAGSystem(mock_config)

    def test_add_course_folder_with_new_courses(self, rag_system_with_mocks, sample_course, sample_course_chunk):
        """Test adding folder with new courses"""
        system = rag_system_with_mocks

        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            open(os.path.join(temp_dir, "course1.txt"), 'w').close()
            open(os.path.join(temp_dir, "course2.txt"), 'w').close()

            # Mock vector store to return no existing courses
            system.vector_store.get_existing_course_titles.return_value = []

            # Mock document processor to return different courses
            course1 = Course(title="Course 1", course_link=None, instructor=None, lessons=[])
            course2 = Course(title="Course 2", course_link=None, instructor=None, lessons=[])
            chunks = [sample_course_chunk]

            system.document_processor.process_course_document.side_effect = [
                (course1, chunks),
                (course2, chunks)
            ]

            total_courses, total_chunks = system.add_course_folder(temp_dir)

            # Verify results
            assert total_courses == 2
            assert total_chunks == 2

            # Verify both courses were added
            assert system.vector_store.add_course_metadata.call_count == 2
            assert system.vector_store.add_course_content.call_count == 2

    def test_add_course_folder_skips_existing_courses(self, rag_system_with_mocks, sample_course, sample_course_chunk):
        """Test that existing courses are skipped"""
        system = rag_system_with_mocks

        with tempfile.TemporaryDirectory() as temp_dir:
            open(os.path.join(temp_dir, "course1.txt"), 'w').close()

            # Mock vector store to return existing course
            system.vector_store.get_existing_course_titles.return_value = ["Testing Fundamentals"]

            # Mock document processor
            chunks = [sample_course_chunk]
            system.document_processor.process_course_document.return_value = (sample_course, chunks)

            total_courses, total_chunks = system.add_course_folder(temp_dir)

            # Verify course was not added
            assert total_courses == 0
            assert total_chunks == 0
            system.vector_store.add_course_metadata.assert_not_called()

    def test_add_course_folder_with_clear_existing(self, rag_system_with_mocks, sample_course, sample_course_chunk):
        """Test clearing existing data before adding"""
        system = rag_system_with_mocks

        with tempfile.TemporaryDirectory() as temp_dir:
            open(os.path.join(temp_dir, "course1.txt"), 'w').close()

            system.vector_store.get_existing_course_titles.return_value = []
            chunks = [sample_course_chunk]
            system.document_processor.process_course_document.return_value = (sample_course, chunks)

            total_courses, total_chunks = system.add_course_folder(temp_dir, clear_existing=True)

            # Verify clear was called
            system.vector_store.clear_all_data.assert_called_once()

    def test_add_course_folder_nonexistent_folder(self, rag_system_with_mocks):
        """Test handling of nonexistent folder"""
        system = rag_system_with_mocks

        total_courses, total_chunks = system.add_course_folder("/nonexistent/folder")

        assert total_courses == 0
        assert total_chunks == 0

    def test_add_course_folder_ignores_non_document_files(self, rag_system_with_mocks, sample_course, sample_course_chunk):
        """Test that non-document files are ignored"""
        system = rag_system_with_mocks

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create various file types
            open(os.path.join(temp_dir, "course.txt"), 'w').close()
            open(os.path.join(temp_dir, "image.jpg"), 'w').close()
            open(os.path.join(temp_dir, "data.json"), 'w').close()

            system.vector_store.get_existing_course_titles.return_value = []
            chunks = [sample_course_chunk]
            system.document_processor.process_course_document.return_value = (sample_course, chunks)

            total_courses, total_chunks = system.add_course_folder(temp_dir)

            # Only .txt file should be processed
            assert system.document_processor.process_course_document.call_count == 1

    def test_add_course_folder_handles_processing_errors(self, rag_system_with_mocks, capsys):
        """Test handling of errors during folder processing"""
        system = rag_system_with_mocks

        with tempfile.TemporaryDirectory() as temp_dir:
            open(os.path.join(temp_dir, "bad_course.txt"), 'w').close()

            system.vector_store.get_existing_course_titles.return_value = []
            system.document_processor.process_course_document.side_effect = Exception("Parse error")

            total_courses, total_chunks = system.add_course_folder(temp_dir)

            # Verify error was handled
            assert total_courses == 0
            assert total_chunks == 0

            captured = capsys.readouterr()
            assert "Error processing" in captured.out


class TestRAGSystemQuery:
    """Test query processing and RAG pipeline"""

    @pytest.fixture
    def rag_system(self, mock_document_processor, mock_vector_store, mock_ai_generator, mock_session_manager, mock_tool_manager):
        """Create RAGSystem with mocked components"""
        with patch('rag_system.DocumentProcessor', return_value=mock_document_processor), \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator', return_value=mock_ai_generator), \
             patch('rag_system.SessionManager', return_value=mock_session_manager):

            mock_config = Mock()
            mock_config.CHUNK_SIZE = 800
            mock_config.CHUNK_OVERLAP = 100
            mock_config.CHROMA_PATH = "./test_chroma"
            mock_config.EMBEDDING_MODEL = "test-model"
            mock_config.MAX_RESULTS = 5
            mock_config.ANTHROPIC_API_KEY = "test_key"
            mock_config.ANTHROPIC_MODEL = "test_model"
            mock_config.MAX_HISTORY = 2

            system = RAGSystem(mock_config)
            # Replace tool_manager with mock for easier testing
            system.tool_manager = mock_tool_manager
            return system

    def test_query_without_session(self, rag_system, mock_ai_generator):
        """Test query processing without session ID"""
        response, sources = rag_system.query("What is testing?")

        # Verify AI generator was called without history
        mock_ai_generator.generate_response.assert_called_once()
        call_args = mock_ai_generator.generate_response.call_args[1]
        assert "What is testing?" in call_args["query"]
        assert call_args["conversation_history"] is None
        assert call_args["tools"] is not None
        assert call_args["tool_manager"] is not None

        # Verify response is returned
        assert response == "Generated response from AI"

    def test_query_with_session(self, rag_system, mock_ai_generator, mock_session_manager):
        """Test query processing with session ID"""
        mock_session_manager.get_conversation_history.return_value = "Previous conversation"

        response, sources = rag_system.query("Follow-up question", session_id="session_123")

        # Verify history was retrieved
        mock_session_manager.get_conversation_history.assert_called_once_with("session_123")

        # Verify AI generator was called with history
        call_args = mock_ai_generator.generate_response.call_args[1]
        assert call_args["conversation_history"] == "Previous conversation"

        # Verify conversation was updated
        mock_session_manager.add_exchange.assert_called_once_with(
            "session_123",
            "Follow-up question",
            "Generated response from AI"
        )

    def test_query_retrieves_sources(self, rag_system, mock_tool_manager):
        """Test that sources are retrieved from tool manager"""
        test_sources = [{"text": "Course 1 - Lesson 1", "url": "http://example.com"}]
        mock_tool_manager.get_last_sources.return_value = test_sources

        response, sources = rag_system.query("What is testing?")

        # Verify sources were retrieved
        mock_tool_manager.get_last_sources.assert_called_once()
        assert sources == test_sources

    def test_query_resets_sources_after_retrieval(self, rag_system, mock_tool_manager):
        """Test that sources are reset after being retrieved"""
        rag_system.query("What is testing?")

        # Verify sources were reset
        mock_tool_manager.reset_sources.assert_called_once()

    def test_query_passes_tools_to_ai(self, rag_system, mock_ai_generator, mock_tool_manager):
        """Test that tool definitions are passed to AI generator"""
        tool_defs = [{"name": "search_tool", "description": "Search"}]
        mock_tool_manager.get_tool_definitions.return_value = tool_defs

        rag_system.query("What is testing?")

        # Verify tools were passed
        call_args = mock_ai_generator.generate_response.call_args[1]
        assert call_args["tools"] == tool_defs
        assert call_args["tool_manager"] == mock_tool_manager

    def test_query_formats_prompt_correctly(self, rag_system, mock_ai_generator):
        """Test that query is formatted into proper prompt"""
        rag_system.query("What is unit testing?")

        call_args = mock_ai_generator.generate_response.call_args[1]
        prompt = call_args["query"]

        assert "What is unit testing?" in prompt
        assert "Answer this question about course materials:" in prompt


class TestRAGSystemAnalytics:
    """Test course analytics functionality"""

    @pytest.fixture
    def rag_system(self, mock_document_processor, mock_vector_store, mock_ai_generator, mock_session_manager):
        """Create RAGSystem with mocked components"""
        with patch('rag_system.DocumentProcessor', return_value=mock_document_processor), \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator', return_value=mock_ai_generator), \
             patch('rag_system.SessionManager', return_value=mock_session_manager):

            mock_config = Mock()
            mock_config.CHUNK_SIZE = 800
            mock_config.CHUNK_OVERLAP = 100
            mock_config.CHROMA_PATH = "./test_chroma"
            mock_config.EMBEDDING_MODEL = "test-model"
            mock_config.MAX_RESULTS = 5
            mock_config.ANTHROPIC_API_KEY = "test_key"
            mock_config.ANTHROPIC_MODEL = "test_model"
            mock_config.MAX_HISTORY = 2

            return RAGSystem(mock_config)

    def test_get_course_analytics(self, rag_system, mock_vector_store):
        """Test getting course analytics"""
        mock_vector_store.get_course_count.return_value = 3
        mock_vector_store.get_existing_course_titles.return_value = ["Course 1", "Course 2", "Course 3"]

        analytics = rag_system.get_course_analytics()

        # Verify analytics structure
        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
        assert "Course 1" in analytics["course_titles"]

    def test_get_course_analytics_empty(self, rag_system, mock_vector_store):
        """Test analytics when no courses exist"""
        mock_vector_store.get_course_count.return_value = 0
        mock_vector_store.get_existing_course_titles.return_value = []

        analytics = rag_system.get_course_analytics()

        assert analytics["total_courses"] == 0
        assert analytics["course_titles"] == []


class TestRAGSystemIntegration:
    """Integration tests with real ChromaDB (optional)"""

    @pytest.mark.integration
    def test_end_to_end_with_real_chromadb(self, sample_course, sample_course_chunk):
        """Test end-to-end flow with actual ChromaDB instance"""
        # This test would use a real temporary ChromaDB instance
        # Skipped in normal test runs, run with: pytest -m integration
        pytest.skip("Integration test - requires real ChromaDB setup")
