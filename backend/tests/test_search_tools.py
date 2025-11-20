"""Tests for search_tools module - CourseSearchTool, CourseOutlineTool, and ToolManager"""

from unittest.mock import Mock

import pytest
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""

    def test_execute_successful_search_no_filters(
        self, mock_vector_store, sample_search_results
    ):
        """Test successful search without any filters"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="testing basics")

        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="testing basics", course_name=None, lesson_number=None
        )

        # Verify result contains formatted content
        assert "[Testing Fundamentals - Lesson 1]" in result
        assert "Content from lesson 1" in result
        assert "[Testing Fundamentals - Lesson 2]" in result
        assert "Content from lesson 2" in result

    def test_execute_with_course_name_filter(
        self, mock_vector_store, sample_search_results
    ):
        """Test search with course name filter"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(
            query="testing basics", course_name="Testing Fundamentals"
        )

        # Verify search was called with course name
        mock_vector_store.search.assert_called_once_with(
            query="testing basics",
            course_name="Testing Fundamentals",
            lesson_number=None,
        )

        assert "Testing Fundamentals" in result

    def test_execute_with_lesson_number_filter(
        self, mock_vector_store, sample_search_results
    ):
        """Test search with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="testing basics", lesson_number=1)

        # Verify search was called with lesson number
        mock_vector_store.search.assert_called_once_with(
            query="testing basics", course_name=None, lesson_number=1
        )

        assert "Lesson 1" in result

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test search with both course name and lesson number filters"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(
            query="testing basics", course_name="Testing Fundamentals", lesson_number=2
        )

        # Verify search was called with both filters
        mock_vector_store.search.assert_called_once_with(
            query="testing basics", course_name="Testing Fundamentals", lesson_number=2
        )

        assert "Testing Fundamentals" in result
        assert "Lesson 2" in result

    def test_execute_empty_results_no_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Test handling of empty results without filters"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="nonexistent topic")

        assert result == "No relevant content found."

    def test_execute_empty_results_with_course_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Test handling of empty results with course filter"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(
            query="testing basics", course_name="Testing Fundamentals"
        )

        assert "No relevant content found in course 'Testing Fundamentals'." == result

    def test_execute_empty_results_with_lesson_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Test handling of empty results with lesson filter"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="testing basics", lesson_number=1)

        assert "No relevant content found in lesson 1." == result

    def test_execute_empty_results_with_both_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Test handling of empty results with both filters"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(
            query="testing basics", course_name="Testing Fundamentals", lesson_number=1
        )

        assert (
            "No relevant content found in course 'Testing Fundamentals' in lesson 1."
            == result
        )

    def test_execute_error_handling(self, mock_vector_store, error_search_results):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="testing basics")

        assert result == "Course not found"

    def test_source_tracking_with_urls(self, mock_vector_store, sample_search_results):
        """Test that last_sources is correctly populated with URLs"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="testing basics")

        # Verify sources are tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Testing Fundamentals - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/course/lesson-1"
        assert tool.last_sources[1]["text"] == "Testing Fundamentals - Lesson 2"

    def test_source_tracking_without_lesson_number(self, mock_vector_store):
        """Test source tracking when metadata has no lesson number"""
        # Create results without lesson numbers
        results = SearchResults(
            documents=["General course content"],
            metadata=[{"course_title": "Testing Fundamentals"}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = results
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="testing basics")

        # Verify source without lesson info
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Testing Fundamentals"
        assert tool.last_sources[0]["url"] is None

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly structured"""
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]

    def test_format_results_with_various_metadata(self, mock_vector_store):
        """Test _format_results with different metadata configurations"""
        tool = CourseSearchTool(mock_vector_store)

        # Test with full metadata
        results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None},
            ],
            distances=[0.1, 0.2],
            error=None,
        )

        formatted = tool._format_results(results)

        assert "[Course A - Lesson 1]" in formatted
        assert "[Course B]" in formatted
        assert "Content 1" in formatted
        assert "Content 2" in formatted


class TestCourseOutlineTool:
    """Test cases for CourseOutlineTool"""

    def test_execute_successful_outline_retrieval(self, mock_vector_store):
        """Test successful retrieval of course outline"""
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="Testing Fundamentals")

        # Verify course name was resolved
        mock_vector_store._resolve_course_name.assert_called_once_with(
            "Testing Fundamentals"
        )

        # Verify outline contains all expected information
        assert "Course: Testing Fundamentals" in result
        assert "Course Link: https://example.com/course" in result
        assert "Instructor: Dr. Test" in result
        assert "Total Lessons: 2" in result
        assert "1. Introduction to Testing" in result
        assert "2. Advanced Testing Techniques" in result

    def test_execute_course_not_found_during_resolution(self, mock_vector_store):
        """Test error when course name cannot be resolved"""
        mock_vector_store._resolve_course_name.return_value = None
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="Nonexistent Course")

        assert result == "No course found matching 'Nonexistent Course'"

    def test_execute_course_not_in_metadata(self, mock_vector_store):
        """Test error when resolved course is not in metadata"""
        mock_vector_store._resolve_course_name.return_value = "Different Course"
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="Testing")

        assert result == "Could not retrieve outline for 'Different Course'"

    def test_format_outline_with_complete_metadata(self, mock_vector_store):
        """Test _format_outline with complete course metadata"""
        tool = CourseOutlineTool(mock_vector_store)

        course_meta = {
            "title": "Complete Course",
            "course_link": "https://example.com/complete",
            "instructor": "Prof. Complete",
            "lesson_count": 3,
            "lessons": [
                {
                    "lesson_number": 1,
                    "lesson_title": "Lesson One",
                    "lesson_link": "https://example.com/lesson-1",
                },
                {
                    "lesson_number": 2,
                    "lesson_title": "Lesson Two",
                    "lesson_link": "https://example.com/lesson-2",
                },
                {
                    "lesson_number": 3,
                    "lesson_title": "Lesson Three",
                    "lesson_link": None,
                },
            ],
        }

        result = tool._format_outline(course_meta)

        assert "Course: Complete Course" in result
        assert "Course Link: https://example.com/complete" in result
        assert "Instructor: Prof. Complete" in result
        assert "Total Lessons: 3" in result
        assert "1. Lesson One - https://example.com/lesson-1" in result
        assert "2. Lesson Two - https://example.com/lesson-2" in result
        assert "3. Lesson Three" in result

    def test_format_outline_without_optional_fields(self, mock_vector_store):
        """Test _format_outline with minimal metadata"""
        tool = CourseOutlineTool(mock_vector_store)

        course_meta = {"title": "Minimal Course", "lesson_count": 1, "lessons": []}

        result = tool._format_outline(course_meta)

        assert "Course: Minimal Course" in result
        assert "Total Lessons: 1" in result
        # Should not have instructor or course link
        assert "Instructor:" not in result
        assert "Course Link:" not in result

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly structured"""
        tool = CourseOutlineTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "course_name" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_name"]


class TestToolManager:
    """Test cases for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test registering a tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool

    def test_register_multiple_tools(self, mock_vector_store):
        """Test registering multiple tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        assert len(manager.tools) == 2
        assert "search_course_content" in manager.tools
        assert "get_course_outline" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        assert any(d["name"] == "search_course_content" for d in definitions)
        assert any(d["name"] == "get_course_outline" for d in definitions)

    def test_execute_tool_success(self, mock_vector_store, sample_search_results):
        """Test executing a registered tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="testing basics")

        assert "Testing Fundamentals" in result
        mock_vector_store.search.assert_called_once()

    def test_execute_tool_not_found(self, mock_vector_store):
        """Test executing a non-existent tool"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test getting sources from last search"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search to populate sources
        manager.execute_tool("search_course_content", query="testing basics")

        sources = manager.get_last_sources()

        assert len(sources) > 0
        assert sources[0]["text"] == "Testing Fundamentals - Lesson 1"

    def test_get_last_sources_when_empty(self, mock_vector_store):
        """Test getting sources when no search has been performed"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        sources = manager.get_last_sources()

        assert sources == []

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources after retrieval"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute search and verify sources exist
        manager.execute_tool("search_course_content", query="testing basics")
        assert len(manager.get_last_sources()) > 0

        # Reset and verify sources are cleared
        manager.reset_sources()
        assert manager.get_last_sources() == []

    def test_register_tool_without_name_raises_error(self):
        """Test that registering a tool without name raises ValueError"""
        manager = ToolManager()

        # Create a mock tool with invalid definition
        invalid_tool = Mock()
        invalid_tool.get_tool_definition.return_value = {"description": "No name"}

        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(invalid_tool)
