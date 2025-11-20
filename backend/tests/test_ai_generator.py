"""Tests for ai_generator module - AIGenerator and tool calling functionality"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator"""

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AIGenerator instance with mocked client"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
            generator.client = mock_anthropic_client  # Ensure mock is used
            return generator

    def test_initialization(self):
        """Test AIGenerator initialization"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(api_key="test_key", model="test_model")

            mock_anthropic.assert_called_once_with(api_key="test_key")
            assert generator.model == "test_model"
            assert generator.base_params["model"] == "test_model"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800

    def test_generate_response_simple_no_tools(self, ai_generator, mock_anthropic_response_simple):
        """Test simple response generation without tools"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_simple

        result = ai_generator.generate_response(query="What is testing?")

        # Verify API was called correctly
        ai_generator.client.messages.create.assert_called_once()
        call_args = ai_generator.client.messages.create.call_args[1]

        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["messages"] == [{"role": "user", "content": "What is testing?"}]
        assert AIGenerator.SYSTEM_PROMPT in call_args["system"]

        # Verify response
        assert result == "This is a simple response from Claude."

    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_response_simple):
        """Test response generation with conversation history"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_simple

        history = "User: Previous question\nAssistant: Previous answer"
        result = ai_generator.generate_response(
            query="Follow-up question",
            conversation_history=history
        )

        # Verify history is included in system prompt
        call_args = ai_generator.client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args["system"]
        assert "Previous question" in call_args["system"]
        assert "Previous answer" in call_args["system"]

    def test_generate_response_with_tools_no_tool_use(self, ai_generator, mock_anthropic_response_simple):
        """Test response with tools available but not used"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_simple

        tools = [{"name": "test_tool", "description": "A test tool"}]
        result = ai_generator.generate_response(
            query="What is testing?",
            tools=tools
        )

        # Verify tools were passed to API
        call_args = ai_generator.client.messages.create.call_args[1]
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}

        # Verify result is returned directly (no tool execution)
        assert result == "This is a simple response from Claude."

    def test_generate_response_with_tool_use_single_call(
        self, ai_generator, mock_anthropic_response_with_tool, mock_tool_manager
    ):
        """Test response with single tool call"""
        # Setup: initial response with tool use
        ai_generator.client.messages.create.return_value = mock_anthropic_response_with_tool

        # Create final response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="Final answer using tool results")]

        # Make client return different responses on subsequent calls
        ai_generator.client.messages.create.side_effect = [
            mock_anthropic_response_with_tool,
            final_response
        ]

        tools = [{"name": "search_course_content", "description": "Search tool"}]
        result = ai_generator.generate_response(
            query="What is testing?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="testing basics"
        )

        # Verify second API call was made for final response
        assert ai_generator.client.messages.create.call_count == 2

        # Verify result is from final response
        assert result == "Final answer using tool results"

    def test_generate_response_with_multiple_tool_calls(
        self, ai_generator, mock_tool_manager
    ):
        """Test response with multiple tool calls"""
        # Create response with multiple tool use blocks
        multi_tool_response = Mock()
        multi_tool_response.stop_reason = "tool_use"

        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "testing basics"}

        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "get_course_outline"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"course_name": "Testing Course"}

        multi_tool_response.content = [tool_block_1, tool_block_2]

        # Create final response
        final_response = Mock()
        final_response.content = [Mock(text="Final answer using multiple tools")]

        ai_generator.client.messages.create.side_effect = [
            multi_tool_response,
            final_response
        ]

        tools = [
            {"name": "search_course_content", "description": "Search tool"},
            {"name": "get_course_outline", "description": "Outline tool"}
        ]
        result = ai_generator.generate_response(
            query="What is testing?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="testing basics"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline",
            course_name="Testing Course"
        )

    def test_system_prompt_without_history(self, ai_generator, mock_anthropic_response_simple):
        """Test system prompt construction without conversation history"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_simple

        ai_generator.generate_response(query="Test query")

        call_args = ai_generator.client.messages.create.call_args[1]
        assert call_args["system"] == AIGenerator.SYSTEM_PROMPT
        assert "Previous conversation:" not in call_args["system"]

    def test_system_prompt_with_history(self, ai_generator, mock_anthropic_response_simple):
        """Test system prompt construction with conversation history"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_simple

        history = "User: Hello\nAssistant: Hi there!"
        ai_generator.generate_response(query="Test query", conversation_history=history)

        call_args = ai_generator.client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args["system"]
        assert "User: Hello" in call_args["system"]
        assert AIGenerator.SYSTEM_PROMPT in call_args["system"]

    def test_api_parameters_validation(self, ai_generator, mock_anthropic_response_simple):
        """Test that all API parameters are set correctly"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_simple

        ai_generator.generate_response(query="Test")

        call_args = ai_generator.client.messages.create.call_args[1]

        # Verify all required parameters
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert "messages" in call_args
        assert "system" in call_args

    def test_tool_choice_parameter_with_tools(self, ai_generator, mock_anthropic_response_simple):
        """Test that tool_choice is set to auto when tools are provided"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_simple

        tools = [{"name": "test_tool"}]
        ai_generator.generate_response(query="Test", tools=tools)

        call_args = ai_generator.client.messages.create.call_args[1]
        assert call_args["tool_choice"] == {"type": "auto"}

    def test_tool_choice_parameter_without_tools(self, ai_generator, mock_anthropic_response_simple):
        """Test that tool_choice is not set when tools are not provided"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_simple

        ai_generator.generate_response(query="Test")

        call_args = ai_generator.client.messages.create.call_args[1]
        assert "tool_choice" not in call_args

    def test_response_text_extraction(self, ai_generator):
        """Test extraction of text from response content"""
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        text_block = Mock()
        text_block.text = "This is the response text"
        mock_response.content = [text_block]

        ai_generator.client.messages.create.return_value = mock_response

        result = ai_generator.generate_response(query="Test")

        assert result == "This is the response text"

    def test_multiple_sequential_calls(self, ai_generator, mock_anthropic_response_simple):
        """Test that multiple sequential generate_response calls work correctly"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_simple

        result1 = ai_generator.generate_response(query="First query")
        result2 = ai_generator.generate_response(query="Second query")

        assert result1 == "This is a simple response from Claude."
        assert result2 == "This is a simple response from Claude."
        assert ai_generator.client.messages.create.call_count == 2


class TestSequentialToolCalling:
    """Test cases for sequential tool calling (multi-round) behavior"""

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AIGenerator instance with mocked client"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
            generator.client = mock_anthropic_client
            return generator

    def test_two_rounds_sequential_different_tools(
        self, ai_generator, mock_two_round_responses, mock_tool_manager
    ):
        """Test two sequential rounds with different tools (search then outline)"""
        ai_generator.client.messages.create.side_effect = mock_two_round_responses

        tools = [
            {"name": "search_course_content", "description": "Search tool"},
            {"name": "get_course_outline", "description": "Outline tool"}
        ]

        result = ai_generator.generate_response(
            query="What is in lesson 1 of Testing Fundamentals?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify 3 API calls: round 1, round 2, final
        assert ai_generator.client.messages.create.call_count == 3

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="testing basics")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Testing Fundamentals")

        # Verify final response
        assert result == "Final answer synthesizing tool results"

    def test_two_rounds_same_tool(self, ai_generator, mock_tool_manager):
        """Test two rounds using the same tool with different parameters"""
        # Create responses for same tool used twice
        round1 = Mock(stop_reason="tool_use")
        tool_block_1 = Mock(spec=['type', 'name', 'id', 'input'])
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "course A"}
        round1.content = [tool_block_1]

        round2 = Mock(stop_reason="tool_use")
        tool_block_2 = Mock(spec=['type', 'name', 'id', 'input'])
        tool_block_2.type = "tool_use"
        tool_block_2.name = "search_course_content"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"query": "course B"}
        round2.content = [tool_block_2]

        final = Mock(stop_reason="end_turn")
        final.content = [Mock(text="Comparison of both courses")]

        ai_generator.client.messages.create.side_effect = [round1, round2, final]

        tools = [{"name": "search_course_content", "description": "Search tool"}]
        result = ai_generator.generate_response(
            query="Compare courses A and B",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify both searches executed with different parameters
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="course A")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="course B")

        assert result == "Comparison of both courses"

    def test_max_rounds_enforcement(
        self, ai_generator, mock_max_rounds_responses, mock_tool_manager
    ):
        """Test that system enforces maximum 2 rounds and makes final call"""
        ai_generator.client.messages.create.side_effect = mock_max_rounds_responses

        tools = [{"name": "search_course_content", "description": "Search tool"}]
        result = ai_generator.generate_response(
            query="Complex query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify exactly 3 API calls: 2 tool rounds + 1 final without tools
        assert ai_generator.client.messages.create.call_count == 3

        # Verify only 2 tool executions (respects max rounds)
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify final response is returned
        assert result == "Final answer synthesizing tool results"

    def test_early_termination_round_one(self, ai_generator, mock_anthropic_response_simple):
        """Test early termination when Claude doesn't use tools in round 1"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_simple

        tools = [{"name": "search_course_content", "description": "Search tool"}]
        result = ai_generator.generate_response(
            query="What is 2+2?",
            tools=tools
        )

        # Verify only 1 API call (Claude answered directly)
        assert ai_generator.client.messages.create.call_count == 1

        # Verify response returned immediately
        assert result == "This is a simple response from Claude."

    def test_early_termination_round_two(
        self, ai_generator, mock_early_termination_responses, mock_tool_manager
    ):
        """Test early termination when Claude uses tools once then answers directly"""
        ai_generator.client.messages.create.side_effect = mock_early_termination_responses

        tools = [{"name": "search_course_content", "description": "Search tool"}]
        result = ai_generator.generate_response(
            query="Simple question",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify 2 API calls: round 1 with tool, round 2 direct answer
        assert ai_generator.client.messages.create.call_count == 2

        # Verify only 1 tool execution
        assert mock_tool_manager.execute_tool.call_count == 1

        # Verify final response
        assert result == "Final answer synthesizing tool results"

    def test_multiple_tools_single_round(self, ai_generator, mock_tool_manager):
        """Test multiple tools called in a single round"""
        # Create response with multiple tool blocks
        round1 = Mock(stop_reason="tool_use")
        tool_block_1 = Mock(spec=['type', 'name', 'id', 'input'])
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "testing"}

        tool_block_2 = Mock(spec=['type', 'name', 'id', 'input'])
        tool_block_2.type = "tool_use"
        tool_block_2.name = "get_course_outline"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"course_name": "Test"}

        round1.content = [tool_block_1, tool_block_2]

        final = Mock(stop_reason="end_turn")
        final.content = [Mock(text="Combined results")]

        ai_generator.client.messages.create.side_effect = [round1, final]

        tools = [
            {"name": "search_course_content", "description": "Search tool"},
            {"name": "get_course_outline", "description": "Outline tool"}
        ]
        result = ai_generator.generate_response(
            query="Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify both tools executed in same round
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify counts as 1 round (2 API calls total)
        assert ai_generator.client.messages.create.call_count == 2

    def test_message_history_accumulation(self, ai_generator, mock_two_round_responses, mock_tool_manager):
        """Test that message history is built correctly across rounds"""
        ai_generator.client.messages.create.side_effect = mock_two_round_responses

        tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]
        ai_generator.generate_response(
            query="Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Check the final API call's messages structure
        final_call_args = ai_generator.client.messages.create.call_args_list[2][1]
        messages = final_call_args["messages"]

        # Should have: user, assistant_1, user_results_1, assistant_2, user_results_2
        assert len(messages) == 5

        # Verify role alternation
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

        # Verify first message is original query
        assert messages[0]["content"] == "Test query"

        # Verify tool results are properly formatted
        assert isinstance(messages[2]["content"], list)
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[4]["content"][0]["type"] == "tool_result"

    def test_no_tool_manager_provided(self, ai_generator, mock_anthropic_response_with_tool):
        """Test graceful handling when tool_manager is None but Claude wants to use tools"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_with_tool

        tools = [{"name": "search_course_content"}]
        result = ai_generator.generate_response(
            query="Test query",
            tools=tools,
            tool_manager=None  # No tool manager
        )

        # Should terminate immediately when tool_manager is None
        assert ai_generator.client.messages.create.call_count == 1

        # Since we can't execute tools, returns empty (handled by content[0].text)
        # In real scenario, this would be a TextBlock, but mock returns ToolUseBlock
        # The actual implementation would handle this gracefully
