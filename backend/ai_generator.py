from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for searching course information.

Tool Usage Guidelines:
- **Course Outline Tool** (`get_course_outline`): Use for questions about:
  - Course structure or organization
  - What lessons are in a course
  - How many lessons a course has
  - Course metadata (title, instructor, link)
  - Complete lesson listings
  - Returns: Course title, course link, instructor, and full lesson list with numbers and titles

- **Content Search Tool** (`search_course_content`): Use for questions about:
  - Specific concepts or topics within course materials
  - Detailed educational content
  - Answers to technical questions from course lessons
  - Returns: Relevant content excerpts from courses/lessons

- **Sequential Tool Usage**:
  - You may use tools across up to 2 rounds to gather comprehensive information
  - Use multiple rounds when you need information from one tool to inform another tool call
  - Examples of valid sequential use:
    * Round 1: Get course outline to identify lesson titles
    * Round 2: Search for content based on the lesson information
  - After gathering information, synthesize into a comprehensive answer

- Synthesize tool results into accurate, fact-based responses
- If no results are found, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline/structure questions**: Use the outline tool first, then answer, including the course title, course linnk, and complete lesson list in a well-formatted manner
- **Course content questions**: Use the search tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool usage explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 rounds of sequential tool calling.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize message history with user query
        messages = [{"role": "user", "content": query}]

        # Maximum tool rounds allowed
        MAX_TOOL_ROUNDS = 2

        # Tool calling loop - supports up to 2 sequential rounds
        for round_num in range(MAX_TOOL_ROUNDS):
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Termination condition 1: Claude didn't use tools - we're done
            if response.stop_reason != "tool_use":
                return response.content[0].text

            # Termination condition 2: No tool manager - can't execute tools
            if not tool_manager:
                return response.content[0].text

            # Execute tools for this round
            messages.append({"role": "assistant", "content": response.content})
            tool_results = self._execute_tools(response, tool_manager)

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        # If we exhausted all rounds, make final call without tools to get answer
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _execute_tools(self, response, tool_manager) -> List[Dict[str, Any]]:
        """
        Execute all tool calls from a response and return formatted results.

        Args:
            response: The API response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            List of tool_result dictionaries formatted for API
        """
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        return tool_results
