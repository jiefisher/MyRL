import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ParsedToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    raw_text: str
    start_pos: int  # character position in text
    end_pos: int


class ToolCallParser:
    """Parses LLM output to extract tool calls. Supports multiple formats."""

    def __init__(self, format: str = "qwen3"):
        self.format = format
        self._parsers = {
            "qwen3": self._parse_qwen3,
            "react": self._parse_react,
            "function_call": self._parse_function_call,
        }

    def parse(self, text: str) -> Optional[ParsedToolCall]:
        """Parse text for a tool call. Returns None if no tool call found (final answer)."""
        parser = self._parsers.get(self.format)
        if parser is None:
            raise ValueError(f"Unknown tool call format: {self.format}")
        return parser(text)

    def is_final_answer(self, text: str) -> bool:
        """Check if the text is a final answer (no tool call)."""
        return self.parse(text) is None

    def _parse_qwen3(self, text: str) -> Optional[ParsedToolCall]:
        """Parse Qwen3 <tool_call> format.

        Expected format:
        <tool_call>
        {"name": "tool_name", "arguments": {"key": "value"}}
        </tool_call>
        """
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            return None
        try:
            call_data = json.loads(match.group(1))
            tool_name = call_data.get("name", "")
            arguments = call_data.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return ParsedToolCall(
                tool_name=tool_name,
                arguments=arguments,
                raw_text=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _parse_react(self, text: str) -> Optional[ParsedToolCall]:
        """Parse ReAct format.

        Expected format:
        Action: tool_name
        Action Input: {"key": "value"}
        """
        action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', text)
        input_match = re.search(r'Action Input:\s*(.+?)(?:\n|$)', text, re.DOTALL)
        if action_match is None:
            return None
        tool_name = action_match.group(1).strip()
        arguments = {}
        if input_match:
            try:
                arguments = json.loads(input_match.group(1).strip())
            except json.JSONDecodeError:
                arguments = {"input": input_match.group(1).strip()}

        start_pos = action_match.start()
        end_pos = input_match.end() if input_match else action_match.end()
        return ParsedToolCall(
            tool_name=tool_name,
            arguments=arguments,
            raw_text=text[start_pos:end_pos],
            start_pos=start_pos,
            end_pos=end_pos,
        )

    def _parse_function_call(self, text: str) -> Optional[ParsedToolCall]:
        """Parse function_call JSON format.

        Expected format:
        ```function_call
        {"name": "tool_name", "arguments": {"key": "value"}}
        ```
        """
        pattern = r'```function_call\s*\n(\{.*?\})\s*\n```'
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            return None
        try:
            call_data = json.loads(match.group(1))
            return ParsedToolCall(
                tool_name=call_data.get("name", ""),
                arguments=call_data.get("arguments", {}),
                raw_text=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
            )
        except (json.JSONDecodeError, KeyError):
            return None
